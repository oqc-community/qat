# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2024 Oxford Quantum Circuits Ltd


from compiler_config.config import TketOptimizations
from pytket._tket.architecture import Architecture
from pytket._tket.predicates import MaxNQubitsPredicate
from pytket.passes import SequencePass
from pytket.qasm import circuit_to_qasm_str
from pytket.qasm.qasm import QASMUnsupportedError

from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.integrations.tket import (
    TketBuilder,
    TketQasmParser,
    _full_stopalize,
    apply_default_transforms,
    check_validity,
    fetch_default_passes,
    optimize_circuit,
)
from qat.purr.utils.logger import get_default_logger
from qat.utils.graphs import get_connected_subgraphs

log = get_default_logger()


def run_pyd_tket_optimizations(qasm_string, opts, hardware: PydHardwareModel) -> str:
    """
    Runs tket-based optimizations and modifications given a Pydantic hardware model.
    Routing will always happen no matter the level.

    Will run optimizations in sections if a full suite fails until a minimal subset of
    passing optimizations is found.
    """

    try:
        tket_builder: TketBuilder = TketQasmParser().parse(TketBuilder(), qasm_string)
        circ = tket_builder.circuit
        log.info(f"Number of gates before tket optimization: {circ.n_gates}")
    except Exception as e:  # Parsing is too fragile, can cause almost any exception.
        log.warning(
            f"Tket failed during QASM parsing with error: {_full_stopalize(e)}. "
            "Skipping this optimization pass."
        )
        return qasm_string

    couplings = []
    coupling_qualities = []
    optimizations_failed = False
    architecture = None

    # Without default remapping pass multi-qubit gates don't get moved around, so
    # trying to apply them to a limited subset of qubits provides no value.
    logical_qubit_map = {q_i: i for i, q_i in enumerate(hardware.qubits.keys())}

    for q, coupled_qs in hardware.logical_connectivity.items():
        logical_q = logical_qubit_map[q]
        for coupled_q in coupled_qs:
            logical_coupled_q = logical_qubit_map[coupled_q]

            couplings.append((logical_q, logical_coupled_q))
            coupling_qualities.append(hardware.logical_connectivity_quality[(q, coupled_q)])

    if TketOptimizations.DefaultMappingPass not in opts:
        architecture = Architecture(couplings)
        optimizations_failed = not optimize_circuit(circ, architecture, opts)
    else:
        sorted_coupling_qualities = sorted(coupling_qualities, reverse=True)
        for quality_level in sorted_coupling_qualities:
            filtered_couplings = [
                couplings[i]
                for i, quality in enumerate(coupling_qualities)
                if quality >= quality_level
            ]

            subgraphs_nodes, subgraphs_edges = get_connected_subgraphs(filtered_couplings)
            for nodes, edges in zip(subgraphs_nodes, subgraphs_edges):
                if circ.n_qubits <= len(nodes):
                    architecture = Architecture(edges)
                    optimizations_failed = not optimize_circuit(circ, architecture, opts)
                    if not optimizations_failed:
                        break
                else:
                    optimizations_failed = True
            if not optimizations_failed:
                break

    # If our optimizations failed but we want the mapping pass, apply that by itself.
    if optimizations_failed:
        if architecture is None:
            raise ValueError(
                "Unable to resolve hardware instance for fall-back optimizations."
            )

        delay_failed = False
        try:
            # DelayMeasure throws on failure, and we want to raise our own errors for
            # this.
            SequencePass(fetch_default_passes(architecture, opts)).apply(circ)
        except RuntimeError:
            delay_failed = True

        # If the delay fails, try with a more limited subset.
        if delay_failed:
            try:
                # Tket just throws an exception if the list is none, so skip if that's
                # the case.
                default_passes = fetch_default_passes(architecture, opts, add_delay=False)
                if len(default_passes) > 0:
                    SequencePass(default_passes).apply(circ)
            except RuntimeError as e:
                message = str(e)
                if MaxNQubitsPredicate.__name__ in message:
                    raise ValueError(
                        f"Circuit uses {len(circ.qubits)} qubits, "
                        f"only {len(architecture.nodes)} available."
                    ) from e

                raise e

        apply_default_transforms(circ, architecture, opts)
        check_validity(circ, architecture)

    try:
        qasm_string = circuit_to_qasm_str(circ)
        log.info(f"Number of gates after tket optimization: {circ.n_gates}")
    except (QASMUnsupportedError, RuntimeError) as e:
        log.warning(
            f"Error generating QASM from Tket circuit: {_full_stopalize(e)}. "
            "Skipping this optimization pass."
        )

    # TODO: Return result object with more information about compilation/errors
    return qasm_string
