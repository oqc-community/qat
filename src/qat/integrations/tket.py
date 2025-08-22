# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2024 Oxford Quantum Circuits Ltd


from compiler_config.config import TketOptimizations
from pytket import Circuit, Qubit
from pytket._tket.architecture import Architecture
from pytket._tket.predicates import MaxNQubitsPredicate
from pytket.circuit import Node
from pytket.passes import SequencePass
from pytket.placement import Placement
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


def run_tket_optimizations(qasm_string, opts, hardware: PydHardwareModel) -> str:
    """
    Runs tket-based optimizations and modifications given a Pydantic hardware model.
    Routing will always happen no matter the level.

    Will run optimizations in sections if a full suite fails until a minimal subset of
    passing optimizations is found.
    """

    optimiser = TketOptimisationHelper(qasm_string, opts, hardware)
    if optimiser.circ is None:
        return qasm_string

    n_qubits = optimiser.circ.n_qubits
    match n_qubits:
        case 1:
            optimiser.run_one_qubit_optimizations()
        case 2:
            optimiser.run_multi_qubit_optimizations(use_1q_quality=True)
        case _:
            optimiser.run_multi_qubit_optimizations()

    return optimiser.convert_to_qasm_string()


class TketOptimisationHelper:
    """
    Helper class to run pydantic tket optimizations
    """

    def __init__(
        self, qasm_string: str, opts: TketOptimizations, hardware: PydHardwareModel
    ):
        self.hardware = hardware
        self.opts = opts
        self.circ = self._get_circuit_from_qasm(qasm_string)
        self.coupling_qualities = {}
        self.architecture = None
        self.logical_qubit_map = self._get_logical_qubit_map()

    def run_one_qubit_optimizations(self):
        qubit_qualities = {}
        for log_q in self.logical_qubit_map.values():
            quality = self.hardware.qubit_quality(log_q)
            qubit_qualities[log_q] = quality if quality != 1 else 0

        best_qubit = max(qubit_qualities, key=qubit_qualities.get)

        q_map = {Qubit(0): Node(best_qubit), Qubit(best_qubit): Node(0)}
        Placement.place_with_map(self.circ, q_map)

    def run_multi_qubit_optimizations(self, use_1q_quality: bool = False):
        optimisations_failed = False
        if TketOptimizations.DefaultMappingPass not in self.opts:
            couplings = self._get_logical_qubit_couplings()
            optimisations_failed = self._optimise_circuit(couplings)
        else:
            self._calculate_multi_qubit_qualities(use_1q_quality)
            sorted_coupling_qualities = dict(
                sorted(
                    self.coupling_qualities.items(), key=lambda item: item[1], reverse=True
                )
            )

            for quality_level in sorted_coupling_qualities.values():
                filtered_couplings = [
                    key
                    for key, quality in self.coupling_qualities.items()
                    if quality >= quality_level
                ]

                subgraphs_nodes, subgraphs_edges = get_connected_subgraphs(
                    filtered_couplings
                )
                for nodes, edges in zip(subgraphs_nodes, subgraphs_edges):
                    if self.circ.n_qubits <= len(nodes):
                        if not (optimisations_failed := self._optimise_circuit(edges)):
                            break
                    else:
                        optimisations_failed = True
                if not optimisations_failed:
                    break

        if optimisations_failed:
            self._handle_failed_optimisation()

    def convert_to_qasm_string(self):
        try:
            qasm_string = circuit_to_qasm_str(self.circ)
            log.info(f"Number of gates after tket optimization: {self.circ.n_gates}")
            return qasm_string
        except (QASMUnsupportedError, RuntimeError) as e:
            log.warning(
                f"Error generating QASM from Tket circuit: {_full_stopalize(e)}. "
                "Skipping this optimization pass."
            )

    def _get_logical_qubit_couplings(self):
        couplings = []
        for q, coupled_qs in self.hardware.logical_connectivity.items():
            logical_q = self.logical_qubit_map[q]
            for coupled_q in coupled_qs:
                logical_coupled_q = self.logical_qubit_map[coupled_q]
                couplings.append((logical_q, logical_coupled_q))
        return couplings

    def _get_logical_qubit_map(self):
        """
        Without default remapping pass multi-qubit gates don't get moved around, so
        trying to apply them to a limited subset of qubits provides no value.
        """
        return {q_i: i for i, q_i in enumerate(self.hardware.qubits.keys())}

    @staticmethod
    def _get_circuit_from_qasm(qasm_string: str) -> Circuit | None:
        """
        Runs tket-based optimizations and modifications given a Pydantic hardware model.
        Routing will always happen no matter the level.
        """
        try:
            tket_builder: TketBuilder = TketQasmParser().parse(TketBuilder(), qasm_string)
            circ = tket_builder.circuit
            log.info(f"Number of gates before tket optimization: {circ.n_gates}")
            return circ

        except Exception as e:  # Parsing is too fragile, can cause almost any exception.
            log.warning(
                f"Tket failed during QASM parsing with error: {_full_stopalize(e)}. "
                "Skipping this optimization pass."
            )

    def _calculate_multi_qubit_qualities(self, use_1q_quality: bool = False):
        """
        Calculate couplings and coupling qualities from the hardware model.
        """
        for q, coupled_qs in self.hardware.logical_connectivity.items():
            logical_q = self.logical_qubit_map[q]
            for coupled_q in coupled_qs:
                logical_coupled_q = self.logical_qubit_map[coupled_q]

                quality = self.hardware.logical_connectivity_quality[(q, coupled_q)]
                if use_1q_quality:
                    qubit_quality = self.hardware.qubit_quality(logical_q)
                    coupled_qubit_quality = self.hardware.qubit_quality(logical_coupled_q)
                    quality *= qubit_quality * coupled_qubit_quality
                self.coupling_qualities[(logical_q, logical_coupled_q)] = quality

    def _optimise_circuit(self, graph_edges):
        self.architecture = Architecture(graph_edges)
        return not optimize_circuit(self.circ, self.architecture, self.opts)

    def _handle_failed_optimisation(self):
        """
        Check if the optimizations failed.
        If our optimizations failed, but we want the mapping pass, apply that by itself.
        """
        if self.architecture is None:
            raise ValueError(
                "Unable to resolve hardware instance for fall-back optimizations."
            )

        delay_failed = False
        try:
            # DelayMeasure throws on failure, and we want to raise our own errors for
            # this.
            SequencePass(fetch_default_passes(self.architecture, self.opts)).apply(
                self.circ
            )
        except RuntimeError:
            delay_failed = True

        # If the delay fails, try with a more limited subset.
        if delay_failed:
            try:
                # Tket just throws an exception if the list is none, so skip if that's
                # the case.
                default_passes = fetch_default_passes(
                    self.architecture, self.opts, add_delay=False
                )
                if len(default_passes) > 0:
                    SequencePass(default_passes).apply(self.circ)
            except RuntimeError as e:
                message = str(e)
                if MaxNQubitsPredicate.__name__ in message:
                    raise ValueError(
                        f"Circuit uses {len(self.circ.qubits)} qubits, "
                        f"only {len(self.architecture.nodes)} available."
                    ) from e

                raise e

        apply_default_transforms(self.circ, self.architecture, self.opts)
        check_validity(self.circ, self.architecture)


run_pyd_tket_optimizations = run_tket_optimizations
