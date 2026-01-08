# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2025 Oxford Quantum Circuits Ltd

from math import pi

from compiler_config.config import TketOptimizations
from pytket import Bit, Circuit, Qubit
from pytket._tket.architecture import Architecture
from pytket._tket.predicates import MaxNQubitsPredicate
from pytket.circuit import Node
from pytket.passes import SequencePass
from pytket.placement import Placement
from pytket.qasm import circuit_to_qasm_str
from pytket.qasm.qasm import QASMUnsupportedError
from sympy import pi as sympy_pi
from sympy import sympify

from qat.ir.instruction_builder import InstructionBuilder, QuantumInstructionBuilder
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.integrations.tket import (
    TketBuilder as LegacyTketBuilder,
)
from qat.purr.integrations.tket import (
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


def run_tket_optimizations(
    circuit: str | Circuit,
    opts,
    hardware: PydHardwareModel,
    return_as_qasm_str: bool = True,
) -> str | QuantumInstructionBuilder:
    """
    Runs tket-based optimizations and modifications given a Pydantic hardware model.
    Routing will always happen no matter the level.

    Will run optimizations in sections if a full suite fails until a minimal subset of
    passing optimizations is found.
    """

    optimiser = TketOptimisationHelper(circuit, opts, hardware)
    if optimiser.circ is None:
        return circuit
    optimiser.run_optimizations()

    if return_as_qasm_str:
        return optimiser.convert_to_qasm_string()
    else:
        return optimiser.circ


class TketOptimisationHelper:
    """
    Helper class to run pydantic tket optimizations
    """

    def __init__(
        self, circuit: Circuit | str, opts: TketOptimizations, hardware: PydHardwareModel
    ):
        self.hardware = hardware
        self.opts = opts
        if isinstance(circuit, str):
            self.circ = self._get_circuit_from_qasm(circuit)
        else:
            # Helps handle cases where more qubits are allocated than needed
            self.circ = circuit
            self.circ.remove_blank_wires()
        self.coupling_qualities = {}
        self.architecture = None
        self.logical_qubit_map = self._get_logical_qubit_map()

    def run_optimizations(self):
        n_qubits = self.circ.n_qubits
        match n_qubits:
            case 1:
                self.run_one_qubit_optimizations()
            case 2:
                self.run_multi_qubit_optimizations(use_1q_quality=True)
            case _:
                self.run_multi_qubit_optimizations()

    def run_one_qubit_optimizations(self):
        qubit_qualities = {}
        for phys_q, log_q in self.logical_qubit_map.items():
            quality = self.hardware.qubit_quality(phys_q)
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
            tket_builder: LegacyTketBuilder = TketQasmParser().parse(
                LegacyTketBuilder(), qasm_string
            )
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
                    qubit_quality = self.hardware.qubit_quality(q)
                    coupled_qubit_quality = self.hardware.qubit_quality(coupled_q)
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


class TketBuilder(InstructionBuilder):
    """Assembles a TKET circuit using the InstructionBuilder interface."""

    def __init__(self, hardware_model: PydHardwareModel):
        super().__init__(hardware_model)
        n_qubits = len(hardware_model.qubits)
        self.circuit = Circuit(n_qubits)
        self._output_variables: dict[int, str] = {}
        self._bit_ctr = 0

    def get_physical_qubit(self, index: int) -> Qubit:
        if index >= len(self.circuit.qubits):
            raise IndexError(
                f"Qubit index {index} out of range for circuit with "
                f"{len(self.circuit.qubits)} qubits."
            )
        return self.circuit.qubits[index]

    def get_logical_qubit(self, index: int) -> Qubit:
        """The TketBuilder doesn't distinguish between physical and logical qubits, so this
        is identical to get_physical_qubit."""
        return self.get_physical_qubit(index)

    @property
    def qubits(self) -> list[Qubit]:
        return self.circuit.qubits

    def X(self, qubit: Qubit, theta: float | None = None):
        if theta is not None:
            self.circuit.Rx(self._normalize_angle(theta), qubit)
        else:
            self.circuit.X(qubit)

    def Y(self, qubit: Qubit, theta: float | None = None):
        if theta is not None:
            self.circuit.Ry(self._normalize_angle(theta), qubit)
        else:
            self.circuit.Y(qubit)
        return self

    def Z(self, qubit: Qubit, theta: float | None = None):
        if theta is not None:
            self.circuit.Rz(self._normalize_angle(theta), qubit)
        else:
            self.circuit.Z(qubit)
        return self

    def U(self, qubit: Qubit, theta: float, phi: float, lamd: float):
        self.circuit.U3(
            self._normalize_angle(theta),
            self._normalize_angle(phi),
            self._normalize_angle(lamd),
            qubit,
        )
        return self

    def swap(self, qubit1: Qubit, qubit2: Qubit):
        self.circuit.SWAP(qubit1, qubit2)
        return self

    def controlled(self, *args):
        raise NotImplementedError("Does not yet support custom control gates.")

    def cX(self, control: Qubit, target: Qubit, theta: float = pi):
        self.circuit.CRx(self._normalize_angle(theta), control, target)
        return self

    def cY(self, control: Qubit, target: Qubit, theta: float = pi):
        self.circuit.CRy(self._normalize_angle(theta), control, target)
        return self

    def cZ(self, control: Qubit, target: Qubit, theta: float = pi):
        self.circuit.CRz(self._normalize_angle(theta), control, target)
        return self

    def cnot(self, control: Qubit, target: Qubit):
        self.circuit.CX(control, target)
        return self

    def ccnot(self, control1: Qubit, control2: Qubit, target: Qubit):
        self.circuit.CCX(control1, control2, target)
        return self

    def ECR(self, qubit1: Qubit, qubit2: Qubit):
        self.circuit.ECR(qubit1, qubit2)
        return self

    def measure_single_shot_z(
        self, qubit: Qubit, output_variable: str | None = None, **kwargs
    ):
        self._output_variables[self._bit_ctr] = output_variable or str(self._bit_ctr)
        cbit = Bit(self._output_variables[self._bit_ctr], self._bit_ctr)
        self.circuit.add_bit(cbit)
        self.circuit.Measure(qubit, cbit)
        self._bit_ctr += 1
        return self

    def reset(self, qubit: Qubit, **kwargs):
        self.circuit.Reset(qubit)
        return self

    def _normalize_angle(self, theta: float) -> float:
        """Tket likes angles to be in units of pi."""
        return theta / pi


class TketToQatIRConverter:
    """Converts a Tket circuit into Qat IR.

    Essentially the "parser" for Tket circuits.

    .. warning::
        This converter is only intended to be used to convert a TKET circuit into QAT IR
        after being parsed from QIR. It does not account for multiple quantum and classical
        registers, and might give undesired behaviour if used outside of this use case.
    """

    @staticmethod
    def convert_parameter(arg: str):
        r"""A parameter stored in a Tket operation is in units of :math:`\pi`. Parameters
        are returned as a string expression, e.g. sometimes containing multiplication and
        division. These expressions are parsed using sympy."""

        return float(sympy_pi * sympify(arg))

    def convert(
        self, qat_builder: InstructionBuilder, tket_builder: TketBuilder
    ) -> InstructionBuilder:
        """Converts a Tket circuit into Qat IR, adding any necesarry assigns and returns.

        :param circuit: Program as a Tket circuit.
        :param result_format: Specifies how measurement results are formatted.
        """

        for command in tket_builder.circuit.to_dict()["commands"]:
            # Retrieves the qubit / clbit indices for each operation
            args = [arg[1][0] for arg in command["args"]]

            match command["op"]["type"]:
                # One-qubit gates
                case "X":
                    qat_builder.X(qat_builder.get_logical_qubit(args[0]))
                case "Y":
                    qat_builder.Y(qat_builder.get_logical_qubit(args[0]))
                case "Z":
                    qat_builder.Z(qat_builder.get_logical_qubit(args[0]))
                case "H":
                    qat_builder.had(qat_builder.get_logical_qubit(args[0]))
                case "SX":
                    qat_builder.SX(qat_builder.get_logical_qubit(args[0]))
                case "SXdg":
                    qat_builder.SXdg(qat_builder.get_logical_qubit(args[0]))
                case "S":
                    qat_builder.S(qat_builder.get_logical_qubit(args[0]))
                case "Sdg":
                    qat_builder.Sdg(qat_builder.get_logical_qubit(args[0]))
                case "T":
                    qat_builder.T(qat_builder.get_logical_qubit(args[0]))
                case "Tdg":
                    qat_builder.Tdg(qat_builder.get_logical_qubit(args[0]))
                case "Rx":
                    qat_builder.X(
                        qat_builder.get_logical_qubit(args[0]),
                        self.convert_parameter(command["op"]["params"][0]),
                    )
                case "Ry":
                    qat_builder.Y(
                        qat_builder.get_logical_qubit(args[0]),
                        self.convert_parameter(command["op"]["params"][0]),
                    )
                case "Rz":
                    qat_builder.Z(
                        qat_builder.get_logical_qubit(args[0]),
                        self.convert_parameter(command["op"]["params"][0]),
                    )
                case "U1":
                    qat_builder.Z(
                        qat_builder.get_logical_qubit(args[0]),
                        self.convert_parameter(command["op"]["params"][0]),
                    )
                case "U2":
                    qat_builder.U(
                        qat_builder.get_logical_qubit(args[0]),
                        pi / 2,
                        self.convert_parameter(command["op"]["params"][0]),
                        self.convert_parameter(command["op"]["params"][1]),
                    )
                case "U3":
                    qat_builder.U(
                        qat_builder.get_logical_qubit(args[0]),
                        self.convert_parameter(command["op"]["params"][0]),
                        self.convert_parameter(command["op"]["params"][1]),
                        self.convert_parameter(command["op"]["params"][2]),
                    )

                # Two-qubit gates
                case "CX":
                    qat_builder.cnot(
                        qat_builder.get_logical_qubit(args[0]),
                        qat_builder.get_logical_qubit(args[1]),
                    )
                case "ECR":
                    qat_builder.ECR(
                        qat_builder.get_logical_qubit(args[0]),
                        qat_builder.get_logical_qubit(args[1]),
                    )
                case "SWAP":
                    qat_builder.swap(
                        qat_builder.get_logical_qubit(args[0]),
                        qat_builder.get_logical_qubit(args[1]),
                    )

                # Operations
                case "Measure":
                    output_var = tket_builder._output_variables[int(str(args[1]))]
                    qat_builder.measure_single_shot_z(
                        qat_builder.get_logical_qubit(args[0]),
                        output_variable=output_var,
                    )
                case "Barrier":
                    qat_builder.synchronize(
                        [qat_builder.get_logical_qubit(arg) for arg in args]
                    )
                case "Reset":
                    qat_builder.reset(qat_builder.get_logical_qubit(args[0]))
                case _:
                    raise NotImplementedError(
                        f"Command {command['op']['type']} not implemented."
                    )

        qat_builder += tket_builder
        return qat_builder
