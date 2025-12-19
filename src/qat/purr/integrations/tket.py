# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2025 Oxford Quantum Circuits Ltd
from copy import deepcopy
from numbers import Number
from typing import List, Union

import numpy as np
from compiler_config.config import InlineResultsProcessing, TketOptimizations
from pytket import Bit, Circuit, OpType, Qubit
from pytket._tket.architecture import Architecture, RingArch
from pytket._tket.circuit import CustomGateDef
from pytket._tket.predicates import (
    ConnectivityPredicate,
    DirectednessPredicate,
    MaxNQubitsPredicate,
    NoMidMeasurePredicate,
)
from pytket._tket.transform import Transform
from pytket.circuit import Node
from pytket.passes import (
    AutoRebase,
    CliffordSimp,
    ContextSimp,
    DecomposeArbitrarilyControlledGates,
    DecomposeBoxes,
    DefaultMappingPass,
    FullPeepholeOptimise,
    GlobalisePhasedX,
    KAKDecomposition,
    PeepholeOptimise2Q,
    RemoveBarriers,
    RemoveDiscarded,
    RemoveRedundancies,
    SequencePass,
    SimplifyMeasured,
    SynthesiseTket,
    ThreeQubitSquash,
)
from pytket.placement import Placement
from pytket.qasm import circuit_to_qasm_str
from pytket.qasm.qasm import NOPARAM_COMMANDS, PARAM_COMMANDS, QASMUnsupportedError
from qiskit.circuit.library import CXGate, ECRGate, UGate
from sympy import pi, sympify

from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.execution import InstructionExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Variable
from qat.purr.integrations.qasm import BitRegister, Qasm2Parser, QasmContext, QubitRegister
from qat.purr.integrations.qir import QIRParser
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class TketBuilder:
    """
    Builds a Tket circuit using a builder API. No direct relation to our other builders
    as the API is a little too different.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.circuit = Circuit()

    def qreg(self, name, size):
        self.circuit.add_q_register(name, size)

    def creg(self, name, size):
        self.circuit.add_c_register(name, size)

    def barrier(self, qubits):
        qubits = [qubits] if not isinstance(qubits, List) else qubits
        self.circuit.add_barrier(qubits)

    def ECR(self, qubit1, qubit2, *args):
        self.circuit.ECR(qubit1, qubit2)

    def measure(self, qubits, bits, conditions=None):
        qubits = [qubits] if not isinstance(qubits, List) else qubits
        bits = [bits] if not isinstance(bits, List) else bits
        conditions = conditions or {}
        for qubit, bit in zip(qubits, bits):
            self.circuit.Measure(qubit, bit, **conditions)

    def is_basic_gate(self, name):
        return self._get_basic_gate_def(name) is not None

    def _get_basic_gate_def(self, name):
        return PARAM_COMMANDS.get(name, None) or NOPARAM_COMMANDS.get(name, None)

    def gate(self, name, qubits, params=None, conditions=None):
        target_gate = self._get_basic_gate_def(name)
        if target_gate is None:
            raise ValueError(f"Can't find a definition for {name}.")

        conditions = conditions or {}
        params = params or []
        qubits = [qubits] if not isinstance(qubits, List) else qubits
        params = [params] if not isinstance(params, List) else params
        self.circuit.add_gate(target_gate, params, qubits, **conditions)

    def custom_gate(self, gate_def: CustomGateDef, qubits, params=None, conditions=None):
        conditions = conditions or {}
        params = params or []
        qubits = [qubits] if not isinstance(qubits, List) else qubits
        params = [params] if not isinstance(params, List) else params
        self.circuit.add_custom_gate(gate_def, params, qubits, **conditions)


class TketQasmParser(Qasm2Parser):
    """
    QASM parser that turns QASM into Tket structures. Switch to Tkets QASM parser when it's
    more mature.
    """

    def process_qreg(self, node, context: QasmContext, builder: TketBuilder, **kwargs):
        if node.name in context.registers.quantum:
            return
        context.registers.quantum[node.name] = QubitRegister(
            [Qubit(node.name, val) for val in range(node.size)]
        )
        builder.qreg(node.name, node.size)

    def process_creg(self, node, context, builder: TketBuilder, **kwargs):
        if node.name in context.registers.classic:
            return
        context.registers.classic[node.name] = BitRegister(
            [Bit(node.name, val) for val in range(node.size)]
        )
        builder.creg(node.name, node.size)

    def process_gate_definition(self, node, context, _, **kwargs):
        # Just a declaration, gates are used when CustomUnitary is used
        context.gates[node.name] = node

    def _get_staggered_qubit_params(self, node, context):
        def transform_to_halfturn(num):
            try:
                return sympify(num) / pi
            except Exception:
                raise ValueError("Cannot parse angle: {}".format(num))

        res = self._expand_to_match_registers(
            self._get_qubits(node, context),
            [
                transform_to_halfturn(param) if isinstance(param, Number) else param
                for param in self._get_parameters(node, context)
            ],
        )

        return res

    def process_gate(
        self,
        method,
        context: QasmContext,
        builder: TketBuilder,
        conditions=None,
        **kwargs,
    ):
        # For ease-of-use we just process CX and U as a normal gate, but the QASM nodes
        # don't have names. Infer them in this case.
        if isinstance(method.op, UGate):
            gate_name = "U"
        elif isinstance(method.op, CXGate):
            gate_name = "CX"
        elif isinstance(method.op, ECRGate):
            gate_name = "ECR"
        else:
            gate_name = method.name

        # TODO: Right now we just flatten all gates. Should move to use the
        #   MethodGateDef of Tket, but as that functionality is currently broken we
        #   lose nothing here.
        for qbs, args in self._get_staggered_qubit_params(method, context):
            if builder.is_basic_gate(gate_name):
                builder.gate(gate_name, qbs, args, conditions)
            else:
                super().process_gate(
                    method, context, builder, conditions=conditions, **kwargs
                )

    def process_if(self, node, context, builder: TketBuilder, *kwargs):
        left = self._resolve_variable(node.children[0], context)
        right = self._resolve_value(node.children[1], context)

        if isinstance(left, BitRegister):
            left = left.bits
        else:
            left = [left]

        conditions = {"condition_bits": left, "condition_value": right}
        self.walk_node(node.children[2], context, builder, conditions=conditions)

    def process_reset(self, node, context, builder, conditions=None, **kwargs):
        for qubits, _ in self._get_staggered_qubit_params(node, context):
            builder.gate("reset", qubits, conditions)

    def process_measure(
        self,
        node,
        context: QasmContext,
        builder: TketBuilder,
        conditions=None,
        **kwargs,
    ):
        for qubits, bits in self._get_staggered_qubit_params(node, context):
            builder.measure(qubits, bits, conditions)

    def process_unitary(
        self, node, context, builder: TketBuilder, conditions=None, **kwargs
    ):
        self.process_gate(node, context, builder, conditions, **kwargs)

    def process_cnot(self, node, context, builder, conditions=None, **kwargs):
        self.process_gate(node, context, builder, conditions, **kwargs)

    def process_barrier(self, node, context, builder: TketBuilder, **kwargs):
        barrier_qubits = []
        for qubits, _ in self._get_staggered_qubit_params(node, context):
            if isinstance(qubits, List):
                barrier_qubits.extend(qubits)
            else:
                barrier_qubits.append(qubits)

        builder.barrier(barrier_qubits)

    def process_program(self, builder, qasm):
        self._walk_program(builder, qasm, QasmContext())
        return builder


class TketQIRParser(QIRParser):
    """QIR parser than turns circuits into Tket structures."""

    def __init__(self, hardware: Union[QuantumHardwareModel, InstructionExecutionEngine]):
        if isinstance(hardware, InstructionExecutionEngine):
            hardware = hardware.model

        self.hardware: QuantumHardwareModel = hardware
        self.circuit: Circuit = Circuit(len(hardware.qubits), len(hardware.qubits))
        self.results_format = InlineResultsProcessing.Program
        self.result_variables = []

    def _get_qubit(self, id_: int):
        return self.hardware.qubits[id_]

    @staticmethod
    def normalize_parameter(param):
        return param / np.pi

    def ccx(self, control1, control2, target):
        self.circuit.CCX(control1, control2, target)

    def cx(self, control: int, target: int):
        self.circuit.CX(control, target)

    def cz(self, control: int, target: int):
        self.circuit.CZ(control, target)

    def h(self, target: int):
        self.circuit.H(target)

    def mz(self, qubit: int, target: int):
        self.circuit.Measure(qubit, target)

    def reset(self, target: int):
        self.circuit.Reset(target)

    def rx(self, theta: float, qubit: int):
        self.circuit.Rx(self.normalize_parameter(theta), qubit)

    def ry(self, theta: float, qubit: int):
        self.circuit.Ry(self.normalize_parameter(theta), qubit)

    def rz(self, theta: float, qubit: int):
        self.circuit.Rz(self.normalize_parameter(theta), qubit)

    def s(self, qubit: int):
        self.circuit.S(qubit)

    def s_adj(self, qubit: int):
        self.circuit.Sdg(qubit)

    def t(self, qubit: int):
        self.circuit.T(qubit)

    def t_adj(self, qubit: int):
        self.circuit.Tdg(qubit)

    def x(self, qubit: int):
        self.circuit.X(qubit)

    def y(self, qubit: int):
        self.circuit.Y(qubit)

    def z(self, qubit: int):
        self.circuit.Z(qubit)

    def returns(self, result_name=None):
        """Returns are dealt with upon converting back to QatIR."""
        pass

    def assign(self, name, value):
        """Assigns are dealt with upon converting back to QatIR."""
        pass

    @property
    def builder(self):
        """Returns the circuit and variables for where results are stored in Qat IR.

        This property overwrites the "builder" in the :class:`QIRParser` so that the circuit
        and output variables are returned in its place. It also removes unused qubits which
        can greatly increase the effectiveness of the placement of logical-to-physical
        qubits.
        """

        self.circuit.remove_blank_wires()
        return self.circuit, self.result_variables

    @builder.setter
    def builder(self, _):
        """The QIR parser resets the builder after being parsed (possibly for reuse). This
        setter is just to reset."""

        self.circuit = Circuit(len(self.hardware.qubits), len(self.hardware.qubits))


class TketToQatIRConverter:
    """Converts a Tket circuit into Qat IR.

    .. warning::
        This converter is only intended to be used to convert a TKET circuit into QAT IR
        after being parsed from QIR. It does not account for multiple quantum and classical
        registers, and might give undesired behaviour if used outside of this use case.
    """

    def __init__(self, model: QuantumHardwareModel):
        self.model = model
        self.builder = model.create_builder()
        self.output_variables = []

    def get_qubit(self, index: int):
        """Maps a Tket logical qubit index onto a physical qubit."""
        return self.model.qubits[index]

    @staticmethod
    def convert_parameter(arg: str):
        r"""A parameter stored in a Tket operation is in units of :math:`\pi`. Parameters
        are returned as a string, sometimes in fractional form: we need to convert it to an
        absolute value."""

        if "/" in arg:
            a, b = arg.split("/")
            arg = float(a) / float(b)
        else:
            arg = float(arg)
        return np.pi * arg

    def convert(self, circuit: Circuit):
        """Converts a Tket circuit into Qat IR, adding any necesarry assigns and returns.

        :param circuit: Program as a Tket circuit.
        :param result_format: Specifies how measurement results are formatted.
        """

        for command in circuit.to_dict()["commands"]:
            # Retrieves the qubit / clbit indices for each operation
            args = [arg[1][0] for arg in command["args"]]

            match command["op"]["type"]:
                # One-qubit gates
                case "X":
                    self.builder.X(self.get_qubit(args[0]))
                case "Y":
                    self.builder.Y(self.get_qubit(args[0]))
                case "Z":
                    self.builder.Z(self.get_qubit(args[0]))
                case "H":
                    self.builder.had(self.get_qubit(args[0]))
                case "SX":
                    self.builder.SX(self.get_qubit(args[0]))
                case "SXdg":
                    self.builder.SXdg(self.get_qubit(args[0]))
                case "S":
                    self.builder.S(self.get_qubit(args[0]))
                case "Sdg":
                    self.builder.Sdg(self.get_qubit(args[0]))
                case "T":
                    self.builder.T(self.get_qubit(args[0]))
                case "Tdg":
                    self.builder.Tdg(self.get_qubit(args[0]))
                case "Rx":
                    self.builder.X(
                        self.get_qubit(args[0]),
                        self.convert_parameter(command["op"]["params"][0]),
                    )
                case "Ry":
                    self.builder.Y(
                        self.get_qubit(args[0]),
                        self.convert_parameter(command["op"]["params"][0]),
                    )
                case "Rz":
                    self.builder.Z(
                        self.get_qubit(args[0]),
                        self.convert_parameter(command["op"]["params"][0]),
                    )
                case "U1":
                    self.builder.Z(
                        self.get_qubit(args[0]),
                        self.convert_parameter(command["op"]["params"][0]),
                    )
                case "U2":
                    self.builder.U(
                        self.get_qubit(args[0]),
                        np.pi / 2,
                        self.convert_parameter(command["op"]["params"][0]),
                        self.convert_parameter(command["op"]["params"][1]),
                    )
                case "U3":
                    self.builder.U(
                        self.get_qubit(args[0]),
                        self.convert_parameter(command["op"]["params"][0]),
                        self.convert_parameter(command["op"]["params"][1]),
                        self.convert_parameter(command["op"]["params"][2]),
                    )

                # Two-qubit gates
                case "CX":
                    self.builder.cnot(self.get_qubit(args[0]), self.get_qubit(args[1]))
                case "ECR":
                    self.builder.ECR(self.get_qubit(args[0]), self.get_qubit(args[1]))
                case "SWAP":
                    self.builder.swap(self.get_qubit(args[0]), self.get_qubit(args[1]))

                # Operations
                case "Measure":
                    output_var = str(args[1])
                    self.builder.measure_single_shot_z(
                        self.get_qubit(args[0]), output_variable=output_var
                    )
                    self.output_variables.append(output_var)
                case "Barrier":
                    self.builder.synchronize([self.get_qubit(arg) for arg in args])
                case "Reset":
                    # Reset operation not implemented: do nothing instead of throwing an
                    # error to maintain support with non-optimised QIR.
                    continue
                case _:
                    raise NotImplementedError(
                        f"Command {command['op']['type']} not implemented."
                    )

        builder = self.builder
        self.builder = self.model.create_builder()
        return builder


def fetch_default_passes(architecture, opts, pass_list: List = None, add_delay=True):
    pass_list = pass_list or []
    if TketOptimizations.DefaultMappingPass in opts:
        pass_list.append(DefaultMappingPass(architecture, add_delay))

    return pass_list


def _full_stopalize(e):
    exception = str(e)
    return exception if exception.endswith(".") else f"{exception}."


def apply_default_transforms(circuit, architecture, opts):
    transform_list = []
    if TketOptimizations.DirectionalCXGates in opts:
        transform_list.append(Transform.DecomposeCXDirected(architecture))
        transform_list.append(Transform.RemoveRedundancies())

    for transform in transform_list:
        transform.apply(circuit)


def check_validity(circuit, architecture):
    predicates = [
        (
            "Failed to apply circuit to hardware topology.",
            ConnectivityPredicate(architecture),
        ),
        (
            "Failed to satisfy 2QB gate constraints.",
            DirectednessPredicate(architecture),
        ),
        ("No mid-circuit measurements allowed.", NoMidMeasurePredicate()),
    ]

    failed_preds = [message for message, pred in predicates if not pred.verify(circuit)]
    if any(failed_preds):
        raise ValueError(f"Verification failed: {' '.join(failed_preds)}")


def optimize_circuit(circ, architecture, opts):
    """
    Apply all Tket optimizations to the circuit provided as defined in the options flag.
    """
    try:
        # By default we want to decompose Tket-specific circuits and then rebase into
        # the form that Tket can optimize most efficiently.
        passes = [DecomposeBoxes(), SynthesiseTket()]

        if TketOptimizations.ContextSimp in opts:
            passes.append(ContextSimp(allow_classical=True))

        # TODO: FullPeepholeOptimise fails due to bug in tket. Find out what's wrong and
        #   fix/flag.
        if TketOptimizations.FullPeepholeOptimise in opts:
            passes.append(FullPeepholeOptimise())

        if TketOptimizations.CliffordSimp in opts:
            passes.append(CliffordSimp(allow_swaps=True))

        if TketOptimizations.DecomposeArbitrarilyControlledGates in opts:
            passes.append(DecomposeArbitrarilyControlledGates())

        # if TketOptimizations.EulerAngleReduction in opts:
        #     passes.append(EulerAngleReduction())

        if TketOptimizations.GlobalisePhasedX in opts:
            passes.append(GlobalisePhasedX())

        # if TketOptimizations.GuidedPauliSimp in opts:
        #     passes.append(GuidedPauliSimp())

        if TketOptimizations.KAKDecomposition in opts:
            passes.append(KAKDecomposition())

        # if TketOptimizations.OptimisePhaseGadgets in opts:
        #     passes.append(OptimisePhaseGadgets())
        #
        # if TketOptimizations.PauliSimp in opts:
        #     passes.append(PauliSimp())
        #
        # if TketOptimizations.PauliSquash in opts:
        #     passes.append(PauliSquash())

        if TketOptimizations.PeepholeOptimise2Q in opts:
            passes.append(PeepholeOptimise2Q())

        if TketOptimizations.RemoveDiscarded in opts:
            passes.append(RemoveDiscarded())

        if TketOptimizations.RemoveBarriers in opts:
            passes.append(RemoveBarriers())

        if TketOptimizations.RemoveRedundancies in opts:
            passes.append(RemoveRedundancies())

        if TketOptimizations.ThreeQubitSquash in opts:
            passes.append(ThreeQubitSquash())

        if TketOptimizations.SimplifyMeasured in opts:
            passes.append(SimplifyMeasured())

        # TODO: Use HW model object to create fetch calibration data for optimal
        #   routing.
        passes = fetch_default_passes(architecture, opts, passes)

        passes += [AutoRebase({OpType.CX, OpType.U3})]

        # Not everything in the list is a pass, we've also got transforms.
        # Everything in the list should have an apply function though.
        SequencePass(passes, strict=False).apply(circ)
        apply_default_transforms(circ, architecture, opts)
        check_validity(circ, architecture)

        log.info("Tket optimizations complete.")
        return True
    except (ValueError, IndexError, RuntimeError) as e:
        log.warning(
            f"Tket could not compile circuit due do exception: {_full_stopalize(e)}. "
            "Skipping this optimization pass."
        )
        return False


def get_coupling_subgraphs(couplings):
    """
    Given a list of connections which potentially describe a disconnected graph,
    this returns a list of connected subgraphs
    """
    subgraphs = []
    for coupling in couplings:
        subgraph_added_to = None
        for subgraph in subgraphs:
            # check whether coupling joins to subgraph
            if set(coupling) & {qubit for connection in subgraph for qubit in connection}:
                if subgraph_added_to is None:
                    subgraph.append(coupling)  # add the coupling to the subgraph
                    subgraph_added_to = subgraph
                else:
                    subgraph_added_to.extend(
                        subgraph
                    )  # if coupling joins to 2nd subgraph, the subgraphs combine
                    subgraphs.remove(subgraph)
        if (
            subgraph_added_to is None
        ):  # if coupling doesn't join to any of the current subgraphs, make it a new one
            subgraphs.append([coupling])
    return subgraphs


def run_1Q_tket_optimizations(circ: Circuit, hardware: QuantumHardwareModel) -> Circuit:
    logical_qubit_map = {q.index: i for i, q in enumerate(hardware.qubits)}

    qubit_qualities = {}
    for phys_q, log_q in logical_qubit_map.items():
        qubit_qualities[log_q] = (
            hardware.qubit_quality(phys_q) if hardware.qubit_quality(phys_q) != 1 else 0
        )
    best_qubit = max(qubit_qualities, key=qubit_qualities.get)

    q_map = {Qubit(0): Node(best_qubit), Qubit(best_qubit): Node(0)}
    Placement.place_with_map(circ, q_map)

    return circ


def run_multiQ_tket_optimizations(
    circ: Circuit, opts, hardware: QuantumHardwareModel, use_1Q_quality: bool = False
) -> Circuit:
    couplings = deepcopy(hardware.qubit_direction_couplings)
    optimizations_failed = False
    architecture = None
    if any(couplings):
        if TketOptimizations.DefaultMappingPass not in opts:
            architecture = Architecture([val.direction for val in couplings])
            optimizations_failed = not optimize_circuit(circ, architecture, opts)

        else:
            # Without default remapping pass multi-qubit gates don't get moved around, so
            # trying to apply them to a limited subset of qubits provides no value.
            logical_qubit_map = {q.index: i for i, q in enumerate(hardware.qubits)}
            for coupling in couplings:
                control_ind, target_ind = coupling.direction
                coupling.direction = (
                    logical_qubit_map[control_ind],
                    logical_qubit_map[target_ind],
                )
                if use_1Q_quality:
                    coupling.quality *= hardware.qubit_quality(
                        control_ind
                    ) * hardware.qubit_quality(target_ind)
            else:
                coupling_qualities = list({val.quality for val in couplings})
            coupling_qualities.sort(reverse=True)

            for quality_level in coupling_qualities:
                filtered_couplings = [
                    val.direction for val in couplings if val.quality >= quality_level
                ]
                coupling_subgraphs = get_coupling_subgraphs(filtered_couplings)

                for subgraph in coupling_subgraphs:
                    if circ.n_qubits <= len(
                        set([qubit for coupling in subgraph for qubit in coupling])
                    ):
                        architecture = Architecture(subgraph)
                        optimizations_failed = not optimize_circuit(
                            circ, architecture, opts
                        )
                        if not optimizations_failed:
                            break
                    else:
                        optimizations_failed = True
                if not optimizations_failed:
                    break
    else:
        architecture = RingArch(len(hardware.qubits))
        optimizations_failed = not optimize_circuit(circ, architecture, opts)

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
                    )

                raise e

        apply_default_transforms(circ, architecture, opts)
        check_validity(circ, architecture)
    return circ


def run_tket_optimizations(circ: Circuit, opts, hardware: QuantumHardwareModel) -> Circuit:
    """
    Runs tket-based optimizations and modifications. Routing will always happen no
    matter the level.

    Will run optimizations in sections if a full suite fails until a minimal subset of
    passing optimizations is found.

    :param circ: Tket circuit to optimize. The source file must be already parsed to a
        TKET circuit.
    :param opts: Specifies which TKET optimizations to run.
    :param hardware: The hardware model is used for routing and placement purposes.
    """
    if circ.n_qubits == 1 and hardware.error_mitigation:
        return run_1Q_tket_optimizations(circ, hardware)
    elif circ.n_qubits == 2 and hardware.error_mitigation:
        return run_multiQ_tket_optimizations(circ, opts, hardware, use_1Q_quality=True)
    else:
        return run_multiQ_tket_optimizations(circ, opts, hardware, use_1Q_quality=False)


def run_tket_optimizations_qasm(qasm_string, opts, hardware: QuantumHardwareModel) -> str:
    """
    Runs tket-based optimizations and modifications. Routing will always happen no
    matter the level.

    Will run optimizations in sections if a full suite fails until a minimal subset of
    passing optimizations is found.

    :param qasm_string: The circuit as a QASM string.
    :param opts: Specifies which TKET optimizations to run.
    :param hardware: The hardware model is used for routing and placement purposes.
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

    circ = run_tket_optimizations(circ, opts, hardware)

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


def run_tket_optimizations_qir(
    file_or_str,
    opts,
    hardware: QuantumHardwareModel,
    results_format: InlineResultsProcessing = None,
) -> InstructionBuilder:
    """
    Runs tket-based optimizations and modifications. Routing will always happen no
    matter the level.

    Will run optimizations in sections if a full suite fails until a minimal subset of
    passing optimizations is found.

    :param file_or_str: The QIR program as a string, or its file path.
    :param opts: Specifies which TKET optimizations to run.
    :param hardware: The hardware model is used for routing and placement purposes.
    """
    results_format = results_format or InlineResultsProcessing.Program
    circ, output_variables = TketQIRParser(hardware).parse(file_or_str)
    log.info(f"Number of gates before tket optimization: {circ.n_gates}")

    circ = run_tket_optimizations(circ, opts, hardware)

    builder = TketToQatIRConverter(hardware).convert(circ)
    for output_variable in output_variables:
        builder.results_processing(output_variable[0].name, results_format)
    if any(output_variables):
        potential_names = [val[1] for val in output_variables if len(val[1] or "") != 0]
        if not any(potential_names):
            result_name = Variable.generate_name()
        else:
            result_name = "_".join(potential_names)
        builder.assign(result_name, [val[0] for val in output_variables])
        builder.returns(result_name)
    else:
        builder.returns()
    log.info(f"Number of gates after tket optimization: {circ.n_gates}")
    return builder
