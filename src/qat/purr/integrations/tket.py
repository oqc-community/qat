# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from copy import deepcopy
from numbers import Number
from typing import List

from compiler_config.config import TketOptimizations
from pytket import Bit, Circuit, Qubit
from pytket._tket.architecture import Architecture, RingArch
from pytket._tket.circuit import CustomGateDef
from pytket._tket.predicates import (
    ConnectivityPredicate,
    DirectednessPredicate,
    MaxNQubitsPredicate,
    NoMidMeasurePredicate,
)
from pytket._tket.transform import Transform
from pytket.passes import (
    CliffordSimp,
    ContextSimp,
    DecomposeArbitrarilyControlledGates,
    DecomposeBoxes,
    DefaultMappingPass,
    FullPeepholeOptimise,
    GlobalisePhasedX,
    KAKDecomposition,
    PeepholeOptimise2Q,
    RebaseTket,
    RemoveBarriers,
    RemoveDiscarded,
    RemoveRedundancies,
    SequencePass,
    SimplifyMeasured,
    SynthesiseTket,
    ThreeQubitSquash,
)
from pytket.qasm import circuit_to_qasm_str
from pytket.qasm.qasm import NOPARAM_COMMANDS, PARAM_COMMANDS, QASMUnsupportedError
from qiskit.circuit.library import CXGate, UGate
from sympy import pi, sympify

from qat.purr.compiler.execution import QuantumHardwareModel
from qat.purr.integrations.qasm import BitRegister, Qasm2Parser, QasmContext, QubitRegister
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
            except:
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

        passes += [RebaseTket()]

        # Not everything in the list is a pass, we've also got transforms.
        # Everything in the list should have an apply function though.
        # for pass_ in passes:
        SequencePass(passes).apply(circ)
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


def run_tket_optimizations(qasm_string, opts, hardware: QuantumHardwareModel) -> str:
    """
    Runs tket-based optimizations and modifications. Routing will always happen no
    matter the level.

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

    couplings = deepcopy(hardware.qubit_direction_couplings)
    optimizations_failed = False
    architecture = None
    if any(couplings):
        # Without default remapping pass multi-qubit gates don't get moved around, so
        # trying to apply them to a limited subset of qubits provides no value.
        logical_qubit_map = {q.index: i for i, q in enumerate(hardware.qubits)}
        for coupling in couplings:
            control_ind, target_ind = coupling.direction
            coupling.direction = (
                logical_qubit_map[control_ind],
                logical_qubit_map[target_ind],
            )
        if TketOptimizations.DefaultMappingPass not in opts:
            architecture = Architecture([val.direction for val in couplings])
            optimizations_failed = not optimize_circuit(circ, architecture, opts)
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
