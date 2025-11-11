# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


from compiler_config.config import Languages
from qiskit import qasm2
from qiskit.circuit import (
    Barrier,
    ClassicalRegister,
    Delay,
    Gate,
    IfElseOp,
    Measure,
    QuantumRegister,
    Reset,
)
from qiskit.circuit.library import CXGate, UGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGNode, DAGOpNode, DAGOutNode
from qiskit.qasm2 import CustomInstruction

from qat.frontend.parsers.qasm.base import AbstractParser, QasmContext
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class Qasm2Parser(AbstractParser):
    ecr_gate = Gate("ecr", 2, [])

    def __init__(self, order_result_vars=False, raw_results=False):
        super().__init__()
        self.order_result_vars = order_result_vars
        self.raw_results = raw_results
        self._cached_parses: dict[int, DAGCircuit] = dict()

    def __repr__(self):
        return self.__class__.__name__

    def parser_language(self) -> Languages:
        return Languages.Qasm2

    def _get_intrinsics(self, qasm="") -> list[CustomInstruction]:
        instrinsics = list(qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        if "gate ecr" not in qasm:
            instrinsics.append(
                qasm2.CustomInstruction("ecr", 0, 2, lambda: self.ecr_gate, builtin=True)
            )
        return instrinsics

    def _fetch_or_parse(self, qasm: str) -> DAGCircuit:
        # If we have seen this file before
        qasm_id = hash(qasm)
        if (cached_value := self._cached_parses.get(qasm_id, None)) is not None:
            return cached_value

        circ = qasm2.loads(qasm, custom_instructions=self._get_intrinsics(qasm))
        program = circuit_to_dag(circ)
        from importlib.metadata import version

        if version("qiskit") < "2.0.0":
            # TODO: Drop conversion when we enforce Qiskit 2.x - COMPILER-658
            from qiskit.transpiler.passes.utils.convert_conditions_to_if_ops import (
                ConvertConditionsToIfOps,
            )

            program = ConvertConditionsToIfOps().run(program)

        self._cached_parses[qasm_id] = program
        return program

    def parse(
        self, builder: QuantumInstructionBuilder, qasm: str
    ) -> QuantumInstructionBuilder:
        # Parse or pick up the cached version, then remove it as we are about to
        # interpret it.
        program = self._fetch_or_parse(qasm)
        if (qasm_id := hash(qasm)) in self._cached_parses:
            del self._cached_parses[qasm_id]
        return self.process_program(builder, program)

    def validate(self, circ: DAGCircuit):
        pass

    def modify(self, circ: DAGCircuit):
        """
        Allows children to transform the program before validation/transforming into our
        AST occurs.
        """
        pass

    def _walk_program(
        self, builder: QuantumInstructionBuilder, circ: DAGCircuit, context: QasmContext
    ):
        self.modify(circ)
        self.validate(circ)
        self._current_dag = circ

        for i, node in enumerate(circ.nodes()):
            self.walk_node(node, context, builder)

    def process_program(
        self, builder: QuantumInstructionBuilder, circ: DAGCircuit
    ) -> QuantumInstructionBuilder:
        context = QasmContext()
        self._walk_program(builder, circ, context)

        # 'raw' results in this case simply means return the base measurement array in
        # the order of execution.
        register_keys = context.registers.classic.keys()
        if self.order_result_vars:
            register_keys = sorted(register_keys)

        for key in register_keys:
            builder.assign(key, [val.value for val in context.registers.classic[key].bits])

        builder.returns([key for key in register_keys])
        return builder

    def process_gate(
        self,
        node: DAGOpNode,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
        **kwargs,
    ):
        """Process a gate call."""
        if not isinstance(node.op, Gate):
            raise ValueError(
                f"Node {str(node)} is not actually a gate but being treated like one."
            )

        gate_def = node.op.definition
        if gate_def is None or len(gate_nodes := circuit_to_dag(gate_def).gate_nodes()) < 1:
            self.process_intrinsic(node, context, builder)
            return
        for gate_node in gate_nodes:
            qs = tuple([node.qargs[gate_def.qubits.index(q)] for q in gate_node.qargs])
            gate_node.qargs = qs
            self.walk_node(gate_node, context, builder)

    def process_intrinsic(
        self, node: DAGOpNode, context: QasmContext, builder: QuantumInstructionBuilder
    ):
        # Only process if it's an ECR gate and, more pertinently, our ECR gate.
        if node.name == "ecr":
            # TODO: Define precisely what rotation this should be, including argument.
            qubits = self._get_qubits(node, context)
            self._add_ecr(qubits, builder)
        else:
            raise ValueError(f"Gate {node.name} isn't intrinsic and has no body.")

    def process_barrier(
        self,
        node: DAGOpNode,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
        **kwargs,
    ):
        pass

    def process_delay(
        self,
        node: DAGOpNode,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
        **kwargs,
    ):
        qubits = self._get_qubits(node, context)
        delay = node.op.duration
        self._add_delay(delay, qubits, builder)

    def process_gate_definition(self, node: DAGOpNode, context: QasmContext, _, **kwargs):
        context.gates[node.name] = node

    def process_creg(
        self,
        node: ClassicalRegister,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
        **kwargs,
    ):
        self._add_creg(node.name, node.size, context)

    def process_qreg(
        self,
        node: QuantumRegister,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
        **kwargs,
    ):
        self._add_qreg(node.name, node.size, context, builder)

    def process_measure(
        self,
        node: DAGOpNode,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
        **kwargs,
    ):
        self._add_measure(
            self._get_qubits(node, context),
            self._get_clbits(node, context),
            builder,
        )

    def process_unitary(
        self,
        node: DAGOpNode,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
        **kwargs,
    ):
        """Unitary in QASM terms is just ``U(...)``."""
        theta, phi, _lambda = node.op.params
        self._add_unitary(theta, phi, _lambda, self._get_qubits(node, context), builder)

    def process_cnot(
        self,
        node: DAGOpNode,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
        **kwargs,
    ):
        self._add_cnot(
            *self._get_qubits(node, context),
            builder,
        )

    def process_reset(
        self,
        node: DAGOpNode,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
        **kwargs,
    ):
        self._add_reset(self._get_qubits(node, context), builder)

    def walk_node(
        self,
        node: DAGNode,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
        **kwargs,
    ):
        """
        Process each individual QASM node, builds context or forwards processing to
        relevant ``process_x`` method associated with each node type.
        """
        if isinstance(node, (DAGInNode, DAGOutNode)):
            for register, _ in self._current_dag.find_bit(node.wire).registers:
                if isinstance(register, QuantumRegister):
                    self.process_qreg(register, context, builder)
                elif isinstance(register, ClassicalRegister):
                    self.process_creg(register, context, builder)
        elif isinstance(node, DAGOpNode):
            match node.op:
                case Delay():
                    self.process_delay(node, context, builder, **kwargs)
                case Barrier():
                    self.process_barrier(node, context, builder, **kwargs)
                case Measure():
                    self.process_measure(node, context, builder, **kwargs)
                case Reset():
                    self.process_reset(node, context, builder, **kwargs)
                case UGate():
                    self.process_unitary(node, context, builder, **kwargs)
                case CXGate():
                    self.process_cnot(node, context, builder, **kwargs)
                case IfElseOp():
                    raise NotImplementedError("IfElseOp is not currently supported.")
                case Gate():
                    self.process_gate(node, context, builder, **kwargs)
                case _:
                    raise NotImplementedError(
                        "DAGNode action not implemented for '{type(node)}'."
                    )

    def _get_parameters(self, node: DAGOpNode, context: QasmContext) -> list:
        """Get the params of a gate. These are the non-qubit values of a gate."""
        if isinstance(node.op, Gate):
            return node.op.params
        if isinstance(node.op, Measure):
            bits = self._get_clbits(node, context)
            if not isinstance(bits, list):
                bits = [bits]

            return bits
        return []

    def _get_qubits(self, node: DAGOpNode, context: QasmContext) -> list:
        """
        Resolve what qubits or qubit registers this node relates too.

        If a value was originally a qubit register, it will be appended as a list,
        otherwise if it was a single qubit target, just a normal object.

        :Example:

        .. code-block:: python

            [[q0, q1, q2], q2]

        The above means the first target was a register referencing q1-3, whereas the
        second was a single qubit referenced directly. If we say our register is just
        called qb, the above would equate to an argument list of ``[qb, qb[2]]``.
        """
        qubits = []
        for qubit in node.qargs:
            reg, ind = self._current_dag.find_bit(qubit).registers[0]
            register = context.registers.quantum.get(reg.name, None)
            if register is not None:
                qubits.append(register.qubits[ind])
        return qubits

    def _get_clbits(self, source: DAGOpNode | tuple, context: QasmContext) -> list:
        if isinstance(source, DAGOpNode):
            return self._get_clbits(source.cargs, context)
        else:
            clbits = []
            for clbit in source:
                reg, ind = self._current_dag.find_bit(clbit).registers[0]
                register = context.registers.classic.get(reg.name, None)
                if register is not None:
                    clbits.append(register.bits[ind])
            return clbits


class RestrictedQasm2Parser(Qasm2Parser):
    """Parser which only allows certain gates to be passed."""

    def __init__(self, allowed_gates=None, disable_if=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_gates = allowed_gates
        self.disable_if = disable_if

    def validate(self, circ: DAGCircuit):
        if self.allowed_gates is not None:
            intrinsic_gates = {node.name: node for node in qasm2.LEGACY_CUSTOM_INSTRUCTIONS}

            # Look at both the main script and custom gate intrinsic usage.
            gate_nodes = {
                val.name for val in circ.gate_nodes() if val.name in intrinsic_gates
            }

            invalid_gates = gate_nodes.difference(self.allowed_gates)
            if any(invalid_gates):
                raise ValueError(
                    f"Gates [{', '.join(invalid_gates)}] are currently unable to be used."
                )

        if self.disable_if and any(
            [node.op for node in circ.op_nodes() if isinstance(node.op, IfElseOp)]
        ):
            raise ValueError("If's are currently unable to be used.")


class CloudQasmParser(RestrictedQasm2Parser):
    """
    QASM parser used in our QCaaS system.
    """

    def __init__(self):
        super().__init__(allowed_gates=None, disable_if=True, order_result_vars=True)
