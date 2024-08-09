# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import abc
import math
import re
from copy import deepcopy
from numbers import Number
from os.path import dirname, join
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import qiskit.qasm.node as qasm_ast
from lark import Lark, Token, Tree, UnexpectedCharacters
from lark.visitors import Interpreter
from numpy import append, array, exp, linspace
from qiskit.qasm import Qasm
from qiskit.qasm.node import Gate
from qiskit.qasm.qasmparser import QasmParser as QiskitQasmParser

from qat.purr.backends.utilities import evaluate_shape
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import InlineResultsProcessing, Languages
from qat.purr.compiler.devices import PhysicalChannel, PulseChannel, Qubit
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CustomPulse,
    FrequencyShift,
    NotEquals,
    PhaseReset,
    PhaseShift,
    PostProcessType,
    ProcessAxis,
    Pulse,
    PulseShapeType,
    Variable,
)
from qat.purr.compiler.waveforms import get_waveform_type
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class ParseResults:
    """
    Results object for attempted parse. When coerced to a boolean matches
    against if parse was successful.
    """

    def __init__(self, success, errors=None):
        self.success = success
        self.errors = errors or ""

    def __bool__(self):
        return self.success

    @staticmethod
    def success():
        return ParseResults(True)

    @staticmethod
    def failure(message):
        return ParseResults(False, message)


def get_qasm_parser(qasm_str: str):
    """Gets the appropriate QASM parser for the passed-in QASM string."""
    parsers = [CloudQasmParser(), Qasm3Parser()]
    attempts = []
    for parser in parsers:
        parse_attempt = parser.can_parse(qasm_str)
        if parse_attempt:
            return parser
        attempts.append((parser.parser_language().name, parse_attempt.errors))

    raise ValueError(
        "No valid parser could be found. Attempted: "
        f"{', '.join([f'{a} with error {b}' for a, b in attempts])}"
    )


def qasm_from_file(file_path):
    """Get QASM from a file."""
    with open(file_path) as ifile:
        return ifile.read()


class QubitRegister:
    def __init__(self, qubits=None):
        self.qubits: List[Qubit] = qubits or []

    def __repr__(self):
        return "QubitRegister: " + str(self.qubits)


class BitRegister:
    def __init__(self, bits=None):
        self.bits: List[CregIndexValue] = bits or []

    def __repr__(self):
        return "Register: " + str(self.bits)


class Registers:
    def __init__(self):
        self.quantum: Dict[str, QubitRegister] = dict()
        self.classic: Dict[str, BitRegister] = dict()


class QasmContext:
    """
    Container object for all data relating to the scope/pass of QASM currently under
    analysis.
    """

    def __init__(self, registers=None, gates=None, variables=None):
        self.registers: Registers = registers or Registers()
        self.variables: Dict[str, Variable] = variables or dict()
        self.gates: Dict[str, Any] = gates or dict()


class CregIndexValue:
    """
    Used to reference when we're looking at a particular index in a creg variable.
    """

    def __init__(self, register_name: str, index: int, value: Any):
        self.register_name = register_name
        self.index = index
        self.value = value

    @property
    def variable(self):
        return f"{self.register_name}[{self.index}]"

    def __repr__(self):
        return self.variable


def fetch_gate_node(qasm, gate_name):
    """
    Used to fetch particular nodes out of a minimalistic QASM program. Not used for
    execution, more for retrieving particular things for insertion into Qiskit's QASM
    parser later on.
    """
    program = Qasm(None, qasm).parse()
    return [
        val for val in program.children if isinstance(val, Gate) and val.name == gate_name
    ][0]


class AbstractParser:
    def __init__(self):
        self.results_format = InlineResultsProcessing.Program

    def can_parse(self, qasm_str) -> ParseResults:
        return ParseResults.success()

    def parser_language(self) -> Languages:
        return Languages.Empty

    @abc.abstractmethod
    def walk_node(self, node, context: QasmContext, builder, **kwargs):
        pass

    def _add_delay(self, delay, qubits, builder: InstructionBuilder):
        registers = self._expand_to_match_registers(qubits, flatten_results=True)
        for qubit in registers:
            builder.delay(qubit, delay)

    def _add_qreg(self, reg_name, reg_length, context: QasmContext, builder):
        index_range = self._get_qreg_index_range(reg_length, context, builder)
        context.registers.quantum[reg_name] = QubitRegister(
            [builder.model.get_qubit(val) for val in index_range]
        )

    def _add_creg(self, reg_name, reg_length, context):
        context.registers.classic[reg_name] = BitRegister(
            [CregIndexValue(reg_name, val, 0) for val in range(reg_length)]
        )

    def _add_measure(self, qubits, bits, builder):
        registers = self._expand_to_match_registers(qubits, bits, flatten_results=True)

        for qubit, creg in registers:
            res_id = f"{creg}_{qubit.index}"
            builder.measure_single_shot_z(qubit, output_variable=res_id)
            creg.value = Variable(res_id)
            builder.results_processing(res_id, self.results_format)

    def _add_unitary(
        self,
        theta,
        phi,
        _lambda,
        qubit_or_register: List[Union[Qubit, QubitRegister]],
        builder,
    ):
        """Unitary in QASM terms is just ``U(...)``."""
        qubits = self._expand_to_match_registers(qubit_or_register, flatten_results=True)
        for qubit in qubits:
            builder.Z(qubit, _lambda).Y(qubit, theta).Z(qubit, phi)

    def _add_cnot(self, control_qbs, target_qbs, builder):
        for cqb, tqb in self._expand_to_match_registers(
            control_qbs, target_qbs, flatten_results=True
        ):
            builder.cnot(cqb, tqb)

    def _add_reset(self, qubits, builder):
        qubits = self._expand_to_match_registers(qubits, flatten_results=True)
        builder.reset(qubits)

    def _add_if(self, left, right, if_body, context, builder):
        label = builder.create_label()

        # This is the condition when a jump _should_ happen, so it's inverted from the
        # if condition.
        builder.jump(label, NotEquals(left, right))
        self.walk_node(if_body, context, builder)
        builder.add(label)

    def _add_ecr(self, qubits, builder):
        if len(qubits) != 2:
            raise ValueError(
                "Definition of ECR intrinsic is wrong. Can only take 2 "
                "qubits as arguments."
            )
        builder.ECR(qubits[0], qubits[1])

    def _get_qreg_index_range(self, reg_length, context, builder):
        next_free = 0
        available_indices = [qubit.index for qubit in builder.model.qubits]

        max_used = max(
            [-1]
            + [
                qubit.index
                for qubit_reg in context.registers.quantum.values()
                for qubit in qubit_reg.qubits
            ]
        )
        if max_used > -1:
            next_free = available_indices.index(max_used) + 1
        index_range = available_indices[next_free : next_free + reg_length]
        if len(index_range) < reg_length:
            raise ValueError("Attempted to allocate more qubits than available.")

        return index_range

    def _is_register_target(self, values: List[Any]):
        """
        Does it look like the passed-in qubit or parameter list contains a register.
        """
        return isinstance(values, Iterable) and any(
            isinstance(val, (QubitRegister, BitRegister)) for val in values
        )

    def _expand_to_match_registers(self, *args, tuple_return=True, flatten_results=False):
        """
        Expands and zips registers/non-registers together so they can be processed.

        QASM treats any registers of bits or qubits as calling the gate * register size
        times with each individual register value. This is a helper method for
        centralizing this expansion/mapping.

        :Example:

        .. code-block:: python

            [[q1, q2, q3], [q4, q5, q6],[c1]]

        With the first 2 lists being expanded qubit registers, should resolve into:

        .. code-block:: pyhton

            [[q1, q4, c1], [q2, q5, c1], [q3, q6, c1]]
        """
        args = [val if isinstance(val, List) else [val] for val in args]

        def _flatten_registers(value):
            res = []
            if not isinstance(value, List):
                value = [value]

            # If we have multiple registers it means they should be zipped together as
            # one.
            # So p, r = [ [ p[0], p[1] ], [ r[0], r[1] ] ] = [ [ p0, r0 ], [ p1, r1 ] ]
            registers = [
                val for val in value if isinstance(val, (QubitRegister, BitRegister))
            ]
            if len(registers) >= 2:
                value.extend(
                    self._expand_to_match_registers(*registers, tuple_return=False)
                )
                for removable_reg in registers:
                    value.remove(removable_reg)

            for val in value:
                if isinstance(val, QubitRegister):
                    res.extend(val.qubits)
                elif isinstance(val, BitRegister):
                    res.extend(val.bits)
                else:
                    res.append(val)

            return res

        # Flatten registers out, so they are individual lists holding their bit/qubits
        # for easy zipping.
        args = [
            (
                _flatten_registers(val)
                if self._is_register_target(val)
                else [_flatten_registers(val)]
            )
            for val in args
        ]
        max_length = max([len(val) for val in args])
        results = list(zip(*[val * max_length if len(val) == 1 else val for val in args]))
        if flatten_results:
            results = [
                tuple(
                    [
                        val[0] if isinstance(val, List) and len(val) == 1 else val
                        for val in tup
                    ]
                )
                for tup in results
            ]

        return [
            (val[0] if len(val) == 1 else tuple(val)) if tuple_return else list(val)
            for val in results
        ]


class Qasm2Parser(AbstractParser):
    ecr_qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    gate ecr q0, q1 { }
    """

    ecr_gate = fetch_gate_node(ecr_qasm_str, "ecr")

    def __init__(self, order_result_vars=False, raw_results=False):
        super().__init__()
        self.order_result_vars = order_result_vars
        self.raw_results = raw_results
        self._cached_parses: Dict[int, qasm_ast.Program] = dict()

    def __repr__(self):
        return self.__class__.__name__

    def parser_language(self) -> Languages:
        return Languages.Qasm2

    def _get_gate_variables(self, node: qasm_ast.CustomUnitary, context: QasmContext):
        """
        Looks at the gate definition values and maps across argument to parameter. Gate
        definition needs to be recognized in the context, as both callsite and
        declaration required.

        For a concrete example:

        .. code-block:: python

            gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
            u2(0,pi) a

        This maps ``phi=0``, ``lambda=pi`` and ``q=a`` when parsing ``u2`` to make sure
        when we flatten everything the variables are correct.
        """
        gate_def = context.gates.get(node.name, None)
        variables = dict()
        if gate_def is None:
            return variables

        if gate_def.arguments is not None:
            variables.update(
                dict(
                    zip(
                        [val.name for val in gate_def.arguments.children],
                        self._get_parameters(node, context),
                    )
                )
            )

        if gate_def.bitlist is not None:
            variables.update(
                dict(
                    zip(
                        [val.name for val in gate_def.bitlist.children],
                        self._get_qubits(node, context),
                    )
                )
            )

        return variables

    def _add_intrinsics_to_parser(self, parser, qasm=""):
        """Adds details of our intrinsics to the parser if needed."""

        # Can't parse if someone uses the ECR intrinsic without it set as a global
        # symbol, but if they define their own gate with that name we can't know until
        # it's parsed and failed due to duplicate definitions. As we currently don't
        # allow imports this, while rough, is accurate.
        if "gate ecr" not in qasm:
            parser.global_symtab["ecr"] = self.ecr_gate

    def _get_qiskit_parser(self):
        """
        Should return a pre-setup qiskit QASM parser for use in parsing the program.
        """
        qiskit_parser = QiskitQasmParser("")
        qiskit_parser.parse_debug(False)
        return qiskit_parser

    def _fetch_or_parse(self, qasm: str) -> qasm_ast.Program:
        # If we've seen this file before
        qasm_id = hash(qasm)
        if (cached_value := self._cached_parses.get(qasm_id, None)) is not None:
            return cached_value

        with self._get_qiskit_parser() as parser:
            self._add_intrinsics_to_parser(parser, qasm)
            program = parser.parse(qasm)

        self._cached_parses[qasm_id] = program
        return program

    def parse(self, builder, qasm: str):
        # Parse or pick up the cached version, then remove it as we're about to
        # interpret it.
        program = self._fetch_or_parse(qasm)
        if (qasm_id := hash(qasm)) in self._cached_parses:
            del self._cached_parses[qasm_id]
        return self.process_program(builder, program)

    def validate(self, qasm: qasm_ast.Program):
        pass

    def modify(self, qasm: qasm_ast.Program):
        """
        Allows children to transform the program before validation/transforming into our
        AST occurs.
        """
        # ECR is added by default, if we don't see a definition then add one.
        if not any(
            filter(
                lambda node: isinstance(node, qasm_ast.Gate) and node.name == "ecr",
                qasm.children,
            )
        ):
            qasm.children.insert(1, self.ecr_gate)

    def can_parse(self, qasm: str) -> ParseResults:
        try:
            self._fetch_or_parse(qasm)
            return ParseResults.success()
        except Exception as ex:
            return ParseResults.failure(str(ex))

    def _walk_program(self, builder, qasm: qasm_ast.Program, context):
        self.modify(qasm)
        self.validate(qasm)

        for node in qasm.children:
            self.walk_node(node, context, builder)

    def process_program(self, builder, qasm: qasm_ast.Program):
        context = QasmContext()
        self._walk_program(builder, qasm, context)

        # 'raw' results in this case simply means return the base measurement array in
        # the order of execution.
        register_keys = context.registers.classic.keys()
        if self.order_result_vars:
            register_keys = sorted(register_keys)

        for key in register_keys:
            builder.assign(key, [val.value for val in context.registers.classic[key].bits])

        builder.returns([key for key in register_keys])
        return builder

    def process_gate(self, method, context: QasmContext, builder, **kwargs):
        """Process a gate call."""
        if not isinstance(method, qasm_ast.CustomUnitary):
            raise ValueError(
                f"Node {str(method)} is not actually a gate but being treated like one."
            )

        gate_def = context.gates.get(method.name, None)
        if gate_def is None or len(gate_def.body.children) == 0:
            self.process_intrinsic(method, context, builder)
        else:
            current_context = QasmContext(
                Registers(), context.gates, self._get_gate_variables(method, context)
            )
            for node in gate_def.body.children:
                self.walk_node(node, current_context, builder)

    def process_intrinsic(self, method, context: QasmContext, builder):
        # Only process if it's an ECR gate and, more pertinently, our ECR gate.
        if method.name == "ecr":
            # TODO: Define precisely what rotation this should be, including argument.
            qubits = self._get_gate_variables(method, context)
            self._add_ecr([qb for qb in qubits.values()], builder)
        else:
            raise ValueError(f"Gate {method.name} isn't intrinsic and has no body.")

    def process_barrier(self, node, context, builder, **kwargs):
        pass

    def process_gate_definition(self, node, context, _, **kwargs):
        # Just a declaration, gates are used when CustomUnitary is used
        context.gates[node.name] = node

    def process_creg(self, node, context, builder, **kwargs):
        self._add_creg(node.name, node.index, context)

    def process_qreg(self, node, context: QasmContext, builder, **kwargs):
        self._add_qreg(node.name, node.id.index, context, builder)

    def process_measure(self, node, context: QasmContext, builder, **kwargs):
        self._add_measure(
            self._get_qubits(node, context),
            self._get_parameters(node, context),
            builder,
        )

    def process_unitary(self, node, context, builder, **kwargs):
        """Unitary in QASM terms is just ``U(...)``."""
        theta, phi, _lambda = (
            self._resolve_value(val, context) for val in node.children[0].children
        )

        self._add_unitary(theta, phi, _lambda, self._get_qubits(node, context), builder)

    def process_if(self, node, context, builder, **kwargs):
        left = self._resolve_value(node.children[0], context)
        right = self._resolve_value(node.children[1], context)
        self._add_if(left, right, node.children[2], context, builder)

    def process_cnot(self, node, context, builder, **kwargs):
        self._add_cnot(
            self._get_qubits(node.children[0], context),
            self._get_qubits(node.children[1], context),
            builder,
        )

    def process_reset(self, node, context, builder, **kwargs):
        self._add_reset(self._get_qubits(node, context), builder)

    def walk_node(self, node, context: QasmContext, builder, **kwargs):
        """
        Process each individual QASM node, builds context or forwards processing to
        relevant ``process_x`` method associated with each node type.
        """
        if isinstance(node, qasm_ast.CustomUnitary):
            self.process_gate(node, context, builder, **kwargs)
        elif isinstance(node, qasm_ast.Qreg):
            self.process_qreg(node, context, builder, **kwargs)
        elif isinstance(node, qasm_ast.Gate):
            self.process_gate_definition(node, context, builder, **kwargs)
        elif isinstance(node, qasm_ast.Measure):
            self.process_measure(node, context, builder, **kwargs)
        elif isinstance(node, qasm_ast.UniversalUnitary):
            self.process_unitary(node, context, builder, **kwargs)
        elif isinstance(node, qasm_ast.Cnot):
            self.process_cnot(node, context, builder, **kwargs)
        elif isinstance(node, qasm_ast.Reset):
            self.process_reset(node, context, builder, **kwargs)
        elif isinstance(node, qasm_ast.if_.If):
            self.process_if(node, context, builder, **kwargs)
        elif isinstance(node, qasm_ast.Creg):
            self.process_creg(node, context, builder, **kwargs)
        elif isinstance(node, qasm_ast.Barrier):
            self.process_barrier(node, context, builder, **kwargs)

    def _get_parameters(self, node, context):
        """Get the params of a gate. These are the non-qubit values of a gate."""
        if isinstance(node, (qasm_ast.CustomUnitary, qasm_ast.UniversalUnitary)):
            if node.arguments is not None:
                args = []
                for val in node.arguments.children:
                    arg = self._resolve_value(val, context)
                    if isinstance(arg, Iterable):
                        args.extend(arg)
                    else:
                        args.append(arg)

                return args
        elif isinstance(node, qasm_ast.Measure):
            bits = self._resolve_variable(node.children[1], context)
            if not isinstance(bits, List):
                bits = [bits]

            return bits
        return []

    def _get_qubits(self, node, context: QasmContext, follow_variable=False):
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

        ``follow_variable`` means that variables will be followed and their actual
        register value returned.
        """
        if isinstance(node, qasm_ast.UniversalUnitary):
            return self._get_qubits(node.bitlist, context, follow_variable)
        elif isinstance(node, qasm_ast.Cnot):
            return [
                qb
                for qbl in node.children
                for qb in self._get_qubits(qbl, context, follow_variable)
            ]
        elif isinstance(node, qasm_ast.Barrier):
            return [
                qb
                for qbl in node.children[0].children
                for qb in self._get_qubits(qbl, context, follow_variable)
                if qb
            ]
        elif isinstance(node, qasm_ast.Reset):
            return self._get_qubits(node.children[0], context, follow_variable)
        elif isinstance(node, qasm_ast.Measure):
            return self._get_qubits(node.children[0], context, follow_variable)
        elif isinstance(node, qasm_ast.CustomUnitary):
            if node.bitlist is None:
                return []

            qubits = []
            for val in node.bitlist.children:
                qubit = self._get_qubits(val, context, follow_variable)
                qubits.extend(qubit)

            return qubits

        elif isinstance(node, qasm_ast.Id):
            qb_register = context.registers.quantum.get(node.name, None)
            if qb_register is not None:
                return [qb_register]

            # Sometimes our qubits are in a variable, such as calling new gates with an
            # argument.
            variable = context.variables.get(node.name, None)
            if variable is not None:
                if isinstance(variable, Variable) and follow_variable:
                    variable = context.registers.quantum[variable.name]

                if not isinstance(variable, List):
                    variable = [variable]

                return variable

        elif isinstance(node, qasm_ast.IndexedId):
            register = context.registers.quantum.get(node.name, None)
            if register is not None:
                return [register.qubits[node.index]]

        raise ValueError(f"Cannot resolve qubit from {str(node)}")

    def _resolve_variable(self, node, context):
        """If the value is a variable pointing to a classic register, resolve it."""
        value = self._resolve_value(node, context)
        if isinstance(value, Variable):
            classic_reg = context.registers.classic.get(value.name, value)
            return classic_reg
        return value

    def _resolve_value(self, node, context: QasmContext):
        """
        Resolves the values being used in node calls, such as parameters and constants.
        """
        if isinstance(node, qasm_ast.Id):
            variable = context.registers.classic.get(node.name, None)
            if variable is not None:
                return Variable(node.name)

            constant = context.variables.get(node.name, None)
            if constant is not None:
                return constant

        elif isinstance(node, qasm_ast.IndexedId):
            var_array = context.registers.classic.get(node.name, None)
            if var_array is not None:
                return var_array.bits[node.index]

        elif isinstance(node, qasm_ast.BinaryOp):
            operator = self._resolve_value(node.children[0], context)
            left = self._resolve_value(node.children[1], context)
            right = self._resolve_value(node.children[2], context)
            return eval(f"{left}{operator}{right}")
        elif isinstance(node, qasm_ast.Prefix):
            prefix = self._resolve_value(node.children[0], context)
            expression = self._resolve_value(node.children[1], context)
            return eval(f"{prefix}{expression}")
        elif isinstance(
            node, (qasm_ast.Int, qasm_ast.BinaryOperator, qasm_ast.UnaryOperator)
        ):
            return node.value
        elif isinstance(node, qasm_ast.Real):
            # Reals are represented as strings and need transformation.
            return float(node.value)

        raise ValueError(f"Cannot resolve value from {str(node)}")


class RestrictedQasm2Parser(Qasm2Parser):
    """Parser which only allows certain gates to be passed."""

    def __init__(self, allowed_gates=None, disable_if=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_gates = allowed_gates
        self.disable_if = disable_if

    def validate(self, qasm: qasm_ast.Program):
        if self.allowed_gates is not None:
            intrinsic_gates = {
                node.name: node
                for node in qasm.children
                if isinstance(node, qasm_ast.Gate) and node.file.endswith("qelib1.inc")
            }

            # Look at both the main script and custom gate intrinsic usage.
            gate_nodes = {
                val.name
                for val in qasm.children
                + [
                    body_node
                    for node in qasm.children
                    if isinstance(node, qasm_ast.Gate) and node.name not in intrinsic_gates
                    for body_node in node.body.children
                ]
                if isinstance(val, qasm_ast.CustomUnitary) and val.name in intrinsic_gates
            }

            invalid_gates = gate_nodes.difference(self.allowed_gates)
            if any(invalid_gates):
                raise ValueError(
                    f"Gates [{', '.join(invalid_gates)}] "
                    "are currently unable to be used."
                )

        if self.disable_if and any(
            [val for val in qasm.children if isinstance(val, qasm_ast.if_.If)]
        ):
            raise ValueError("If's are currently unable to be used.")


class CloudQasmParser(RestrictedQasm2Parser):
    """
    QASM parser used in our QCaaS system.
    """

    def __init__(self):
        super().__init__(allowed_gates=None, disable_if=True, order_result_vars=True)


def _create_lark_parser():
    with open(
        join(dirname(__file__), "grammars", "partial_qasm3.lark"), "r", encoding="utf-8"
    ) as lark_grammar_file:
        lark_grammar_str = lark_grammar_file.read()
    return Lark(lark_grammar_str, regex=True)


class LarkOpenPulseContext(QasmContext):
    def __init__(self, registers=None, gates=None, variables=None, cali_methods=None):
        super().__init__(registers, gates, variables)
        self.calibration_methods: Dict[str, Any] = cali_methods or dict()


class QasmMethodWrapper(qasm_ast.CustomUnitary):
    def __init__(self, qasm_method, qubit_args, classic_args):
        self.qasm_method = qasm_method
        self.qubit_args = qubit_args
        self.classic_args = classic_args

        # Shadowed fields that get called in calling code.
        self.name = qasm_method.name


class LarkPatchingParser(Qasm2Parser):
    """Parser built to route the lark limited QASM 3 requests into and through."""

    base_include_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    """

    def load_default_gates(self, builder):
        """Loads the default QASM 2 gates for further processing."""
        context = LarkOpenPulseContext()
        with self._get_qiskit_parser() as parser:
            program: qasm_ast.Program = parser.parse(LarkPatchingParser.base_include_str)
            self._walk_program(builder.get_child_builder(), program, context)

        return context

    def _get_gate_variables(self, node: QasmMethodWrapper, context: QasmContext):
        if not isinstance(node, QasmMethodWrapper):
            return super()._get_gate_variables(node, context)

        variables = dict()
        if node.qasm_method.arguments is not None:
            variables.update(
                dict(
                    zip(
                        [val.name for val in node.qasm_method.arguments.children],
                        node.classic_args,
                    )
                )
            )

        if node.qasm_method.bitlist is not None:
            variables.update(
                dict(
                    zip(
                        [val.name for val in node.qasm_method.bitlist.children],
                        node.qubit_args,
                    )
                )
            )
        return variables

    def add_delay(self, delay, qubits, builder: InstructionBuilder):
        self._add_delay(delay, qubits, builder)

    def add_qreg(self, reg_name, reg_length, context: QasmContext, builder):
        self._add_qreg(reg_name, reg_length, context, builder)

    def add_creg(self, reg_name, reg_length, context):
        self._add_creg(reg_name, reg_length, context)

    def add_measure(self, qubits, bits, builder):
        self._add_measure(qubits, bits, builder)

    def add_unitary(
        self,
        theta,
        phi,
        _lambda,
        qubit_or_register: List[Union[Qubit, QubitRegister]],
        builder,
    ):
        self._add_unitary(theta, phi, _lambda, qubit_or_register, builder)

    def add_cnot(self, control_qbs, target_qbs, builder):
        self._add_cnot(control_qbs, target_qbs, builder)

    def add_reset(self, qubits, builder):
        self._add_reset(qubits, builder)

    def add_if(self, left, right, if_body, context, builder):
        self._add_if(left, right, if_body, context, builder)

    def add_ecr(self, qubits, builder):
        self._add_ecr(qubits, builder)


def get_frame_mappings(model: QuantumHardwareModel):
    """
    Generate the names for frames we allow in open pulse 'extern' statements.
    Returns a dictionary mapping name->pulse channel.
    """
    frames = {}
    for qubit in model.qubits:
        for pulse_channel in qubit.get_all_channels():
            frame_pattern = pulse_channel.id.replace(".", "_").lower()
            frames[frame_pattern] = pulse_channel

    return frames


def extern_port_name(physical_channel):
    match = re.search("[0-9]+", physical_channel.id)
    index = None
    if match is not None:
        start, end = match.regs[0]
        index = int(physical_channel.id[start:end])

    if index is None:
        raise ValueError(f"Port ID for {str(physical_channel)} unable to be evaluated.")

    return f"channel_{index}"


def get_port_mappings(model: QuantumHardwareModel):
    """
    Generate the names for ports we allow in open pulse 'extern' statements.
    Returns a dictionary mapping name->physical channel.
    """
    ports = {}
    for physical_channel in model.physical_channels.values():
        name = extern_port_name(physical_channel)
        ports[name] = physical_channel

    return ports


class UntargetedPulse:
    """Pulse that currently has no device to send it down."""

    def __init__(self, pulse_class: type, *args, **kwargs):
        self._ref_instance: Union[CustomPulse, Pulse] = pulse_class(
            None, *args, ignore_channel_scale=True, **kwargs
        )
        self._built = False

    @property
    def ref_instance(self):
        # This defers the deep copy to only if an UntargedPulse is accessed after it has
        # been already built.
        if self._built:
            self._ref_instance = deepcopy(self._ref_instance)
            self._ref_instance.quantum_targets = []
            self._built = False

        return self._ref_instance

    def build_with_target(self, channel: PulseChannel):
        pulse = self.ref_instance
        pulse.quantum_targets.append(channel)
        self._built = True
        return pulse

    def __repr__(self):
        return f"Partial instance of {type(self._ref_instance).__name__}"


class Qasm3Parser(Interpreter, AbstractParser):
    lark_parser = _create_lark_parser()

    def __init__(self):
        super().__init__()
        self.builder: Optional[InstructionBuilder] = None
        self._general_context: Optional[LarkOpenPulseContext] = None
        self._calibration_context: Optional[LarkOpenPulseContext] = None
        self._current_context: Optional[LarkOpenPulseContext] = None
        self._q2_patcher = LarkPatchingParser()
        self._port_mappings: Dict[str, PhysicalChannel] = dict()
        self._frame_mappings: Dict[str, PulseChannel] = dict()
        self._cached_parses: Dict[int, Any] = dict()

        self._has_qasm_version = False
        self._has_calibration_version = False
        self._has_open_pulse = False

    def parser_language(self) -> Languages:
        return Languages.Qasm3

    def _fetch_or_parse(self, qasm_str: str):
        # If we've seen this file before
        qasm_id = hash(qasm_str)
        if (cached_value := self._cached_parses.get(qasm_id, None)) is not None:
            return cached_value

        try:
            program = self.lark_parser.parse(qasm_str)
        except UnexpectedCharacters as e:
            invalid_string = e._context.strip(" ^\t\n\r")
            raise ValueError(f"Invalid QASM 3 syntax: '{invalid_string}'.")

        self._cached_parses[qasm_id] = program
        return program

    def can_parse(self, qasm_str: str) -> ParseResults:
        try:
            self._fetch_or_parse(qasm_str)
            return ParseResults.success()
        except Exception as ex:
            return ParseResults.failure(str(ex))

    def _reset_and_return(self):
        builder = self.builder
        self.builder = None
        self._general_context = None
        self._calibration_context = None
        self._current_context = None
        self._port_mappings = dict()
        self._frame_mappings = dict()
        self._has_calibration_version = False
        self._has_qasm_version = False
        self._has_open_pulse = False
        return builder

    def walk_node(self, node, context: QasmContext, builder, **kwargs):
        self.visit_children(node)

    def initalize(self, builder: InstructionBuilder):
        self.builder = builder

        # Both contexts share global state except for variables.
        self._general_context = self._q2_patcher.load_default_gates(self.builder)
        self._calibration_context = LarkOpenPulseContext(
            registers=self._general_context.registers,
            gates=self._general_context.gates,
            cali_methods=self._general_context.calibration_methods,
        )
        self._current_context = self._general_context
        self._frame_mappings = get_frame_mappings(builder.model)
        self._port_mappings = get_port_mappings(builder.model)

    def get_waveform_samples(self, wf: UntargetedPulse):
        if wf.ref_instance is CustomPulse:
            samples = wf.ref_instance.samples
            # TODO: Can be a normal list of complex too, doesn't have to be numpy.
            if not isinstance(samples, array):
                raise TypeError("should be complex array")
            return samples
        else:
            waveform = wf.ref_instance
            # TODO: how do we do this arbitarily?
            dt = 0.5e-9
            samples = int(waveform.width / dt)
            midway_time = waveform.width / 2

            t = linspace(-midway_time, midway_time, samples)
            return evaluate_shape(waveform, t)

    def _perform_signal_processing(self, name, args):
        if name == "mix":
            wf1, wf2 = args
            # TODO: just make getwfsamp take args
            samples1, samples2 = [self.get_waveform_samples(w) for w in (wf1, wf2)]
            output_length, pulse_length = (
                max(len(samples1), len(samples2)),
                min(len(samples1), len(samples2)),
            )
            output = append(
                array([1] * pulse_length, dtype=complex),
                array([0] * (output_length - pulse_length), dtype=complex),
            )
            for wave in (samples1, samples2):
                for i, val in enumerate(wave):
                    output[i] *= val

            return UntargetedPulse(CustomPulse, output)

        elif name == "sum":
            wf1, wf2 = args
            # TODO: just make getwfsamp take args
            samples1, samples2 = [self.get_waveform_samples(w) for w in (wf1, wf2)]
            pulse_length = max(len(samples1), len(samples2))
            output = array([0] * pulse_length, dtype=complex)
            for wave in (samples1, samples2):
                for i, val in enumerate(wave):
                    output[i] += val
            return UntargetedPulse(CustomPulse, output)

        elif name == "phase_shift":
            wf1: UntargetedPulse
            wf1, shift = args
            exp_shift = exp(1j * shift)
            if isinstance(wf1.ref_instance, CustomPulse):
                wf1.ref_instance.samples = [
                    exp_shift * val for val in wf1.ref_instance.samples
                ]
            else:
                wf1.ref_instance.phase += shift
            return wf1

        elif name == "scale":
            wf1, scale = args
            if wf1.ref_instance is CustomPulse:
                wf1.ref_instance.samples = [scale * val for val in wf1.ref_instance.samples]
            else:
                wf1.ref_instance.scale_factor *= scale
            return wf1

        return None

    def _attempt_declaration(self, var: Variable):
        if var.name in self._current_context.variables:
            raise ValueError(f"Can't redeclare variable {var.name}")
        self._current_context.variables[var.name] = var

    def transform_to_value(self, child_tree, walk_variable=True, return_variable=False):
        if isinstance(child_tree, List):
            return [
                self.transform_to_value(val, walk_variable, return_variable)
                for val in child_tree
            ]

        if not isinstance(child_tree, (Tree, Token)):
            return child_tree

        # TODO: For now it's fine, but want to change all visitor methods to return
        #   appropriate values and chain from there, as right now arguments and root
        #   calls are treated differently.
        def _walk_to_single_value(node: Tree):
            if not isinstance(node, Tree):
                return node, None

            if (
                any([isinstance(node, Token) for node in node.children])
                or len(node.children) > 1
            ):
                if len(node.children) == 1:
                    return node.children[0], node.data
                return node, node.data
            else:
                return _walk_to_single_value(node.children[0])

        node, data = _walk_to_single_value(child_tree)
        if isinstance(node, Tree):
            if data == "real_number":
                op, value = self.transform_to_value(node.children)
                if op == "-":
                    return -value
                elif op == "+":
                    return +value
            elif data == "unary_expression":
                op, value = self.transform_to_value(node.children)
                if op == "-":
                    return -value
                elif op == "~":
                    return ~value
                elif op == "!":
                    return not value
            elif data == "index_identifier":
                registers = self.transform_to_value(node.children[0])
                if isinstance(registers, BitRegister):
                    reg_index = self.transform_to_value(node.children[1])
                    return next(
                        (val for val in registers.bits if val.index == reg_index), None
                    )

                if isinstance(registers, QubitRegister):
                    reg_index = self.transform_to_value(node.children[1])
                    if len(qubits := registers.qubits) > reg_index:
                        return qubits[reg_index]
                    return None

            if data == "additive_expression":
                return self.transform_to_value(node.children[0]) + self.transform_to_value(
                    node.children[2]
                )

            elif data == "subtraction_expression":
                return self.transform_to_value(node.children[0]) - self.transform_to_value(
                    node.children[2]
                )

            elif data == "division_expression":
                return self.transform_to_value(node.children[0]) / self.transform_to_value(
                    node.children[2]
                )

            elif data == "multiplicative_expression":
                return self.transform_to_value(node.children[0]) * self.transform_to_value(
                    node.children[2]
                )

            elif data == "complex_number":
                return complex(
                    "".join([str(self.transform_to_value(val)) for val in node.children])
                )

            elif data == "timing_literal":
                value = self.transform_to_value(node.children[0])
                unit = self.transform_to_value(node.children[1])
                # TODO: [JF] Work out what DT is meant to be here.
                if unit == "dt":
                    # DT hardcoded at .5ns
                    return value / 2000000000
                elif unit == "ns":
                    return value / 1000000000
                elif unit == "us" or unit == "µs":
                    return value / 1000000
                elif unit == "ms":
                    return value / 1000
                elif unit == "s":
                    return value

            elif data == "extern_or_subroutine_call":
                return self.visit(node)

            # Return tuple of 'type' and 'name' if we're a classical argument.
            elif data == "classical_argument":
                results = [
                    self.transform_to_value(val, walk_variable, return_variable)
                    for val in node.children
                ]
                if len(results) == 1:
                    return None, results[0]

                return tuple(results)

            return [
                self.transform_to_value(val, walk_variable, return_variable)
                for val in node.children
            ]

        node: Token
        if data == "imag_number":
            return complex(0, float(node.value))
        elif node.type == "FLOAT_LITERAL":
            return float(node.value)
        elif node.type == "DECIMAL_INTEGER_LITERAL":
            return int(node.value)
        elif node.type == "CONSTANT":
            if node.value in ("pi", "π"):
                return math.pi
            elif node.value in ("tau", "𝜏"):
                return math.tau
            elif node.value in ("euler", "ℇ"):
                return math.e
        elif node.type == "IDENTIFIER":
            # $num just means get qubit at that index
            id_value: str = node.value
            if id_value.startswith("$"):
                return self.builder.model.get_qubit(int(id_value.strip("$")))

            qubits = self._current_context.registers.quantum.get(id_value, None)
            if qubits is not None:
                return qubits

            bits = self._current_context.registers.classic.get(id_value, None)
            if bits is not None:
                return bits

            def _walk_variables(var_id, guard: Set[str] = None):
                if guard is None:
                    guard = set()
                variable = self._current_context.variables.get(var_id, None)
                if variable is not None:
                    if walk_variable:
                        if isinstance(variable.value, Variable):
                            if var_id not in guard:
                                guard.add(var_id)
                                return _walk_variables(variable.name, guard)

                        return variable if return_variable else variable.value
                return var_id

            return _walk_variables(id_value)

        # If we have encompassing quotes just strip them as we're already a string.
        node_value = node.value
        if node_value.count('"') == 2:
            node_value = node_value.lstrip('"')
            node_value = node_value.rstrip('"')

        return node_value

    def version(self, tree: Tree):
        version_number = self.transform_to_value(tree)
        if version_number != 3:
            raise ValueError("OpenQASM version has to be 3.")
        self._has_qasm_version = True

    def calibration_grammar_declaration(self, tree: Tree):
        version_number = self.transform_to_value(tree)[1]
        if version_number != "openpulse":
            raise ValueError("Calibration grammar has to be openpulse.")
        self._has_calibration_version = True

    def frame_definition(self, tree: Tree):
        name = self.transform_to_value(tree.children[1])
        args = self.transform_to_value(tree.children[4])
        port = args[0]
        frequency = (
            self._get_frequency(args[1][0]) if isinstance(args[1], list) else args[1]
        )
        phase = 0.0 if len(args) <= 2 else args[2]

        if isinstance(port, PulseChannel):
            port = port.physical_channel
        elif not isinstance(port, PhysicalChannel):
            raise TypeError(
                f"Cannot create new frame from variable '{name}'. "
                "Must be either type Port or Frame."
            )

        pulse_channel = port.create_pulse_channel(name, frequency)
        self.builder.phase_shift(pulse_channel, phase)

        self._attempt_declaration(Variable(name, PulseChannel, pulse_channel))

    def waveform_definition(self, tree: Tree):
        if len(tree.children) == 4:
            assigned_variable = self.transform_to_value(tree.children[1])
            intrinsic_name = None
        elif len(tree.children) == 5:
            assigned_variable = self.transform_to_value(tree.children[1])
            intrinsic_name = self.transform_to_value(tree.children[3])
        else:
            raise ValueError("Unknown waveform definition.")

        _empty = tuple()

        def _validate_waveform_args(
            *,
            width=_empty,
            amp=_empty,
            beta=_empty,
            zero_at_edges=_empty,
            rise=_empty,
            std_dev=_empty,
            frequency=_empty,
            phase=_empty,
            square_width=_empty,
            drag=_empty,
        ):
            if width is not _empty and not isinstance(width, float):
                raise ValueError(
                    f"Width '{str(width)}' used in {intrinsic_name} is not a float."
                )
            if drag is not _empty and not isinstance(drag, float):
                raise ValueError(
                    f"Drag '{str(width)}' used in {intrinsic_name} is not a float."
                )
            if amp is not _empty and not isinstance(amp, Number):
                raise ValueError(
                    f"Amp '{str(amp)}' used in {intrinsic_name} is not a float."
                )
            if beta is not _empty and not isinstance(beta, float):
                raise ValueError(
                    f"Beta '{str(beta)}' used in {intrinsic_name} is not a float."
                )
            if zero_at_edges is not _empty and not isinstance(zero_at_edges, int):
                raise ValueError(
                    f"Zero at edges '{str(zero_at_edges)}' used in {intrinsic_name} "
                    "is not a int."
                )
            if rise is not _empty and not isinstance(rise, float):
                raise ValueError(
                    f"Rise '{str(rise)}' used in {intrinsic_name} is not a float."
                )
            if std_dev is not _empty and not isinstance(std_dev, (float, int)):
                raise ValueError(
                    f"Standard deviation '{str(std_dev)}' used in {intrinsic_name} "
                    "is not a float."
                )
            if frequency is not _empty and not isinstance(frequency, float):
                raise ValueError(
                    f"Frequency '{str(frequency)}' used in {intrinsic_name} "
                    "is not a float."
                )
            if phase is not _empty and not isinstance(phase, float):
                raise ValueError(
                    f"Phase '{str(phase)}' used in {intrinsic_name} is not a float."
                )
            if square_width is not _empty and not isinstance(square_width, float):
                raise ValueError(
                    f"Square width '{str(square_width)}' used in {intrinsic_name} "
                    "is not a float."
                )

        def _validate_arg_length(arg_tree, *lengths):
            """As results length is dynamic, centralize validation and messages."""
            lengths = set(lengths)
            args = self.transform_to_value(arg_tree)
            is_iterable = isinstance(args, (tuple, list))
            arg_length = len(args) if is_iterable else 1
            if (is_iterable and arg_length not in lengths) or (
                not is_iterable and 1 not in lengths
            ):
                raise ValueError(
                    f"Waveform '{intrinsic_name}' has incorrect number of arguments. "
                    f"Needs {','.join([str(val) for val in lengths])}, "
                    f"has {arg_length}."
                )
            return args + ([None] * (max(lengths) - arg_length))

        if intrinsic_name is None:
            # This is a flat array of pulse values.
            array_contents = self.transform_to_value(tree.children[3])
            waveform = UntargetedPulse(CustomPulse, array_contents)

        # TODO: implement non intrinsic waveforms.
        elif intrinsic_name == "constant":
            width, amp = _validate_arg_length(tree.children[4], 2)
            _validate_waveform_args(width=width, amp=amp)
            waveform = UntargetedPulse(Pulse, PulseShapeType.SQUARE, width, amp=amp)

        elif intrinsic_name == "rounded_square":
            width, std_dev, rise_time, amp = _validate_arg_length(tree.children[4], 4)
            _validate_waveform_args(width=width, std_dev=std_dev, amp=amp, rise=rise_time)
            waveform = UntargetedPulse(
                Pulse,
                PulseShapeType.ROUNDED_SQUARE,
                width,
                std_dev=std_dev,
                amp=amp,
                rise=rise_time,
            )

        elif intrinsic_name == "drag":
            amp, width, std_dev, beta, zero_at_edges = _validate_arg_length(
                tree.children[4], 4, 5
            )
            zero_at_edges = 0 if not zero_at_edges else 1
            _validate_waveform_args(
                width=width,
                amp=amp,
                beta=beta,
                zero_at_edges=zero_at_edges,
                std_dev=std_dev,
            )
            waveform = UntargetedPulse(
                Pulse,
                PulseShapeType.GAUSSIAN_DRAG,
                width=width,
                amp=amp,
                zero_at_edges=zero_at_edges,
                beta=beta,
                std_dev=std_dev,
            )

        elif intrinsic_name == "gaussian":
            amp, width, std_dev = _validate_arg_length(tree.children[4], 3)
            _validate_waveform_args(width=width, amp=amp, std_dev=std_dev)
            waveform = UntargetedPulse(
                Pulse,
                PulseShapeType.GAUSSIAN_ZERO_EDGE,
                width=width,
                amp=amp,
                std_dev=std_dev,
            )

        elif intrinsic_name == "gaussian_zero_edge":
            amp, width, std_dev, zero_at_edges = _validate_arg_length(tree.children[4], 4)
            zero_at_edges = 0 if not zero_at_edges else 1
            _validate_waveform_args(
                width=width, amp=amp, zero_at_edges=zero_at_edges, std_dev=std_dev
            )
            waveform = UntargetedPulse(
                Pulse,
                PulseShapeType.GAUSSIAN_ZERO_EDGE,
                width=width,
                amp=amp,
                zero_at_edges=zero_at_edges,
                std_dev=std_dev,
            )

        elif intrinsic_name == "sech":
            amp, width, std_dev = _validate_arg_length(tree.children[4], 3)
            waveform = UntargetedPulse(
                Pulse, PulseShapeType.SECH, width=width, amp=amp, std_dev=std_dev
            )

        elif intrinsic_name == "gaussian_square":
            amp, width, square_width, std_dev = _validate_arg_length(tree.children[4], 4)
            _validate_waveform_args(
                width=width, amp=amp, square_width=square_width, std_dev=std_dev
            )
            raise ValueError("Gaussian square waveform currently not supported.")

        elif intrinsic_name == "sine":
            amp, width, frequency, phase = _validate_arg_length(tree.children[4], 4)
            _validate_waveform_args(width=width, amp=amp, frequency=frequency, phase=phase)
            waveform = UntargetedPulse(
                Pulse,
                PulseShapeType.SIN,
                amp=amp,
                width=width,
                frequency=frequency,
                phase=phase,
            )

        elif intrinsic_name == "gaussian_rise":
            amp, width, rise, drag, phase = _validate_arg_length(tree.children[4], 5)
            _validate_waveform_args(width=width, rise=rise, amp=amp, drag=drag, phase=phase)
            waveform = UntargetedPulse(
                Pulse,
                PulseShapeType.GAUSSIAN,
                amp=amp,
                width=width,
                rise=rise,
                drag=drag,
                phase=phase,
            )

        elif intrinsic_name == "soft_square_rise":
            amp, width, rise, drag, phase = _validate_arg_length(tree.children[4], 5)
            _validate_waveform_args(width=width, rise=rise, amp=amp, drag=drag, phase=phase)
            waveform = UntargetedPulse(
                Pulse,
                PulseShapeType.SOFT_SQUARE,
                amp=amp,
                width=width,
                rise=rise,
                drag=drag,
                phase=phase,
            )

        # Intrinsic waveform shapes
        elif (internal_waveform := get_waveform_type(intrinsic_name)) is not None:
            width, amp = _validate_arg_length(tree.children[4], 2)
            _validate_waveform_args(width=width, amp=amp)
            waveform = UntargetedPulse(internal_waveform, width, amp)
        else:
            raise ValueError(f"Unknown waveform {intrinsic_name}.")

        self._attempt_declaration(Variable(assigned_variable, UntargetedPulse, waveform))

    def timing_instruction(self, tree: Tree):
        """This is actually a delay instruction."""
        delay_time, target = self.transform_to_value(tree)
        self.builder.delay(target, delay_time)

    def quantum_measurement_assignment_statement(self, tree: Tree):
        args = self.transform_to_value(tree)
        if len(args) == 2:
            qubits = args[0]
            bits = args[1]
        else:
            bits = args[0]
            qubits = args[2]

        if not isinstance(bits, List):
            bits = [bits]

        if not isinstance(qubits, List):
            qubits = [qubits]

        # If the measure for this particular qubit has been overriden the functionality
        # is distinctly different.
        if self._has_defcal_override("measure", qubits):
            results = self._call_gate("measure", qubits)
            if results is not None:
                results = results if isinstance(results, List) else [results]
                if len(bits) != len(results):
                    raise ValueError("Can't flatten overriden measure into assignment.")

                bit: CregIndexValue
                for bit, result in zip(bits, results):
                    bit.value = result
        else:
            self._q2_patcher.add_measure(qubits, bits, self.builder)

    def _has_defcal_override(
        self,
        name: str,
        qubits: List[Qubit],
        argument_values: Optional[List[Any]] = None,
    ):
        """
        Returns whether this gate has been overriden, either in a generic
        or qubit-specific manner.
        """
        argument_values = argument_values or []
        qubits = qubits if isinstance(qubits, List) else [qubits]
        qubit_specific_name = self._create_qb_specific_gate_suffix(name, qubits)

        is_calibration = name in self._current_context.calibration_methods
        is_qubit_specific = qubit_specific_name in self._current_context.calibration_methods
        is_list_expr = (
            self.generate_expr_list_defcal_name(name, argument_values)
            in self._current_context.calibration_methods
        )

        return is_calibration or is_qubit_specific or is_list_expr

    def generate_expr_list_defcal_name(self, name, expr_list):
        if not isinstance(expr_list, list):
            expr_list = [expr_list]
        return name + "_" + "_".join([str(i) for i in expr_list])

    def _call_gate(self, name, method_args, throw_on_missing=True):
        qubits, others = [], []

        def _strip_qubits(value):
            if isinstance(value, List):
                for val in value:
                    _strip_qubits(val)
            else:
                if isinstance(value, (Qubit, QubitRegister)):
                    qubits.append(value)
                else:
                    others.append(value)

        _strip_qubits(method_args)

        gate_def = self._current_context.gates.get(name, None)
        cal_def = self._current_context.calibration_methods.get(name, None)
        new_name = self.generate_expr_list_defcal_name(name, others)
        expr_list = False
        cal_def_new = self._current_context.calibration_methods.get(new_name, None)
        cal_def = cal_def_new or cal_def
        if cal_def_new is not None:
            expr_list = True

        qb_specific_cal_def = self._current_context.calibration_methods.get(
            self._create_qb_specific_gate_suffix(name, qubits), None
        )

        qb_specific_cal_def_expr_list = self._current_context.calibration_methods.get(
            self._create_qb_specific_gate_suffix(new_name, qubits), None
        )

        # Functionally the methods are processed the same, but the qb-specific override
        # takes presidence.
        if qb_specific_cal_def is not None:
            cal_def = qb_specific_cal_def

        if qb_specific_cal_def_expr_list is not None:
            expr_list = True
            cal_def = qb_specific_cal_def_expr_list

        if name in ("cnot", "CNOT"):
            self._q2_patcher.add_cnot(qubits[0], qubits[1], self.builder)
        elif name in ("u", "U"):
            # TODO: Untested as not in grammar.
            self._q2_patcher.add_unitary(
                others[0], others[1], others[2], qubits, self.builder
            )
        elif name in ("ecr", "ECR"):
            self._q2_patcher.add_ecr(qubits, self.builder)

        # Prioritize calibration definitions here if people override the base functions.
        # We also don't care about qubit scoping and restrictions.
        elif cal_def is not None:
            # Implied 'barrier' between defcals. To make things simple just assume that
            # everything in the defcal focuses on the qubits coming in.
            self.builder.synchronize(qubits)
            existing_context = self._current_context
            self._current_context = LarkOpenPulseContext(
                self._general_context.registers,
                self._general_context.gates,
                dict(self._calibration_context.variables),
                self._general_context.calibration_methods,
            )

            arg_mappings, qubit_mappings, body = cal_def
            if not expr_list:
                # Strip off the type as we have no need for it here.
                # No type given if an expression list
                arg_mappings = [name for type_, name in arg_mappings]

            if not isinstance(arg_mappings, List):
                arg_mappings = [arg_mappings]

            if not isinstance(qubit_mappings, List):
                qubit_mappings = [qubit_mappings]

            if len(arg_mappings) != len(others):
                raise ValueError(
                    f"Call to '{name}' needs {len(arg_mappings)} arguments. "
                    f"Has {len(others)}."
                )

            for arg_name, value in zip(arg_mappings, others):
                self._attempt_declaration(Variable(arg_name, type(value), value))

            for qb_name, value in zip(qubit_mappings, qubits):
                # If we resolved to a qubit already, we're a physical qubit.
                if isinstance(qb_name, (QubitRegister, Qubit)):
                    continue

                self._attempt_declaration(Variable(qb_name, type(value), value))

            cali_results = self.visit_children(body)
            self._current_context = existing_context

            # We imply that the final returned value is the 'return' from the
            # calibration block.
            # If people want to return something distinct, use a return.
            ret_value = cali_results[-1]
            return ret_value

        elif gate_def is not None:
            # Our wrapper exposes any fields required and we override the argument
            # gathering.
            wrapper = QasmMethodWrapper(gate_def, qubits, others)
            self._q2_patcher.process_gate(wrapper, self._current_context, self.builder)
        elif throw_on_missing:
            raise ValueError(
                f"Can't find gate implementation for '{name}' with supplied arguments."
            )

    def quantum_gate_call(self, tree: Tree):
        args = self.transform_to_value(tree)
        name = args[0]
        method_args = args[1:]
        self._call_gate(name, method_args)

    def quantum_reset(self, tree: Tree):
        qubits = self.transform_to_value(tree)
        self.builder.reset(qubits)

    def quantum_barrier(self, tree: Tree):
        args = self.transform_to_value(tree)
        self.builder.synchronize(args)

    def quantum_measurement(self, tree: Tree):
        """Throwaway measurement that doesn't store results into any variable."""
        qubits = self.transform_to_value(tree)

        if self._has_defcal_override("measure", qubits):
            self._call_gate("measure", qubits)
        else:
            self.builder.measure_single_shot_z(qubits)

    def return_statement(self, tree: Tree):
        variable = self.transform_to_value(tree, walk_variable=False)

        # Expressions return variables directly, so no need for lookup.
        if not isinstance(variable, Variable):
            if isinstance(variable, str):
                var_name = variable
                if (
                    variable := self._current_context.variables.get(var_name, None)
                ) is None:
                    raise ValueError(f"Can't return {var_name} as it doesn't exist.")
            else:
                variable = Variable.with_random_name(
                    self.builder.existing_names, type(variable), variable
                )
                self._attempt_declaration(variable)

        return variable

    # Timed box, not used right now.
    def timing_box(self, tree: Tree):
        pass

    def quantum_declaration(self, tree: Tree):
        args = self.transform_to_value(tree)
        if isinstance(args, str):
            length = 1
            variable = args
        else:
            # If you're using the soon-to-be-depreciated qreg,
            # arguments are switched.
            if isinstance(args[0], str):
                length = args[1]
                variable = args[0]
            else:
                length = args[0]
                variable = args[1]

        self._q2_patcher.add_qreg(variable, length, self._current_context, self.builder)

    def bit_declaration_statement(self, tree: Tree):
        args = self.transform_to_value(tree)
        if isinstance(args, str):
            length = 1
            variable = args
        else:
            # If you're using the soon-to-be-depreciated creg,
            # arguments are switched.
            if isinstance(args[0], str):
                length = args[1]
                variable = args[0]
            else:
                length = args[0]
                variable = args[1]

        self._q2_patcher.add_creg(variable, length, self._current_context)

    def complex_declaration_statement(self, tree: Tree):
        _, name = self.transform_to_value(tree)

        self._attempt_declaration(Variable(name, complex))

    def _create_qb_specific_gate_suffix(self, name, target_qubits):
        return f"{name}[{','.join([str(qb) for qb in target_qubits])}]"

    def calibration_definition(self, tree: Tree):
        self._has_open_pulse = True
        gate_name: str = self.transform_to_value(tree.children[1])
        is_expr_list = False
        if len(tree.children) == 4:
            classic_args = []
            target_qubits = self.transform_to_value(tree.children[2], walk_variable=False)
            body = tree.children[3]
        else:
            is_expr_list = tree.children[2].children[0].data == "expression_list"
            classic_args = self.transform_to_value(tree.children[2], walk_variable=False)
            target_qubits = self.transform_to_value(tree.children[3], walk_variable=False)
            body = tree.children[4]

        if not isinstance(target_qubits, list):
            target_qubits = [target_qubits]

        if not isinstance(classic_args, list):
            classic_args = [classic_args]

        if is_expr_list:
            gate_name = self.generate_expr_list_defcal_name(gate_name, classic_args)

        if all(isinstance(val, (Qubit, QubitRegister)) for val in target_qubits):
            gate_name = self._create_qb_specific_gate_suffix(gate_name, target_qubits)
        self._current_context.calibration_methods[gate_name] = (
            classic_args,
            target_qubits,
            body,
        )

    def gate_definition(self, tree: Tree):
        gate_name: str = self.transform_to_value(tree.children[0])
        if len(tree.children) == 3:
            classic_args = []
            target_qubits = self.transform_to_value(tree.children[1], walk_variable=False)
            body = tree.children[2]
        else:
            classic_args = self.transform_to_value(tree.children[1], walk_variable=False)
            target_qubits = self.transform_to_value(tree.children[2], walk_variable=False)
            body = tree.children[3]

        if not isinstance(target_qubits, list):
            target_qubits = [target_qubits]
        if all(isinstance(val, (Qubit, QubitRegister)) for val in target_qubits):
            gate_name = self._create_qb_specific_gate_suffix(gate_name, target_qubits)

        # Not technically a calibration method, but the way to call is the same.
        self._current_context.calibration_methods[gate_name] = (
            classic_args,
            target_qubits,
            body,
        )

    def _get_phase(self, channel: PulseChannel):
        phase = 0
        for inst in self.builder.instructions:
            if isinstance(inst, PhaseShift) and channel == inst.channel:
                phase += inst.phase
            elif isinstance(inst, PhaseReset) and channel in inst.quantum_targets:
                phase = 0
        return phase

    def _get_frequency(self, channel: PulseChannel):
        frequency = channel.frequency
        for inst in self.builder.instructions:
            if isinstance(inst, FrequencyShift) and inst.channel == channel:
                frequency += inst.frequency
        return frequency

    def _capture_iq_value(
        self, pulse_channel: PulseChannel, time: float, output_variable, filter=None
    ):
        # The acquire integrator mode means that for every acquired resonator response
        # signal within a group of shots, we integrate along the time axis to find the
        # average amplitude of the response. Before averaging, the acquired waveform is
        # down converted. Since the down conversion and mean are performed on the FPGA
        # these post processing operations are removed for executions on live hardware.
        #
        # The returned value for each shot after postprocessing is a complex iq value.
        acquire = Acquire(
            channel=pulse_channel,
            time=time,
            mode=AcquireMode.INTEGRATOR,
            output_variable=output_variable,
            filter=filter,
        )
        self.builder.add(acquire)
        self.builder.post_processing(
            acquire, PostProcessType.DOWN_CONVERT, ProcessAxis.TIME
        )
        self.builder.post_processing(acquire, PostProcessType.MEAN, ProcessAxis.TIME)

        return acquire

    def _validate_channel_args(self, channel, val_type: str, value=None):
        if not isinstance(channel, PulseChannel):
            raise ValueError(f"{str(channel)} is not a valid pulse channel.")

        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"{str(value)} is not a valid {val_type}.")

        return channel, value

    def _validate_phase_args(self, channel, phase=None):
        return self._validate_channel_args(channel, "phase", phase)

    def _validate_freq_args(self, channel, frequency=None):
        return self._validate_channel_args(channel, "frequency", frequency)

    def extern_or_subroutine_call(self, tree: Tree):
        name = self.transform_to_value(tree.children[0])
        args = self.transform_to_value(tree.children[1])

        if name in ("mix", "sum", "phase_shift", "scale"):
            return self._perform_signal_processing(name, args)

        elif name == "play":
            ut_pulse: UntargetedPulse = args[1]
            if not isinstance(ut_pulse, UntargetedPulse):
                variable_name = self.transform_to_value(
                    tree.children[1].children[1], walk_variable=False
                )
                raise ValueError(
                    f"Play frame argument {variable_name} hasn't been linked to a " "frame."
                )

            pulse_target = args[0]
            if not isinstance(pulse_target, PulseChannel):
                variable_name = self.transform_to_value(tree.children[1].children[0])
                raise ValueError(
                    f"Play waveform argument {variable_name} doesn't point to a "
                    "waveform."
                )

            self.builder.add(ut_pulse.build_with_target(pulse_target))
        elif name == "shift_phase":
            channel, phase = self._validate_phase_args(args[0], args[1])
            self.builder.add(PhaseShift(channel, phase))
        elif name == "set_phase":
            channel, phase_arg = self._validate_phase_args(args[0], args[1])
            phase = self._get_phase(channel)
            self.builder.add(PhaseShift(channel, phase_arg - phase))
        elif name == "get_phase":
            channel, _ = self._validate_phase_args(args)
            return self._get_phase(channel)
        elif name == "set_frequency":
            channel, freq_arg = self._validate_freq_args(args[0], args[1])
            frequency = self._get_frequency(channel)
            self.builder.add(FrequencyShift(channel, freq_arg - frequency))
        elif name == "get_frequency":
            channel, _ = self._validate_freq_args(args)
            return self._get_frequency(channel)
        elif name == "shift_frequency":
            channel, freq_arg = self._validate_freq_args(args[0], args[1])
            self.builder.add(FrequencyShift(channel, freq_arg))
        elif name == "capture_v0":
            # Not sure what this method should return.
            pass
        elif name == "capture_v1":
            # A capture command that returns an iq value
            variable: Variable = Variable.with_random_name(self.builder.existing_names)
            self._capture_iq_value(
                args[0], args[1], variable.name, args[2] if len(args) > 2 else None
            )

            self._attempt_declaration(variable)
            return variable
        elif name == "capture_v2":
            # A capture command that returns a discriminated bit
            # The first part of this capture is the same as capture_v1 but we take the
            # complex iq value and perform a linear complex to real map which is used to
            # adjust the iq values to a form which can be discriminated into a bit. If
            # the value is positive we return 0, qubit is in ground state, if the value
            # is negative we return 1, qubit is in excited state.
            pulse_channel = args[0]
            variable: Variable = Variable.with_random_name(self.builder.existing_names)
            acquire = self._capture_iq_value(
                pulse_channel,
                args[1],
                variable.name,
                args[2] if len(args) > 2 else None,
            )  # TODO: Fill in output_variable
            mean_z_map_args = None
            if len(args) > 3:
                mean_z_map_args = args[3]
            else:
                resonator = self.builder.model.get_devices_from_physical_channel(
                    pulse_channel.physical_channel_id
                )[0]
                for qb in self.builder.model.qubits:
                    if qb.measure_device == resonator:
                        mean_z_map_args = qb.mean_z_map_args
                        break
            if mean_z_map_args is None:
                keys = next(
                    key
                    for key, value in self._frame_mappings.items()
                    if value == pulse_channel
                )
                raise ValueError(f"Could not find mean_z_map_args for frame {keys}.")
            self.builder.post_processing(
                acquire,
                PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                args=mean_z_map_args,
            )
            self.builder.results_processing(variable.name, InlineResultsProcessing.Program)

            self._attempt_declaration(variable)
            return variable
        elif name == "capture_v3":
            # A capture command that returns a raw waveform data
            #
            # The acquire scope mode will average the resonator response signal over all
            # the shots within a group.
            #
            # Unlike the integrator mode, these will not be integrated over time to get
            # an average amplitude but will instead keep the entire waveform, averaging
            # over all the shots. This means you might get unexpected results for
            # greater than 1 shots if you're measuring a qubit in superposition, as the
            # different signal responses from the different measurement results will
            # average out.
            variable: Variable = Variable.with_random_name(
                self.builder.existing_names, CustomPulse
            )
            acquire = Acquire(
                channel=args[0],
                time=args[1],
                mode=AcquireMode.SCOPE,
                output_variable=variable.name,
                filter=args[2] if len(args) > 2 else None,
            )
            self.builder.add(acquire)
            self.builder.post_processing(
                acquire, PostProcessType.MEAN, ProcessAxis.SEQUENCE
            )
            self.builder.post_processing(
                acquire, PostProcessType.DOWN_CONVERT, ProcessAxis.TIME
            )

            self._attempt_declaration(variable)
            return variable
        elif name == "capture_v4":
            # Not relevant to us
            pass
        else:
            raise ValueError(f"Extern {name} not implemented.")

    def extern_frame(self, tree: Tree):
        name = self.transform_to_value(tree, walk_variable=False)
        pulse_channel = self._frame_mappings.get(name, None)
        if pulse_channel is None:
            raise ValueError(f"Could not find extern Frame with name '{name}'.")

        self._attempt_declaration(Variable(name, PulseChannel, pulse_channel))

    def extern_port(self, tree: Tree):
        name = self.transform_to_value(tree, walk_variable=False)
        physical_channel = self._port_mappings.get(name, None)
        if physical_channel is None:
            raise ValueError(f"Could not find extern Port with name '{name}'.")

        self._attempt_declaration(Variable(name, PhysicalChannel, physical_channel))

    def frame_attribute_assignment(self, tree: Tree):
        args = self.transform_to_value(tree)
        pulse_channel = args[0][0]
        if not isinstance(pulse_channel, PulseChannel):
            raise ValueError("Tried to assign to a frame that doesn't exist.")

        field = args[0][1]
        op = args[1]
        value = args[2]
        if field == "phase":
            inst = PhaseShift
            getter = self._get_phase
        elif field == "frequency":
            inst = FrequencyShift
            getter = self._get_frequency
        else:
            raise ValueError(f"Attempted to assign to an unknown frame field '{field}'.")

        if op == "=":
            current_value = getter(pulse_channel)
            self.builder.add(inst(pulse_channel, value - current_value))
        elif op == "+=":
            self.builder.add(inst(pulse_channel, value))
        elif op == "-=":
            self.builder.add(inst(pulse_channel, -value))
        else:
            raise ValueError(f"Attempted to use an unknown frame operator '{op}'.")

    def assignment(self, tree: Tree):
        name = self.transform_to_value(tree.children[0], walk_variable=False)
        op = self.transform_to_value(tree.children[1])
        value = self.transform_to_value(tree.children[2])

        if op != "=":
            raise ValueError(f"Assignment operator {op} is unsupported.")

        existing = self._current_context.variables.get(name, None)
        if existing is not None:
            existing.value = value
        else:
            self._attempt_declaration(
                value if isinstance(value, Variable) else Variable(name, type(value), value)
            )

    def cal_block(self, tree: Tree):
        self._has_open_pulse = True
        existing_context = self._current_context
        self._current_context = self._calibration_context
        self.visit_children(tree)
        self._current_context = existing_context

    def parse(self, builder: InstructionBuilder, qasm_str: str):
        self.initalize(builder)
        parsed = self._fetch_or_parse(qasm_str)

        if (qasm_id := hash(qasm_str)) in self._cached_parses:
            del self._cached_parses[qasm_id]

        # If we have a new results format, propagate it.
        self._q2_patcher.results_format = self.results_format

        self.visit(parsed)
        if not self._has_qasm_version:
            raise ValueError("Ambiguous QASM version, need OPENQASM header.")

        if self._has_open_pulse and not self._has_calibration_version:
            raise ValueError("Uses pulse definitions without defcalgrammar header.")

        # If we're purely QASM act like the previous version in regards to results.
        register_keys = self._general_context.registers.classic.keys()
        if any(register_keys):
            for key in register_keys:
                builder.assign(
                    key,
                    [
                        val.value
                        for val in self._general_context.registers.classic[key].bits
                    ],
                )

            builder.returns([key for key in register_keys])

        return self._reset_and_return()
