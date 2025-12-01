# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import math
from copy import deepcopy
from functools import singledispatchmethod
from importlib.util import find_spec
from numbers import Number
from os.path import join
from pathlib import Path
from pydoc import locate
from typing import Any

import numpy as np
from compiler_config.config import InlineResultsProcessing, Languages
from lark import Lark, Token, Tree, UnexpectedCharacters
from lark.visitors import Interpreter
from openqasm3 import ast
from openqasm3.parser import parse as oq3_parse
from openqasm3.visitor import QASMVisitor

from qat.frontend.parsers.qasm.base import AbstractParser, ParseResults, QasmContext
from qat.frontend.register import BitRegister, CregIndexValue, QubitRegister, Registers
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import (
    FrequencyShift,
    PhaseReset,
    PhaseShift,
    QuantumInstruction,
    Variable,
)
from qat.ir.measure import Acquire, AcquireMode, PostProcessType, ProcessAxis
from qat.ir.pulse_channel import PulseChannel as IRPulseChannel
from qat.ir.waveforms import (
    AbstractWaveform,
    DragGaussianWaveform,
    GaussianSquareWaveform,
    GaussianWaveform,
    GaussianZeroEdgeWaveform,
    Pulse,
    RoundedSquareWaveform,
    SampledWaveform,
    SechWaveform,
    SinWaveform,
    SoftSquareWaveform,
    SquareWaveform,
    Waveform,
)
from qat.model.device import (
    AcquirePulseChannel,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    QubitPulseChannels,
    ResonatorPulseChannels,
)
from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.utils.pydantic import ValidatedList

log = get_default_logger()


class OpenPulseContext(QasmContext):
    calibration_methods: dict[str, Any] = dict()


def get_frame_mappings(hw: PhysicalHardwareModel):
    """
    Generate the names for frames we allow in open pulse 'extern' statements.
    Returns a dictionary mapping name->pulse channel.
    """
    frames = {}
    for q1_id, q1 in hw.qubits.items():
        for channel_type in QubitPulseChannels.model_fields:
            if "cross_resonance" in channel_type:
                for q2_id, pulse_channel in getattr(
                    q1.pulse_channels, channel_type
                ).items():
                    qubit_prefix = f"q{q1_id}_q{q2_id}"
                    channel_type = channel_type.replace("_channels", "")
                    frames[f"{qubit_prefix}_{channel_type}"] = pulse_channel
            else:
                qubit_prefix = f"q{q1_id}"
                frames[f"{qubit_prefix}_{channel_type}"] = getattr(
                    q1.pulse_channels, channel_type
                )

        for channel_type in ResonatorPulseChannels.model_fields:
            res_prefix = f"r{q1_id}"
            frames[f"{res_prefix}_{channel_type}"] = getattr(
                q1.resonator.pulse_channels, channel_type
            )

    return frames


def get_port_mappings(hw: PhysicalHardwareModel):
    """
    Generate the names for ports we allow in open pulse 'extern' statements.
    Returns a dictionary mapping name->physical channel.
    """
    ports = {}
    for index, channel in hw.physical_channel_map.items():
        name = f"channel_{index}"
        ports[name] = channel

    return ports


class Qasm3ParserBase(AbstractParser, QASMVisitor):
    def parser_language(self) -> Languages:
        return Languages.Qasm3

    def _includes_standard_gates(self, qasm_str: str) -> bool:
        return any(
            [
                include in qasm_str
                for include in ('include "qelib1.inc";', 'include "stdgates.inc";')
            ]
        )

    def load_default_gates(self, context: QasmContext) -> QasmContext:
        node = ast.Include(filename="stdgates.inc")
        self.visit(node, context)
        return context

    def parse(
        self, builder: QuantumInstructionBuilder, qasm: str
    ) -> QuantumInstructionBuilder:
        self.builder = builder
        # Parse or pick up the cached version, then remove it as we are about to
        # interpret it.
        program = self._fetch_or_parse(qasm)
        if (qasm_id := hash(qasm)) in self._cached_parses:
            del self._cached_parses[qasm_id]
        context = QasmContext()
        if not self._includes_standard_gates(qasm):
            self.load_default_gates(context)
        return self.process_program(program, context)

    def process_program(
        self, prog: ast.Program, context: QasmContext | None
    ) -> QuantumInstructionBuilder:
        context = context or QasmContext()
        self._walk_program(prog, context)
        self._assign_returns(context)
        return self.builder

    def _fetch_or_parse(self, qasm_str: str) -> ast.Program:
        if 'defcalgrammar "openpulse"' in qasm_str:
            raise ValueError("QASM3ParserBase can not parse OpenPulse programs.")

        # If we have seen this file before.
        qasm_id = hash(qasm_str)
        if (cached_value := self._cached_parses.get(qasm_id, None)) is not None:
            return cached_value

        try:
            program = oq3_parse(qasm_str)
        except Exception as e:
            invalid_string = str(e)
            raise ValueError(f"Invalid QASM 3 syntax: '{invalid_string}'.")

        self._cached_parses[qasm_id] = program
        return program

    def validate(self, prog: ast.Program):
        pass

    def modify(self, prog: ast.Program):
        """
        Allows children to transform the program before validation/transforming into our
        AST occurs.
        """
        pass

    def _walk_program(self, prog: ast.Program, context: QasmContext):
        self.modify(prog)
        self.validate(prog)

        for node in prog.statements:
            self.visit(node, context)

    def visit_Include(self, node: ast.Include, context: QasmContext):
        if node.filename == "qelib1.inc":
            file_path = Path(
                find_spec("qiskit.qasm.libs").submodule_search_locations[0], node.filename
            )
        elif node.filename == "stdgates.inc":
            file_path = Path(Path(__file__).parents[1], "grammars", node.filename)
        else:
            file_path = Path(node.filename)
        if not file_path.is_file():
            raise ValueError(f"File not found for '{str(file_path)}'.")
        with file_path.open(encoding="utf-8") as f:
            self._walk_program(oq3_parse(f.read()), context)

    def visit_QuantumGateDefinition(
        self, node: ast.QuantumGateDefinition, context: QasmContext
    ):
        context.gates[node.name.name] = node

    @singledispatchmethod
    def _get_qubits(self, input_, context: QasmContext):
        raise NotImplementedError(f"Cannot get qubits from {type(input_)}")

    @_get_qubits.register(list)
    def _(self, input_, context: QasmContext):
        qubits = []
        for item in input_:
            qbs = self._get_qubits(item, context)
            if isinstance(qbs, list):
                qubits.extend(qbs)
            else:
                qubits.append(qbs)
        return qubits

    @_get_qubits.register(ast.QASMNode)
    def _(self, input_, context: QasmContext):
        return self.visit(input_, context)

    @_get_qubits.register(Qubit)
    @_get_qubits.register(QubitRegister)
    def _(self, input_, context: QasmContext):
        return input_

    def _create_qb_specific_gate_suffix(
        self, name: str, target_qubits: list[Qubit | QubitRegister]
    ) -> str:
        return f"{name}[{','.join([str(qb) for qb in target_qubits])}]"

    def _attempt_declaration(self, variable: Variable, context: QasmContext):
        if (old_var := context.variables.get(variable.name, None)) is None:
            context.variables[variable.name] = variable
        elif old_var != variable:
            raise ValueError(f"Can't redeclare variable {variable.name}")

    def visit_ClassicalDeclaration(
        self, node: ast.ClassicalDeclaration, context: QasmContext
    ):
        self.add_creg(node.identifier.name, node.type.size.value, context)

    def visit_QubitDeclaration(self, node: ast.QubitDeclaration, context: QasmContext):
        self.add_qreg(node.qubit.name, node.size.value, context, self.builder)

    def _get_node_arguments(self, node: ast.QASMNode, context: QasmContext) -> list:
        args = []
        for arg in node.arguments:
            if isinstance(arg, ast.QASMNode):
                args.append(self.visit(arg, context))
            else:
                args.append(arg)
        return args

    def visit_BinaryExpression(self, node: ast.BinaryExpression, context: QasmContext):
        lhs = self.visit(node.lhs, context)
        rhs = self.visit(node.rhs, context)
        op = node.op.name

        match op:
            case "+":
                return lhs + rhs
            case "-":
                return lhs - rhs
            case "/":
                return lhs / rhs
            case "*":
                return lhs * rhs
            case ">":
                return lhs > rhs
            case "<":
                return lhs < rhs
            case ">=":
                return lhs >= rhs
            case "<=":
                return lhs <= rhs
            case "==":
                return lhs == rhs
            case "!=":
                return lhs != rhs
            case "&&":
                return lhs and rhs
            case "||":
                return lhs or rhs
            case _:
                # | ^ & << >> % **
                raise NotImplementedError(f"Unsupported operator '{op}'")

    def visit_UnaryExpression(self, node: ast.UnaryExpression, context: QasmContext):
        ex = self.visit(node.expression, context)
        op = node.op.name
        match op:
            case "-":
                return -ex
            case "~":
                return ~ex
            case "!":
                return not ex
            case _:
                raise NotImplementedError(f"Unsupported operator '{op}'")

    def visit_IntegerLiteral(self, node: ast.IntegerLiteral, context: QasmContext) -> int:
        return node.value

    def visit_Identifier(self, node: ast.Identifier, context: QasmContext):
        id_name = node.name

        if id_name.startswith("$"):
            return self.builder.hw.qubit_with_index(int(id_name.strip("$")))

        qubits = context.registers.quantum.get(id_name, None)
        if qubits is not None:
            return qubits

        bits = context.registers.classic.get(id_name, None)
        if bits is not None:
            return bits

        variable = context.variables.get(id_name, None)
        if variable is not None:
            return variable.value

        # Return constant values
        match id_name:
            case "pi" | "π":
                return math.pi
            case "tau" | "𝜏":
                return math.tau
            case "euler" | "ℇ":
                return math.e

        return id_name

    def visit_IndexedIdentifier(
        self, node: ast.IndexedIdentifier, context: QasmContext
    ) -> list:
        indices = node.indices
        if isinstance(indices, list):
            indices = indices[0]
        frame = self.visit(node.name, context)
        match frame:
            case QubitRegister():
                return [frame.qubits[self.visit(ind, context)] for ind in indices]
            case BitRegister():
                return [frame.bits[self.visit(ind, context)] for ind in indices]
            case _:
                raise NotImplementedError(
                    f"IndexIdentifier not implemented for '{type(frame)}' indexing."
                )

    def visit_QuantumGate(self, node: ast.QuantumGate, context: QasmContext):
        target_qubits = self._get_qubits(node.qubits, context)
        gate_name = node.name.name
        arguments = self._get_node_arguments(node, context)

        match gate_name:
            case "u" | "U":
                theta, phi, _lambda = arguments
                return self.add_unitary(theta, phi, _lambda, target_qubits, self.builder)
            case "cx" | "CX":
                return self.add_cnot(*target_qubits, self.builder)
            case "ecr" | "ECR":
                return self.add_ecr(target_qubits, self.builder)

        if len(node.modifiers) > 0:
            raise NotImplementedError("Gate modifiers are not yet supported.")

        gate_context = QasmContext(registers=Registers(), gates=context.gates)

        if (gate_def := context.gates.get(gate_name, None)) is not None:
            for arg, value in zip(gate_def.arguments, arguments):
                if (known_var := context.variables.get(arg.name, None)) is not None:
                    self._attempt_declaration(
                        Variable(name=known_var.name, var_type=type(value), value=value),
                        gate_context,
                    )
                else:
                    self._attempt_declaration(
                        Variable(
                            name=self.visit(arg, context), var_type=type(value), value=value
                        ),
                        gate_context,
                    )
            for qb_name, value in zip(gate_def.qubits, target_qubits):
                if isinstance(qb_name, (QubitRegister, Qubit)):
                    continue
                self._attempt_declaration(
                    Variable(name=qb_name.name, var_type=type(value), value=value),
                    gate_context,
                )
            for n in gate_def.body:
                self.visit(n, gate_context)

    def visit_QuantumMeasurementStatement(
        self, node: ast.QuantumMeasurementStatement, context: QasmContext
    ):
        qubits = self.visit(node.measure, context)
        bits = self.visit(node.target, context)
        self.add_measure(qubits, bits, self.builder)

    def visit_QuantumMeasurement(self, node: ast.QuantumMeasurement, context: QasmContext):
        return self.visit(node.qubit, context)

    def walk_node(self, node, context: QasmContext, builder, **kwargs):
        pass

    def _assign_returns(self, context: QasmContext):
        register_keys = context.registers.classic.keys()
        if any(register_keys):
            for key in register_keys:
                self.builder.assign(
                    key,
                    [val.value for val in context.registers.classic[key].bits],
                )

            self.builder.returns([key for key in register_keys])

    def add_delay(
        self, delay: float, qubits: list[Qubit], builder: QuantumInstructionBuilder
    ):
        self._add_delay(delay, qubits, builder)

    def add_qreg(
        self,
        reg_name: str,
        reg_length: int,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
    ):
        self._add_qreg(reg_name, reg_length, context, builder)

    def add_creg(self, reg_name: str, reg_length: int, context: QasmContext):
        self._add_creg(reg_name, reg_length, context)

    def add_measure(
        self,
        qubits: list[Qubit],
        bits: list[CregIndexValue],
        builder: QuantumInstructionBuilder,
    ):
        self._add_measure(qubits, bits, builder)

    def add_unitary(
        self,
        theta: float,
        phi: float,
        _lambda: float,
        qubit_or_register: list[Qubit | QubitRegister],
        builder: QuantumInstructionBuilder,
    ):
        self._add_unitary(theta, phi, _lambda, qubit_or_register, builder)

    def add_cnot(
        self,
        control_qbs: list[Qubit],
        target_qbs: list[Qubit],
        builder: QuantumInstructionBuilder,
    ):
        self._add_cnot(control_qbs, target_qbs, builder)

    def add_reset(self, qubits, builder: QuantumInstructionBuilder):
        self._add_reset(qubits, builder)

    def add_if(
        self, left, right, if_body, context: QasmContext, builder: QuantumInstructionBuilder
    ):
        self._add_if(left, right, if_body, context, builder)

    def add_ecr(self, qubits, builder: QuantumInstructionBuilder):
        self._add_ecr(qubits, builder)


def _create_lark_parser():
    with open(
        join(
            Path(__file__).parents[3],
            "frontend",
            "parsers",
            "grammars",
            "partial_qasm3.lark",
        ),
        "r",
        encoding="utf-8",
    ) as lark_grammar_file:
        lark_grammar_str = lark_grammar_file.read()
    return Lark(lark_grammar_str, regex=True)


class Qasm3Parser(Interpreter, AbstractParser):
    lark_parser = _create_lark_parser()

    def __init__(self):
        super().__init__()
        self.builder: QuantumInstructionBuilder | None = None
        self._general_context: OpenPulseContext | None = None
        self._calibration_context: OpenPulseContext | None = None
        self._current_context: OpenPulseContext | None = None
        self._q3_patcher: Qasm3ParserBase = Qasm3ParserBase()
        self._port_mappings: dict[str, PhysicalChannel] = dict()
        self._frame_mappings: dict[str, PulseChannel] = dict()
        self._cached_parses: dict[int, Any] = dict()

        self._has_qasm_version = False
        self._has_calibration_version = False
        self._has_open_pulse = False

    def parser_language(self) -> Languages:
        return Languages.Qasm3

    def _fetch_or_parse(self, qasm_str: str):
        # If we have seen this file before.
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
            return ParseResults(success=True)
        except Exception as ex:
            return ParseResults(success=True, errors=str(ex))

    def include(self, tree):
        filename = self.transform_to_value(tree.children[0])
        self._q3_patcher.visit(ast.Include(filename), self._current_context)

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

    def initialize(self, builder: QuantumInstructionBuilder):
        self.builder = builder
        self._q3_patcher.builder = builder

        # Both contexts share global state except for variables.
        self._general_context = self._q3_patcher.load_default_gates(OpenPulseContext())
        self._calibration_context = OpenPulseContext(
            registers=self._general_context.registers,
            gates=self._general_context.gates,
            calibration_methods=self._general_context.calibration_methods,
        )
        self._current_context = self._general_context
        self._frame_mappings = get_frame_mappings(builder.hw)
        self._port_mappings = get_port_mappings(builder.hw)

    def get_waveform_samples(self, waveform: Waveform | SampledWaveform) -> np.ndarray:
        if isinstance(waveform, SampledWaveform):
            return waveform.samples
        else:
            # TODO: how do we do this arbitarily? (COMPILER-752)
            dt = 0.5e-9
            samples = int(waveform.width / dt)
            midway_time = waveform.width / 2

            t = np.linspace(-midway_time, midway_time, samples)
            return waveform.sample(t).samples

    def _perform_signal_processing(
        self, name: str, args
    ) -> Waveform | SampledWaveform | None:
        if name == "mix":
            wf1, wf2 = args
            # TODO: just make getwfsamp take args
            samples1, samples2 = [self.get_waveform_samples(w) for w in (wf1, wf2)]
            output_length, pulse_length = (
                max(len(samples1), len(samples2)),
                min(len(samples1), len(samples2)),
            )
            output = np.append(
                np.array([1] * pulse_length, dtype=complex),
                np.array([0] * (output_length - pulse_length), dtype=complex),
            )
            for wave in (samples1, samples2):
                for i, val in enumerate(wave):
                    output[i] *= val

            return SampledWaveform(samples=output)

        elif name == "sum":
            wf1, wf2 = args
            # TODO: just make getwfsamp take args
            samples1, samples2 = [self.get_waveform_samples(w) for w in (wf1, wf2)]
            pulse_length = max(len(samples1), len(samples2))
            output = np.array([0] * pulse_length, dtype=complex)
            for wave in (samples1, samples2):
                for i, val in enumerate(wave):
                    output[i] += val
            return SampledWaveform(samples=output)

        elif name == "phase_shift":
            wf1: Waveform | SampledWaveform
            wf1, shift = args
            exp_shift = np.exp(1j * shift)
            if isinstance(wf1, SampledWaveform):
                wf1.samples = [exp_shift * val for val in wf1.samples]
            else:
                wf1.phase += shift
            return wf1

        elif name == "scale":
            wf1, scale = args
            if isinstance(wf1, SampledWaveform):
                wf1.samples = [scale * val for val in wf1.samples]
            else:
                wf1.scale_factor *= scale
            return wf1

        return None

    def _attempt_declaration(self, var: Variable):
        if var.name in self._current_context.variables:
            raise ValueError(f"Can't redeclare variable {var.name}")
        self._current_context.variables[var.name] = var

    def transform_to_value(self, child_tree, walk_variable=True, return_variable=False):
        if isinstance(child_tree, list):
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
                return self.builder.hw.qubit_with_index(int(id_value.strip("$")))

            qubits = self._current_context.registers.quantum.get(id_value, None)
            if qubits is not None:
                return qubits

            bits = self._current_context.registers.classic.get(id_value, None)
            if bits is not None:
                return bits

            def _walk_variables(var_id, guard: set[str] = None):
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

        if isinstance(port, IRPulseChannel):
            physical_channel_id = port.physical_channel_id
        elif isinstance(port, PhysicalChannel):
            physical_channel_id = port.uuid
        else:
            raise TypeError(
                f"Cannot create new frame from variable '{name}'. "
                "Must be either type Port or Frame."
            )

        pulse_channel = self.builder.create_pulse_channel(
            frequency=frequency, physical_channel=physical_channel_id
        )
        self.builder.phase_shift(target=pulse_channel, theta=phase)

        self._attempt_declaration(
            Variable(name=name, var_type=IRPulseChannel, value=pulse_channel)
        )

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
                    f"Frequency '{str(frequency)}' used in {intrinsic_name} is not a float."
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
            """As results length is dynamic, centralise validation and messages."""
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

        match intrinsic_name:
            case None:
                # This is a flat array of pulse values.
                array_contents = self.transform_to_value(tree.children[3])
                waveform = SampledWaveform(samples=array_contents)

            # TODO: implement non intrinsic waveforms.
            case "constant":
                width, amp = _validate_arg_length(tree.children[4], 2)
                _validate_waveform_args(width=width, amp=amp)
                waveform = SquareWaveform(width=width, amp=amp)

            case "rounded_square":
                width, std_dev, rise_time, amp = _validate_arg_length(tree.children[4], 4)
                _validate_waveform_args(
                    width=width, std_dev=std_dev, amp=amp, rise=rise_time
                )
                waveform = RoundedSquareWaveform(
                    width=width,
                    std_dev=std_dev,
                    amp=amp,
                    rise=rise_time,
                )

            case "drag":
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
                waveform = DragGaussianWaveform(
                    width=width,
                    amp=amp,
                    zero_at_edges=zero_at_edges,
                    beta=beta,
                    std_dev=std_dev,
                )

            case "gaussian":
                amp, width, std_dev = _validate_arg_length(tree.children[4], 3)
                _validate_waveform_args(width=width, amp=amp, std_dev=std_dev)
                waveform = GaussianZeroEdgeWaveform(
                    width=width,
                    amp=amp,
                    std_dev=std_dev,
                    zero_at_edges=0,
                )

            case "gaussian_zero_edge":
                amp, width, std_dev, zero_at_edges = _validate_arg_length(
                    tree.children[4], 4
                )
                zero_at_edges = bool(zero_at_edges)
                _validate_waveform_args(
                    width=width, amp=amp, zero_at_edges=zero_at_edges, std_dev=std_dev
                )
                waveform = GaussianZeroEdgeWaveform(
                    width=width,
                    amp=amp,
                    zero_at_edges=zero_at_edges,
                    std_dev=std_dev,
                )

            case "sech":
                amp, width, std_dev = _validate_arg_length(tree.children[4], 3)
                waveform = SechWaveform(width=width, amp=amp, std_dev=std_dev)

            case "gaussian_square":
                amp, width, square_width, std_dev, zero_at_edges = _validate_arg_length(
                    tree.children[4], 4, 5
                )
                zero_at_edges = bool(zero_at_edges)
                _validate_waveform_args(
                    width=width,
                    amp=amp,
                    square_width=square_width,
                    std_dev=std_dev,
                    zero_at_edges=zero_at_edges,
                )
                waveform = GaussianSquareWaveform(
                    width=width,
                    std_dev=std_dev,
                    amp=amp,
                    zero_at_edges=zero_at_edges,
                    square_width=square_width,
                )

            case "sine":
                amp, width, frequency, phase = _validate_arg_length(tree.children[4], 4)
                _validate_waveform_args(
                    width=width, amp=amp, frequency=frequency, phase=phase
                )
                waveform = SinWaveform(
                    amp=amp,
                    width=width,
                    frequency=frequency,
                    phase=phase,
                )

            case "gaussian_rise":
                amp, width, rise, drag, phase = _validate_arg_length(tree.children[4], 5)
                _validate_waveform_args(
                    width=width, rise=rise, amp=amp, drag=drag, phase=phase
                )
                waveform = GaussianWaveform(
                    amp=amp,
                    width=width,
                    rise=rise,
                    drag=drag,
                    phase=phase,
                )

            case "soft_square_rise":
                amp, width, rise, drag, phase = _validate_arg_length(tree.children[4], 5)
                _validate_waveform_args(
                    width=width, rise=rise, amp=amp, drag=drag, phase=phase
                )
                waveform = SoftSquareWaveform(
                    amp=amp,
                    width=width,
                    rise=rise,
                    drag=drag,
                    phase=phase,
                )

            case _:

                def snake_case_to_CamelCase(input: str):
                    return input.replace("_", " ").title().replace(" ", "")

                waveform_type = locate(
                    "qat.ir.waveforms."
                    + snake_case_to_CamelCase(intrinsic_name)
                    + "Waveform"
                )
                if waveform_type is None:
                    raise ValueError(f"Unknown waveform {intrinsic_name}.")

                width, amp = _validate_arg_length(tree.children[4], 2)
                _validate_waveform_args(width=width, amp=amp)
                waveform = waveform_type(width=width, amp=amp)

        self._attempt_declaration(
            Variable(name=assigned_variable, var_type=AbstractWaveform, value=waveform)
        )

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

        qubits = qubits if isinstance(qubits, list) else [qubits]

        if isinstance(bits, BitRegister):
            bits = bits.bits
        bits = bits if isinstance(bits, list | ValidatedList) else [bits]

        # If the measure for this particular qubit has been overriden the functionality
        # is distinctly different.
        if self._has_defcal_override("measure", qubits):
            results = self._call_gate("measure", qubits)
            if results is not None:
                results = results if isinstance(results, list) else [results]
                if len(bits) != len(results):
                    raise ValueError("Can't flatten overriden measure into assignment.")

                bit: CregIndexValue
                for bit, result in zip(bits, results):
                    bit.value = result
        else:
            self._q3_patcher.add_measure(qubits, bits, self.builder)

    def _has_defcal_override(
        self,
        name: str,
        qubits: list[Qubit],
        argument_values: list[Any] | None = None,
    ):
        """
        Returns whether this gate has been overriden, either in a generic
        or qubit-specific manner.
        """
        argument_values = argument_values or []
        qubits = qubits if isinstance(qubits, list) else [qubits]
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
            if isinstance(value, list):
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

        if name in ("u", "U"):
            # u is not in openpulse grammar so cannot be overridden...
            self._q3_patcher.add_unitary(
                others[0], others[1], others[2], qubits, self.builder
            )

        # Prioritize calibration definitions here if people override the base functions.
        # We also don't care about qubit scoping and restrictions.
        elif cal_def is not None:
            # Implied 'barrier' between defcals. To make things simple just assume that
            # everything in the defcal focuses on the qubits coming in.
            self.builder.synchronize(qubits)
            existing_context = self._current_context
            self._current_context = OpenPulseContext(
                registers=self._general_context.registers,
                gates=self._general_context.gates,
                variables=dict(self._calibration_context.variables),
                calibration_methods=self._general_context.calibration_methods,
            )

            arg_mappings, qubit_mappings, body = cal_def
            if not expr_list:
                # Strip off the type as we have no need for it here.
                # No type given if an expression list
                arg_mappings = [name for _, name in arg_mappings]

            if not isinstance(arg_mappings, list):
                arg_mappings = [arg_mappings]

            if not isinstance(qubit_mappings, list):
                qubit_mappings = [qubit_mappings]

            if len(arg_mappings) != len(others):
                raise ValueError(
                    f"Call to '{name}' needs {len(arg_mappings)} arguments. "
                    f"Has {len(others)}."
                )

            for arg_name, value in zip(arg_mappings, others):
                self._attempt_declaration(
                    Variable(name=str(arg_name), var_type=type(value), value=value)
                )

            for qb_name, value in zip(qubit_mappings, qubits):
                # If we resolved to a qubit already, we're a physical qubit.
                if isinstance(qb_name, (QubitRegister, Qubit)):
                    continue

                self._attempt_declaration(
                    Variable(name=qb_name, var_type=type(value), value=value)
                )

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
            node = ast.QuantumGate(
                name=ast.Identifier(name), qubits=qubits, arguments=others, modifiers=[]
            )
            self._q3_patcher.visit(node, self._current_context)
        elif name in ("cnot", "CNOT"):
            self._q3_patcher.add_cnot(qubits[0], qubits[1], self.builder)
        elif name in ("ecr", "ECR"):
            self._q3_patcher.add_ecr(qubits, self.builder)
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
        targets = self.transform_to_value(tree)
        self.builder.reset(targets)

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
                variable = self._current_context.variables.get(var_name, None)
                if variable is None:
                    raise ValueError(f"Can't return {var_name} as it doesn't exist.")
            else:
                variable = Variable.with_random_name()
                variable.value = variable
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
                variable = str(args[1])

        self._q3_patcher.add_qreg(variable, length, self._current_context, self.builder)

    def bit_declaration_statement(self, tree: Tree):
        args = self.transform_to_value(tree)
        if isinstance(args, str):
            length = 1
            variable = args
        else:
            # If you are using the soon-to-be-depreciated creg,
            # arguments are switched.
            if isinstance(args[0], str):
                length = args[1]
                variable = args[0]
            else:
                length = args[0]
                variable = args[1]

        self._q3_patcher.add_creg(variable, length, self._current_context)

    def complex_declaration_statement(self, tree: Tree):
        _, name = self.transform_to_value(tree)

        self._attempt_declaration(Variable(name=name, var_type=complex))

    def _create_qb_specific_gate_suffix(self, name: str, target_qubits: list[Qubit]):
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

        target_qubits = (
            target_qubits if isinstance(target_qubits, list) else [target_qubits]
        )
        classic_args = classic_args if isinstance(classic_args, list) else [classic_args]

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

        target_qubits = (
            target_qubits if isinstance(target_qubits, list) else [target_qubits]
        )
        if all(isinstance(val, (Qubit, QubitRegister)) for val in target_qubits):
            gate_name = self._create_qb_specific_gate_suffix(gate_name, target_qubits)

        # Not technically a calibration method, but the way to call is the same.
        self._current_context.calibration_methods[gate_name] = (
            classic_args,
            target_qubits,
            body,
        )

    def _get_phase(self, pulse_channel: IRPulseChannel):
        phase = 0
        for inst in self.builder.instructions:
            inst: QuantumInstruction
            if isinstance(inst, PhaseShift) and pulse_channel.uuid in inst.targets:
                phase += inst.phase
            elif isinstance(inst, PhaseReset) and pulse_channel.uuid in inst.targets:
                phase = 0
        return phase

    def _get_frequency(self, pulse_channel: IRPulseChannel):
        frequency = pulse_channel.frequency
        for inst in self.builder.instructions:
            if isinstance(inst, FrequencyShift) and pulse_channel.uuid in inst.targets:
                frequency += inst.frequency
        return frequency

    def _capture_iq_value(
        self,
        pulse_channel: IRPulseChannel,
        duration: float,
        output_variable: str,
        filter: Pulse = None,
    ):
        # The acquire integrator mode means that for every acquired resonator response
        # signal within a group of shots, we integrate along the time axis to find the
        # average amplitude of the response.
        #
        # The returned value for each shot after postprocessing is a complex iq value.
        qubit = self.builder.hw.qubit_for_physical_channel_id(
            pulse_channel.physical_channel_id
        )
        if qubit is None:
            raise TypeError(
                f"Pulse channel {pulse_channel} is not assigned to any known qubit."
            )

        # This kind of logic to add delays is really part of a measure definition, and
        # not within the semantics of a capture command, which is essentially an acqurire.
        # I don't think this is necessary to do, but I won't change it for now...
        hw_pulse_channel = self.builder.hw.pulse_channel_with_id(pulse_channel.uuid)
        if isinstance(hw_pulse_channel, AcquirePulseChannel):
            delay = hw_pulse_channel.acquire.delay
        else:
            log.warning(
                f"The acquire channel {pulse_channel.uuid} is not an acquire channel: "
                "setting the delay to 0.0."
            )
            delay = 0.0

        acquire = Acquire(
            targets=pulse_channel.uuid,
            duration=duration,
            mode=AcquireMode.INTEGRATOR,
            output_variable=output_variable,
            filter=filter,
            delay=delay,
        )
        self.builder.add(acquire)
        self.builder.post_processing(
            target=qubit,
            process_type=PostProcessType.MEAN,
            axes=[ProcessAxis.TIME],
            output_variable=output_variable,
        )

        return acquire

    def _validate_channel_args(
        self, pulse_channel: IRPulseChannel, val_type: str, value=None
    ):
        if not isinstance(pulse_channel, IRPulseChannel):
            raise ValueError(f"{str(pulse_channel)} is not a valid pulse channel.")

        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"{str(value)} is not a valid {val_type}.")

        return pulse_channel, value

    def _validate_phase_args(self, pulse_channel: IRPulseChannel, phase=None):
        return self._validate_channel_args(pulse_channel, "phase", phase)

    def _validate_freq_args(self, pulse_channel: IRPulseChannel, frequency=None):
        return self._validate_channel_args(pulse_channel, "frequency", frequency)

    def extern_or_subroutine_call(self, tree: Tree):
        name = self.transform_to_value(tree.children[0])
        args = self.transform_to_value(tree.children[1])

        if name in ("mix", "sum", "phase_shift", "scale"):
            return self._perform_signal_processing(name, args)

        elif name == "play":
            waveform: Waveform | SampledWaveform = args[1]
            if not isinstance(waveform, Waveform | SampledWaveform):
                variable_name = self.transform_to_value(
                    tree.children[1].children[1], walk_variable=False
                )
                raise ValueError(
                    f"Play frame argument {variable_name} has not been linked to a frame."
                )

            pulse_target = args[0]
            if not isinstance(pulse_target, IRPulseChannel):
                variable_name = self.transform_to_value(tree.children[1].children[0])
                raise ValueError(
                    f"Play waveform argument {variable_name} does not point to a waveform."
                )

            waveform_blob = deepcopy(waveform)
            pulse = Pulse(
                targets=pulse_target.uuid, waveform=waveform_blob, ignore_channel_scale=True
            )

            self.builder.add(pulse)

        elif name == "shift_phase":
            pulse_channel, phase = self._validate_phase_args(args[0], args[1])
            self.builder.add(PhaseShift(targets=pulse_channel.uuid, phase=phase))

        elif name == "set_phase":
            pulse_channel, phase_arg = self._validate_phase_args(args[0], args[1])
            phase = self._get_phase(pulse_channel)
            self.builder.add(
                PhaseShift(targets=pulse_channel.uuid, phase=phase_arg - phase)
            )

        elif name == "get_phase":
            pulse_channel, _ = self._validate_phase_args(args)
            return self._get_phase(pulse_channel)

        elif name == "set_frequency":
            pulse_channel, freq_arg = self._validate_freq_args(args[0], args[1])
            frequency = self._get_frequency(pulse_channel)
            self.builder.add(
                FrequencyShift(targets=pulse_channel.uuid, frequency=freq_arg - frequency)
            )

        elif name == "get_frequency":
            pulse_channel, _ = self._validate_freq_args(args)
            return self._get_frequency(pulse_channel)

        elif name == "shift_frequency":
            pulse_channel, freq_arg = self._validate_freq_args(args[0], args[1])
            self.builder.add(FrequencyShift(targets=pulse_channel.uuid, frequency=freq_arg))

        elif name == "capture_v0":
            # Not sure what this method should return.
            raise NotImplementedError(
                "capture_v0 is not yet implemented. Please use capture_v1, capture_v2 or "
                "capture_v3 instead."
            )

        elif name == "capture_v1":
            # A capture command that returns an iq value
            variable: Variable = Variable.with_random_name()
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
            variable: Variable = Variable.with_random_name()
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
                for q in self.builder.hw.qubits.values():
                    if q.resonator.pulse_channels.pulse_channel_with_id(pulse_channel.uuid):
                        mean_z_map_args = q.mean_z_map_args
                        break
            if mean_z_map_args is None:
                keys = next(
                    key
                    for key, value in self._frame_mappings.items()
                    if value == pulse_channel
                )
                raise ValueError(f"Could not find mean_z_map_args for frame {keys}.")
            self.builder.post_processing(
                target=q,
                process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                args=mean_z_map_args,
                output_variable=acquire.output_variable,
            )
            self.builder.results_processing(
                variable=variable.name, res_format=InlineResultsProcessing.Program
            )

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
            # greater than 1 shots if you are measuring a qubit in superposition, as the
            # different signal responses from the different measurement results will
            # average out.
            variable: Variable = Variable.with_random_name()
            pulse_channel = args[0]

            acquire = Acquire(
                targets=pulse_channel.uuid,
                duration=args[1],
                mode=AcquireMode.SCOPE,
                output_variable=variable.name,
                filter=args[2] if len(args) > 2 else None,
            )
            qubit = None
            for q in self.builder.hw.qubits.values():
                if q.resonator.pulse_channels.pulse_channel_with_id(pulse_channel.uuid):
                    qubit = q
            if not isinstance(qubit, Qubit):
                raise ValueError(
                    f"Pulse channel with id {pulse_channel.uuid} is not associated with any quantum device."
                )

            self.builder.add(acquire)
            self.builder.post_processing(
                target=qubit,
                process_type=PostProcessType.MEAN,
                axes=[ProcessAxis.SEQUENCE],
                output_variable=acquire.output_variable,
            )
            self._attempt_declaration(variable)
            return variable

        elif name == "capture_v4":
            raise NotImplementedError(
                "capture_v4 is not implemented. Please use capture_v1, capture_v2 or "
                "capture_v3 instead."
            )
        else:
            raise ValueError(f"Extern {name} not implemented.")

    def extern_frame(self, tree: Tree):
        name = self.transform_to_value(tree, walk_variable=False)
        hwm_pulse_channel = self._frame_mappings.get(name, None)
        if hwm_pulse_channel is None:
            raise ValueError(f"Could not find extern Frame with name '{name}'.")
        pulse_channel = self.builder.get_pulse_channel(hwm_pulse_channel.uuid)

        self._attempt_declaration(
            Variable(name=name, var_type=IRPulseChannel, value=pulse_channel)
        )

    def extern_port(self, tree: Tree):
        name = self.transform_to_value(tree, walk_variable=False)
        physical_channel = self._port_mappings.get(name, None)
        if physical_channel is None:
            raise ValueError(f"Could not find extern Port with name '{name}'.")

        self._attempt_declaration(
            Variable(name=name, var_type=PhysicalChannel, value=physical_channel)
        )

    def frame_attribute_assignment(self, tree: Tree):
        args = self.transform_to_value(tree)
        pulse_channel = args[0][0]
        if not isinstance(pulse_channel, IRPulseChannel):
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
            instr_info = {"targets": pulse_channel.uuid, field: value - current_value}
        elif op == "+=":
            instr_info = {"targets": pulse_channel.uuid, field: value}
        elif op == "-=":
            instr_info = {"targets": pulse_channel.uuid, field: -value}
        else:
            raise ValueError(f"Attempted to use an unknown frame operator '{op}'.")
        self.builder.add(inst(**instr_info))

    def assignment(self, tree: Tree):
        register = self.transform_to_value(tree.children[0], walk_variable=False)
        op = self.transform_to_value(tree.children[1])
        value = self.transform_to_value(tree.children[2])

        if op != "=":
            raise ValueError(f"Assignment operator {op} is unsupported.")

        if isinstance(register, CregIndexValue):
            register = register.register_name

        existing = self._current_context.variables.get(register, None)
        if existing is not None:
            existing.value = value
        else:
            self._attempt_declaration(
                value
                if isinstance(value, Variable)
                else Variable(name=register, var_type=type(value), value=value)
            )

    def cal_block(self, tree: Tree):
        self._has_open_pulse = True
        existing_context = self._current_context
        self._current_context = self._calibration_context
        self.visit_children(tree)
        self._current_context = existing_context

    def parse(self, builder: QuantumInstructionBuilder, qasm_str: str):
        self.initialize(builder)
        parsed = self._fetch_or_parse(qasm_str)

        if (qasm_id := hash(qasm_str)) in self._cached_parses:
            del self._cached_parses[qasm_id]

        # If we have a new results format, propagate it.
        self._q3_patcher.results_format = self.results_format

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
