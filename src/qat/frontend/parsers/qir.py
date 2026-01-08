# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2025 Oxford Quantum Circuits Ltd
import uuid
from enum import Enum
from importlib.metadata import version

from compiler_config.config import InlineResultsProcessing

from qat.ir.instruction_builder import InstructionBuilder
from qat.purr.utils.logging_utils import log_duration

qir_available = True
try:
    from pyqir import (
        Call,
        Constant,
        Context,
        FloatConstant,
        IntConstant,
        Module,
        extract_byte_string,
        is_entry_point,
    )

    if version("pyqir") < "0.12.0":
        # TODO: Drop PyTket < 0.12.0 compatibility, COMPILER-909
        from pyqir import (
            qubit_id,
            result_id,
        )

        def _convert_args(
            builder: InstructionBuilder,
            args: list[Constant],
            expected_types: list["ArgumentType"],
        ) -> list:
            """Converts QIR arguments to appropriate types for the instruction builder."""

            for i, arg in enumerate(args):
                if isinstance(arg, (IntConstant, FloatConstant)):
                    args[i] = arg.value
                elif (qid := qubit_id(arg)) is not None:
                    # Checks to see if this is a qubit
                    args[i] = builder.get_logical_qubit(qid)
                elif (rid := result_id(arg)) is not None:
                    # Checks to see if this is a result
                    args[i] = str(rid)
                elif expected_types[i] == ArgumentType.Label:
                    if not arg.is_null and (byte_string := extract_byte_string(arg)):
                        args[i] = byte_string.decode("utf-8").rstrip("\00")
                    else:
                        args[i] = None

            return args

    else:
        from pyqir import ptr_id

        def _convert_args(
            builder: InstructionBuilder,
            args: list[Constant],
            expected_types: list["ArgumentType"],
        ) -> list:
            """Converts QIR arguments to appropriate types for the instruction builder,
            given expected argument types.
            """

            for i, (arg, expected_type) in enumerate(zip(args, expected_types)):
                if expected_type == ArgumentType.Qubit:
                    if (qid := ptr_id(arg)) is not None:
                        args[i] = builder.get_logical_qubit(qid)
                    else:
                        raise ValueError(f"Argument {i} is not a valid qubit.")
                elif expected_type == ArgumentType.Result:
                    if (rid := ptr_id(arg)) is not None:
                        args[i] = str(rid)
                    else:
                        raise ValueError(f"Argument {i} is not a valid result.")
                elif expected_type == ArgumentType.Number:
                    if isinstance(arg, (IntConstant, FloatConstant)):
                        args[i] = arg.value
                    else:
                        raise ValueError(f"Argument {i} is not a valid number.")
                elif expected_type == ArgumentType.Label:
                    if not arg.is_null and (byte_string := extract_byte_string(arg)):
                        args[i] = byte_string.decode("utf-8").rstrip("\00")
                    else:
                        args[i] = None

            return args

except ImportError:
    qir_available = False


class ArgumentType(Enum):
    Qubit = 1
    Result = 2
    Number = 3
    Label = 4


class QIRParser:
    """Used to parse QIR files and walk the AST to assemble a quantum program.

    The parser makes API calls to the instruction builder passed at instantiation to allow
    for dynamic building of programs. This object is only responsible for interpreting
    and walking the QIR program.
    """

    def __init__(
        self,
        results_format: InlineResultsProcessing | None = None,
    ):
        self._results_format = (
            results_format
            if results_format is not None
            else InlineResultsProcessing.Program
        )
        self._result_variables = []

    def parse(self, builder: InstructionBuilder, qir_file: str):
        if not qir_available:
            raise RuntimeError("QIR parser unavailable.")

        with log_duration("QIR parsing completed, took {} seconds."):
            # Load the file if required
            if isinstance(qir_file, str):
                if qir_file.endswith(".bc"):
                    with open(qir_file, "rb") as f:
                        qir_file = f.read()
                elif qir_file.endswith(".ll"):
                    with open(qir_file) as f:
                        qir_file = f.read()

            # Create a module by inferring the type
            if isinstance(qir_file, str):
                mod = Module.from_ir(Context(), qir_file)
            elif isinstance(qir_file, bytes):
                mod = Module.from_bitcode(Context(), qir_file)
            else:
                raise ValueError(f"Expected type str | bytes, got {type(mod)} instead.")
            self._process_instructions(builder, mod)
            return builder

    @staticmethod
    def _throw_on_invalid_args(intrinsic_name: str, actual_args: int, expected_args: int):
        if actual_args != expected_args:
            raise ValueError(
                f"{intrinsic_name} has {actual_args} arguments, expected {expected_args}."
            )

    def _process_instruction(self, builder: InstructionBuilder, instruction):
        """Dispatches the correct calls to the builder given the type of instruction.

        Considered using a state dispatch approach, but the details get a little tricky
        and might add confusion. The benefit would be being able to write the instrinsic
        name next to explicit arguments and builder calls.
        """
        if not isinstance(instruction, Call):
            return

        intrinsic_name = instruction.callee.name
        args = instruction.args
        num_args = len(args)

        match intrinsic_name:
            case "__quantum__qis__ccx__body" | "__quantum____qis__ccnot__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 3)
                args = _convert_args(
                    builder,
                    args,
                    [ArgumentType.Qubit, ArgumentType.Qubit, ArgumentType.Qubit],
                )
                builder.ccnot(*args)
            case "__quantum__qis__cnot__body" | "__quantum__qis__cx__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 2)
                args = _convert_args(
                    builder, args, [ArgumentType.Qubit, ArgumentType.Qubit]
                )
                builder.cnot(*args)
            case "__quantum__qis__cz__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 2)
                args = _convert_args(
                    builder, args, [ArgumentType.Qubit, ArgumentType.Qubit]
                )
                builder.cZ(*args)
            case "__quantum__qis__h__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 1)
                args = _convert_args(builder, args, [ArgumentType.Qubit])
                builder.had(*args)
            case "__quantum__qis__mz__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 2)
                args = _convert_args(
                    builder, args, [ArgumentType.Qubit, ArgumentType.Result]
                )
                builder.measure_single_shot_z(args[0], output_variable=args[1])
                builder.results_processing(args[1], self._results_format)
            case "__quantum__qis__reset__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 1)
                args = _convert_args(builder, args, [ArgumentType.Qubit])
                builder.reset(*args)
            case "__quantum__qis__rx__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 2)
                args = _convert_args(
                    builder, args, [ArgumentType.Number, ArgumentType.Qubit]
                )
                builder.X(args[1], args[0])
            case "__quantum__qis__ry__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 2)
                args = _convert_args(
                    builder, args, [ArgumentType.Number, ArgumentType.Qubit]
                )
                builder.Y(args[1], args[0])
            case "__quantum__qis__rz__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 2)
                args = _convert_args(
                    builder, args, [ArgumentType.Number, ArgumentType.Qubit]
                )
                builder.Z(args[1], args[0])
            case "__quantum__qis__s__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 1)
                args = _convert_args(builder, args, [ArgumentType.Qubit])
                builder.S(*args)
            case "__quantum__qis__s_adj":
                self._throw_on_invalid_args(intrinsic_name, num_args, 1)
                args = _convert_args(builder, args, [ArgumentType.Qubit])
                builder.Sdg(*args)
            case "__quantum__qis__t__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 1)
                args = _convert_args(builder, args, [ArgumentType.Qubit])
                builder.T(*args)
            case "__quantum__qis__t__adj":
                self._throw_on_invalid_args(intrinsic_name, num_args, 1)
                args = _convert_args(builder, args, [ArgumentType.Qubit])
                builder.Tdg(*args)
            case "__quantum__qis__x__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 1)
                args = _convert_args(builder, args, [ArgumentType.Qubit])
                builder.X(*args)
            case "__quantum__qis__y__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 1)
                args = _convert_args(builder, args, [ArgumentType.Qubit])
                builder.Y(*args)
            case "__quantum__qis__z__body":
                self._throw_on_invalid_args(intrinsic_name, num_args, 1)
                args = _convert_args(builder, args, [ArgumentType.Qubit])
                builder.Z(*args)
            case "__quantum__rt__result_record_output":
                self._throw_on_invalid_args(intrinsic_name, num_args, 2)

                # not exactly clear if there is always a way to identify the type without
                # the instrinsic for context, so handle this here for now
                args = _convert_args(
                    builder,
                    args,
                    [ArgumentType.Result, ArgumentType.Label],
                )
                res = args[0]
                label = args[1]
                if label is None:
                    label = "generated_name_" + str(uuid.uuid4())
                self._result_variables.append((str(res), label))

            # currently do nothing; here for documentation purposes
            case "__quantum__rt__initialize":
                pass
            case "__quantum__rt__tuple_record_output":
                pass
            case "__quantum__rt__array_record_output":
                pass
            case _:
                pass

    def _process_instructions(self, builder: InstructionBuilder, module: Module):
        """Processes all instructions in the entry point function of the module."""

        entry_point = next(
            iter(filter(lambda fnc: is_entry_point(fnc), module.functions)), None
        )
        if entry_point is None:
            raise ValueError("Entry point unable to be found in QIR file.")

        instructions = [inst for bb in entry_point.basic_blocks for inst in bb.instructions]
        for inst in instructions:
            self._process_instruction(builder, inst)

        if any(self._result_variables):
            result_name = "_".join([val[1] for val in self._result_variables])

            builder.assign(result_name, [val[0] for val in self._result_variables])
            builder.returns(result_name)
        else:
            builder.returns()
