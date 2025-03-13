# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2024 Oxford Quantum Circuits Ltd
import uuid
from typing import List, Union

from compiler_config.config import InlineResultsProcessing

from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
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
        qubit_id,
        result_id,
    )
except ImportError:
    qir_available = False


class QIRParser:
    def __init__(
        self,
        hw_model: PydHardwareModel,
        builder: QuantumInstructionBuilder = None,
    ):
        self.hw_model = hw_model
        self.builder = builder or QuantumInstructionBuilder(hw_model)
        self.results_format = InlineResultsProcessing.Program
        self.result_variables = []

    def _get_qubit(self, i: int):
        return self.hw_model.qubit_with_index(i)

    def ccx(self, control1: int, control2: int, target: int):
        self.builder.ccnot(
            [self._get_qubit(control1), self._get_qubit(control2)],
            self._get_qubit(target),
        )

    def cx(self, control: int, target: int):
        self.builder.cnot(self._get_qubit(control), self._get_qubit(target))

    def cz(self, control: int, target: int):
        self.builder.cZ(self._get_qubit(control), self._get_qubit(target))

    def h(self, target: int):
        self.builder.had(self._get_qubit(target))

    def mz(self, qubit: int, target: int):
        result_var = str(target)
        self.builder.measure_single_shot_z(
            self._get_qubit(qubit), output_variable=result_var
        )
        self.builder.results_processing(result_var, self.results_format)

    def reset(self, target: int):
        self.builder.reset(self._get_qubit(target))

    def rx(self, theta: float, qubit: int):
        self.builder.X(self._get_qubit(qubit), theta)

    def ry(self, theta: float, qubit: int):
        self.builder.Y(self._get_qubit(qubit), theta)

    def rz(self, theta: float, qubit: int):
        self.builder.Z(self._get_qubit(qubit), theta)

    def s(self, qubit: int):
        self.builder.S(self._get_qubit(qubit))

    def s_adj(self, qubit: int):
        self.builder.Sdg(self._get_qubit(qubit))

    def t(self, qubit: int):
        self.builder.T(self._get_qubit(qubit))

    def t_adj(self, qubit: int):
        self.builder.Tdg(self._get_qubit(qubit))

    def x(self, qubit: int):
        self.builder.X(self._get_qubit(qubit))

    def y(self, qubit: int):
        self.builder.Y(self._get_qubit(qubit))

    def z(self, qubit: int):
        self.builder.Z(self._get_qubit(qubit))

    def process_instructions(self, instructions):
        for inst in instructions:
            if isinstance(inst, Call):

                def throw_on_invalid_args(actual_args, expected_args):
                    if actual_args != expected_args:
                        raise ValueError(
                            f"{intrinsic_name} has {actual_args} arguments, "
                            f"expected {expected_args}."
                        )

                args: List[Constant] = inst.args
                intrinsic_name = inst.callee.name
                if intrinsic_name in (
                    "__quantum__qis__ccx__body",
                    "__quantum__qis__ccnot__body",
                ):
                    throw_on_invalid_args(len(args), 3)
                    self.ccx(
                        qubit_id(args[0]),
                        qubit_id(args[1]),
                        qubit_id(args[2]),
                    )
                elif intrinsic_name in (
                    "__quantum__qis__cnot__body",
                    "__quantum__qis__cx__body",
                ):
                    throw_on_invalid_args(len(args), 2)
                    self.cx(qubit_id(args[0]), qubit_id(args[1]))
                elif intrinsic_name == "__quantum__qis__cz__body":
                    throw_on_invalid_args(len(args), 2)
                    self.cz(qubit_id(args[0]), qubit_id(args[1]))
                elif intrinsic_name == "__quantum__qis__h__body":
                    throw_on_invalid_args(len(args), 1)
                    self.h(qubit_id(args[0]))
                elif intrinsic_name == "__quantum__qis__mz__body":
                    throw_on_invalid_args(len(args), 2)
                    self.mz(qubit_id(args[0]), result_id(args[1]))
                elif intrinsic_name == "__quantum__qis__reset__body":
                    throw_on_invalid_args(len(args), 1)
                    self.reset(qubit_id(args[0]))
                elif intrinsic_name == "__quantum__qis__rx__body":
                    throw_on_invalid_args(len(args), 2)
                    radii: Union[IntConstant, FloatConstant] = args[0]
                    self.rx(radii.value, qubit_id(args[1]))
                elif intrinsic_name == "__quantum__qis__ry__body":
                    throw_on_invalid_args(len(args), 2)
                    radii: Union[IntConstant, FloatConstant] = args[0]
                    self.ry(radii.value, qubit_id(args[1]))
                elif intrinsic_name == "__quantum__qis__rz__body":
                    throw_on_invalid_args(len(args), 2)
                    radii: Union[IntConstant, FloatConstant] = args[0]
                    self.rz(radii.value, qubit_id(args[1]))
                elif intrinsic_name == "__quantum__qis__s__body":
                    throw_on_invalid_args(len(args), 1)
                    self.s(qubit_id(args[0]))
                elif intrinsic_name == "__quantum__qis__s_adj":
                    throw_on_invalid_args(len(args), 1)
                    self.s_adj(qubit_id(args[0]))
                elif intrinsic_name == "__quantum__qis__t__body":
                    throw_on_invalid_args(len(args), 1)
                    self.t(qubit_id(args[0]))
                elif intrinsic_name == "__quantum__qis__t__adj":
                    throw_on_invalid_args(len(args), 1)
                    self.t_adj(qubit_id(args[0]))
                elif intrinsic_name == "__quantum__qis__x__body":
                    throw_on_invalid_args(len(args), 1)
                    self.x(qubit_id(args[0]))
                elif intrinsic_name == "__quantum__qis__y__body":
                    throw_on_invalid_args(len(args), 1)
                    self.y(qubit_id(args[0]))
                elif intrinsic_name == "__quantum__qis__z__body":
                    throw_on_invalid_args(len(args), 1)
                    self.z(qubit_id(args[0]))
                elif intrinsic_name == "__quantum__rt__initialize":
                    pass
                elif intrinsic_name == "__quantum__rt__tuple_record_output":
                    pass
                elif intrinsic_name == "__quantum__rt__array_record_output":
                    pass
                elif intrinsic_name == "__quantum__rt__result_record_output":
                    throw_on_invalid_args(len(args), 2)
                    res = result_id(args[0])
                    label_ptr = args[1]
                    label = ""
                    if (not label_ptr.is_null) and (
                        byte_string := extract_byte_string(label_ptr)
                    ):
                        label = byte_string.decode("utf-8").rstrip("\x00")
                    else:
                        label = "generated-name_" + str(uuid.uuid4())
                    self.result_variables.append((str(res), label))

    def parse(self, qir_file: str):
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

            entry_point = next(
                iter(filter(lambda fnc: is_entry_point(fnc), mod.functions)), None
            )
            if entry_point is None:
                raise ValueError("Entry point unable to be found in QIR file.")

            self.process_instructions(
                [inst for bb in entry_point.basic_blocks for inst in bb.instructions]
            )

            if any(self.result_variables):
                result_name = "_".join([val[1] for val in self.result_variables])

                self.builder.assign(result_name, [val[0] for val in self.result_variables])
                self.builder.returns(result_name)
            else:
                self.builder.returns()

            complete_builder = self.builder
            self.builder = QuantumInstructionBuilder(self.hw_model)
            return complete_builder
