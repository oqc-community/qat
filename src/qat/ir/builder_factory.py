# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from functools import singledispatchmethod

from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.compiler.builders import (
    QuantumInstructionBuilder as PurrQuantumInstructionBuilder,
)
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class BuilderFactory:
    @singledispatchmethod
    @staticmethod
    def create_builder(model):
        raise TypeError(
            f"Cannot find a builder for hardware model with type {type(model)}."
        )

    @create_builder.register
    @staticmethod
    def _(model: PhysicalHardwareModel) -> PydQuantumInstructionBuilder:
        return PydQuantumInstructionBuilder(model)

    @create_builder.register
    @staticmethod
    def _(model: QuantumHardwareModel) -> PurrQuantumInstructionBuilder:
        return model.create_builder()
