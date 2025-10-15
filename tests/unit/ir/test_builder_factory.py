# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import pytest

from qat.ir.builder_factory import BuilderFactory
from qat.ir.instruction_builder import QuantumInstructionBuilder as PydInstructionBuilder
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.compiler.builders import QuantumInstructionBuilder as PurrInstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel as PurrHardwareModel
from qat.utils.hardware_model import generate_hw_model


class TestBuilderFactory:
    def test_create_builder_with_purr_hw(self):
        purr_hw = PurrHardwareModel()
        builder = BuilderFactory.create_builder(purr_hw)
        assert builder is not None
        assert isinstance(builder, PurrInstructionBuilder)

    def test_create_builder_with_pyd_hw(self):
        pyd_hw: PydHardwareModel = generate_hw_model(4)
        builder = BuilderFactory.create_builder(pyd_hw)
        assert builder is not None
        assert isinstance(builder, PydInstructionBuilder)

    @pytest.mark.parametrize("invalid_type", [42, 3.14, None, [], {}])
    def test_create_builder_with_invalid_type(self, invalid_type):
        with pytest.raises(
            TypeError,
            match=f"Cannot find a builder for hardware model with type {type(invalid_type)}.",
        ):
            BuilderFactory.create_builder(invalid_type)
