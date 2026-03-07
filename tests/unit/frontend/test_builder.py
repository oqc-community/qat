# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import pytest

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.frontend.builder import BuilderFrontend
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.model.loaders.purr import EchoModelLoader
from qat.purr.compiler.builders import QuantumInstructionBuilder as PurrBuilder


class TestBuilderFrontend:
    purr_model = EchoModelLoader().load()
    pyd_model = convert_purr_echo_hw_to_pydantic(purr_model)
    frontend = BuilderFrontend(pyd_model)

    @pytest.mark.parametrize(
        "src", ["not an instruction builder", 123, QuantumInstructionBuilder(pyd_model)]
    )
    def test_builder_frontend_rejects_non_builders(self, src):
        assert self.frontend.check_and_return_source(src) == False

    def test_builder_frontend_accepts_purr_builders(self):
        purr_builder = PurrBuilder(self.purr_model)
        result = self.frontend.check_and_return_source(purr_builder)
        assert result is purr_builder

    def test_emit(self):
        purr_builder = PurrBuilder(self.purr_model)
        qat_ir_builder = self.frontend.emit(purr_builder, ResultManager(), MetricsManager())
        assert isinstance(qat_ir_builder, QuantumInstructionBuilder)
