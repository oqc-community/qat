# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from dataclasses import dataclass

import pytest

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import AnalysisPass, PassManager, TransformPass, ValidationPass
from qat.core.result_base import ResultInfoMixin, ResultManager
from qat.model.loaders.purr import EchoModelLoader
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import Instruction, Repeat, Sweep

from tests.unit.utils.builder_nuggets import resonator_spect


class InvalidInstruction:
    pass


@dataclass
class DummyResult(ResultInfoMixin):
    num_instructions: int = 0


class DummyAnalysis(AnalysisPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        result = DummyResult()
        result.num_instructions = len(builder.instructions)
        res_mgr.add(result)
        return builder


class DummyValidation(ValidationPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        for inst in builder.instructions:
            if not isinstance(inst, Instruction):
                raise ValueError(f"{inst} is not an valid instruction")
        return builder


class DummyTransform(TransformPass):
    def run(
        self,
        builder: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        builder.instructions = builder.instructions[::-1]
        return builder


def test_pass_manager():
    model = EchoModelLoader().load()
    builder = resonator_spect(model)
    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    pipline = PassManager() | DummyValidation() | DummyTransform() | DummyAnalysis()

    # Add an invalid instruction
    assert not isinstance(InvalidInstruction(), Instruction)
    builder.add(InvalidInstruction())
    with pytest.raises(ValueError):
        pipline.run(builder, res_mgr, met_mgr)

    builder = resonator_spect(model)
    pipline.run(builder, res_mgr, met_mgr)

    # Get the analysis result
    dummy_result = res_mgr.lookup_by_type(DummyResult)
    assert dummy_result.num_instructions == len(builder.instructions)

    # Verify that the instructions list has been reversed
    assert isinstance(builder.instructions[0], Repeat)
    assert isinstance(builder.instructions[-2], Sweep)
