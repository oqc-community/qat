from dataclasses import dataclass

import pytest

from qat.ir.pass_base import AnalysisPass, PassManager, TransformPass, ValidationPass
from qat.ir.result_base import ResultInfoMixin, ResultManager
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import Instruction, Repeat, Sweep
from tests.qat.utils.builder_nuggets import resonator_spect


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


class DummyValidation(ValidationPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        for inst in builder.instructions:
            if not isinstance(inst, Instruction):
                raise ValueError(f"{inst} is not an valid instruction")


class DummyTransform(TransformPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        builder.instructions = builder.instructions[::-1]


def test_pass_manager():
    model = get_default_echo_hardware()
    builder = resonator_spect(model)
    res_mgr = ResultManager()
    pipline = PassManager() | DummyValidation() | DummyTransform() | DummyAnalysis()

    # Add an invalid instruction
    assert not isinstance(InvalidInstruction(), Instruction)
    builder.add(InvalidInstruction())
    with pytest.raises(ValueError):
        pipline.run(builder, res_mgr)

    builder = resonator_spect(model)
    pipline.run(builder, res_mgr)

    # Get the analysis result
    dummy_result: DummyResult = res_mgr.lookup_by_type(DummyResult)
    assert dummy_result.num_instructions == len(builder.instructions)

    # Verify that the instructions list has been reversed
    assert isinstance(builder.instructions[0], Repeat)
    assert isinstance(builder.instructions[-1], Sweep)
