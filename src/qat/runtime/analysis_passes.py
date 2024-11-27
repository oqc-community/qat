from dataclasses import dataclass, field
from typing import List

from compiler_config.config import CompilerConfig

from qat.ir.pass_base import AnalysisPass, QatIR
from qat.ir.result_base import ResultInfoMixin, ResultManager
from qat.purr.backends.calibrations.remote import find_calibration
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.runtime import CalibrationWithArgs


@dataclass
class CalibrationAnalysisResult(ResultInfoMixin):
    calibration_executables: List[CalibrationWithArgs] = field(default_factory=list)


class CalibrationAnalysis(AnalysisPass):
    def __init__(self, compiler_config: CompilerConfig):
        self.compiler_config = compiler_config

    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        cal_blocks = [
            find_calibration(arg) for arg in self.compiler_config.active_calibrations
        ]
        res_mgr.add(CalibrationAnalysisResult(cal_blocks))
