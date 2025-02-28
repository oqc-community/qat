import abc
from typing import Optional

from compiler_config.config import CompilerConfig

from qat.backend.validation_passes import HardwareConfigValidity
from qat.compiler.transform_passes import PhaseOptimisation, PostProcessingSanitisation
from qat.compiler.validation_passes import ReadoutValidation
from qat.passes.metrics_base import MetricsManager
from qat.passes.pass_base import PassManager
from qat.passes.result_base import ResultManager
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.analysis_passes import CalibrationAnalysis


class BaseMiddleend(abc.ABC):
    def __init__(self, model: None | QuantumHardwareModel):
        self.model = model

    @abc.abstractmethod
    def emit(
        self,
        ir,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
    ): ...


class CustomMiddleend(BaseMiddleend):
    def __init__(
        self, model: None | QuantumHardwareModel, pipeline: None | PassManager = None
    ):
        self.pipeline = pipeline
        super().__init__(model=model)

    def emit(
        self,
        ir,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
    ):
        ir = self.pipeline.run(ir, res_mgr, met_mgr, compiler_config=compiler_config)
        return ir


class FallthroughMiddleend(CustomMiddleend):
    def __init__(self, model: None = None):
        super().__init__(model=None, pipeline=PassManager())


class DefaultMiddleend(CustomMiddleend):
    def __init__(self, model: QuantumHardwareModel):
        pipeline = self.build_pass_pipeline(model)
        super().__init__(model=model, pipeline=pipeline)

    @staticmethod
    def build_pass_pipeline(model) -> PassManager:
        return (
            PassManager()
            | HardwareConfigValidity(model)
            | CalibrationAnalysis()
            | PhaseOptimisation()
            | PostProcessingSanitisation()
            | ReadoutValidation(model)
        )
