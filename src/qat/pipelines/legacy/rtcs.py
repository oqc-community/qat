# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.fallthrough import FallthroughBackend
from qat.core.pass_base import PassManager
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.legacy.transform import (
    IntegratorAcquireSanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
)
from qat.middleend.passes.legacy.validation import (
    HardwareConfigValidity,
    InstructionValidation,
    ReadoutValidation,
)
from qat.model.loaders.legacy import RTCSModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.legacy.base import LegacyPipeline
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import PipelineConfig
from qat.purr.backends.realtime_chip_simulator import RealtimeSimHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.legacy import LegacyRuntime
from qat.runtime.passes.legacy.analysis import CalibrationAnalysis

log = get_default_logger()


class LegacyRTCSPipelineConfig(PipelineConfig):
    """Configuration for the :class:`LegacyRTCSPipeline`."""

    name: str = "legacy_rtcs"


class LegacyRTCSPipeline(LegacyPipeline):
    """A pipeline that compiles programs using the legacy echo backend and executes them
    using the :class:`LegacyRuntime`.

    Implements a custom pipeline to make instructions suitable for the legacy echo engine.
    """

    @staticmethod
    def _build_pipeline(
        config: LegacyRTCSPipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None = None,
        engine: None = None,
    ) -> Pipeline:
        if not isinstance(model, RealtimeSimHardwareModel):
            raise ValueError("Model must be an instance of RealtimeSimHardwareModel.")

        if engine is not None:
            log.warning(
                "An engine was provided to the LegacyRTCSPipeline, but it will be ignored. "
                "The legacy RealTimeChipSimEngine is used directly."
            )

        target_data = target_data if target_data is not None else TargetData.default()
        return Pipeline(
            name=config.name,
            model=model,
            target_data=target_data if target_data is not None else TargetData.default(),
            frontend=AutoFrontend(model),
            middleend=CustomMiddleend(
                model,
                pipeline=LegacyRTCSPipeline._middleend_pipeline(
                    model=model, target_data=target_data
                ),
            ),
            backend=FallthroughBackend(model),
            runtime=LegacyRuntime(
                engine=model.create_engine(),
                results_pipeline=LegacyRTCSPipeline._results_pipeline(model),
            ),
        )

    @staticmethod
    def _middleend_pipeline(
        model: QuantumHardwareModel, target_data: TargetData | None = None
    ) -> PassManager:
        return (
            PassManager()
            | HardwareConfigValidity(model)
            | CalibrationAnalysis()
            | PhaseOptimisation()
            | IntegratorAcquireSanitisation()
            | PostProcessingSanitisation()
            | InstructionValidation(target_data)
            | ReadoutValidation(model)
        )


legacy_rtcs2 = LegacyRTCSPipeline(
    config=LegacyRTCSPipelineConfig(name="legacy_rtcs2"),
    loader=RTCSModelLoader(),
).pipeline
