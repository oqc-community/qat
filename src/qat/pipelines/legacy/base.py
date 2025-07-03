# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pydantic import field_validator

from qat.backend.fallthrough import FallthroughBackend
from qat.core.pass_base import PassManager
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.legacy.transform import (
    LegacyPhaseOptimisation,
    PostProcessingSanitisation,
)
from qat.middleend.passes.legacy.validation import (
    HardwareConfigValidity,
    InstructionValidation,
    ReadoutValidation,
)
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.backends.live import LiveHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.connection import ConnectionMode
from qat.runtime.legacy import LegacyRuntime
from qat.runtime.passes.legacy.analysis import CalibrationAnalysis, IndexMappingAnalysis
from qat.runtime.passes.transform import ErrorMitigation, ResultTransform

log = get_default_logger()


class LegacyPipelineConfig(PipelineConfig):
    """Configuration for the :class:`LegacyPipeline`, extending :class:`PipelineConfig` with
    configurable connection modes."""

    name: str = "legacy"
    connection_mode: ConnectionMode = ConnectionMode.MANUAL

    @field_validator("connection_mode", mode="before")
    @classmethod
    def _flag_as_string(cls, v):
        if isinstance(v, str):
            return ConnectionMode[v]
        return v


class LegacyPipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the legacy backend and executes them using
    the :class:`LegacyRuntime`.

    The piepline uses the engine provided by the legacy model, and cannot be provided to
    the factory.
    """

    @staticmethod
    def _build_pipeline(
        config: LegacyPipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None = None,
        engine: None = None,
    ) -> Pipeline:
        if engine is not None:
            log.warning(
                "The engine for the LegacyPipeline is expected to be provided by the "
                "model, and the provided engine will be ignored."
            )

        if isinstance(model, LiveHardwareModel):
            # Let the Runtime handle startup based on ConnectionMode
            engine = model.create_engine(startup_engine=False)
        else:
            engine = model.create_engine()

        return Pipeline(
            name=config.name,
            model=model,
            target_data=target_data if target_data is not None else TargetData.default(),
            frontend=AutoFrontend(model),
            middleend=CustomMiddleend(
                model,
                pipeline=LegacyPipeline._middleend_pipeline(
                    model=model, target_data=target_data
                ),
            ),
            backend=FallthroughBackend(model),
            runtime=LegacyRuntime(
                engine=engine,
                results_pipeline=LegacyPipeline._results_pipeline(model),
                connection_mode=config.connection_mode,
            ),
        )

    @staticmethod
    def _middleend_pipeline(
        model: QuantumHardwareModel, target_data: TargetData
    ) -> PassManager:
        return (
            PassManager()
            | HardwareConfigValidity(model)
            | CalibrationAnalysis()
            | LegacyPhaseOptimisation()
            | PostProcessingSanitisation()
            | InstructionValidation(target_data)
            | ReadoutValidation(model)
        )

    @staticmethod
    def _results_pipeline(model: QuantumHardwareModel) -> PassManager:
        return (
            PassManager()
            | ResultTransform()
            | IndexMappingAnalysis(model)
            | ErrorMitigation(model)
        )
