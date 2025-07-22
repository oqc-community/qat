# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pydantic import field_validator

from qat.core.pass_base import PassManager
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import ExecutePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.backends.live import LiveHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.connection import ConnectionMode
from qat.runtime.legacy import LegacyRuntime
from qat.runtime.passes.purr.analysis import IndexMappingAnalysis
from qat.runtime.passes.transform import ErrorMitigation, ResultTransform

log = get_default_logger()


def results_pipeline(model: QuantumHardwareModel) -> PassManager:
    return (
        PassManager()
        | ResultTransform()
        | IndexMappingAnalysis(model)
        | ErrorMitigation(model)
    )


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


class LegacyExecutePipeline(UpdateablePipeline):
    """A pipeline that executes programs using the legacy backend.

    .. warning::

        This pipeline is for execution purposes only and does not compile programs. Please
        select an appropriate compilation pipeline if you wish to compile programs before
        execution.
    """

    @staticmethod
    def _build_pipeline(
        config: LegacyPipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None = None,
        engine: None = None,
    ) -> ExecutePipeline:
        if engine is not None:
            log.warning(
                "The engine for the LegacyExecutionPipeline is expected to be provided by "
                "the model, and the provided engine will be ignored."
            )

        if isinstance(model, LiveHardwareModel):
            # Let the Runtime handle startup based on ConnectionMode
            engine = model.create_engine(startup_engine=False)
        else:
            engine = model.create_engine()

        target_data = target_data if target_data is not None else TargetData.default()
        return ExecutePipeline(
            name=config.name,
            model=model,
            target_data=target_data,
            runtime=LegacyRuntime(
                engine=engine,
                results_pipeline=results_pipeline(model),
                connection_mode=config.connection_mode,
            ),
        )
