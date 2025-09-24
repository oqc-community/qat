# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.engines.waveform_v1 import EchoEngine
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import ExecutePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.utils.logger import get_default_logger
from qat.runtime import SimpleRuntime
from qat.runtime.results_pipeline import (
    get_experimental_results_pipeline,
)

log = get_default_logger()


class EchoExecutePipeline(UpdateablePipeline):
    """A pipeline that executes
    :class:`WaveformV1Executable <qat.backend.waveform_v1.executeable.WaveformV1Executable>`
    using the :class:`EchoEngine`.

    An engine cannot be provided to the pipeline, as the EchoEngine is used directly.

    .. warning::

        This pipeline is experimental and still in progress. Please use with caution.

        It is intended for executing compiled programs, and is not capable of compilation.
        Please use an appropriate compilation pipeline to prepare programs for execution.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: PhysicalHardwareModel,
        target_data: TargetData | None,
        engine: None = None,
    ) -> ExecutePipeline:
        """Constructs a pipeline equipped with the :class:`PydWaveformV1Backend`
        and :class:`EchoEngine`."""

        if engine is not None:
            log.warning(
                "An engine was provided to the EchoPipeline, but it will be ignored. "
                "The EchoEngine is used directly."
            )

        target_data = target_data if target_data is not None else TargetData.default()
        results_pipeline = get_experimental_results_pipeline(model, target_data)
        return ExecutePipeline(
            model=model,
            target_data=target_data,
            runtime=SimpleRuntime(engine=EchoEngine(), results_pipeline=results_pipeline),
            name=config.name,
        )
