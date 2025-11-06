# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from qat.engines.waveform_v1 import EchoEngine
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import ExecutePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime import SimpleRuntime
from qat.runtime.results_pipeline import (
    get_default_results_pipeline,
)

log = get_default_logger()


class EchoExecutePipeline(UpdateablePipeline):
    """A pipeline that executes :class:`Executable <qat.executables.Executable>` with
    :class:`WaveformV1Program <qat.backend.waveform_v1.executable.WaveformV1Program>`
    packages using the :class:`EchoEngine`, which simply passes through the waveform
    buffers.

    .. warning::

        This pipeline is for execution purposes only and does not compile programs. Please
        select an appropriate compilation pipeline if you wish to compile programs before
        execution.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None,
        engine: None = None,
    ) -> ExecutePipeline:
        if engine is not None:
            log.warning(
                "An engine was provided to the EchoPipeline, but it will be ignored. "
                "The EchoEngine is used directly."
            )

        target_data = target_data if target_data is not None else TargetData.default()
        results_pipeline = get_default_results_pipeline(model)
        return ExecutePipeline(
            model=model,
            target_data=target_data,
            runtime=SimpleRuntime(engine=EchoEngine(), results_pipeline=results_pipeline),
            name=config.name,
        )
