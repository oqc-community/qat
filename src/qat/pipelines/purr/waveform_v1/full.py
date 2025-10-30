# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from qat.backend.waveform_v1.purr.codegen import WaveformV1Backend
from qat.engines.waveform_v1 import EchoEngine
from qat.frontend import AutoFrontend
from qat.middleend import DefaultMiddleend
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime import SimpleRuntime
from qat.runtime.results_pipeline import (
    get_default_results_pipeline,
)

log = get_default_logger()


class EchoPipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the :class:`WaveformV1Backend` and executes
    them using the :class:`EchoEngine`.

    An engine cannot be provided to the pipeline, as the EchoEngine is used directly.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None,
        engine: None = None,
    ) -> Pipeline:
        """Constructs a pipeline equipped with the :class:`WaveformV1Backend` and
        :class:`EchoEngine`."""

        if engine is not None:
            log.warning(
                "An engine was provided to the EchoPipeline, but it will be ignored. "
                "The EchoEngine is used directly."
            )

        target_data = target_data if target_data is not None else TargetData.default()
        results_pipeline = get_default_results_pipeline(model)
        return Pipeline(
            model=model,
            target_data=target_data,
            frontend=AutoFrontend.default_for_purr(model),
            middleend=DefaultMiddleend(model, target_data),
            backend=WaveformV1Backend(model),
            runtime=SimpleRuntime(engine=EchoEngine(), results_pipeline=results_pipeline),
            name=config.name,
        )
