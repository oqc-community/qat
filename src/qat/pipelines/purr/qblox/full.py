# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from qat.backend.qblox.codegen import QbloxBackend2
from qat.backend.qblox.config.constants import QbloxTargetData
from qat.engines.qblox.execution import QbloxEngine
from qat.frontend import AutoFrontend
from qat.middleend import DefaultMiddleend
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime import SimpleRuntime
from qat.runtime.results_pipeline import (
    get_default_results_pipeline,
)

log = get_default_logger()


class QbloxPipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the :class:`QbloxBackend2` and executes
    them using the :class:`EchoEngine`.

    An engine cannot be provided to the pipeline, as the EchoEngine is used directly.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QbloxLiveHardwareModel,
        target_data: QbloxTargetData | None,
        engine: QbloxEngine = None,
    ) -> Pipeline:
        target_data = target_data if target_data is not None else QbloxTargetData.default()
        results_pipeline = get_default_results_pipeline(model)
        return Pipeline(
            model=model,
            target_data=target_data,
            frontend=AutoFrontend(model),
            middleend=DefaultMiddleend(model, target_data),
            backend=QbloxBackend2(model),
            runtime=SimpleRuntime(
                engine=engine,
                results_pipeline=results_pipeline,
            ),
            name=config.name,
        )
