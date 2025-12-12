# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from qat.backend.qblox.codegen import QbloxBackend1, QbloxBackend2
from qat.backend.qblox.config.constants import QbloxTargetData
from qat.engines.qblox.execution import QbloxEngine
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.purr.qblox.compile import (
    backend_pipeline1,
    backend_pipeline2,
    middleend_pipeline1,
    middleend_pipeline2,
)
from qat.pipelines.purr.qblox.execute import get_results_pipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime import SimpleRuntime
from qat.runtime.aggregator import QBloxAggregator

log = get_default_logger()


class QbloxPipeline1(UpdateablePipeline):
    """
    A pipeline that compiles programs using the :class:`QbloxBackend1` and executes
    them using the :class:`QbloxEngine`.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QbloxLiveHardwareModel,
        target_data: QbloxTargetData | None,
        engine: QbloxEngine = None,
    ) -> Pipeline:
        target_data = target_data if target_data is not None else QbloxTargetData.default()
        results_pipeline = get_results_pipeline(model)
        return Pipeline(
            model=model,
            target_data=target_data,
            frontend=AutoFrontend(model),
            middleend=CustomMiddleend(
                model=model,
                pipeline=middleend_pipeline1(model=model, target_data=target_data),
            ),
            backend=QbloxBackend1(
                model=model,
                pipeline=backend_pipeline1(),
            ),
            runtime=SimpleRuntime(
                engine=engine,
                results_pipeline=results_pipeline,
                aggregator=QBloxAggregator(),
            ),
            name=config.name,
        )


class QbloxPipeline2(UpdateablePipeline):
    """
    A pipeline that compiles programs using the :class:`QbloxBackend2` and executes
    them using the :class:`QbloxEngine`.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QbloxLiveHardwareModel,
        target_data: QbloxTargetData | None,
        engine: QbloxEngine = None,
    ) -> Pipeline:
        target_data = target_data if target_data is not None else QbloxTargetData.default()
        results_pipeline = get_results_pipeline(model)
        return Pipeline(
            model=model,
            target_data=target_data,
            frontend=AutoFrontend(model),
            middleend=CustomMiddleend(
                model=model,
                pipeline=middleend_pipeline2(model=model, target_data=target_data),
            ),
            backend=QbloxBackend2(
                model=model,
                pipeline=backend_pipeline2(),
            ),
            runtime=SimpleRuntime(
                engine=engine,
                results_pipeline=results_pipeline,
                aggregator=QBloxAggregator(),
            ),
            name=config.name,
        )
