# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.qblox.codegen import QbloxBackend2
from qat.backend.qblox.config.constants import QbloxTargetData
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.pipelines.legacy.base import results_pipeline
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.legacy import LegacyRuntime

from ...purr.qblox.compile import backend_pipeline2
from .compile import middleend_pipeline

log = get_default_logger()


class LegacyQbloxPipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the legacy qblox backend and executes them
    using the :class:`LegacyRuntime`.

    Implements a custom pipeline to make instructions suitable for the legacy qblox engine,
    and cannot be configured with a custom engine.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: QbloxTargetData = None,
        engine=None,
    ) -> Pipeline:
        if engine is not None:
            log.warning(
                "An engine was provided to the LegacyQbloxPipeline, but it will be ignored. "
                "The legacy QbloxEngine is used directly."
            )

        target_data = target_data or QbloxTargetData.default()
        return Pipeline(
            name=config.name,
            model=model,
            target_data=target_data,
            frontend=AutoFrontend.default_for_legacy(model),
            middleend=CustomMiddleend(
                model,
                pipeline=middleend_pipeline(model=model, target_data=target_data),
            ),
            backend=QbloxBackend2(
                model,
                pipeline=backend_pipeline2(),
            ),
            runtime=LegacyRuntime(
                engine=model.create_engine(),
                results_pipeline=results_pipeline(model),
            ),
        )
