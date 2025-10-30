# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.fallthrough import FallthroughBackend
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.model.target_data import TargetData
from qat.pipelines.legacy.base import results_pipeline
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.backends.echo import EchoEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.legacy import LegacyRuntime

from .compile import middleend_pipeline

log = get_default_logger()


class LegacyEchoPipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the legacy echo backend and executes them
    using the :class:`LegacyRuntime`.

    Implements a custom pipeline to make instructions suitable for the legacy echo engine,
    and cannot be configured with a custom engine.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None = None,
        engine=None,
    ) -> Pipeline:
        if engine is not None:
            log.warning(
                "An engine was provided to the LegacyEchoPipeline, but it will be ignored. "
                "The legacy EchoEngine is used directly."
            )

        target_data = target_data if target_data is not None else TargetData.default()
        return Pipeline(
            name=config.name,
            model=model,
            target_data=target_data,
            frontend=AutoFrontend.default_for_legacy(model),
            middleend=CustomMiddleend(
                model,
                pipeline=middleend_pipeline(model=model, target_data=target_data),
            ),
            backend=FallthroughBackend(model),
            runtime=LegacyRuntime(
                engine=EchoEngine(model=model),
                results_pipeline=results_pipeline(model),
            ),
        )
