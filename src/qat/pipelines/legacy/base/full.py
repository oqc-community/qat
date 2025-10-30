# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.fallthrough import FallthroughBackend
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import UpdateablePipeline
from qat.purr.backends.live import LiveHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.legacy import LegacyRuntime

from .compile import middleend_pipeline
from .execute import LegacyPipelineConfig, results_pipeline

log = get_default_logger()


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
            frontend=AutoFrontend.default_for_legacy(model),
            middleend=CustomMiddleend(
                model,
                pipeline=middleend_pipeline(model=model, target_data=target_data),
            ),
            backend=FallthroughBackend(model),
            runtime=LegacyRuntime(
                engine=engine,
                results_pipeline=results_pipeline(model),
                connection_mode=config.connection_mode,
            ),
        )
