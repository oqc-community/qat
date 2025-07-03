# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.qblox.codegen import QbloxBackend2
from qat.backend.qblox.config.constants import QbloxTargetData
from qat.core.pass_base import PassManager
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.legacy.transform import (
    DeviceUpdateSanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
)
from qat.middleend.passes.legacy.validation import (
    InstructionValidation,
    ReadoutValidation,
)
from qat.pipelines.legacy.base import LegacyPipeline
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import PipelineConfig
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.legacy import LegacyRuntime

log = get_default_logger()


class LegacyQbloxPipelineConfig(PipelineConfig):
    """Configuration for the :class:`LegacyQbloxPipeline`."""

    name: str = "legacy_qblox"


class LegacyQbloxPipeline(LegacyPipeline):
    """A pipeline that compiles programs using the legacy qblox backend and executes them
    using the :class:`LegacyRuntime`.

    Implements a custom pipeline to make instructions suitable for the legacy qblox engine,
    and cannot be configured with a custom engine.
    """

    @staticmethod
    def _build_pipeline(
        config: LegacyQbloxPipelineConfig,
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
            frontend=AutoFrontend(model),
            middleend=CustomMiddleend(
                model,
                pipeline=LegacyQbloxPipeline._middleend_pipeline(
                    model=model, target_data=target_data
                ),
            ),
            backend=QbloxBackend2(model),
            runtime=LegacyRuntime(
                engine=model.create_engine(),
                results_pipeline=LegacyQbloxPipeline._results_pipeline(model),
            ),
        )

    @staticmethod
    def _middleend_pipeline(
        model: QuantumHardwareModel, target_data: QbloxTargetData
    ) -> PassManager:
        """Factory for creating middleend pipelines for Qblox legacy models.

        Includes a list of passes that replicate the responsibilities of
        :meth:`LiveDeviceEngine.optimize <qat.purr.backends.live.LiveDeviceEngine.optimize>` and
        :meth:`LiveDeviceEngine.validate <qat.purr.backends.live.LiveDeviceEngine.validate>`.

        :param model: The hardware model is required for ReadoutValidation.
        :return: The pipeline as a pass manager.
        """
        return (
            PassManager()
            | DeviceUpdateSanitisation()
            | PhaseOptimisation()
            | PostProcessingSanitisation()
            | InstructionValidation(target_data)
            | ReadoutValidation(model)
        )


def get_pipeline(model: QuantumHardwareModel, name: str = "legacy_qblox") -> Pipeline:
    """A factory for creating pipelines that performs the responsibilities of the legacy
    runtime :class:`EchoEngine <qat.purr.compiler.runtime.QuantumRuntime>`

    :param model: The Qblox hardware model.
    :param name: The name of the pipeline, defaults to "legacy_qblox"
    :return: The complete pipeline, including the runtime and engine.
    """
    return LegacyQbloxPipeline(
        config=LegacyQbloxPipelineConfig(name=name), model=model
    ).pipeline
