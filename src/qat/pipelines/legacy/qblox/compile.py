# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.qblox.codegen import QbloxBackend2
from qat.backend.qblox.config.constants import QbloxTargetData
from qat.core.pass_base import PassManager
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.middleend.passes.purr.transform import (
    DeviceUpdateSanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
)
from qat.middleend.passes.purr.validation import (
    InstructionValidation,
    ReadoutValidation,
)
from qat.pipelines.pipeline import CompilePipeline
from qat.pipelines.purr.qblox.compile import backend_pipeline2
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def middleend_pipeline(
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


class LegacyQbloxCompilePipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the legacy qblox backend.

    Implements a custom pipeline to make instructions suitable for the legacy qblox engine,
    and cannot be configured with a custom engine.

    .. warning::

        This pipeline is for compilation purposes only and does not execute programs. Please
        select an appropriate execution pipeline if you wish to run the compiled programs.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: QbloxTargetData = None,
        engine=None,
    ) -> CompilePipeline:
        if engine is not None:
            log.warning(
                "The compilation pipeline does not require an engine. It will be ignored."
            )

        target_data = target_data or QbloxTargetData.default()
        return CompilePipeline(
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
        )
