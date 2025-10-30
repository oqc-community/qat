# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.fallthrough import FallthroughBackend
from qat.core.pass_base import PassManager
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.middleend.passes.purr.transform import (
    IntegratorAcquireSanitisation,
    LegacyPhaseOptimisation,
)
from qat.middleend.passes.purr.validation import (
    HardwareConfigValidity,
    InstructionValidation,
    ReadoutValidation,
)
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import CompilePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.passes.purr.analysis import CalibrationAnalysis

log = get_default_logger()


def middleend_pipeline(
    model: QuantumHardwareModel, target_data: TargetData | None = None
) -> PassManager:
    return (
        PassManager()
        | HardwareConfigValidity(model)
        | CalibrationAnalysis()
        | LegacyPhaseOptimisation()
        | IntegratorAcquireSanitisation()
        | InstructionValidation(target_data)
        | ReadoutValidation(model)
    )


class LegacyEchoCompilePipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the legacy echo backend.

    .. warning::

        This pipeline is for compilation purposes only and does not execute programs. Please
        select an appropriate execution pipeline if you wish to run the compiled programs.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None = None,
        engine=None,
    ) -> CompilePipeline:
        if engine is not None:
            log.warning(
                "The compilation pipeline does not require an engine. It will be ignored."
            )

        target_data = target_data if target_data is not None else TargetData.default()
        return CompilePipeline(
            name=config.name,
            model=model,
            target_data=target_data,
            frontend=AutoFrontend.default_for_legacy(model),
            middleend=CustomMiddleend(
                model,
                pipeline=middleend_pipeline(model=model, target_data=target_data),
            ),
            backend=FallthroughBackend(model),
        )
