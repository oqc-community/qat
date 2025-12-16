# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.passes.purr.analysis import (
    BindingPass,
    CFGPass,
    TILegalisationPass,
    TriagePass,
)
from qat.backend.passes.purr.transform import DesugaringPass
from qat.backend.qblox.codegen import QbloxBackend1, QbloxBackend2
from qat.backend.qblox.config.constants import QbloxTargetData
from qat.backend.qblox.passes.analysis import PreCodegenPass, QbloxLegalisationPass
from qat.core.pass_base import PassManager
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.middleend.passes.purr.transform import (
    DeviceUpdateSanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
    RepeatSanitisation,
    ReturnSanitisation,
    ScopeSanitisation,
)
from qat.middleend.passes.purr.validation import InstructionValidation, ReadoutValidation
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import CompilePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def middleend_pipeline1(
    model: QuantumHardwareModel, target_data: TargetData
) -> PassManager:
    return (
        PassManager()
        | PhaseOptimisation()
        | PostProcessingSanitisation()
        | DeviceUpdateSanitisation()
        | InstructionValidation(target_data)
        | ReadoutValidation(model)
        | RepeatSanitisation(model, target_data)
        | ScopeSanitisation()
        | ReturnSanitisation()
        | DesugaringPass()
        | TriagePass()
        | BindingPass()
    )


def middleend_pipeline2(model: QuantumHardwareModel, target_data: TargetData):
    return (
        PassManager()
        | PhaseOptimisation()
        | PostProcessingSanitisation()
        | DeviceUpdateSanitisation()
        | InstructionValidation(target_data)
        | ReadoutValidation(model)
        | RepeatSanitisation(model, target_data)
        | ScopeSanitisation()
        | ReturnSanitisation()
        | DesugaringPass()
        | TriagePass()
        | BindingPass()
        | TILegalisationPass()
    )


def backend_pipeline1():
    return PassManager() | PreCodegenPass() | CFGPass()


def backend_pipeline2():
    return PassManager() | QbloxLegalisationPass() | PreCodegenPass() | CFGPass()


class QbloxCompilePipeline1(UpdateablePipeline):
    """A pipeline that compiles programs using the :class:`QbloxBackend1`.

    .. warning::

        This pipeline is for compilation purposes only and does not execute programs. Please
        select an appropriate execution pipeline if you wish to run the compiled programs.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: QbloxTargetData | None,
        engine: None = None,
    ) -> CompilePipeline:
        if engine is not None:
            log.warning(
                "An engine was provided to the QbloxCompilePipeline, but it will "
                "be ignored. "
            )

        target_data = target_data if target_data is not None else QbloxTargetData.default()
        return CompilePipeline(
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
            name=config.name,
        )


class QbloxCompilePipeline2(UpdateablePipeline):
    """A pipeline that compiles programs using the :class:`QbloxBackend2`.

    .. warning::

        This pipeline is for compilation purposes only and does not execute programs. Please
        select an appropriate execution pipeline if you wish to run the compiled programs.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: QbloxTargetData | None,
        engine: None = None,
    ) -> CompilePipeline:
        if engine is not None:
            log.warning(
                "An engine was provided to the QbloxCompilePipeline, but it will "
                "be ignored. "
            )

        target_data = target_data if target_data is not None else QbloxTargetData.default()
        return CompilePipeline(
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
            name=config.name,
        )
