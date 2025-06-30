# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.fallthrough import FallthroughBackend
from qat.core.pass_base import PassManager
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.legacy.transform import (
    IntegratorAcquireSanitisation,
    LegacyPhaseOptimisation,
)
from qat.middleend.passes.legacy.validation import (
    HardwareConfigValidity,
    InstructionValidation,
    ReadoutValidation,
)
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.legacy.base import LegacyPipeline
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import PipelineConfig
from qat.purr.backends.echo import EchoEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.legacy import LegacyRuntime
from qat.runtime.passes.legacy.analysis import CalibrationAnalysis

log = get_default_logger()


class LegacyEchoPipelineConfig(PipelineConfig):
    """Configuration for the :class:`LegacyEchoPipeline`."""

    name: str = "legacy_echo"


class LegacyEchoPipeline(LegacyPipeline):
    """A pipeline that compiles programs using the legacy echo backend and executes them
    using the :class:`LegacyRuntime`.

    Implements a custom pipeline to make instructions suitable for the legacy echo engine,
    and cannot be configured with a custom engine.
    """

    @staticmethod
    def _build_pipeline(
        config: LegacyEchoPipelineConfig,
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
            frontend=AutoFrontend(model),
            middleend=CustomMiddleend(
                model,
                pipeline=LegacyEchoPipeline._middleend_pipeline(
                    model=model, target_data=target_data
                ),
            ),
            backend=FallthroughBackend(model),
            runtime=LegacyRuntime(
                engine=EchoEngine(model=model),
                results_pipeline=LegacyEchoPipeline._results_pipeline(model),
            ),
        )

    @staticmethod
    def _middleend_pipeline(
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


def _create_pipeline_instance(num_qubits: int) -> Pipeline:
    return LegacyEchoPipeline(
        config=LegacyEchoPipelineConfig(name=f"legacy_echo{num_qubits}"),
        loader=EchoModelLoader(qubit_count=num_qubits),
    ).pipeline


legacy_echo8 = _create_pipeline_instance(8)
legacy_echo16 = _create_pipeline_instance(16)
legacy_echo32 = _create_pipeline_instance(32)
