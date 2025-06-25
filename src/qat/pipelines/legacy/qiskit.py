# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.fallthrough import FallthroughBackend
from qat.core.pass_base import PassManager
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.legacy.transform import (
    QiskitInstructionsWrapper,
)
from qat.middleend.passes.legacy.validation import (
    HardwareConfigValidity,
    QiskitResultsFormatValidation,
)
from qat.model.loaders.legacy import QiskitModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.backends.qiskit_simulator import QiskitHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.legacy import LegacyRuntime
from qat.runtime.passes.legacy.transform import (
    QiskitErrorMitigation,
    QiskitSimplifyResults,
    QiskitStripMetadata,
)

log = get_default_logger()


class LegacyQiskitPipelineConfig(PipelineConfig):
    """Configuration for the :class:`LegacyQiskitPipeline`."""

    name: str = "legacy_qiskit"


class LegacyQiskitPipeline(UpdateablePipeline):
    """A pipeline that executes programs using the :class:`QiskitEngine` and the
    :class:`LegacyRuntime`.

    Implements a custom pipeline to make instructions suitable for the legacy Qiskit engine,
    and has a custom post-processing pipeline.
    """

    @staticmethod
    def _build_pipeline(
        config: LegacyQiskitPipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None = None,
        engine: None = None,
    ) -> Pipeline:
        if not isinstance(model, QiskitHardwareModel):
            raise TypeError("Model must be an instance of QiskitHardwareModel")

        if engine is not None:
            log.warning(
                "An engine was provided to the LegacyQiskitPipeline, but it will be ignored. "
                "The legacy Qiskit engine is used directly."
            )

        return Pipeline(
            name=config.name,
            model=model,
            target_data=target_data if target_data is not None else TargetData.default(),
            frontend=AutoFrontend(model),
            middleend=CustomMiddleend(
                model,
                pipeline=LegacyQiskitPipeline._middleend_pipeline(model=model),
            ),
            backend=FallthroughBackend(model),
            runtime=LegacyRuntime(
                engine=model.create_engine(),
                results_pipeline=LegacyQiskitPipeline._results_pipeline(),
            ),
        )

    @staticmethod
    def _middleend_pipeline(model: QiskitHardwareModel) -> PassManager:
        return (
            PassManager()
            | QiskitResultsFormatValidation()
            | HardwareConfigValidity(model)
            | QiskitInstructionsWrapper()
        )

    @staticmethod
    def _results_pipeline() -> PassManager:
        return (
            PassManager()
            | QiskitStripMetadata()
            | QiskitErrorMitigation()
            | QiskitSimplifyResults()
        )


def _create_pipeline_instance(num_qubits: int) -> Pipeline:
    return LegacyQiskitPipeline(
        config=LegacyQiskitPipelineConfig(name=f"legacy_qiskit{num_qubits}"),
        loader=QiskitModelLoader(qubit_count=num_qubits),
    ).pipeline


legacy_qiskit8 = _create_pipeline_instance(8)
legacy_qiskit16 = _create_pipeline_instance(16)
legacy_qiskit32 = _create_pipeline_instance(32)
