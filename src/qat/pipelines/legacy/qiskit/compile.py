# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.fallthrough import FallthroughBackend
from qat.core.pass_base import PassManager
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.middleend.passes.purr.transform import (
    QiskitInstructionsWrapper,
)
from qat.middleend.passes.purr.validation import (
    HardwareConfigValidity,
    QiskitResultsFormatValidation,
)
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import CompilePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.backends.qiskit_simulator import QiskitHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def middleend_pipeline(model: QiskitHardwareModel) -> PassManager:
    return (
        PassManager()
        | QiskitResultsFormatValidation()
        | HardwareConfigValidity(model)
        | QiskitInstructionsWrapper()
    )


class LegacyQiskitCompilePipeline(UpdateablePipeline):
    """A pipeline that compiles programs for the Qiskit backend.

    .. warning::

        This pipeline is for compilation purposes only and does not execute programs. Please
        use the :class:`LegacyQiskitExecutePipeline <.execute.LegacyQiskitExecutePipeline`
        for execution.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None = None,
        engine: None = None,
    ) -> CompilePipeline:
        if not isinstance(model, QiskitHardwareModel):
            raise TypeError("Model must be an instance of QiskitHardwareModel")

        if engine is not None:
            log.warning(
                "The compilation pipeline does not require an engine. It will be ignored."
            )

        return CompilePipeline(
            name=config.name,
            model=model,
            target_data=target_data if target_data is not None else TargetData.default(),
            frontend=AutoFrontend.default_for_legacy(model),
            middleend=CustomMiddleend(
                model,
                pipeline=middleend_pipeline(model=model),
            ),
            backend=FallthroughBackend(model),
        )
