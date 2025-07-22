# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.core.pass_base import PassManager
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import ExecutePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.backends.qiskit_simulator import QiskitHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.legacy import LegacyRuntime
from qat.runtime.passes.purr.transform import (
    QiskitErrorMitigation,
    QiskitSimplifyResults,
    QiskitStripMetadata,
)

log = get_default_logger()


def results_pipeline() -> PassManager:
    return (
        PassManager()
        | QiskitStripMetadata()
        | QiskitErrorMitigation()
        | QiskitSimplifyResults()
    )


class LegacyQiskitExecutePipeline(UpdateablePipeline):
    """A pipeline that executes programs using the :class:`QiskitEngine` and the
    :class:`LegacyRuntime`.

    Implements a custom pipeline to make instructions suitable for the legacy Qiskit engine,
    and has a custom post-processing pipeline.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None = None,
        engine: None = None,
    ) -> ExecutePipeline:
        if not isinstance(model, QiskitHardwareModel):
            raise TypeError("Model must be an instance of QiskitHardwareModel")

        if engine is not None:
            log.warning(
                "An engine was provided to the LegacyQiskitExecutePipeline, but it will be "
                "ignored. The legacy Qiskit engine is used directly."
            )

        return ExecutePipeline(
            name=config.name,
            model=model,
            target_data=target_data if target_data is not None else TargetData.default(),
            runtime=LegacyRuntime(
                engine=model.create_engine(),
                results_pipeline=results_pipeline(),
            ),
        )
