# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from qat.backend.qblox.config.constants import QbloxTargetData
from qat.pipelines.pipeline import ExecutePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime import SimpleRuntime
from qat.runtime.results_pipeline import (
    get_default_results_pipeline,
)

log = get_default_logger()


class QbloxExecutePipeline(UpdateablePipeline):
    """A pipeline that executes
    :class:`QbloxExecutables <qat.backend.qblox.execution.QbloxExecutable>`
    packages using the :class:`QbloxLiveEngineAdapter`.

    .. warning::

        This pipeline is for execution purposes only and does not compile programs. Please
        select an appropriate compilation pipeline if you wish to compile programs before
        execution.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: QbloxTargetData | None,
        engine: None = None,
    ) -> ExecutePipeline:
        if engine is not None:
            log.warning(
                "An engine was provided to the QbloxExecutePipeline, but it will be ignored. "
                "The QbloxLiveEngineAdapter is used directly."
            )

        target_data = target_data if target_data is not None else QbloxTargetData.default()
        results_pipeline = get_default_results_pipeline(model)
        return ExecutePipeline(
            model=model,
            target_data=target_data,
            runtime=SimpleRuntime(
                # TODO: Pipelines using `QbloxEngine`: COMPILER-730
                engine=model.create_engine(),
                results_pipeline=results_pipeline,
            ),
            name=config.name,
        )
