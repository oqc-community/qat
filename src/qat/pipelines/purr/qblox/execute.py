# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from qat.backend.qblox.config.constants import QbloxTargetData
from qat.core.pass_base import PassManager
from qat.engines.qblox.execution import QbloxEngine
from qat.pipelines.pipeline import ExecutePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime import SimpleRuntime
from qat.runtime.aggregator import QBloxAggregator
from qat.runtime.passes.purr.analysis import IndexMappingAnalysis
from qat.runtime.passes.transform import (
    AssignResultsTransform,
    ErrorMitigation,
    InlineResultsProcessingTransform,
    QBloxAcquisitionPostProcessing,
    ResultTransform,
)

log = get_default_logger()


def get_results_pipeline(model: QuantumHardwareModel) -> PassManager:
    """Factory for creating the default results pipeline.

    :param model: The quantum hardware model to use for the pipeline.
    """

    return (
        PassManager()
        | QBloxAcquisitionPostProcessing()
        | InlineResultsProcessingTransform()
        | AssignResultsTransform()
        | ResultTransform()
        | IndexMappingAnalysis(model)
        | ErrorMitigation(model)
    )


class QbloxExecutePipeline(UpdateablePipeline):
    """A pipeline that executes :class:`Executable <qat.executable.Executable>`s with
    :class:`QbloxProgram <qat.backend.qblox.execution.QbloxProgram>`s
    packages using the :class:`QbloxLiveEngineAdapter`.

    .. warning::

        This pipeline is for execution purposes only and does not compile programs. Please
        select an appropriate compilation pipeline if you wish to compile programs before
        execution.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QbloxLiveHardwareModel,
        target_data: QbloxTargetData | None,
        engine: QbloxEngine = None,
    ) -> ExecutePipeline:
        target_data = target_data if target_data is not None else QbloxTargetData.default()
        results_pipeline = get_results_pipeline(model)
        return ExecutePipeline(
            model=model,
            target_data=target_data,
            runtime=SimpleRuntime(
                engine=engine,
                results_pipeline=results_pipeline,
                aggregator=QBloxAggregator(),
            ),
            name=config.name,
        )
