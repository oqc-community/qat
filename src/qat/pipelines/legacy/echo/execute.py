# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.target_data import TargetData
from qat.pipelines.legacy.base import results_pipeline
from qat.pipelines.pipeline import ExecutePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.backends.echo import EchoEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime.legacy import LegacyRuntime

log = get_default_logger()


class LegacyEchoExecutePipeline(UpdateablePipeline):
    """A pipeline that executes compiled using the legacy echo backend.

    .. warning::

        This pipeline is for execution purposes only and does not compile programs. Please
        select an appropriate compilation pipeline if you wish to compile programs before
        execution.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None = None,
        engine=None,
    ) -> ExecutePipeline:
        if engine is not None:
            log.warning(
                "The engine for the LegacyEchoExecutePipeline is expected to be provided by "
                "the model, and the provided engine will be ignored."
            )

        target_data = target_data if target_data is not None else TargetData.default()
        return ExecutePipeline(
            name=config.name,
            model=model,
            target_data=target_data,
            runtime=LegacyRuntime(
                engine=EchoEngine(model=model),
                results_pipeline=results_pipeline(model),
            ),
        )
