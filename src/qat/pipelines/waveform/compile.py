# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.waveform.codegen import PydWaveformBackend
from qat.frontend import AutoFrontendWithFlattenedIR
from qat.middleend import PydDefaultMiddleend
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import CompilePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class WaveformCompilePipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the :class:`PydWaveformBackend`.

    .. warning::

        This pipeline is for compilation purposes only and does not execute programs.
        Please select an appropriate execution pipeline if you wish to execute compiled
        programs.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: PhysicalHardwareModel,
        target_data: TargetData | None,
        engine: None = None,
    ) -> CompilePipeline:
        """Constructs a pipeline equipped with the :class:`PydWaveformBackend`
        and :class:`EchoEngine`."""

        if engine is not None:
            log.warning(
                "The compilation pipeline does not require an engine. It will be ignored."
            )

        target_data = target_data if target_data is not None else TargetData.default()
        return CompilePipeline(
            model=model,
            target_data=target_data,
            frontend=AutoFrontendWithFlattenedIR.default_for_pydantic(model),
            middleend=PydDefaultMiddleend(model, target_data),
            backend=PydWaveformBackend(model, target_data),
            name=config.name,
        )
