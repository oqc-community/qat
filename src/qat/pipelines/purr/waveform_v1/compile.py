# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.waveform_v1.codegen import WaveformV1Backend
from qat.frontend import AutoFrontend
from qat.middleend.middleends import DefaultMiddleend
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import CompilePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class WaveformV1CompilePipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the :class:`WaveformV1Backend`.

    .. warning::

        This pipeline is for compilation purposes only and does not execute programs. Please
        select an appropriate execution pipeline if you wish to run the compiled programs.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None,
        engine: None = None,
    ) -> CompilePipeline:
        if engine is not None:
            log.warning(
                "An engine was provided to the WaveformV1CompilationPipeline, but it will "
                "be ignored. "
            )

        target_data = target_data if target_data is not None else TargetData.default()
        return CompilePipeline(
            model=model,
            target_data=target_data,
            frontend=AutoFrontend(model),
            middleend=DefaultMiddleend(model, target_data),
            backend=WaveformV1Backend(model),
            name=config.name,
        )
