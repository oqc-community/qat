# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.waveform_v1.codegen import PydWaveformV1Backend
from qat.frontend import AutoFrontend
from qat.middleend.middleends import ExperimentalDefaultMiddleend
from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import CompilePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class WaveformV1CompilePipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the :class:`PydWaveformV1Backend`.

    .. warning::

        This pipeline is for compilation purposes only and does not execute programs.
        Please select an appropriate execution pipeline if you wish to execute compiled
        programs.

        This pipeline is experimental and still in progress. Please use with caution.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None,
        engine: None = None,
    ) -> CompilePipeline:
        """Constructs a pipeline equipped with the :class:`PydWaveformV1Backend`
        and :class:`EchoEngine`."""

        if engine is not None:
            log.warning(
                "The compilation pipeline does not require an engine. It will be ignored."
            )

        target_data = target_data if target_data is not None else TargetData.default()
        pyd_model = convert_purr_echo_hw_to_pydantic(model)
        return CompilePipeline(
            model=model,
            target_data=target_data,
            frontend=AutoFrontend(model),
            middleend=ExperimentalDefaultMiddleend(model, pyd_model, target_data),
            backend=PydWaveformV1Backend(pyd_model, model, target_data),
            name=config.name,
        )
