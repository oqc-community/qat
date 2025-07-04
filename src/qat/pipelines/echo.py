# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from qat.backend.waveform_v1.codegen import PydWaveformV1Backend, WaveformV1Backend
from qat.engines.waveform_v1 import EchoEngine
from qat.frontend import AutoFrontend
from qat.middleend.middleends import DefaultMiddleend, ExperimentalDefaultMiddleend
from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.runtime import SimpleRuntime
from qat.runtime.results_pipeline import (
    get_default_results_pipeline,
    get_experimental_results_pipeline,
)

log = get_default_logger()


class EchoPipelineConfig(PipelineConfig):
    """Configuration for the :class:`EchoPipeline`."""

    name: str = "echo"


class EchoPipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the :class:`WaveformV1Backend` and executes
    them using the :class:`EchoEngine`.

    An engine cannot be provided to the pipeline, as the EchoEngine is used directly.
    """

    @staticmethod
    def _build_pipeline(
        config: EchoPipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None,
        engine: None = None,
    ) -> Pipeline:
        """Constructs a pipeline equipped with the :class:`WaveformV1Backend` and
        :class:`EchoEngine`."""

        if engine is not None:
            log.warning(
                "An engine was provided to the EchoPipeline, but it will be ignored. "
                "The EchoEngine is used directly."
            )

        target_data = target_data if target_data is not None else TargetData.default()
        results_pipeline = get_default_results_pipeline(model)
        return Pipeline(
            model=model,
            target_data=target_data,
            frontend=AutoFrontend(model),
            middleend=DefaultMiddleend(model, target_data),
            backend=WaveformV1Backend(model),
            runtime=SimpleRuntime(engine=EchoEngine(), results_pipeline=results_pipeline),
            name=config.name,
        )


class ExperimentalEchoPipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the :class:`PydWaveformV1Backend`
    and executes them using the :class:`EchoEngine`.

    An engine cannot be provided to the pipeline, as the EchoEngine is used directly.
    """

    @staticmethod
    def _build_pipeline(
        config: EchoPipelineConfig,
        model: QuantumHardwareModel,
        target_data: TargetData | None,
        engine: None = None,
    ) -> Pipeline:
        """Constructs a pipeline equipped with the :class:`PydWaveformV1Backend`
        and :class:`EchoEngine`."""

        if engine is not None:
            log.warning(
                "An engine was provided to the EchoPipeline, but it will be ignored. "
                "The EchoEngine is used directly."
            )

        target_data = target_data if target_data is not None else TargetData.default()
        pyd_model = convert_purr_echo_hw_to_pydantic(model)
        results_pipeline = get_experimental_results_pipeline(model, pyd_model)
        return Pipeline(
            model=model,
            target_data=target_data,
            frontend=AutoFrontend(model),
            middleend=ExperimentalDefaultMiddleend(model, pyd_model, target_data),
            backend=PydWaveformV1Backend(pyd_model, model, target_data),
            runtime=SimpleRuntime(engine=EchoEngine(), results_pipeline=results_pipeline),
            name=config.name,
        )


def _create_pipeline_instance(num_qubits: int) -> Pipeline:
    return EchoPipeline(
        config=EchoPipelineConfig(name=f"echo{num_qubits}"),
        loader=EchoModelLoader(qubit_count=num_qubits),
    ).pipeline


def _create_experimental_pipeline_instance(num_qubits: int) -> Pipeline:
    return ExperimentalEchoPipeline(
        config=EchoPipelineConfig(name=f"experimental_echo{num_qubits}"),
        loader=EchoModelLoader(qubit_count=num_qubits),
    ).pipeline


echo8 = _create_pipeline_instance(8)
echo16 = _create_pipeline_instance(16)
echo32 = _create_pipeline_instance(32)
