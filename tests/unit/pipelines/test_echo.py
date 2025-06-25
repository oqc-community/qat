# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.waveform_v1.codegen import PydWaveformV1Backend, WaveformV1Backend
from qat.engines.waveform_v1 import EchoEngine
from qat.frontend import AutoFrontend
from qat.middleend import DefaultMiddleend
from qat.middleend.middleends import ExperimentalDefaultMiddleend
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.echo import EchoPipeline, EchoPipelineConfig, ExperimentalEchoPipeline
from qat.pipelines.pipeline import Pipeline
from qat.runtime import SimpleRuntime


class TestEchoPipeline:
    def test_build_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = EchoModelLoader(qubit_count=4).load()
        pipeline = EchoPipeline._build_pipeline(
            config=EchoPipelineConfig(), model=model, target_data=None
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "echo"
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, DefaultMiddleend)
        assert isinstance(pipeline.backend, WaveformV1Backend)
        assert isinstance(pipeline.runtime, SimpleRuntime)
        assert isinstance(pipeline.target_data, TargetData)
        assert pipeline.target_data == TargetData.default()
        assert isinstance(pipeline.engine, EchoEngine)


class TestExperimentalEchoPipeline:
    def test_build_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = EchoModelLoader(qubit_count=4).load()
        pipeline = ExperimentalEchoPipeline._build_pipeline(
            config=EchoPipelineConfig(), model=model, target_data=None
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "echo"
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, ExperimentalDefaultMiddleend)
        assert isinstance(pipeline.backend, PydWaveformV1Backend)
        assert isinstance(pipeline.runtime, SimpleRuntime)
        assert isinstance(pipeline.target_data, TargetData)
        assert pipeline.target_data == TargetData.default()
        assert isinstance(pipeline.engine, EchoEngine)
