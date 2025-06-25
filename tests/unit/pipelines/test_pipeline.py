# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.backend import WaveformV1Backend
from qat.engines import ZeroEngine
from qat.frontend import AutoFrontend
from qat.middleend import DefaultMiddleend
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import Pipeline
from qat.runtime import SimpleRuntime


class TestPipeline:
    def test_correct_pipeline_initialization(self):
        model = EchoModelLoader(qubit_count=4).load()
        pipeline = Pipeline(
            name="TestPipeline",
            model=model,
            frontend=AutoFrontend(model=model),
            middleend=DefaultMiddleend(model=model),
            backend=WaveformV1Backend(model=model),
            runtime=SimpleRuntime(engine=ZeroEngine()),
            target_data=TargetData.default(),
        )

        assert pipeline.name == "TestPipeline"
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, DefaultMiddleend)
        assert isinstance(pipeline.backend, WaveformV1Backend)
        assert isinstance(pipeline.runtime, SimpleRuntime)
        assert isinstance(pipeline.target_data, TargetData)

    def test_pipeline_with_inconsistent_model_raises_error(self):
        model1 = EchoModelLoader(qubit_count=4).load()
        model2 = EchoModelLoader(qubit_count=6).load()
        with pytest.raises(ValueError):
            Pipeline(
                name="TestPipeline",
                model=model1,
                frontend=AutoFrontend(model=model1),
                middleend=DefaultMiddleend(model=model1),
                backend=WaveformV1Backend(model=model2),
                runtime=SimpleRuntime(engine=ZeroEngine()),
                target_data=TargetData.default(),
            )

    def test_copy_returns_new_instance_same_components(self):
        model = EchoModelLoader(qubit_count=4).load()
        pipeline = Pipeline(
            name="TestPipeline",
            model=model,
            frontend=AutoFrontend(model=model),
            middleend=DefaultMiddleend(model=model),
            backend=WaveformV1Backend(model=model),
            runtime=SimpleRuntime(engine=ZeroEngine()),
            target_data=TargetData.default(),
        )

        copied_pipeline = pipeline.copy()
        assert copied_pipeline is not pipeline
        assert copied_pipeline.name == pipeline.name
        assert copied_pipeline.model is pipeline.model
        assert copied_pipeline.frontend is pipeline.frontend
        assert copied_pipeline.middleend is pipeline.middleend
        assert copied_pipeline.backend is pipeline.backend
        assert copied_pipeline.runtime is pipeline.runtime
        assert copied_pipeline.target_data is pipeline.target_data

    def test_copy_with_name(self):
        model = EchoModelLoader(qubit_count=4).load()
        pipeline = Pipeline(
            name="TestPipeline",
            model=model,
            frontend=AutoFrontend(model=model),
            middleend=DefaultMiddleend(model=model),
            backend=WaveformV1Backend(model=model),
            runtime=SimpleRuntime(engine=ZeroEngine()),
            target_data=TargetData.default(),
        )

        copied_pipeline = pipeline.copy_with_name(name="NewPipelineName")
        assert pipeline.name == "TestPipeline"
        assert copied_pipeline.name == "NewPipelineName"
        assert copied_pipeline.model is pipeline.model
        assert copied_pipeline.frontend is pipeline.frontend
        assert copied_pipeline.middleend is pipeline.middleend
        assert copied_pipeline.backend is pipeline.backend
        assert copied_pipeline.runtime is pipeline.runtime
        assert copied_pipeline.target_data is pipeline.target_data
