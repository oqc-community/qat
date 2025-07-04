# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend import DefaultBackend
from qat.core.config.descriptions import PipelineClassDescription
from qat.core.pass_base import PassManager
from qat.core.pipelines.configurable import ConfigurablePipeline
from qat.engines import ZeroEngine
from qat.frontend import DefaultFrontend
from qat.middleend import DefaultMiddleend
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import DefaultTargetData
from qat.pipelines.pipeline import Pipeline
from qat.runtime import DefaultRuntime

from tests.unit.utils.engines import MockEngineWithModel
from tests.unit.utils.resultsprocessing import get_pipeline


class MockModelInjectee:
    """A mock class to test injection of model and target data."""

    def __init__(self, model):
        self.model = model


class MockTargetDataInjectee:
    """A mock class to test injection of target data."""

    def __init__(self, target_data):
        self.target_data = target_data


class MockModelAndTargetDataInjectee:
    """A mock class to test injection of both model and target data."""

    def __init__(self, model, target_data):
        self.model = model
        self.target_data = target_data


class TestConfigurablePipeline:
    model = EchoModelLoader().load()
    target_data = DefaultTargetData()

    def test_inject_model(self):
        """Test the _inject_model_and_target_data method with a model."""
        injectee = ConfigurablePipeline._inject_model_and_target_data(
            MockModelInjectee, self.model, self.target_data
        )
        assert injectee.model == self.model

    def test_inject_target_data(self):
        """Test the _inject_model_and_target_data method with target data."""
        injectee = ConfigurablePipeline._inject_model_and_target_data(
            MockTargetDataInjectee, self.model, self.target_data
        )
        assert injectee.target_data == self.target_data

    def test_inject_model_and_target_data(self):
        """Test the _inject_model_and_target_data method with both model and target data."""
        injectee = ConfigurablePipeline._inject_model_and_target_data(
            MockModelAndTargetDataInjectee, self.model, self.target_data
        )
        assert injectee.model == self.model
        assert injectee.target_data == self.target_data

    def test_create_engine_without_model(self):
        """Test the _create_engine method without a model."""
        engine = ConfigurablePipeline._create_engine(ZeroEngine, None)
        assert isinstance(engine, ZeroEngine)

    def test_create_engine_with_model(self):
        """Test the _create_engine method with a model."""
        engine = ConfigurablePipeline._create_engine(MockEngineWithModel, self.model)
        assert isinstance(engine, MockEngineWithModel)
        assert engine.model == self.model

    def test_create_results_pipeline(self):
        """Test the _create_results_pipeline method."""
        results_pipeline = ConfigurablePipeline._create_results_pipeline(
            get_pipeline, self.model
        )
        assert isinstance(results_pipeline, PassManager)
        assert len(results_pipeline.passes) == 1

    def test_create_runtime(self):
        """Test the _create_runtime method."""
        engine = ZeroEngine()
        results_pipeline = get_pipeline(self.model)
        runtime = ConfigurablePipeline._create_runtime(
            DefaultRuntime, engine, results_pipeline, self.model
        )
        assert isinstance(runtime, DefaultRuntime)
        assert runtime.engine == engine
        assert runtime.results_pipeline == results_pipeline
        assert isinstance(runtime.engine, ZeroEngine)

    def test_build_pipeline(self):
        """Test the _build_pipeline method."""
        config = PipelineClassDescription(
            name="test_pipeline",
            frontend="qat.frontend.DefaultFrontend",
            middleend="qat.middleend.DefaultMiddleend",
            backend="qat.backend.DefaultBackend",
            engine="qat.engines.ZeroEngine",
            runtime="qat.runtime.DefaultRuntime",
            results_pipeline="tests.unit.utils.resultsprocessing.get_pipeline",
            target_data="qat.model.target_data.DefaultTargetData",
        )
        pipeline = ConfigurablePipeline._build_pipeline(
            config, self.model, self.target_data
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.model == self.model
        assert pipeline.target_data == self.target_data
        assert pipeline.name == "test_pipeline"
        assert isinstance(pipeline.frontend, DefaultFrontend)
        assert isinstance(pipeline.middleend, DefaultMiddleend)
        assert isinstance(pipeline.backend, DefaultBackend)
        assert isinstance(pipeline.runtime, DefaultRuntime)
        assert isinstance(pipeline.engine, ZeroEngine)
        assert isinstance(pipeline.runtime.results_pipeline, PassManager)
        assert len(pipeline.runtime.results_pipeline.passes) == 1
