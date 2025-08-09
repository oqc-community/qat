# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.backend import DefaultBackend
from qat.core.config.descriptions import PipelineClassDescription
from qat.core.pass_base import PassManager
from qat.core.pipelines.configurable import (
    ConfigurableCompilePipeline,
    ConfigurableExecutePipeline,
    ConfigurablePipeline,
    _inject_model_and_target_data,
)
from qat.engines import ZeroEngine
from qat.frontend import DefaultFrontend
from qat.middleend import DefaultMiddleend
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import DefaultTargetData
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.runtime import DefaultRuntime


class MockModelInjectee:
    """A mock class to test injection of model and target data."""

    def __init__(self, model, engine=None):
        self.model = model
        self.engine = engine


class MockTargetDataInjectee:
    """A mock class to test injection of target data."""

    def __init__(self, target_data, engine=None):
        self.target_data = target_data
        self.engine = engine


class MockModelAndTargetDataInjectee:
    """A mock class to test injection of both model and target data."""

    def __init__(self, model, target_data, engine=None):
        self.model = model
        self.target_data = target_data
        self.engine = engine


@pytest.mark.parametrize("with_engine", [True, False])
class TestInjectModelAndTargetData:
    """Tests for the _inject_model_and_target_data function."""

    def test_inject_model(self, with_engine):
        """Test injecting only the model."""
        model = EchoModelLoader().load()
        kwargs = {"engine": ZeroEngine()} if with_engine else {}
        result = _inject_model_and_target_data(
            MockModelInjectee, model, DefaultTargetData(), **kwargs
        )
        assert result.model == model
        if with_engine:
            assert isinstance(result.engine, ZeroEngine)
        else:
            assert result.engine is None

    def test_inject_target_data(self, with_engine):
        """Test injecting only the target data."""
        target_data = DefaultTargetData()
        kwargs = {"engine": ZeroEngine()} if with_engine else {}
        result = _inject_model_and_target_data(
            MockTargetDataInjectee, EchoModelLoader().load(), target_data, **kwargs
        )
        assert result.target_data == target_data
        if with_engine:
            assert isinstance(result.engine, ZeroEngine)
        else:
            assert result.engine is None

    def test_inject_model_and_target_data(self, with_engine):
        """Test injecting both model and target data."""
        model = EchoModelLoader().load()
        target_data = DefaultTargetData()
        kwargs = {"engine": ZeroEngine()} if with_engine else {}
        result = _inject_model_and_target_data(
            MockModelAndTargetDataInjectee, model, target_data, **kwargs
        )
        assert result.model == model
        assert result.target_data == target_data
        if with_engine:
            assert isinstance(result.engine, ZeroEngine)
        else:
            assert result.engine is None


class TestConfigurableCompilePipeline:
    model = EchoModelLoader().load()
    target_data = DefaultTargetData()

    def test_build_pipeline(self):
        """Test the _build_pipeline method."""
        config = PipelineClassDescription(
            name="test_compile_pipeline",
            frontend="qat.frontend.DefaultFrontend",
            middleend="qat.middleend.DefaultMiddleend",
            backend="qat.backend.DefaultBackend",
            target_data="qat.model.target_data.DefaultTargetData",
        )
        pipeline = ConfigurableCompilePipeline._build_pipeline(
            config, self.model, self.target_data, engine=ZeroEngine()
        )
        assert isinstance(pipeline, CompilePipeline)
        assert pipeline.model == self.model
        assert pipeline.target_data == self.target_data
        assert pipeline.name == "test_compile_pipeline"
        assert isinstance(pipeline.frontend, DefaultFrontend)
        assert isinstance(pipeline.middleend, DefaultMiddleend)
        assert isinstance(pipeline.backend, DefaultBackend)


class TestConfigurableExecutePipeline:
    model = EchoModelLoader().load()
    target_data = DefaultTargetData()

    def test_build_pipeline(self):
        """Test the _build_pipeline method."""
        config = PipelineClassDescription(
            name="test_execute_pipeline",
            engine="zero",
            runtime="qat.runtime.DefaultRuntime",
            results_pipeline="tests.unit.utils.resultsprocessing.get_pipeline",
            target_data="qat.model.target_data.DefaultTargetData",
        )
        pipeline = ConfigurableExecutePipeline._build_pipeline(
            config, self.model, self.target_data, engine=ZeroEngine()
        )
        assert isinstance(pipeline, ExecutePipeline)
        assert pipeline.model == self.model
        assert pipeline.target_data == self.target_data
        assert pipeline.name == "test_execute_pipeline"
        assert isinstance(pipeline.runtime, DefaultRuntime)
        assert isinstance(pipeline.engine, ZeroEngine)
        assert isinstance(pipeline.runtime.results_pipeline, PassManager)
        assert len(pipeline.runtime.results_pipeline.passes) == 1


class TestConfigurablePipeline:
    model = EchoModelLoader().load()
    target_data = DefaultTargetData()

    def test_build_pipeline(self):
        """Test the _build_pipeline method."""
        config = PipelineClassDescription(
            name="test_pipeline",
            frontend="qat.frontend.DefaultFrontend",
            middleend="qat.middleend.DefaultMiddleend",
            backend="qat.backend.DefaultBackend",
            engine="zero",
            runtime="qat.runtime.DefaultRuntime",
            results_pipeline="tests.unit.utils.resultsprocessing.get_pipeline",
            target_data="qat.model.target_data.DefaultTargetData",
        )
        pipeline = ConfigurablePipeline._build_pipeline(
            config, self.model, self.target_data, engine=ZeroEngine()
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
