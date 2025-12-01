# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import pytest

from qat.backend import DefaultBackend
from qat.core.config.descriptions import PipelineFactoryDescription
from qat.core.pipelines.factory import PipelineFactory
from qat.engines import ZeroEngine
from qat.engines.waveform import EchoEngine
from qat.frontend import DefaultFrontend
from qat.middleend import DefaultMiddleend
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import DefaultTargetData, TargetData
from qat.pipelines.pipeline import Pipeline
from qat.runtime import DefaultRuntime


def create_mock_pipeline(model, name="test", echo=False) -> Pipeline:
    """Used for mocking a pipeline factory for testing purposes."""

    engine = EchoEngine() if echo else ZeroEngine()
    with pytest.warns(DeprecationWarning):
        return Pipeline(
            name=name,
            frontend=DefaultFrontend(model=model),
            middleend=DefaultMiddleend(model=model),
            backend=DefaultBackend(model=model),
            model=model,
            runtime=DefaultRuntime(engine=engine),
        )


def create_mock_pipeline_with_engine(model, engine, name="test") -> Pipeline:
    """Used for mocking a pipeline factory with a specific engine for testing purposes."""

    with pytest.warns(DeprecationWarning):
        return Pipeline(
            name=name,
            frontend=DefaultFrontend(model=model),
            middleend=DefaultMiddleend(model=model),
            backend=DefaultBackend(model=model),
            model=model,
            runtime=DefaultRuntime(engine=engine),
        )


def create_mock_pipeline_with_target_data(model, target_data, name="test") -> Pipeline:
    """Used for mocking a pipeline factory with target data for testing purposes."""

    with pytest.warns(DeprecationWarning):
        return Pipeline(
            name=name,
            frontend=DefaultFrontend(model=model),
            middleend=DefaultMiddleend(model=model, target_data=target_data),
            backend=DefaultBackend(model=model),
            model=model,
            runtime=DefaultRuntime(engine=ZeroEngine()),
            target_data=target_data,
        )


class TestPipelineFactory:
    def test_build_pipeline(self):
        """Test that the build pipeline method correctly calls the factory function."""

        config = PipelineFactoryDescription(
            name="test_factory",
            pipeline="tests.unit.core.pipelines.test_factory.create_mock_pipeline",
            config={},
        )

        factory = PipelineFactory(config, loader=EchoModelLoader())
        assert factory.name == "test_factory"
        assert isinstance(factory.frontend, DefaultFrontend)
        assert isinstance(factory.middleend, DefaultMiddleend)
        assert isinstance(factory.backend, DefaultBackend)
        assert isinstance(factory.runtime, DefaultRuntime)
        assert isinstance(factory.engine, ZeroEngine)

    def test_build_pipeline_with_custom_parameter(self):
        """Sets the `echo=True` to verify it creates the correct engines, thus properly
        accounting for the config."""

        config = PipelineFactoryDescription(
            name="test_factory",
            pipeline="tests.unit.core.pipelines.test_factory.create_mock_pipeline",
            config={"echo": True},
        )

        factory = PipelineFactory(config, loader=EchoModelLoader())
        assert isinstance(factory.engine, EchoEngine)

    def test_build_pipeline_with_invalid_custom_parameter_throws_error(self):
        """Adds an invalid parameter to the config to verify it raises an error."""

        config = PipelineFactoryDescription(
            name="test_factory",
            pipeline="tests.unit.core.pipelines.test_factory.create_mock_pipeline",
            config={"invalid_param": True},
        )

        with pytest.raises(TypeError):
            PipelineFactory(config, loader=EchoModelLoader())

    def test_build_pipeline_with_custom_engine(self):
        """Tests that the factory can build a pipeline with a custom engine."""

        config = PipelineFactoryDescription(
            name="test_factory",
            pipeline="tests.unit.core.pipelines.test_factory.create_mock_pipeline_with_engine",
            engine="echo",
        )

        factory = PipelineFactory(config, loader=EchoModelLoader(), engine=EchoEngine())
        assert isinstance(factory.engine, EchoEngine)

    def test_has_argument_returns_false_if_target_data_not_required(self):
        """Tests the check to see if the factory requires target data."""

        assert PipelineFactory._has_argument(create_mock_pipeline, "target_data") is False

    def test_has_argument_returns_true_if_target_data_required(self):
        """Tests that the factory creates target data if the factory function requires it."""

        assert PipelineFactory._has_argument(
            create_mock_pipeline_with_target_data, "target_data"
        )

    def test_has_argument_returns_false_if_engine_not_required(self):
        """Tests the check to see if the factory requires target data."""

        assert PipelineFactory._has_argument(create_mock_pipeline, "engine") is False

    def test_has_argument_returns_true_if_engine_required(self):
        """Tests that the factory creates target data if the factory function requires it."""

        assert PipelineFactory._has_argument(create_mock_pipeline_with_engine, "engine")

    def test_build_pipeline_with_custom_target_data(self):
        """Tests that the factory can build a pipeline with custom target data."""

        config = PipelineFactoryDescription(
            name="test_factory",
            pipeline="tests.unit.core.pipelines.test_factory.create_mock_pipeline_with_target_data",
            target_data="qat.model.target_data.DefaultTargetData",
        )

        factory = PipelineFactory(
            config, loader=EchoModelLoader(), target_data=DefaultTargetData()
        )
        assert isinstance(factory.target_data, TargetData)
