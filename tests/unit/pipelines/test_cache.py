# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.model.loaders.purr.echo import EchoModelLoader
from qat.pipelines.cache import CompilePipelineCache, ExecutePipelineCache

from tests.unit.utils.pipelines import (
    MockCompileUpdateablePipeline,
    MockExecuteUpdateablePipeline,
    MockUpdateablePipeline,
)


class TestCompilePipelineCache:
    loader = EchoModelLoader()

    @pytest.mark.parametrize(
        "pipeline",
        [
            MockExecuteUpdateablePipeline(config=dict(name="test"), loader=loader),
            MockExecuteUpdateablePipeline(config=dict(name="test"), loader=loader).pipeline,
            MockCompileUpdateablePipeline(config=dict(name="test"), loader=loader),
            MockCompileUpdateablePipeline(config=dict(name="test"), loader=loader).pipeline,
        ],
    )
    def test_cannot_instantiate_pipeline_cache(self, pipeline):
        with pytest.raises(ValueError):
            CompilePipelineCache(pipeline)

    @pytest.mark.parametrize(
        "pipeline",
        [
            MockUpdateablePipeline(config=dict(name="test"), loader=loader),
            MockUpdateablePipeline(config=dict(name="test"), loader=loader).pipeline,
        ],
    )
    def test_can_instantiate_compile_pipeline_cache(self, pipeline):
        cache = CompilePipelineCache(pipeline)
        assert isinstance(cache, CompilePipelineCache)
        assert cache._pipeline == pipeline
        assert cache.frontend == pipeline.frontend
        assert cache.middleend == pipeline.middleend
        assert cache.backend == pipeline.backend
        assert cache.name == pipeline.name
        assert cache.model == pipeline.model


class TestExecutePipelineCache:
    loader = EchoModelLoader()

    @pytest.mark.parametrize(
        "pipeline",
        [
            MockExecuteUpdateablePipeline(config=dict(name="test"), loader=loader),
            MockExecuteUpdateablePipeline(config=dict(name="test"), loader=loader).pipeline,
            MockCompileUpdateablePipeline(config=dict(name="test"), loader=loader),
            MockCompileUpdateablePipeline(config=dict(name="test"), loader=loader).pipeline,
        ],
    )
    def test_cannot_instantiate_pipeline_cache(self, pipeline):
        with pytest.raises(ValueError):
            ExecutePipelineCache(pipeline)

    @pytest.mark.parametrize(
        "pipeline",
        [
            MockUpdateablePipeline(config=dict(name="test"), loader=loader),
            MockUpdateablePipeline(config=dict(name="test"), loader=loader).pipeline,
        ],
    )
    def test_can_instantiate_compile_pipeline_cache(self, pipeline):
        cache = ExecutePipelineCache(pipeline)
        assert isinstance(cache, ExecutePipelineCache)
        assert cache._pipeline == pipeline
        assert cache.engine == pipeline.engine
        assert cache.runtime == pipeline.runtime
        assert cache.name == pipeline.name
        assert cache.model == pipeline.model
