from qat.backend.waveform_v1.codegen import WaveformV1Backend
from qat.engines.native import NativeEngine
from qat.engines.zero import ZeroEngine
from qat.frontend.auto import AutoFrontend
from qat.middleend.middleends import DefaultMiddleend
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.runtime.results_pipeline import get_default_results_pipeline
from qat.runtime.simple import SimpleRuntime


class MockPipeline(UpdateablePipeline):
    """A mock pipeline for testing purposes."""

    @staticmethod
    def _build_pipeline(config, model, target_data=None, engine=None) -> Pipeline:
        engine = engine if engine is not None else NativeEngine()
        return Pipeline(
            name=config.name,
            model=model,
            frontend=AutoFrontend(model),
            middleend=DefaultMiddleend(model),
            backend=WaveformV1Backend(model),
            runtime=SimpleRuntime(
                engine=engine,
                results_pipeline=get_default_results_pipeline(model),
            ),
        )


def get_mock_pipeline(model, name="test", engine=None) -> Pipeline:
    """A factory for creating a pipeline, mocked up for testing purposes."""
    engine = engine if engine is not None else ZeroEngine()
    return Pipeline(
        name=name,
        model=model,
        frontend=AutoFrontend(model),
        middleend=DefaultMiddleend(model),
        backend=WaveformV1Backend(model),
        runtime=SimpleRuntime(
            engine=engine,
            results_pipeline=get_default_results_pipeline(model),
        ),
    )


def get_mock_compile_pipeline(model, name="test", engine=None) -> CompilePipeline:
    """A factory for creating a compile pipeline, mocked up for testing purposes."""
    engine = engine if engine is not None else ZeroEngine()
    return CompilePipeline(
        name=name,
        model=model,
        frontend=AutoFrontend(model),
        middleend=DefaultMiddleend(model),
        backend=WaveformV1Backend(model),
        target_data=TargetData.default(),
    )


def get_mock_execute_pipeline(model, name="test", engine=None) -> ExecutePipeline:
    """A factory for creating an execute pipeline, mocked up for testing purposes."""
    engine = engine if engine is not None else ZeroEngine()
    return ExecutePipeline(
        name=name,
        model=model,
        runtime=SimpleRuntime(engine=engine),
        target_data=TargetData.default(),
    )


class MockPipelineConfig(PipelineConfig):
    name: str = "test"
    test_attr: bool = False


class MockUpdateablePipeline(UpdateablePipeline):
    """A mock updateable pipeline for testing purposes."""

    @staticmethod
    def _build_pipeline(
        config: MockPipelineConfig, model, target_data=None, engine=None
    ) -> Pipeline:
        engine = engine if engine is not None else ZeroEngine()
        target_data = target_data if target_data is not None else TargetData.default()
        return Pipeline(
            name=config.name,
            model=model,
            frontend=AutoFrontend(model=model),
            middleend=DefaultMiddleend(model=model),
            backend=WaveformV1Backend(model=model),
            runtime=SimpleRuntime(engine=engine),
            target_data=target_data,
        )


class MockCompileUpdateablePipeline(UpdateablePipeline):
    """A mock updateable pipeline that only supports compilation, for testing purposes."""

    @staticmethod
    def _build_pipeline(
        config: MockPipelineConfig, model, target_data=None, engine=None
    ) -> CompilePipeline:
        target_data = target_data if target_data is not None else TargetData.default()
        return CompilePipeline(
            name=config.name,
            model=model,
            frontend=AutoFrontend(model=model),
            middleend=DefaultMiddleend(model=model),
            backend=WaveformV1Backend(model=model),
            target_data=target_data,
        )


class MockExecuteUpdateablePipeline(UpdateablePipeline):
    """A mock updateable pipeline that only supports execution, for testing purposes."""

    @staticmethod
    def _build_pipeline(
        config: MockPipelineConfig, model, target_data=None, engine=None
    ) -> ExecutePipeline:
        engine = engine if engine is not None else ZeroEngine()
        target_data = target_data if target_data is not None else TargetData.default()
        return ExecutePipeline(
            name=config.name,
            model=model,
            runtime=SimpleRuntime(engine=engine),
            target_data=target_data,
        )
