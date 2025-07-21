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


def get_mock_pipeline(model, name="test") -> Pipeline:
    """A factory for creating a pipeline, mocked up for testing purposes."""
    return Pipeline(
        name=name,
        model=model,
        frontend=AutoFrontend(model),
        middleend=DefaultMiddleend(model),
        backend=WaveformV1Backend(model),
        runtime=SimpleRuntime(
            engine=ZeroEngine(),
            results_pipeline=get_default_results_pipeline(model),
        ),
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
