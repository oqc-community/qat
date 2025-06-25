from qat.backend.waveform_v1.codegen import WaveformV1Backend
from qat.engines.native import NativeEngine
from qat.engines.zero import ZeroEngine
from qat.frontend.auto import AutoFrontend
from qat.middleend.middleends import DefaultMiddleend
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import UpdateablePipeline
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
