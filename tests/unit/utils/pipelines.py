from qat.backend.waveform_v1.codegen import WaveformV1Backend
from qat.core.pipeline import Pipeline
from qat.engines.native import NativeEngine
from qat.frontend.auto import AutoFrontend
from qat.middleend.middleends import DefaultMiddleend
from qat.pipelines.echo import get_results_pipeline
from qat.runtime.simple import SimpleRuntime


def get_pipeline(model, engine: NativeEngine, name="echo") -> Pipeline:
    results_pipeline = get_results_pipeline(model=model)

    return Pipeline(
        name=name,
        frontend=AutoFrontend(model),
        middleend=DefaultMiddleend(model),
        backend=WaveformV1Backend(model),
        runtime=SimpleRuntime(engine=engine, results_pipeline=results_pipeline),
        model=model,
    )
