# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# Creating custom pipelines in QAT
# ========================================
#
# We can bring together all the granular components of a pipeline to create a custom pipeline.
# See below.

# %% tags=["remove-cell"]
# used to disable output from logs; not shown in the docs because of the
# remove-cell tag
import logging

logging.disable(logging.CRITICAL)

# %% [markdown]
# Let's recreate the echo pipeline!
# * `AutoFrontend`: dispatches the correct frontend, decided by the input type.
# * `DefaultMiddleend`: Applies default optimization and lowering of pulse-level IR.
# * `WaveformBackend`: Transforms the IR into a binary for "Waveform" hardware.
# * `SimpleRuntime`: Applies a hybrid of execution on the hardware, followed by a results post-processing pipeline with an EchoEngine in this example.
# * `EchoEngine`: Executes the binary against an Echo simulator.

# %%
from qat.backend.waveform import WaveformBackend
from qat.core.pipeline import Pipeline
from qat.engines.waveform import EchoEngine
from qat.frontend import AutoFrontend
from qat.middleend.default import DefaultMiddleend
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.target_data import DefaultTargetData
from qat.runtime import SimpleRuntime
from qat.runtime.results_pipeline import get_results_pipeline

model = LucyModelLoader(qubit_count=8).load()
target_data = DefaultTargetData()
results_pipeline = get_results_pipeline(model=model, target_data=target_data)
new_echo8 = Pipeline(
    name="new_echo8",
    frontend=AutoFrontend.default_for_pydantic(model),
    middleend=DefaultMiddleend(model, target_data),
    backend=WaveformBackend(model, target_data),
    runtime=SimpleRuntime(EchoEngine(), results_pipeline=results_pipeline),
    model=model,
    target_data=target_data,
)

# %% [markdown]
# If we want to make this an updateable pipeline, we can do so as shown below. The `_build_pipeline` acts a factory method with the recipe to build a pipeline provided the model and target data.

# %%
from qat.backend.waveform import WaveformBackend
from qat.core.pipeline import Pipeline
from qat.engines.waveform import EchoEngine
from qat.frontend import AutoFrontend
from qat.middleend.default import DefaultMiddleend
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.target_data import DefaultTargetData, TargetData
from qat.pipelines.updateable import Model, PipelineConfig, UpdateablePipeline
from qat.runtime import SimpleRuntime
from qat.runtime.results_pipeline import get_results_pipeline


class MyCoolPipeline(UpdateablePipeline):
    @staticmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: Model,
        target_data: TargetData | None,
        *args,
        **kwargs,
    ) -> Pipeline:
        results_pipeline = get_results_pipeline(model=model, target_data=target_data)
        return Pipeline(
            name="new_echo8",
            frontend=AutoFrontend.default_for_pydantic(model),
            middleend=DefaultMiddleend(model, target_data),
            backend=WaveformBackend(model, target_data),
            runtime=SimpleRuntime(EchoEngine(), results_pipeline=results_pipeline),
            model=model,
            target_data=target_data,
        )


model = LucyModelLoader(qubit_count=8).load()
target_data = DefaultTargetData()
pipeline_instance = MyCoolPipeline(
    config=dict(name="my_pipeline"), model=model, target_data=target_data
)
