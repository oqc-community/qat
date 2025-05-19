# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from qat.backend.waveform_v1.codegen import WaveformV1Backend
from qat.core.pass_base import PassManager
from qat.core.pipeline import Pipeline
from qat.engines.waveform_v1 import EchoEngine
from qat.frontend import AutoFrontend
from qat.middleend.middleends import DefaultMiddleend
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.runtime import SimpleRuntime
from qat.runtime.passes.analysis import IndexMappingAnalysis
from qat.runtime.passes.transform import (
    AssignResultsTransform,
    ErrorMitigation,
    InlineResultsProcessingTransform,
    PostProcessingTransform,
    ResultTransform,
)


def get_results_pipeline(model) -> PassManager:
    return (
        PassManager()
        | PostProcessingTransform()
        | InlineResultsProcessingTransform()
        | AssignResultsTransform()
        | ResultTransform()
        | IndexMappingAnalysis(model)
        | ErrorMitigation(model)
    )


def get_pipeline(model, name="echo", target_data=TargetData.default()) -> Pipeline:
    results_pipeline = get_results_pipeline(model=model)

    return Pipeline(
        name=name,
        frontend=AutoFrontend(model),
        middleend=DefaultMiddleend(model, target_data),
        backend=WaveformV1Backend(model),
        runtime=SimpleRuntime(engine=EchoEngine(), results_pipeline=results_pipeline),
        model=model,
    )


echo8 = get_pipeline(EchoModelLoader(qubit_count=8).load(), name="echo8")
echo16 = get_pipeline(EchoModelLoader(qubit_count=16).load(), name="echo16")
echo32 = get_pipeline(EchoModelLoader(qubit_count=32).load(), name="echo32")
