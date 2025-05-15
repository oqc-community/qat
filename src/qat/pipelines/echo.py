# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from qat.backend.waveform_v1.codegen import WaveformV1Backend
from qat.core.pass_base import PassManager
from qat.core.pipeline import Pipeline
from qat.engines.waveform_v1 import EchoEngine
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.analysis import ActivePulseChannelAnalysis
from qat.middleend.passes.transform import (
    AcquireSanitisation,
    EndOfTaskResetSanitisation,
    EvaluatePulses,
    InactivePulseChannelSanitisation,
    InitialPhaseResetSanitisation,
    InstructionGranularitySanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
    RepeatSanitisation,
    ReturnSanitisation,
    SynchronizeTask,
)
from qat.middleend.passes.validation import (
    FrequencyValidation,
    HardwareConfigValidity,
    ReadoutValidation,
)
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.runtime import SimpleRuntime
from qat.runtime.passes.analysis import CalibrationAnalysis, IndexMappingAnalysis
from qat.runtime.passes.transform import (
    AssignResultsTransform,
    ErrorMitigation,
    InlineResultsProcessingTransform,
    PostProcessingTransform,
    ResultTransform,
)


def get_middleend_pipeline(model) -> PassManager:
    target_data = TargetData.default()

    return (
        PassManager()
        | HardwareConfigValidity(model)
        | CalibrationAnalysis()
        # Validate first to fail fast:
        | FrequencyValidation(model, target_data)
        # Process a "task" into a program. Could be considered as a frontend pipeline
        # that converts takes a specific task and makes the appropriate qat ir.
        # For example, wrapping the "task" in a repeat / for loop for QASM / QIR.
        | ActivePulseChannelAnalysis(model)
        | InactivePulseChannelSanitisation()
        | RepeatSanitisation(model, target_data)
        | ReturnSanitisation()
        | SynchronizeTask()
        | EndOfTaskResetSanitisation()
        | InitialPhaseResetSanitisation()
        # Basically just "corrections" to bad ir generated from the builder, should
        # eventually be replaced with behaviour from the builder
        | PostProcessingSanitisation()
        | AcquireSanitisation()
        # handles mid-circuit measures + pp validation, should be split up
        # is pp validation needed in a pipeline that explicitly does pp sanitisation?
        | ReadoutValidation(model)
        # Optimisation of the ir:
        | PhaseOptimisation()
        # Preparing the IR for the backend
        | InstructionGranularitySanitisation(model, target_data)
        | EvaluatePulses()
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


def get_pipeline(model, name="echo") -> Pipeline:
    results_pipeline = get_results_pipeline(model=model)

    return Pipeline(
        name=name,
        frontend=AutoFrontend(model),
        middleend=CustomMiddleend(model, get_middleend_pipeline(model)),
        backend=WaveformV1Backend(model),
        runtime=SimpleRuntime(engine=EchoEngine(), results_pipeline=results_pipeline),
        model=model,
    )


echo8 = get_pipeline(EchoModelLoader(qubit_count=8).load(), name="echo8")
echo16 = get_pipeline(EchoModelLoader(qubit_count=16).load(), name="echo16")
echo32 = get_pipeline(EchoModelLoader(qubit_count=32).load(), name="echo32")
