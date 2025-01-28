# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from qat.backend.validation_passes import HardwareConfigValidity
from qat.compiler.analysis_passes import InputAnalysis
from qat.compiler.transform_passes import (
    InputOptimisation,
    Parse,
    PhaseOptimisation,
    PostProcessingOptimisation,
)
from qat.compiler.validation_passes import InstructionValidation, ReadoutValidation
from qat.ir.pass_base import PassManager
from qat.runtime import LegacyRuntime, NativeEngine, SimpleRuntime
from qat.runtime.analysis_passes import CalibrationAnalysis, IndexMappingAnalysis
from qat.runtime.transform_passes import (
    AssignResultsTransform,
    ErrorMitigation,
    InlineResultsProcessingTransform,
    PostProcessingTransform,
    ResultTransform,
)


def DefaultCompile(hardware_model):
    pipeline = PassManager()
    return (
        pipeline
        | InputAnalysis()
        | InputOptimisation(hardware_model)
        | Parse(hardware_model)
    )


def DefaultExecute(hardware_model, engine=None):
    if engine is None:
        engine = hardware_model.create_engine()
    pipeline = PassManager()
    return (
        pipeline
        | HardwareConfigValidity(hardware_model)
        | CalibrationAnalysis()
        | PhaseOptimisation()
        | PostProcessingOptimisation()
        | InstructionValidation(engine)
        | ReadoutValidation(hardware_model)
    )


def DefaultPostProcessing(hardware_model):
    pipeline = PassManager()
    return (
        pipeline
        | ResultTransform()
        | IndexMappingAnalysis(hardware_model)
        | ErrorMitigation(hardware_model)
    )


def DefaultRuntime(engine):
    if isinstance(engine, NativeEngine):
        return SimpleRuntime
    else:
        return LegacyRuntime


def EchoCompile(model):
    return (
        PassManager()
        | InputAnalysis()
        | InputOptimisation(model)
        | Parse(model)
        | HardwareConfigValidity(model)
        | CalibrationAnalysis()
        | PhaseOptimisation()
        | PostProcessingOptimisation()
        | ReadoutValidation(model)
    )


def EchoExecute():
    return PassManager()


def EchoPostProcessing(model):
    return (
        PassManager()
        | PostProcessingTransform()
        | InlineResultsProcessingTransform()
        | AssignResultsTransform()
        | ResultTransform()
        | IndexMappingAnalysis(model)
        | ErrorMitigation(model)
    )
