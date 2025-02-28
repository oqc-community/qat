# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.fallthrough import FallthroughBackend
from qat.backend.validation_passes import HardwareConfigValidity
from qat.compiler.legacy.transform_passes import IntegratorAcquireSanitisation
from qat.compiler.transform_passes import PhaseOptimisation, PostProcessingSanitisation
from qat.compiler.validation_passes import InstructionValidation, ReadoutValidation
from qat.core.pipeline import Pipeline
from qat.frontend.frontends import DefaultFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.passes.pass_base import PassManager
from qat.purr.backends.echo import get_default_echo_hardware
from qat.runtime.analysis_passes import CalibrationAnalysis, IndexMappingAnalysis
from qat.runtime.runtimes.legacy import LegacyRuntime
from qat.runtime.transform_passes import ErrorMitigation, ResultTransform


def get_pipeline(model, name="legacy_echo") -> Pipeline:
    results_pipeline = (
        PassManager()
        | ResultTransform()
        | IndexMappingAnalysis(model)
        | ErrorMitigation(model)
    )

    engine = model.create_engine()
    middleend = (
        PassManager()
        | HardwareConfigValidity(model)
        | CalibrationAnalysis()
        | PhaseOptimisation()
        | IntegratorAcquireSanitisation()
        | PostProcessingSanitisation()
        | InstructionValidation(engine)
        | ReadoutValidation(model)
    )

    return Pipeline(
        name=name,
        frontend=DefaultFrontend(model),
        middleend=CustomMiddleend(model, pipeline=middleend),
        backend=FallthroughBackend(model),
        runtime=LegacyRuntime(engine=engine, results_pipeline=results_pipeline),
        model=model,
    )


legacy_echo8 = get_pipeline(get_default_echo_hardware(qubit_count=8), name="legacy_echo8")
legacy_echo16 = get_pipeline(
    get_default_echo_hardware(qubit_count=16), name="legacy_echo16"
)
legacy_echo32 = get_pipeline(
    get_default_echo_hardware(qubit_count=32), name="legacy_echo32"
)
