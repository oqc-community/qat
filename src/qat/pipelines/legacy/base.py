# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.fallthrough import FallthroughBackend
from qat.backend.validation_passes import HardwareConfigValidity
from qat.compiler.transform_passes import PhaseOptimisation, PostProcessingSanitisation
from qat.compiler.validation_passes import InstructionValidation, ReadoutValidation
from qat.core.pipeline import Pipeline
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.passes.pass_base import PassManager
from qat.runtime.analysis_passes import CalibrationAnalysis, IndexMappingAnalysis
from qat.runtime.runtimes.legacy import LegacyRuntime
from qat.runtime.transform_passes import ErrorMitigation, ResultTransform


def get_pipeline(model, name="legacy") -> Pipeline:
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
        | PostProcessingSanitisation()
        | InstructionValidation(engine)
        | ReadoutValidation(model)
    )

    return Pipeline(
        name=name,
        frontend=AutoFrontend(model),
        middleend=CustomMiddleend(model, pipeline=middleend),
        backend=FallthroughBackend(model),
        runtime=LegacyRuntime(engine=engine, results_pipeline=results_pipeline),
        model=model,
    )
