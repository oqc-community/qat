# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.fallthrough import FallthroughBackend
from qat.backend.passes.validation import HardwareConfigValidity
from qat.core.pass_base import PassManager
from qat.core.pipeline import Pipeline
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.legacy.transform import IntegratorAcquireSanitisation
from qat.middleend.passes.transform import PhaseOptimisation, PostProcessingSanitisation
from qat.middleend.passes.validation import InstructionValidation, ReadoutValidation
from qat.model.loaders.legacy import EchoModelLoader
from qat.runtime.legacy import LegacyRuntime
from qat.runtime.passes.analysis import CalibrationAnalysis, IndexMappingAnalysis
from qat.runtime.passes.transform import ErrorMitigation, ResultTransform


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
        frontend=AutoFrontend(model),
        middleend=CustomMiddleend(model, pipeline=middleend),
        backend=FallthroughBackend(model),
        runtime=LegacyRuntime(engine=engine, results_pipeline=results_pipeline),
        model=model,
    )


legacy_echo8 = get_pipeline(EchoModelLoader(qubit_count=8).load(), name="legacy_echo8")
legacy_echo16 = get_pipeline(EchoModelLoader(qubit_count=16).load(), name="legacy_echo16")
legacy_echo32 = get_pipeline(EchoModelLoader(qubit_count=32).load(), name="legacy_echo32")
