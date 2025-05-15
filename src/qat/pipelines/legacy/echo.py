# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.fallthrough import FallthroughBackend
from qat.core.pass_base import PassManager
from qat.core.pipeline import Pipeline
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.legacy.transform import IntegratorAcquireSanitisation
from qat.middleend.passes.transform import PhaseOptimisation, PostProcessingSanitisation
from qat.middleend.passes.validation import (
    HardwareConfigValidity,
    InstructionValidation,
    ReadoutValidation,
)
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.legacy.base import get_results_pipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.legacy import LegacyRuntime
from qat.runtime.passes.analysis import CalibrationAnalysis


def get_middleend_pipeline(
    model: QuantumHardwareModel, target_data: TargetData = TargetData.default()
) -> PassManager:
    """Factory for creating middleend pipelines for legacy echo models.

    Includes a list of passes that replicate the responsibilities of
    :meth:`EchoEngine.optimize <qat.purr.backends.echo.EchoEngine.optimize>` and
    :meth:`EchoEngine.validate <qat.purr.backends.echo.EchoEngine.validate>`.

    :param model: The hardware model is required for validation.
    :param engine: The echo engine is required to perform validation.
    :return: The pipeline as a pass manager.
    """
    return (
        PassManager()
        | HardwareConfigValidity(model)
        | CalibrationAnalysis()
        | PhaseOptimisation()
        | IntegratorAcquireSanitisation()
        | PostProcessingSanitisation()
        | InstructionValidation(target_data)
        | ReadoutValidation(model)
    )


def get_pipeline(model: QuantumHardwareModel, name: str = "legacy_echo") -> Pipeline:
    """A factory for creating pipelines that performs the responsibilites of the legacy
    :class:`EchoEngine <qat.purr.backends.echo.EchoEngine>` using the
    :class:`LegacyRuntime`.

    :param model: The echo hardware model.
    :param name: The name of the pipeline, defaults to "legacy_echo"
    :return: The complete pipeline, including the runtime and engine.
    """

    engine = model.create_engine()

    return Pipeline(
        name=name,
        frontend=AutoFrontend(model),
        middleend=CustomMiddleend(model, pipeline=get_middleend_pipeline(model)),
        backend=FallthroughBackend(model),
        runtime=LegacyRuntime(engine=engine, results_pipeline=get_results_pipeline(model)),
        model=model,
    )


legacy_echo8 = get_pipeline(EchoModelLoader(qubit_count=8).load(), name="legacy_echo8")
legacy_echo16 = get_pipeline(EchoModelLoader(qubit_count=16).load(), name="legacy_echo16")
legacy_echo32 = get_pipeline(EchoModelLoader(qubit_count=32).load(), name="legacy_echo32")
