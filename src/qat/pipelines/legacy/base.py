# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.fallthrough import FallthroughBackend
from qat.core.pass_base import PassManager
from qat.core.pipeline import Pipeline
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.legacy.transform import PhaseOptimisation
from qat.middleend.passes.transform import PostProcessingSanitisation
from qat.middleend.passes.validation import (
    HardwareConfigValidity,
    InstructionValidation,
    ReadoutValidation,
)
from qat.purr.compiler.execution import InstructionExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.legacy import LegacyRuntime
from qat.runtime.passes.analysis import CalibrationAnalysis, IndexMappingAnalysis
from qat.runtime.passes.transform import ErrorMitigation, ResultTransform


def get_results_pipeline(model: QuantumHardwareModel) -> PassManager:
    """A factory for creating results pipelines for the :class:`LegacyRuntime` to execute.

    The :class:`LegacyRuntime` executes programs using the legacy
    :class:`InstructionExecutionEngine`, which carries out some of the post-processing
    responibilities. This pipeline contains passes that covers the responibilites of the
    :class:`QuantumRuntime <qat.purr.compiler.runtime.QuantumRuntme`.

    :param model: The hardware model.
    :return: A pipeline containing the runtime passes.
    """
    return (
        PassManager()
        | ResultTransform()
        | IndexMappingAnalysis(model)
        | ErrorMitigation(model)
    )


def get_middleend_pipeline(
    model: QuantumHardwareModel, engine: InstructionExecutionEngine
) -> PassManager:
    """A factory for creating middleend pipelines for legacy echo models.

    Includes a list of passes that replicate the responsibilities of
    :meth:`QuantumExecutionEngine.optimize <qat.purr.compiler.execution.QuantumExecutionEngine.optimize`
    and
    :meth:`QuantumExecutionEngine.validate <qat.purr.compiler.execution.QuantumExecutionEngine.validate`.

    :param model: The hardware model is required for validation.
    :param engine: The echo engine is required to perform validation.
    :return: The pipeline as a pass manager.
    """
    return (
        PassManager()
        | HardwareConfigValidity(model)
        | CalibrationAnalysis()
        | PhaseOptimisation()
        | PostProcessingSanitisation()
        | InstructionValidation(engine)
        | ReadoutValidation(model)
    )


def get_pipeline(model: QuantumHardwareModel, name: str = "legacy") -> Pipeline:
    """A factory for building complete compilation and execution pipelines using legacy
    hardware models and legacy engines.

    :param model: The hardware model.
    :param name: The name of the pipeline, defaults to "legacy"
    :return: The complete pipeline.
    """
    engine = model.create_engine()
    return Pipeline(
        name=name,
        frontend=AutoFrontend(model),
        middleend=CustomMiddleend(model, pipeline=get_middleend_pipeline(model, engine)),
        backend=FallthroughBackend(model),
        runtime=LegacyRuntime(engine=engine, results_pipeline=get_results_pipeline(model)),
        model=model,
    )
