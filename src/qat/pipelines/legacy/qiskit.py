# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.fallthrough import FallthroughBackend
from qat.core.pass_base import PassManager
from qat.core.pipeline import Pipeline
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.legacy.transform import QiskitInstructionsWrapper
from qat.middleend.passes.legacy.validation import (
    HardwareConfigValidity,
    QiskitResultsFormatValidation,
)
from qat.model.loaders.legacy import QiskitModelLoader
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime import LegacyRuntime
from qat.runtime.passes.legacy.transform import (
    QiskitErrorMitigation,
    QiskitSimplifyResults,
    QiskitStripMetadata,
)


def get_results_pipeline() -> PassManager:
    """A factory for creating results pipelines for the :class:`LegacyRuntime` to execute
    with the legacy :class:`QiskitEngine <qat.purr.backends.qiskit_simulator.QiskitEngine>`.

    Contains passes that carry out the responibilities in the legacy
    :class:`QiskitRuntime <qat.purr.backends.qiskit_simulator.QiskitRuntime>`.

    :param model: The hardware model.
    :return: A pipeline containing the runtime passes.
    """
    return (
        PassManager()
        | QiskitStripMetadata()
        | QiskitErrorMitigation()
        | QiskitSimplifyResults()
    )


def get_middleend_pipeline(model: QuantumHardwareModel) -> PassManager:
    """A factory for creating middleend pipelines for legacy qiskit models.

    Includes a list of passes that replicate the responsibilities of
    :class:`QiskitEngine.validate <qat.purr.backends.qiskit_simulator.QiskitEngine.validate>`
    and
    :class:`QiskitEngine.optimize <qat.purr.backends.qiskit_simulator.QiskitEngine.optimize>`.

    :param model: The hardware model is required for validation.
    :param engine: The echo engine is required to perform validation.
    :return: The pipeline as a pass manager.
    """

    return (
        PassManager()
        | QiskitResultsFormatValidation()
        | HardwareConfigValidity(model)
        | QiskitInstructionsWrapper()
    )


def get_pipeline(model: QuantumHardwareModel, name: str = "legacy_qiskit") -> Pipeline:
    """Create a default pipeline using the legacy Qiskit backend.

    :param model: The Qiskit hardware model
    :param name: The name of the pipeline, defaults to "legacy"
    :return: The complete pipeline.
    """

    engine = model.create_engine()

    return Pipeline(
        name=name,
        frontend=AutoFrontend(model),
        middleend=CustomMiddleend(model, pipeline=get_middleend_pipeline(model)),
        backend=FallthroughBackend(model),
        runtime=LegacyRuntime(engine=engine, results_pipeline=get_results_pipeline()),
        model=model,
    )


legacy_qiskit8 = get_pipeline(
    QiskitModelLoader(qubit_count=8).load(), name="legacy_qiskit8"
)
legacy_qiskit16 = get_pipeline(
    QiskitModelLoader(qubit_count=16).load(), name="legacy_qiskit16"
)
legacy_qiskit32 = get_pipeline(
    QiskitModelLoader(qubit_count=32).load(), name="legacy_qiskit32"
)
