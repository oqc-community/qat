# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.backend.fallthrough import FallthroughBackend
from qat.backend.passes.validation import HardwareConfigValidity
from qat.core.pass_base import PassManager
from qat.core.pipeline import Pipeline
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.legacy.transform import QiskitInstructionsWrapper
from qat.middleend.passes.legacy.validation import QiskitResultsFormatValidation
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.runtime import LegacyRuntime
from qat.runtime.passes.legacy.transform import (
    QiskitErrorMitigation,
    QiskitSimplifyResults,
    QiskitStripMetadata,
)


def get_pipeline(model, name="legacy_qiskit") -> Pipeline:
    """Create a default pipeline using the legacy Qiskit backend.

    :param model: The Qiskit hardware model
    :param name: The name of the pipeline, defaults to "legacy"
    :return: The complete pipeline.
    """
    results_pipeline = (
        PassManager()
        | QiskitStripMetadata()
        | QiskitErrorMitigation()
        | QiskitSimplifyResults()
    )

    engine = model.create_engine()
    middleend = (
        PassManager()
        | QiskitResultsFormatValidation()
        | HardwareConfigValidity(model)
        | QiskitInstructionsWrapper()
    )

    return Pipeline(
        name=name,
        frontend=AutoFrontend(model),
        middleend=CustomMiddleend(model, pipeline=middleend),
        backend=FallthroughBackend(model),
        runtime=LegacyRuntime(engine=engine, results_pipeline=results_pipeline),
        model=model,
    )


legacy_qiskit8 = get_pipeline(get_default_qiskit_hardware(8), name="legacy_qiskit8")
legacy_qiskit16 = get_pipeline(get_default_qiskit_hardware(16), name="legacy_qiskit16")
legacy_qiskit32 = get_pipeline(get_default_qiskit_hardware(32), name="legacy_qiskit32")
