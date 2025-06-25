# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.qblox.codegen import QbloxBackend
from qat.core.pass_base import PassManager
from qat.core.pipeline import Pipeline
from qat.frontend import AutoFrontend
from qat.middleend.middleends import CustomMiddleend
from qat.middleend.passes.legacy.transform import (
    DeviceUpdateSanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
)
from qat.middleend.passes.legacy.validation import (
    InstructionValidation,
    ReadoutValidation,
)
from qat.model.target_data import TargetData
from qat.pipelines.legacy.base import get_results_pipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.legacy import LegacyRuntime


def get_middleend_pipeline(
    model: QuantumHardwareModel, target_data: TargetData = TargetData.default()
) -> PassManager:
    """Factory for creating middleend pipelines for Qblox legacy models.

    Includes a list of passes that replicate the responsibilities of
    :meth:`LiveDeviceEngine.optimize <qat.purr.backends.live.LiveDeviceEngine.optimize>` and
    :meth:`LiveDeviceEngine.validate <qat.purr.backends.live.LiveDeviceEngine.validate>`.

    :param model: The hardware model is required for ReadoutValidation.
    :return: The pipeline as a pass manager.
    """

    return (
        PassManager()
        | PhaseOptimisation()
        | PostProcessingSanitisation()
        | DeviceUpdateSanitisation()
        | InstructionValidation(target_data)
        | ReadoutValidation(model)
    )


def get_pipeline(model: QuantumHardwareModel, name: str = "legacy_qblox") -> Pipeline:
    """A factory for creating pipelines that performs the responsibilities of the legacy
    runtime :class:`EchoEngine <qat.purr.compiler.runtime.QuantumRuntime>`

    :param model: The Qblox hardware model.
    :param name: The name of the pipeline, defaults to "legacy_qblox"
    :return: The complete pipeline, including the runtime and engine.
    """

    engine = model.create_engine()

    return Pipeline(
        name=name,
        frontend=AutoFrontend(model),
        middleend=CustomMiddleend(model, pipeline=get_middleend_pipeline(model)),
        backend=QbloxBackend(model),
        runtime=LegacyRuntime(engine=engine, results_pipeline=get_results_pipeline(model)),
        model=model,
    )
