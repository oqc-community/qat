# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.pipelines.legacy.base import LegacyExecutePipeline as LegacyQbloxExecutePipeline
from qat.pipelines.updateable import PipelineConfig
from qat.purr.compiler.hardware_models import QuantumHardwareModel

from .compile import LegacyQbloxCompilePipeline
from .full import LegacyQbloxPipeline


def get_pipeline(
    model: QuantumHardwareModel, name: str = "legacy_qblox"
) -> LegacyQbloxPipeline:
    """A factory for creating pipelines that performs the responsibilities of the legacy
    runtime :class:`EchoEngine <qat.purr.compiler.runtime.QuantumRuntime>`

    :param model: The Qblox hardware model.
    :param name: The name of the pipeline, defaults to "legacy_qblox"
    :return: The complete pipeline, including the runtime and engine.
    """
    return LegacyQbloxPipeline(config=PipelineConfig(name=name), model=model).pipeline


__all__ = [
    "LegacyQbloxPipeline",
    "LegacyQbloxCompilePipeline",
    "LegacyQbloxExecutePipeline",
    "PipelineConfig",
    "get_pipeline",
]
