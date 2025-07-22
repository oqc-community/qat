# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.loaders.purr.echo import EchoModelLoader
from qat.pipelines.updateable import PipelineConfig

from .compile import LegacyEchoCompilePipeline
from .execute import LegacyEchoExecutePipeline
from .full import LegacyEchoPipeline


def _create_pipeline_instance(num_qubits: int) -> LegacyEchoPipeline:
    return LegacyEchoPipeline(
        config=PipelineConfig(name=f"legacy_echo{num_qubits}"),
        loader=EchoModelLoader(qubit_count=num_qubits),
    ).pipeline


legacy_echo8 = _create_pipeline_instance(8)
legacy_echo16 = _create_pipeline_instance(16)
legacy_echo32 = _create_pipeline_instance(32)

__all__ = [
    "LegacyEchoPipeline",
    "LegacyEchoCompilePipeline",
    "LegacyEchoExecutePipeline",
    "PipelineConfig",
    "legacy_echo8",
    "legacy_echo16",
    "legacy_echo32",
]
