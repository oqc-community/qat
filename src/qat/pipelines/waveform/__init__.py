# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.model.loaders.lucy import LucyModelLoader
from qat.pipelines.updateable import PipelineConfig

from .compile import WaveformCompilePipeline
from .execute import EchoExecutePipeline
from .full import EchoPipeline


def _create_pipeline_instance(num_qubits: int) -> EchoPipeline:
    return EchoPipeline(
        config=PipelineConfig(name=f"echo{num_qubits}"),
        loader=LucyModelLoader(qubit_count=num_qubits),
    ).pipeline


echo8 = _create_pipeline_instance(8)
echo16 = _create_pipeline_instance(16)
echo32 = _create_pipeline_instance(32)

__all__ = [
    "EchoPipeline",
    "EchoExecutePipeline",
    "WaveformCompilePipeline",
    "PipelineConfig",
    "echo8",
    "echo16",
    "echo32",
]
