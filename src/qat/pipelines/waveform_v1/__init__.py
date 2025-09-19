# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.model.loaders.lucy import LucyModelLoader
from qat.pipelines.updateable import PipelineConfig

from .compile import WaveformV1CompilePipeline
from .execute import EchoExecutePipeline
from .full import EchoPipeline


def _create_experimental_pipeline_instance(num_qubits: int) -> EchoPipeline:
    return EchoPipeline(
        config=PipelineConfig(name=f"experimental_echo{num_qubits}"),
        loader=LucyModelLoader(qubit_count=num_qubits),
    ).pipeline


experimental_echo8 = _create_experimental_pipeline_instance(8)
experimental_echo16 = _create_experimental_pipeline_instance(16)
experimental_echo32 = _create_experimental_pipeline_instance(32)

__all__ = [
    "EchoPipeline",
    "EchoExecutePipeline",
    "WaveformV1CompilePipeline",
    "PipelineConfig",
    "experimental_echo8",
    "experimental_echo16",
    "experimental_echo32",
]
