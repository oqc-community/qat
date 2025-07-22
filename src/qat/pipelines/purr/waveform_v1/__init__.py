# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.model.loaders.purr import EchoModelLoader
from qat.pipelines.updateable import PipelineConfig

from .compile import WaveformV1CompilePipeline
from .execute import EchoExecutePipeline
from .full import EchoPipeline


def _create_pipeline_instance(num_qubits: int) -> EchoPipeline:
    return EchoPipeline(
        config=PipelineConfig(name=f"echo{num_qubits}"),
        loader=EchoModelLoader(qubit_count=num_qubits),
    ).pipeline


echo8 = _create_pipeline_instance(8)
echo16 = _create_pipeline_instance(16)
echo32 = _create_pipeline_instance(32)


__all__ = [
    "EchoPipeline",
    "EchoExecutePipeline",
    "echo8",
    "echo16",
    "echo32",
    "PipelineConfig",
    "WaveformV1CompilePipeline",
]
