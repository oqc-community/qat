# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.loaders.purr.qiskit import QiskitModelLoader
from qat.pipelines.updateable import PipelineConfig

from .compile import LegacyQiskitCompilePipeline
from .execute import LegacyQiskitExecutePipeline
from .full import LegacyQiskitPipeline


def _create_pipeline_instance(num_qubits: int) -> LegacyQiskitPipeline:
    return LegacyQiskitPipeline(
        config=PipelineConfig(name=f"legacy_qiskit{num_qubits}"),
        loader=QiskitModelLoader(qubit_count=num_qubits),
    ).pipeline


legacy_qiskit8 = _create_pipeline_instance(8)
legacy_qiskit16 = _create_pipeline_instance(16)
legacy_qiskit32 = _create_pipeline_instance(32)

__all__ = [
    "LegacyQiskitPipeline",
    "LegacyQiskitCompilePipeline",
    "LegacyQiskitExecutePipeline",
    "PipelineConfig",
    "legacy_qiskit8",
    "legacy_qiskit16",
    "legacy_qiskit32",
]
