# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.loaders.purr.dummy import QbloxDummyModelLoader
from qat.pipelines.updateable import PipelineConfig

from .compile import QbloxCompilePipeline
from .execute import QbloxExecutePipeline
from .full import QbloxPipeline


def _create_pipeline_instance(num_qubits: int) -> QbloxPipeline:
    name = f"dummy{num_qubits}"
    return QbloxPipeline(
        config=PipelineConfig(name=name),
        loader=QbloxDummyModelLoader(name=name, qubit_count=num_qubits),
    ).pipeline


dummy8 = _create_pipeline_instance(8)
dummy16 = _create_pipeline_instance(16)
# dummy32 = _create_pipeline_instance(32) # TODO: 32Q support: COMPILER-728


__all__ = [
    "QbloxPipeline",
    "QbloxExecutePipeline",
    "dummy8",
    "dummy16",
    # "dummy32",  # TODO: 32Q support: COMPILER-728
    "PipelineConfig",
    "QbloxCompilePipeline",
]
