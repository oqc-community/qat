# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.model.loaders.purr.dummy import QbloxDummyModelLoader
from qat.pipelines.base import BasePipeline
from qat.pipelines.purr.qblox.compile import QbloxCompilePipeline1, QbloxCompilePipeline2
from qat.pipelines.purr.qblox.execute import QbloxExecutePipeline
from qat.pipelines.purr.qblox.full import QbloxPipeline1, QbloxPipeline2
from qat.pipelines.updateable import PipelineConfig


def _create_pipeline_instance(num_qubits: int) -> BasePipeline:
    name = f"dummy{num_qubits}"
    return QbloxPipeline2(
        config=PipelineConfig(name=name),
        loader=QbloxDummyModelLoader(name=name, qubit_count=num_qubits),
    ).pipeline


dummy8 = _create_pipeline_instance(8)
dummy16 = _create_pipeline_instance(16)
# dummy32 = _create_pipeline_instance(32) # TODO: 32Q support: COMPILER-728


__all__ = [
    "QbloxPipeline1",
    "QbloxPipeline2",
    "QbloxExecutePipeline",
    "dummy8",
    "dummy16",
    # "dummy32",  # TODO: 32Q support: COMPILER-728
    "PipelineConfig",
    "QbloxCompilePipeline1",
    "QbloxCompilePipeline2",
]
