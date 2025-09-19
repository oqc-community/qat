# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.core.pass_base import PassManager
from qat.pipelines.pipeline import Pipeline
from qat.purr.compiler.builders import QuantumInstructionBuilder

from .compile import CompileSweepPipeline
from .execute import ExecuteSweepPipeline


class FullSweepPipeline(CompileSweepPipeline, ExecuteSweepPipeline):
    """A pipeline for compiling and executing :class:`QuantumInstructionBuilder`s with
    sweeps and device assigns.

    This pipeline combines the functionality of both the :class:`CompileSweepPipeline` and
    :class:`ExecuteSweepPipeline`.
    """

    def __init__(
        self,
        base_pipeline: Pipeline,
        preprocessing_pipeline: PassManager | None = None,
    ):
        CompileSweepPipeline.__init__(
            self, base_pipeline=base_pipeline, preprocessing_pipeline=preprocessing_pipeline
        )
        ExecuteSweepPipeline.__init__(self, base_pipeline=base_pipeline)

    def is_subtype_of(self, cls):
        return isinstance(self, cls) or self._base_pipeline.is_subtype_of(cls)

    def run(self, builder: QuantumInstructionBuilder, compiler_config=None):
        """Compiles and executes a :class:`QuantumInstructionBuilder` with sweeps and device
        assigns.

        :param builder: The instruction builder to compile and execute.
        :param compiler_config: Optional compiler configuration to use for compilation and
            execution.
        :return: The results of the execution.
        """
        executable, compile_metrics = self.compile(builder, compiler_config)
        results, execute_metrics = self.execute(executable, compiler_config)
        return results, compile_metrics.merge(execute_metrics)
