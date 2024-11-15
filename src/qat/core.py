# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from typing import Optional

from compiler_config.config import CompilerConfig, MetricsType

from qat.compiler.analysis_passes import InputAnalysis
from qat.compiler.transform_passes import InputOptimisation, InputOptimisationResult, Parse
from qat.ir.pass_base import PassManager, QatIR, TransformPass
from qat.ir.result_base import ResultManager
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.frontends import QASMFrontend
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.metrics import CompilationMetrics
from qat.purr.qatconfig import QatConfig
from qat.qat import QATInput


class QAT:
    def __init__(self):

        self._compile_pipelines = {}
        self._execute_pipelines = {}

        self.config = QatConfig()

        # This will move to pydantic on QatConfig
        rtcs_hw = get_default_RTCS_hardware()
        self.add_pipeline(
            "rtcs",
            compile_pipeline=self.build_default_compile_pipeline(rtcs_hw),
            execute_pipeline=self.build_default_execute_pipeline(rtcs_hw),
        )

        echo_hw = get_default_echo_hardware()
        self.add_pipeline(
            "echo",
            compile_pipeline=self.build_default_compile_pipeline(echo_hw),
            execute_pipeline=self.build_default_execute_pipeline(echo_hw),
        )
        self.set_default_pipeline("rtcs")

    def set_default_pipeline(
        self,
        pipeline_name: str | None = None,
        compile: str | None = None,
        execute: str | None = None,
    ):
        compile = compile or pipeline_name
        execute = execute or pipeline_name

        if not compile in self._compile_pipelines:
            raise Exception(f"Compilation pipeline {compile} not found")

        if not execute in self._compile_pipelines:
            raise Exception(f"Execution pipeline {compile} not found")

        self._compile_pipelines["default"] = self._compile_pipelines[compile]
        self._execute_pipelines["default"] = self._execute_pipelines[execute]

    def add_pipeline(
        self, name: str, compile_pipeline: PassManager | None, execute_pipeline
    ):
        if not (compile_pipeline or execute_pipeline):
            raise Exception(
                "At least one of compile_pipeline or execute_pipeline must be set"
            )
        if (name in self._compile_pipelines) or (name in self._execute_pipelines):
            raise Exception(
                f"A pipeline with name {name} already exists, remove that pipeline first"
            )

        if compile_pipeline:
            self._compile_pipelines[name] = compile_pipeline

        if execute_pipeline:
            self._execute_pipelines[name] = execute_pipeline

    def remove_pipeline(self, name: str):
        del self._compile_pipelines[name]
        del self._execute_pipelines[name]

    def compile(
        self,
        program: QATInput,
        compiler_config: Optional[CompilerConfig] = None,
        pipeline: PassManager | str = "default",
    ):
        if isinstance(pipeline, str):
            pipeline = self._compile_pipelines[pipeline]

        # TODO: Improve metrics and config handling
        compiler_config = compiler_config or CompilerConfig()
        metrics = CompilationMetrics()
        metrics.enable(compiler_config.metrics)
        compilation_results = ResultManager()
        ir = QatIR(program)
        pipeline.run(ir, compilation_results, compiler_config)
        metrics.record_metric(
            MetricsType.OptimizedCircuit,
            compilation_results.lookup_by_type(InputOptimisationResult).optimised_circuit,
        )
        return ir.value, metrics

    def execute(
        self,
        instructions: InstructionBuilder,
        compiler_config: Optional[CompilerConfig] = None,
        pipeline: PassManager | str = "default",
    ):
        if isinstance(pipeline, str):
            pipeline = self._execute_pipelines[pipeline]

        compiler_config = compiler_config or CompilerConfig()
        compilation_results = ResultManager()
        pipeline.run(instructions, compilation_results, compiler_config)

        frontend = QASMFrontend()
        return frontend.execute(
            instructions, pipeline.passes[0]._pass.hardware, compiler_config
        )

    @staticmethod
    def build_default_compile_pipeline(hardware_model):
        """This will move elsewhere"""
        pipeline = PassManager()
        return (
            pipeline
            | InputAnalysis()
            | InputOptimisation(hardware_model)
            | Parse(hardware_model)
        )

    @staticmethod
    def build_default_execute_pipeline(hardware_model):
        """This will move elsewhere"""
        pipeline = PassManager()
        return pipeline | ExecutionPass(hardware_model)


class ExecutionPass(TransformPass):
    """
    This is temporary hack that just holds a hardware model
    """

    def __init__(self, hardware: QuantumHardwareModel):
        self.hardware = hardware

    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        pass
