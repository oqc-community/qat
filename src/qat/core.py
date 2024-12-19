# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from typing import Optional

from compiler_config.config import CompilerConfig, MetricsType

from qat.compiler.transform_passes import InputOptimisationResult
from qat.ir.pass_base import PassManager, QatIR
from qat.ir.result_base import ResultManager
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.compiler.hardware_models import get_cl2qu_index_mapping
from qat.purr.compiler.metrics import CompilationMetrics
from qat.purr.compiler.runtime import get_runtime
from qat.purr.qatconfig import QatConfig
from qat.qat import QATInput
from qat.runtime.analysis_passes import CalibrationAnalysisResult


class QAT:
    def __init__(self, qatconfig: Optional[QatConfig | str] = None):

        self._compile_pipelines = {}
        self._execute_pipelines = {}
        self._postprocess_pipelines = {}
        self._engines = {}
        self._default_pipeline = None

        if isinstance(qatconfig, str):
            qatconfig = QatConfig.from_yaml(qatconfig)

        self.config = qatconfig or QatConfig()
        self._populate_pipelines()

    def _populate_pipelines(self):
        default = None
        for pipe in self.config.PIPELINES:

            # This to move to some sort of "HardwareLoader"
            match pipe.hardware.hardware_type:
                case "rtcs":
                    hw = get_default_RTCS_hardware()
                case "echo":
                    qubit_count = pipe.hardware.qubit_count
                    hw = get_default_echo_hardware(qubit_count=qubit_count)
                case "qiskit":
                    qubit_count = pipe.hardware.qubit_count
                    hw = get_default_qiskit_hardware(qubit_count=qubit_count)

            engine = hw.create_engine()
            self.add_pipeline(
                pipe.name,
                pipe.compile(hw),
                pipe.execute(hw, engine),
                pipe.postprocess(hw),
                engine=engine,
            )

            if pipe.default:
                default = pipe.name

        if len(self.config.PIPELINES) > 0:
            self.set_default_pipeline(default)

    def set_default_pipeline(self, pipeline_name: str):
        self._default_pipeline = pipeline_name

    def add_pipeline(
        self,
        name: str,
        compile_pipeline: PassManager,
        execute_pipeline: PassManager,
        postprocess_pipeline: PassManager,
        engine: QuantumExecutionEngine,
    ):
        self._compile_pipelines[name] = compile_pipeline
        self._execute_pipelines[name] = execute_pipeline
        self._postprocess_pipelines[name] = postprocess_pipeline
        self._engines[name] = engine

    def remove_pipeline(self, name: str):
        del self._compile_pipelines[name]
        del self._execute_pipelines[name]
        del self._postprocess_pipelines[name]
        del self._engines[name]

    def _get_pipeline(self, pipeline_dict, pipeline: PassManager | str):
        """Gets a pipeline for a dict of pipelines handling defaults and exceptions"""
        if isinstance(pipeline, str):
            if pipeline == "default":
                if self._default_pipeline is None:
                    raise Exception(f"No Default Pipeline Set")
                pipeline = pipeline_dict.get(self._default_pipeline, None)
                if pipeline is None:
                    raise Exception(
                        f"Default Pipeline set to unknown pipeline {self._default_pipeline}"
                    )
            else:
                if not pipeline in pipeline_dict:
                    raise Exception(f"Pipeline {pipeline} not found")
                pipeline = pipeline_dict[pipeline]

        return pipeline

    def _get_engine(
        self, engine: QuantumExecutionEngine | None, pipeline: PassManager | str
    ):
        """Find and return the engine"""
        if engine is None:
            if isinstance(pipeline, str):
                engine = pipeline
            else:
                raise Exception(
                    "Engine could not be identified automatically, please provide explicitly"
                )

        if isinstance(engine, str):
            if engine == "default":
                if self._default_pipeline is None:
                    raise Exception(f"No Default Pipeline Set")
                engine = self._engines.get(self._default_pipeline, None)
                if engine is None:
                    raise Exception(
                        f"Default Pipeline set to unknown pipeline {self._default_pipeline}"
                    )
            elif engine not in self._engines:
                raise Exception(f"Engine {engine} not found")
            else:
                engine = self._engines[engine]

        return engine

    def compile(
        self,
        program: QATInput,
        compiler_config: Optional[CompilerConfig] = None,
        pipeline: PassManager | str = "default",
    ):

        pipeline = self._get_pipeline(self._compile_pipelines, pipeline)

        # TODO: Improve metrics and config handling
        compiler_config = compiler_config or CompilerConfig()
        metrics = CompilationMetrics()
        metrics.enable(compiler_config.metrics)
        compilation_results = ResultManager()
        ir = QatIR(program)
        pipeline.run(ir, compilation_results, compiler_config=compiler_config)
        metrics.record_metric(
            MetricsType.OptimizedCircuit,
            compilation_results.lookup_by_type(InputOptimisationResult).optimised_circuit,
        )
        return ir.value, metrics

    def execute(
        self,
        builder: InstructionBuilder,
        compiler_config: Optional[CompilerConfig] = None,
        pipeline: PassManager | str = "default",
        execute_pipeline: PassManager | None = None,
        postprocess_pipeline: PassManager | None = None,
        engine: QuantumExecutionEngine | None = None,
    ):

        execute_pipeline = execute_pipeline or pipeline
        postprocess_pipeline = postprocess_pipeline or pipeline

        execute_pipeline = self._get_pipeline(self._execute_pipelines, execute_pipeline)
        postprocess_pipeline = self._get_pipeline(
            self._postprocess_pipelines, postprocess_pipeline
        )

        compiler_config = compiler_config or CompilerConfig()
        execution_results = ResultManager()

        metrics = CompilationMetrics()
        metrics.enable(compiler_config.metrics)

        ir = QatIR(builder)
        execute_pipeline.run(
            ir,
            execution_results,
            compiler_config=compiler_config,
        )

        engine = self._get_engine(engine, pipeline)

        # TODO: Improve calibration handling
        # What is this?
        calibrations = execution_results.lookup_by_type(
            CalibrationAnalysisResult
        ).calibration_executables
        active_runtime = get_runtime(engine.model)
        active_runtime.run_quantum_executable(calibrations)

        metrics.record_metric(
            MetricsType.OptimizedInstructionCount, len(builder.instructions)
        )

        results = engine.execute(
            builder.instructions,
        )

        # TODO: Should this be a pass in a pre-execution Pipeline? (Yes!)
        try:
            index_mapping = get_cl2qu_index_mapping(builder.instructions, engine.model)
        except ValueError:
            index_mapping = {}

        # Result processing pipeline
        ir = QatIR(value=results)
        postprocess_pipeline.run(
            ir,
            execution_results,
            compiler_config=compiler_config,
            mapping=index_mapping,
        )
        return ir.value, metrics
