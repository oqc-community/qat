# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from typing import Optional, Union

from compiler_config.config import CompilerConfig

from qat.backend.waveform_v1 import EchoEngine, WaveformV1Emitter
from qat.ir.metrics_base import MetricsManager
from qat.ir.pass_base import PassManager, QatIR
from qat.ir.result_base import ResultManager
from qat.pipelines import DefaultRuntime
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.qatconfig import QatConfig
from qat.qat import QATInput
from qat.runtime import BaseRuntime, NativeEngine
from qat.runtime.executables import Executable


class QAT:
    def __init__(self, qatconfig: Optional[QatConfig | str] = None):

        self._compile_pipelines = {}
        self._execute_pipelines = {}
        self._postprocess_pipelines = {}
        self._engines = {}
        self._runtimes = {}
        self._emitters = {}
        self._default_pipeline = None

        if isinstance(qatconfig, str):
            qatconfig = QatConfig.from_yaml(qatconfig)

        self.config = qatconfig or QatConfig()
        self._populate_pipelines()

    def _populate_pipelines(self):
        default = None
        for pipe in self.config.PIPELINES:
            # This to move to some sort of "HardwareLoader"
            emitter = None
            match pipe.hardware.hardware_type:
                case "rtcs":
                    hw = get_default_RTCS_hardware()
                    engine = hw.create_engine()
                case "echo":
                    qubit_count = pipe.hardware.qubit_count
                    hw = get_default_echo_hardware(qubit_count=qubit_count)
                    engine = EchoEngine()
                    emitter = WaveformV1Emitter(hw)
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
                emitter=emitter,
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
        engine: Union[QuantumExecutionEngine, NativeEngine],
        runtime: Optional[BaseRuntime] = None,
        emitter=None,
    ):
        self._compile_pipelines[name] = compile_pipeline
        self._execute_pipelines[name] = execute_pipeline
        self._postprocess_pipelines[name] = postprocess_pipeline
        self._engines[name] = engine
        self._runtimes[name] = runtime or DefaultRuntime(engine)
        self._emitters[name] = emitter

    def remove_pipeline(self, name: str):
        del self._compile_pipelines[name]
        del self._execute_pipelines[name]
        del self._postprocess_pipelines[name]
        del self._engines[name]
        del self._runtimes[name]
        del self._emitters[name]

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

    def _get_runtime(self, runtime: BaseRuntime | None, pipeline: PassManager | str):
        """
        Finds and returns the runtime.

        For now, the runtime can be "None", which should only be used for legacy engines,
        which handle some of the post-processing responsibilities.
        """

        if runtime is None:
            if isinstance(pipeline, str):
                runtime = pipeline

        if isinstance(runtime, str):
            if runtime == "default":
                if self._default_pipeline is None:
                    raise Exception(f"No Default Pipeline Set")
                runtime = self._runtimes.get(self._default_pipeline, None)
                if runtime is None:
                    raise Exception(
                        f"Default Pipeline set to unknown pipeline {self._default_pipeline}"
                    )
            elif runtime not in self._runtimes:
                raise Exception(f"Runtime for {runtime} is not found")
            else:
                runtime = self._runtimes[runtime]

        return runtime

    def _get_emitter(self, emitter, pipeline: PassManager | str):
        """
        Finds and returns the emitter

        For now, the emitter can be "None", which should only be used for legacy engines which
        handle the code gen responsibilities.
        """

        if emitter is None:
            if isinstance(pipeline, str):
                emitter = pipeline

        if isinstance(emitter, str):
            if emitter == "default":
                if self._default_pipeline is None:
                    raise Exception(f"No Default Pipeline Set")
                emitter = self._emitters.get(self._default_pipeline, None)
                if emitter is None:
                    raise Exception(
                        f"Default Pipeline set to unknown pipeline {self._default_pipeline}"
                    )
            elif emitter not in self._emitters:
                raise Exception(f"Emitter for {emitter} is not found")
            else:
                emitter = self._emitters[emitter]
        return emitter

    def compile(
        self,
        program: QATInput,
        compiler_config: Optional[CompilerConfig] = None,
        pipeline: PassManager | str = "default",
        emitter=None,
    ):

        compilation_pipeline = self._get_pipeline(self._compile_pipelines, pipeline)

        # TODO: Improve metrics and config handling
        compiler_config = compiler_config or CompilerConfig()
        metrics_manager = MetricsManager(compiler_config.metrics)
        compilation_results = ResultManager()
        ir = QatIR(program)
        compilation_pipeline.run(
            ir, compilation_results, metrics_manager, compiler_config=compiler_config
        )

        # TODO: adopt pass-based emitter
        if emitter := self._get_emitter(emitter, pipeline):
            package = emitter.emit(ir)
        else:
            package = ir.value

        return package, metrics_manager

    def execute(
        self,
        package: InstructionBuilder | Executable,
        compiler_config: Optional[CompilerConfig] = None,
        pipeline: PassManager | str = "default",
        execute_pipeline: PassManager | None = None,
        postprocess_pipeline: PassManager | None = None,
        engine: QuantumExecutionEngine | NativeEngine | None = None,
        runtime: BaseRuntime = None,
    ):

        execute_pipeline = execute_pipeline or pipeline
        postprocess_pipeline = postprocess_pipeline or pipeline

        execute_pipeline = self._get_pipeline(self._execute_pipelines, execute_pipeline)
        postprocess_pipeline = self._get_pipeline(
            self._postprocess_pipelines, postprocess_pipeline
        )

        engine = self._get_engine(engine, pipeline)
        runtime = self._get_runtime(runtime, pipeline)

        compiler_config = compiler_config or CompilerConfig()
        execution_results = ResultManager()
        pp_results = ResultManager()
        metrics_manager = MetricsManager(compiler_config.metrics)

        if isinstance(package, InstructionBuilder):
            package = QatIR(package)

        execute_pipeline.run(
            package,
            execution_results,
            metrics_manager,
            compiler_config=compiler_config,
        )

        with runtime(engine, postprocess_pipeline) as runtime_instance:
            results = runtime_instance.execute(
                package,
                res_mgr=pp_results,
                met_mgr=metrics_manager,
                compiler_config=compiler_config,
            )
        return results, metrics_manager
