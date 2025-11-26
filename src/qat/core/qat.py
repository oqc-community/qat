# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from pathlib import Path

from compiler_config.config import CompilerConfig

from qat.core.config.configure import get_qatconfig, override_config
from qat.core.config.session import QatSessionConfig
from qat.core.metrics_base import MetricsManager
from qat.core.pipeline import EngineSet, HardwareLoaders, PipelineManager
from qat.executables import Executable
from qat.pipelines import get_default_pipelines
from qat.pipelines.base import AbstractPipeline
from qat.pipelines.pipeline import Pipeline
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.qatconfig import QatConfig


class QAT:
    def __init__(self, qatconfig: QatSessionConfig | QatConfig | str | Path | None = None):
        if isinstance(qatconfig, (str, Path)):
            qatconfig = QatSessionConfig.from_yaml(qatconfig)

        if type(qatconfig) is QatConfig:
            qatconfig = QatSessionConfig(**qatconfig.model_dump(exclude_defaults=True))

        self.config = qatconfig or QatSessionConfig(
            **get_qatconfig().model_dump(exclude_defaults=True)
        )

        self._available_hardware = HardwareLoaders.from_descriptions(self.config.HARDWARE)
        self._engines = EngineSet.from_descriptions(
            self.config.ENGINES, self._available_hardware
        )

        if (
            self.config.PIPELINES is None
            and self.config.COMPILE is None
            and self.config.EXECUTE is None
        ):
            full_pipelines, compile_pipelines, execute_pipelines = get_default_pipelines()
        else:
            full_pipelines = self.config.PIPELINES
            compile_pipelines = self.config.COMPILE
            execute_pipelines = self.config.EXECUTE

        self.pipelines = PipelineManager.from_descriptions(
            compile_pipelines=compile_pipelines,
            execute_pipelines=execute_pipelines,
            full_pipelines=full_pipelines,
            available_hardware=self._available_hardware,
            available_engines=self._engines,
        )

    def compile(
        self,
        program,
        compiler_config: CompilerConfig | None = None,
        pipeline: Pipeline | str = "default",
        to_json: bool = False,
        **kwargs,
    ) -> tuple[InstructionBuilder | Executable | str, MetricsManager]:
        """Compiles a source program into an executable using the specified pipeline.

        :param program: The source program to compile.
        :param compiler_config: Configuration options for the compiler, such as optimization
            and results formatting.
        :param pipeline: The pipeline to use for compilation. Defaults to "default".
        :param to_json: If True, the output package will be serialized to JSON format.
        :return: A tuple containing the executable of the compiled program for the target
            device and the metrics manager containing metrics collected during compilation.
        """

        with override_config(self.config):
            P = self.pipelines.get_compile_pipeline(pipeline)
            package, metrics_manager = P.compile(
                program, compiler_config=compiler_config, **kwargs
            )
            package = package.serialize() if to_json else package
            return package, metrics_manager

    def execute(
        self,
        package: InstructionBuilder | Executable | str,
        compiler_config: CompilerConfig | None = None,
        pipeline: Pipeline | str = "default",
        **kwargs,
    ) -> tuple[dict, MetricsManager]:
        """Executes a compiled package on the specified pipeline.

        :param package: The compiled package to execute, which can be provided as a JSON
            blob.
        :param compiler_config: Configuration options for the compiler, such as optimization
            and results formatting.
        :param pipeline: The pipeline to use for execution. Defaults to "default".
        :return: A tuple containing the results of the execution and the metrics manager
        """

        with override_config(self.config):
            P = self.pipelines.get_execute_pipeline(pipeline)
            if isinstance(package, str):
                # If the package is a string, deserialize it
                if "py/object" in package:
                    package = InstructionBuilder.deserialize(package)
                else:
                    package = Executable.deserialize(package)
            return P.execute(package, compiler_config=compiler_config, **kwargs)

    def run(
        self,
        program,
        compiler_config: CompilerConfig | None = None,
        pipeline: AbstractPipeline | str = "default",
        compile_pipeline: AbstractPipeline | str | None = None,
        execute_pipeline: AbstractPipeline | str | None = None,
        **kwargs,
    ) -> tuple[dict, MetricsManager]:
        """Compiles and executes a source program using the specified pipeline.

        The compilation and execution pipeline can be specified separately using
        `compile_pipeline` and `execute_pipeline` which take precedence . Alternatively,
        a unified pipeline can be provided using the `pipeline` parameter. If neither are
        provided, the default pipeline will be used for both compilation and execution.

        :param program: The source program to compile and execute.
        :param compiler_config: Configuration options for the compiler, such as optimization
            and results formatting.
        :param pipeline: The pipeline to use for compilation and execution. Defaults to
            "default".
        :param compile_pipeline: The pipeline to use for compilation. If provided, it will
            override the `pipeline` parameter for compilation.
        :param execute_pipeline: The pipeline to use for execution. If provided, it will
            override the `pipeline` parameter for execution.
        :return: A tuple containing the results of the execution and the metrics manager
        """

        compile_pipeline = compile_pipeline if compile_pipeline is not None else pipeline
        execute_pipeline = execute_pipeline if execute_pipeline is not None else pipeline

        pkg, compile_metrics = self.compile(
            program,
            compiler_config=compiler_config,
            pipeline=compile_pipeline,
            **kwargs,
        )
        result, execute_metrics = self.execute(
            pkg,
            compiler_config=compiler_config,
            pipeline=execute_pipeline,
            **kwargs,
        )
        return result, execute_metrics.merge(compile_metrics)

    def reload_all_models(self):
        """Reloads all hardware models and updates the pipelines."""
        self._available_hardware.reload_all_models()
        self._engines.reload_all_models()
        self.pipelines.reload_all_models()

    @property
    def models_up_to_date(self) -> bool:
        """Checks if all hardware models are up-to-date.

        :return: True if all models are up-to-date, False otherwise.
        """
        return self._available_hardware.models_up_to_date
