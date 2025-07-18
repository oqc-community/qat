# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from pathlib import Path
from typing import Optional

from compiler_config.config import CompilerConfig

from qat.core.config.configure import get_qatconfig, override_config
from qat.core.config.session import QatSessionConfig
from qat.core.metrics_base import MetricsManager
from qat.core.pipeline import HardwareLoaders, PipelineSet
from qat.pipelines import get_default_pipelines
from qat.pipelines.pipeline import Pipeline
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.qatconfig import QatConfig
from qat.runtime.executables import Executable


class QAT:
    def __init__(
        self, qatconfig: Optional[QatSessionConfig | QatConfig | str | Path] = None
    ):
        if isinstance(qatconfig, (str, Path)):
            qatconfig = QatSessionConfig.from_yaml(qatconfig)

        if type(qatconfig) is QatConfig:
            qatconfig = QatSessionConfig(**qatconfig.model_dump(exclude_defaults=True))

        self.config = qatconfig or QatSessionConfig(
            **get_qatconfig().model_dump(exclude_defaults=True)
        )

        self._available_hardware = HardwareLoaders.from_descriptions(self.config.HARDWARE)
        self.pipelines = PipelineSet.from_descriptions(
            self.config.PIPELINES or get_default_pipelines(),
            available_hardware=self._available_hardware,
        )

    def compile(
        self,
        program,
        compiler_config: CompilerConfig | None = None,
        pipeline: Pipeline | str = "default",
        to_json: bool = False,
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
            P = self.pipelines.get(pipeline)
            package, metrics_manager = P.compile(program, compiler_config=compiler_config)
            package = package.serialize() if to_json else package
            return package, metrics_manager

    def execute(
        self,
        package: InstructionBuilder | Executable | str,
        compiler_config: CompilerConfig | None = None,
        pipeline: Pipeline | str = "default",
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
            P = self.pipelines.get(pipeline)
            if isinstance(package, str):
                # If the package is a string, deserialize it
                if "py/object" in package:
                    package = InstructionBuilder.deserialize(package)
                else:
                    package = Executable.deserialize(package)
            return P.execute(package, compiler_config=compiler_config)

    def run(
        self,
        program,
        compiler_config: CompilerConfig | None = None,
        pipeline: Pipeline | str = "default",
    ) -> tuple[dict, MetricsManager]:
        """Compiles and executes a source program using the specified pipeline.

        :param program: The source program to compile and execute.
        :param compiler_config: Configuration options for the compiler, such as optimization
            and results formatting.
        :param pipeline: The pipeline to use for compilation and execution. Defaults to
            "default".
        :return: A tuple containing the results of the execution and the metrics manager
        """

        with override_config(self.config):
            P = self.pipelines.get(pipeline)
            pkg, compile_metrics = self.compile(
                program, compiler_config=compiler_config, pipeline=P
            )
            result, execute_metrics = self.execute(
                pkg, compiler_config=compiler_config, pipeline=P
            )
            return result, execute_metrics.merge(compile_metrics)

    def reload_all_models(self):
        """Reloads all hardware models and updates the pipelines."""
        self._available_hardware.reload_all_models()
        self.pipelines.reload_all_models()
