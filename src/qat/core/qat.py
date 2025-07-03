# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from pathlib import Path
from typing import Optional

from compiler_config.config import CompilerConfig

from qat.core.config.configure import get_qatconfig, override_config
from qat.core.config.session import QatSessionConfig
from qat.core.config.validators import MismatchingHardwareModelException
from qat.core.metrics_base import MetricsManager
from qat.core.pipeline import HardwareLoaders, PipelineSet
from qat.core.result_base import ResultManager
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
    ):
        with override_config(self.config):
            P = self.pipelines.get(pipeline)

            # TODO: Improve metrics and config handling
            compiler_config = compiler_config or CompilerConfig()
            metrics_manager = MetricsManager(compiler_config.metrics)
            compilation_results = ResultManager()

            ir = P.frontend.emit(
                program,
                compilation_results,
                metrics_manager,
                compiler_config=compiler_config,
            )

            ir = P.middleend.emit(
                ir,
                compilation_results,
                metrics_manager,
                compiler_config=compiler_config,
            )

            package = P.backend.emit(
                ir,
                compilation_results,
                metrics_manager,
                compiler_config=compiler_config,
            )

            if to_json:
                package = package.serialize()

            return package, metrics_manager

    def execute(
        self,
        package: InstructionBuilder | Executable | str,
        compiler_config: CompilerConfig | None = None,
        pipeline: Pipeline | str = "default",
    ):
        with override_config(self.config):
            P = self.pipelines.get(pipeline)
            if isinstance(package, str):
                # If the package is a string, deserialize it
                if "py/object" in package:
                    package = InstructionBuilder.deserialize(package)
                else:
                    package = Executable.deserialize(package)
            if P.model.calibration_id != package.calibration_id:
                raise MismatchingHardwareModelException(
                    f"Hardware id in the executable package '{P.model.calibration_id}'' does not match the hardware id '{package.calibration_id}' used during compilation."
                )

            compiler_config = compiler_config or CompilerConfig()
            pp_results = ResultManager()
            metrics_manager = MetricsManager(compiler_config.metrics)

            results = P.runtime.execute(
                package,
                res_mgr=pp_results,
                met_mgr=metrics_manager,
                compiler_config=compiler_config,
            )
            return results, metrics_manager

    def run(
        self,
        program,
        compiler_config: CompilerConfig | None = None,
        pipeline: Pipeline | str = "default",
    ):
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
