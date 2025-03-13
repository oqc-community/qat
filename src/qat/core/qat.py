# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from typing import Optional

from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.pipeline import HardwareLoaders, Pipeline, PipelineSet
from qat.core.result_base import ResultManager
from qat.pipelines import get_default_pipelines
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.qatconfig import QatConfig
from qat.runtime.executables import Executable


class QAT:
    def __init__(self, qatconfig: Optional[QatConfig | str] = None):
        if isinstance(qatconfig, str):
            qatconfig = QatConfig.from_yaml(qatconfig)

        self.config = qatconfig or QatConfig()
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
    ):
        P = self.pipelines.get(pipeline)

        # TODO: Improve metrics and config handling
        compiler_config = compiler_config or CompilerConfig()
        metrics_manager = MetricsManager(compiler_config.metrics)
        compilation_results = ResultManager()

        ir = P.frontend.emit(
            program, compilation_results, metrics_manager, compiler_config=compiler_config
        )

        ir = P.middleend.emit(
            ir, compilation_results, metrics_manager, compiler_config=compiler_config
        )

        package = P.backend.emit(
            ir, compilation_results, metrics_manager, compiler_config=compiler_config
        )

        return package, metrics_manager

    def execute(
        self,
        package: InstructionBuilder | Executable,
        compiler_config: CompilerConfig | None = None,
        pipeline: Pipeline | str = "default",
    ):

        P = self.pipelines.get(pipeline)

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
        P = self.pipelines.get(pipeline)
        pkg, _ = self.compile(program, compiler_config=compiler_config, pipeline=P)
        return self.execute(pkg, compiler_config=compiler_config, pipeline=P)
