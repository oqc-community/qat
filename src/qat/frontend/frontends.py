# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import abc
from typing import Optional

from compiler_config.config import CompilerConfig

from qat.compiler.analysis_passes import InputAnalysis
from qat.compiler.transform_passes import InputOptimisation, Parse
from qat.passes.metrics_base import MetricsManager
from qat.passes.pass_base import PassManager
from qat.passes.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.qat import QATInput


class BaseFrontend(abc.ABC):
    def __init__(self, model: None | QuantumHardwareModel = None):
        self.model = model

    @abc.abstractmethod
    def emit(
        self,
        src: QATInput,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
    ) -> InstructionBuilder: ...


class FallthroughFrontend(BaseFrontend):
    def __init__(self, model: None = None):
        self.model = model

    def emit(
        self,
        src: InstructionBuilder,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
    ) -> InstructionBuilder:
        return src


class CustomFrontend(BaseFrontend):
    def __init__(
        self, model: None | QuantumHardwareModel, pipeline: None | PassManager = None
    ):
        self.model = model
        self.pipeline = pipeline

    def emit(
        self,
        src: QATInput,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
    ):
        ir = self.pipeline.run(src, res_mgr, met_mgr, compiler_config=compiler_config)
        return ir


class DefaultFrontend(CustomFrontend):
    def __init__(self, model: QuantumHardwareModel):
        pipeline = self.build_pass_pipeline(model)
        super().__init__(model=model, pipeline=pipeline)

    @staticmethod
    def build_pass_pipeline(model) -> PassManager:
        return PassManager() | InputAnalysis() | InputOptimisation(model) | Parse(model)
