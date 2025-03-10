# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from typing import Optional

from compiler_config.config import CompilerConfig

from qat.passes.metrics_base import MetricsManager
from qat.passes.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.qat import QATInput


class BaseFrontend(ABC):
    """A frontend is the front-facing part of QAT pipelines that accepts some source
    program and handles compilation down to QAT's intermediate representation.

    Generally, frontends are picked to match the source language. For example, a QIR source
    program should be coupled with the :class:`QIRFrontend`. They implement compilation
    details that are specific to the source language via a pipeline, and parse it down to
    QAT IR.
    """

    def __init__(self, model: None | QuantumHardwareModel = None):
        self.model = model

    @abstractmethod
    def emit(
        self,
        src: QATInput,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
    ) -> InstructionBuilder: ...

    @abstractmethod
    def check_and_return_source(self, src): ...
