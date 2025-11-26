# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod

from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.qat import QATInput
from qat.utils.hardware_model import check_type_legacy_or_pydantic


class BaseFrontend(ABC):
    """
    Base class for frontend that scans a high-level language-specific, but target-agnostic, input
    :class:`QatInput` and verifies its syntax and semantics according to a specific source language
    (QASM, QIR, ...). The input is optimised so as to adhere to the underlying topology of the QPU
    and is then compiled to a target-agnostic intermediate representation (IR) :class:`QatIR` that
    can be further optimised in the middle end.

    Generally, frontends are picked to match the source language. For example, a QIR source
    program should be coupled with the :class:`QIRFrontend`. They implement compilation
    details that are specific to the source language via a pipeline, and parse it down to
    QAT IR.
    """

    def __init__(self, model: None | QuantumHardwareModel | PydHardwareModel = None):
        """
        :param model: The hardware model that holds calibrated information on the qubits on the QPU.
        """
        self.model = check_type_legacy_or_pydantic(model)

    @abstractmethod
    def emit(
        self,
        src: QATInput,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ) -> InstructionBuilder:
        """
        Compiles an input :class:`QatInput` down to :class:`QatIR` and emits it.
        :param src: The high-level input.
        :param res_mgr: Collection of analysis results with caching and aggregation
                        capabilities, defaults to None.
        :param met_mgr: Stores useful intermediary metrics that are generated during
                        compilation, defaults to None.
        :param compiler_config: Compiler settings, defaults to None.
        :return: An intermediate representation as an :class:`InstructionBuilder` which
                holds a list of instructions to be executed on the QPU.
        """
        ...

    @abstractmethod
    def check_and_return_source(self, src): ...

    @staticmethod
    def _check_metrics_and_config(res_mgr, met_mgr, compiler_config):
        if res_mgr is None:
            res_mgr = ResultManager()
        if met_mgr is None:
            met_mgr = MetricsManager()
        if compiler_config is None:
            compiler_config = CompilerConfig()

        return res_mgr, met_mgr, compiler_config
