# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from typing import Optional

from compiler_config.config import CompilerConfig

from qat.backend.base import BaseBackend
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder


class FallthroughBackend(BaseBackend):
    """
    A backend that passes through an IR :class:`InstructionBuilder` and does not alter it.
    """

    def __init__(self, model: None = None):
        """
        :param model: The hardware model that holds calibrated information on the qubits on the QPU.
        """
        self.model = model

    def emit(
        self,
        ir: InstructionBuilder,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
    ):
        return ir
