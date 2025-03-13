# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import abc
from typing import Optional

from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.executables import Executable


class BaseBackend(abc.ABC):
    """
    Converts an intermediate representation (IR) to code for a given target
    by selecting target-machine operations to implement for each instruction
    in the IR.
    """

    def __init__(self, model: None | QuantumHardwareModel):
        """
        :param model: The hardware model that holds calibrated information on the qubits on the QPU.
        """
        self.model = model

    @abc.abstractmethod
    def emit(
        self,
        ir: InstructionBuilder,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
    ) -> Executable: ...
