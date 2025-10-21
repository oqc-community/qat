# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import abc
from collections import defaultdict
from typing import Dict, Generic, Optional, Tuple, TypeVar

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.executables import Executable
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.hardware_models import QuantumHardwareModel

EXE = TypeVar("EXE", bound=Executable)


class BaseBackend(Generic[EXE], abc.ABC):
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
        self.allocations: Dict[int, Dict[str, int]] = defaultdict(dict)

    @abc.abstractmethod
    def emit(
        self,
        ir: InstructionBuilder,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        **kwargs,
    ) -> EXE: ...

    def allocate(self, target: PulseChannel) -> Tuple[int, int]:
        slot_idx = target.physical_channel.slot_idx
        target2seq = self.allocations[slot_idx]
        if (seq_idx := target2seq.get(target.full_id(), None)) is None:
            total = set(target.physical_channel.config.sequencers.keys())
            available = total - set(target2seq.values())
            if not available:
                raise ValueError(f"""
                No more available sequencers for {target}
                Physical channel: {target.physical_channel}
                Total available: {total}
                Current allocations: {target2seq}
                """)
            seq_idx = next(iter(available))
            target2seq[target.full_id()] = seq_idx
        return seq_idx, slot_idx
