# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import abc
from collections import defaultdict
from typing import Generic

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.executables import Executable, Program
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class BaseBackend(Generic[Program], abc.ABC):
    """
    Base class for a backend that takes an intermediate representation (IR) :class:`QatIR`
    and lowers it to machine code that can be executed on a given target.
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
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        **kwargs,
    ) -> Executable[Program]:
        """
        Converts an IR :class:`QatIR` to machine instructions of a given target
        architecture.

        How targets convert the IR is at their discretion but they mostly follow
        macro-expansion techniques where target instructions are selected
        for each instruction in the IR.
        """
        ...


class CustomBackend(BaseBackend[Program], Generic[Program], abc.ABC):
    """
    Backends may need to run pre-codegen passes on the IR :class:`QatIR` as emitted
    from the middle end. These passes are specified via a custom pipeline.
    """

    def __init__(self, model: QuantumHardwareModel, pipeline: PassManager = None):
        super().__init__(model=model)
        self.pipeline = pipeline or PassManager()


class AllocatingBackend(CustomBackend[Program], Generic[Program], abc.ABC):
    """
    A backend that's responsible for allocating FPGA card and sequencers AOT.
    """

    def __init__(self, model: QuantumHardwareModel, pipeline: PassManager = None):
        super().__init__(model=model, pipeline=pipeline)
        self.allocations: dict[int, dict[str, int]] = defaultdict(dict)

    def allocate(self, target: PulseChannel) -> tuple[int, int]:
        """
        For a given target, allocate an FPGA card and a sequencer.
        """

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
