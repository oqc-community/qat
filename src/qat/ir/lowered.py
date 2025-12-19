# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from dataclasses import dataclass, field

from qat.ir.instructions import Assign, Instruction, ResultsProcessing, Return
from qat.ir.measure import Acquire, PostProcessing
from qat.ir.pulse_channel import PulseChannel


@dataclass
class PartitionedIR:
    """Contains the IR that has been processed and reduced to a program per pulse channel.
    Also splits out any post-processing instructions to be used by the executor.

    Programs are typically sent to hardware as a package per physical / logical channel,
    and need to be partitioned out. They also need to be free of instructions that are not
    used at runtime. This class does this.

    This will eventually be replaced by some "partitoned module" approach, which might have
    a similar schema, or might just be many modules.
    """

    target_map: dict[str, list[Instruction]] = field(
        default_factory=lambda: defaultdict(list)
    )
    pulse_channels: dict[str, PulseChannel] = field(default_factory=dict)
    shots: int | None = field(default=None)
    compiled_shots: int | None = field(default=None)
    returns: list[Return] = field(default_factory=list)
    assigns: list[Assign] = field(default_factory=list)
    acquire_map: dict[PulseChannel | str, list[Acquire]] = field(
        default_factory=lambda: defaultdict(list)
    )
    pp_map: dict[str, list[PostProcessing]] = field(
        default_factory=lambda: defaultdict(list)
    )
    rp_map: dict[str, ResultsProcessing] = field(default_factory=dict)

    def get_pulse_channel(self, id: str) -> PulseChannel | None:
        """Get a pulse channel by its ID.

        Maintains API with the QuantumInstructionBuilder."""
        return self.pulse_channels.get(id, None)
