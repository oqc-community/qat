# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.core.pass_base import TransformPass
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, PulseChannelView
from qat.purr.compiler.instructions import DeviceUpdate, FrequencySet


class FrequencyAssignSanitisation(TransformPass):
    """Sanitises device assignments that changes the frequency of a channel, by replacing
    it with a FrequencySet instruction.

    If this is done for targets that can support frequency assignment as an IR instruction,
    (i.e. has support for :class:`FrequencySet`), then this pass can be used to avoid
    mutating the hardware model, which can be unsafe when used within pipelines.
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        instructions = []
        for instruction in ir._instructions:
            if (
                isinstance(instruction, DeviceUpdate)
                and isinstance(instruction.target, (PulseChannel, PulseChannelView))
                and instruction.attribute == "frequency"
            ):
                instructions.append(
                    FrequencySet(channel=instruction.target, frequency=instruction.value)
                )
            else:
                instructions.append(instruction)
        ir._instructions = instructions
        return ir
