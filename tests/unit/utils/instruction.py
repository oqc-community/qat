# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import Instruction as PydInstruction
from qat.ir.waveforms import Pulse
from qat.purr.compiler.instructions import Instruction as LegInstruction


def count_number_of_non_sync_instructions(
    instructions: list[PydInstruction | LegInstruction],
):
    """
    Counts the number of non-sync instructions in a list of instructions. This is mainly used
    for comparison between legacy and pydantic instruction builders where we allow a choice between
    syncs between all pulse channels within a qubit and a syncs between all pulse channels of all qubits
    simultaneously. As such, there is a discrepancy in the number of syncs between the legacy and experimental
    code.
    """
    n_instr_no_sync = 0
    for instr in instructions:
        if instr.__class__.__name__ != "Synchronize":
            n_instr_no_sync += 1

    return n_instr_no_sync


def count_number_of_pulses(builder: QuantumInstructionBuilder, pulse_type: str = "measure"):
    hw = builder.hw

    n_pulse = 0
    for instr in builder._ir:
        if (
            isinstance(instr, Pulse)
            and pulse_type.lower() == hw.pulse_channel_with_id(instr.target).pulse_type
        ):
            n_pulse += 1
    return n_pulse
