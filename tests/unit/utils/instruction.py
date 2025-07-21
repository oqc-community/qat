# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import Instruction as PydInstruction
from qat.ir.waveforms import Pulse
from qat.purr.compiler.instructions import Instruction as LegInstruction


def count_number_of_non_sync_non_phase_reset_non_delay_non_post_processing_instructions(
    instructions: list[PydInstruction | LegInstruction],
):
    """
    Counts the number of non-sync, non-phase reset, non-delay, and non-post processing instructions in a list of
    instructions.
    This is mainly used for comparison between legacy and pydantic instruction builders where we allow a choice
    between syncs between all pulse channels within a qubit and a syncs between all pulse channels of all qubits
    simultaneously.
    Also, in contrast to legacy instructions, we only allow the number of quantum targets in a
    phase reset to be <= 1 in pydantic.
    Further, the way that delay and post-processing instructions are handled in the measurement methods have changed
    to reduce the number of passes.
    As such, there is a discrepancy in the number of instructions between the
    legacy and experimental code.
    """
    n_instr_no_sync = 0
    for instr in instructions:
        if (
            instr.__class__.__name__ != "Synchronize"
            and instr.__class__.__name__ != "PhaseReset"
            and instr.__class__.__name__ != "Delay"
            and instr.__class__.__name__ != "PostProcessing"
        ):
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
