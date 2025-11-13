# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class Opcode(Enum):
    # Control
    NOP = "nop"
    STOP = "stop"

    # Jumps
    ADDRESS = ""
    JUMP = "jmp"
    JUMP_GREATER_EQUALS = "jge"
    JUMP_LESS_THAN = "jlt"
    LOOP = "loop"

    # Arithmetic
    MOVE = "move"
    ADD = "add"
    SUB = "sub"
    NOT = "not"
    AND = "and"
    OR = "or"
    XOR = "xor"

    # Parameter operation
    SET_MARKER = "set_mrk"
    SET_NCO_FREQUENCY = "set_freq"
    SET_NCO_PHASE = "set_ph"
    SET_NCO_PHASE_OFFSET = "set_ph_delta"
    RESET_PHASE = "reset_ph"
    SET_AWG_GAIN = "set_awg_gain"
    SET_AWG_OFFSET = "set_awg_offs"

    # Real-time pipeline
    SET_COND = "set_cond"
    UPDATE_PARAMETERS = "upd_param"
    PLAY = "play"
    ACQUIRE = "acquire"
    ACQUIRE_WEIGHED = "acquire_weighed"
    ACQUIRE_TTL = "acquire_ttl"
    SET_LATCH_EN = "set_latch_en"
    LATCH_RST = "latch_rst"
    WAIT = "wait"
    WAIT_SYNC = "wait_sync"
    WAIT_TRIGGER = "wait_trigger"


class Q1asmInstruction:
    def __init__(self, opcode: Opcode, *operands, comment: str = None):
        self.opcode: Opcode = opcode
        self.operands: tuple[Any, ...] = operands
        self.comment: str = comment

    def __str__(self):
        args = ",".join([str(op) for op in self.operands])

        if self.opcode == Opcode.ADDRESS:
            asm_str = f"{args}:"
        elif args:
            asm_str = f"{self.opcode.value} {args}"
        else:
            asm_str = f"{self.opcode.value}"

        if self.comment:
            asm_str += f" # {self.comment}"

        return asm_str

    def __repr__(self):
        return self.__str__()


@dataclass
class Sequence:
    waveforms: dict[str, dict[str, Any]]
    weights: dict[str, Any]
    acquisitions: dict[str, Any]
    program: str


class SequenceBuilder:
    """
    High-level builder apis for sequence codegen.
    This includes q1asm, waveforms, weights, and acquisitions.
    """

    def __init__(self):
        self.waveforms: dict[str, dict[str, Any]] = {}
        self.acquisitions: dict[str, Any] = {}
        self.weights: dict[str, Any] = {}
        self.q1asm_instructions: list[Q1asmInstruction] = []

    def build(self):
        return Sequence(
            waveforms=self.waveforms,
            acquisitions=self.acquisitions,
            weights=self.weights,
            program="\n".join([str(inst) for inst in self.q1asm_instructions]),
        )

    def lookup_waveform_by_data(self, data: np.ndarray):
        return next(
            (
                wf["index"]
                for wf in self.waveforms.values()
                if np.array(wf["data"]).shape == data.shape
                and np.all(np.array(wf["data"]) == data)
            ),
            None,
        )

    def add_waveform(self, name: str, index: int, data: list[float]):
        if name in self.waveforms:
            raise ValueError(f"A waveform named {name} already exists")
        self.waveforms[name] = {"index": index, "data": data}

    def add_acquisition(self, name: str, index: int, num_bins: int):
        if name in self.acquisitions:
            raise ValueError(f"An acquisition named {name} already exists")
        self.acquisitions[name] = {"index": index, "num_bins": num_bins}

    def add_weight(self, name: str, index: int, data: list[float]):
        if name in self.weights:
            raise ValueError(f"A weight named {name} already exists")
        self.weights[name] = {"index": index, "data": data}

    def _add_instruction(self, opcode: Opcode, *operands, comment: str = None):
        self.q1asm_instructions.append(Q1asmInstruction(opcode, *operands, comment=comment))
        return self

    # Control Instructions
    def nop(self, comment: str = None):
        return self._add_instruction(Opcode.NOP, comment=comment)

    def stop(self, comment: str = None):
        return self._add_instruction(Opcode.STOP, comment=comment)

    # Jump Instructions
    def label(self, label: int | str, comment: str = None):
        return self._add_instruction(Opcode.ADDRESS, label, comment=comment)

    def jmp(self, address: int | str, comment: str = None):
        if isinstance(address, str):
            address = f"@{address}"
        return self._add_instruction(Opcode.JUMP, address, comment=comment)

    def jge(self, a: str, b: int, address: int | str, comment: str = None):
        if isinstance(address, str):
            address = f"@{address}"
        return self._add_instruction(
            Opcode.JUMP_GREATER_EQUALS, a, b, address, comment=comment
        )

    def jlt(self, a: str, b: int, address: int | str, comment: str = None):
        if isinstance(address, str):
            address = f"@{address}"
        return self._add_instruction(Opcode.JUMP_LESS_THAN, a, b, address, comment=comment)

    def loop(self, a: str, address: int | str, comment: str = None):
        if isinstance(address, str):
            address = f"@{address}"
        return self._add_instruction(Opcode.LOOP, a, address, comment=comment)

    # Arithmetic instructions
    def move(self, src: int | str, dest: str, comment: str = None):
        return self._add_instruction(Opcode.MOVE, src, dest, comment=comment)

    def add(self, a: str, b: int | str, dest: str, comment: str = None):
        return self._add_instruction(Opcode.ADD, a, b, dest, comment=comment)

    def sub(self, a: str, b: int | str, dest: str, comment: str = None):
        return self._add_instruction(Opcode.SUB, a, b, dest, comment=comment)

    def logic_not(self, src: int | str, dest: str, comment: str = None):
        return self._add_instruction(Opcode.NOT, src, dest, comment=comment)

    def logic_and(self, a: str, b: int | str, dest: str, comment: str = None):
        return self._add_instruction(Opcode.AND, a, b, dest, comment=comment)

    def logic_or(self, a: str, b: int | str, dest: str, comment: str = None):
        return self._add_instruction(Opcode.OR, a, b, dest, comment=comment)

    def logic_xor(self, a: str, b: int | str, dest: str, comment: str = None):
        return self._add_instruction(Opcode.XOR, a, b, dest, comment=comment)

    # Parameter operation instructions
    def set_mrk(self, mask: int | str, comment: str = None):
        return self._add_instruction(Opcode.SET_MARKER, mask, comment=comment)

    def set_freq(self, frequency: int | str, comment: str = None):
        return self._add_instruction(Opcode.SET_NCO_FREQUENCY, frequency, comment=comment)

    def set_ph(self, phase: int | str, comment: str = None):
        return self._add_instruction(Opcode.SET_NCO_PHASE, phase, comment=comment)

    def set_ph_delta(self, phase_delta: int | str, comment: str = None):
        return self._add_instruction(
            Opcode.SET_NCO_PHASE_OFFSET, phase_delta, comment=comment
        )

    def reset_ph(self, comment: str = None):
        return self._add_instruction(Opcode.RESET_PHASE, comment=comment)

    def set_awg_gain(self, gain_i: int | str, gain_q: int | str, comment: str = None):
        return self._add_instruction(Opcode.SET_AWG_GAIN, gain_i, gain_q, comment=comment)

    def set_awg_offs(self, offset_i: int | str, offset_q: int | str, comment: str = None):
        return self._add_instruction(
            Opcode.SET_AWG_OFFSET, offset_i, offset_q, comment=comment
        )

    # Real-time instructions
    def set_cond(
        self,
        enable: int | str,
        mask: int | str,
        operator: int | str,
        else_duration: int,
        comment: str = None,
    ):
        return self._add_instruction(
            Opcode.SET_COND, enable, mask, operator, else_duration, comment=comment
        )

    def upd_param(self, duration: int, comment: str = None):
        return self._add_instruction(Opcode.UPDATE_PARAMETERS, duration, comment=comment)

    def play(
        self,
        wave_i: int | str,
        wave_q: int | str,
        duration: int,
        comment: str = None,
    ):
        return self._add_instruction(Opcode.PLAY, wave_i, wave_q, duration, comment=comment)

    def acquire(self, acq: int, bin: int | str, duration: int, comment: str = None):
        return self._add_instruction(Opcode.ACQUIRE, acq, bin, duration, comment=comment)

    def acquire_weighed(
        self,
        acq: int,
        bin: int | str,
        weight_i: int | str,
        weight_q: int | str,
        duration: int,
        comment: str = None,
    ):
        return self._add_instruction(
            Opcode.ACQUIRE_WEIGHED, acq, bin, weight_i, weight_q, duration, comment=comment
        )

    def acquire_ttl(
        self,
        acq: int,
        bin: int | str,
        enable: int,
        duration: int,
        comment: str = None,
    ):
        return self._add_instruction(
            Opcode.ACQUIRE_TTL, acq, bin, enable, duration, comment=comment
        )

    def set_latch_en(self, enable: int | str, duration: int, comment: str = None):
        return self._add_instruction(Opcode.SET_LATCH_EN, enable, duration, comment=comment)

    def latch_rst(self, duration: int | str, comment: str = None):
        return self._add_instruction(Opcode.LATCH_RST, duration, comment=comment)

    def wait(self, duration: int | str, comment: str = None):
        return self._add_instruction(Opcode.WAIT, duration, comment=comment)

    def wait_trigger(self, trigger: int | str, duration: int | str, comment: str = None):
        return self._add_instruction(
            Opcode.WAIT_TRIGGER, trigger, duration, comment=comment
        )

    def wait_sync(self, duration: int | str, comment: str = None):
        return self._add_instruction(Opcode.WAIT_SYNC, duration, comment=comment)
