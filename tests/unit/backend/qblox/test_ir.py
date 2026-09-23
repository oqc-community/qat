# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.backend.qblox.ir import Opcode, Q1asmInstruction, SequenceBuilder


class TestQ1asmInstruction:
    @pytest.mark.parametrize(
        ("op_code", "operands", "comment"),
        [
            (Opcode.NOP, (), "Stalling"),
            (Opcode.STOP, (), "Stop the sequencer"),
            (Opcode.ADDRESS, ("label",), "Marked address as label"),
            (Opcode.ADD, ("R0", "R1", "R0"), "R0 <- R0 + R1"),
            (Opcode.JUMP, ("label",), "Unconditionally jump to address marked by label"),
            (
                Opcode.JUMP_LESS_THAN,
                (
                    "R0",
                    1000,
                    "label",
                ),
                "If R0 < 1000 then jump to address marked by label",
            ),
        ],
    )
    def test_asm_string(self, op_code, operands, comment):
        instruction = Q1asmInstruction(op_code, *operands, comment=comment)

        args = ",".join([str(op) for op in operands])
        if op_code == Opcode.ADDRESS:
            assert str(instruction) == f"{args}: # {comment}"
        elif args:
            assert str(instruction) == f"{op_code.value} {args} # {comment}"
        else:
            assert str(instruction) == f"{op_code.value} # {comment}"


def test_sequence_builder():
    builder = SequenceBuilder()

    builder.label("foo", "Section for function foo")
    sequence = builder.build()
    assert sequence.program
    assert not sequence.waveforms
    assert not sequence.weights
    assert not sequence.acquisitions

    instruction = Q1asmInstruction(
        Opcode.ADDRESS, "foo", comment="Section for function foo"
    )
    assert sequence.program == str(instruction)
    assert str(instruction) == "foo: # Section for function foo"


def test_q1asm_instruction_repr_matches_assembly():
    instruction = Q1asmInstruction(Opcode.NOP)

    assert repr(instruction) == "nop"


def test_sequence_builder_manages_data_tables():
    builder = SequenceBuilder()

    builder.add_waveform("pulse", 0, [0.25, 0.5])
    builder.add_weight("weight", 1, [1.0, -1.0])
    builder.add_acquisition("readout", 2, 4)

    assert builder.lookup_waveform_by_data(np.array([0.25, 0.5])) == 0
    assert builder.lookup_waveform_by_data(np.array([0.25])) is None

    for method, arguments in [
        ("add_waveform", ("pulse", 3, [0.0])),
        ("add_weight", ("weight", 3, [0.0])),
        ("add_acquisition", ("readout", 3, 1)),
    ]:
        with pytest.raises(ValueError, match="already exists"):
            getattr(builder, method)(*arguments)


@pytest.mark.parametrize(
    ("method", "arguments", "opcode", "operands"),
    [
        ("nop", (), Opcode.NOP, ()),
        ("stop", (), Opcode.STOP, ()),
        ("label", ("target",), Opcode.ADDRESS, ("target",)),
        ("jmp", ("target",), Opcode.JUMP, ("@target",)),
        ("jmp", (4,), Opcode.JUMP, (4,)),
        ("jge", ("R0", 1, "target"), Opcode.JUMP_GREATER_EQUALS, ("R0", 1, "@target")),
        ("jlt", ("R0", 1, 4), Opcode.JUMP_LESS_THAN, ("R0", 1, 4)),
        ("loop", ("R0", "target"), Opcode.LOOP, ("R0", "@target")),
        ("move", (1, "R0"), Opcode.MOVE, (1, "R0")),
        ("add", ("R0", 1, "R1"), Opcode.ADD, ("R0", 1, "R1")),
        ("sub", ("R0", 1, "R1"), Opcode.SUB, ("R0", 1, "R1")),
        ("logic_not", ("R0", "R1"), Opcode.NOT, ("R0", "R1")),
        ("logic_and", ("R0", 1, "R1"), Opcode.AND, ("R0", 1, "R1")),
        ("logic_or", ("R0", 1, "R1"), Opcode.OR, ("R0", 1, "R1")),
        ("logic_xor", ("R0", 1, "R1"), Opcode.XOR, ("R0", 1, "R1")),
        ("set_mrk", (1,), Opcode.SET_MARKER, (1,)),
        ("set_freq", (2,), Opcode.SET_NCO_FREQUENCY, (2,)),
        ("set_ph", (3,), Opcode.SET_NCO_PHASE, (3,)),
        ("set_ph_delta", (4,), Opcode.SET_NCO_PHASE_OFFSET, (4,)),
        ("reset_ph", (), Opcode.RESET_PHASE, ()),
        ("set_awg_gain", (5, 6), Opcode.SET_AWG_GAIN, (5, 6)),
        ("set_awg_offs", (7, 8), Opcode.SET_AWG_OFFSET, (7, 8)),
        ("set_cond", (1, 2, 3, 4), Opcode.SET_COND, (1, 2, 3, 4)),
        ("upd_param", (4,), Opcode.UPDATE_PARAMETERS, (4,)),
        ("play", (0, 1, 4), Opcode.PLAY, (0, 1, 4)),
        ("acquire", (0, "R0", 4), Opcode.ACQUIRE, (0, "R0", 4)),
        (
            "acquire_weighed",
            (0, "R0", 1, 2, 4),
            Opcode.ACQUIRE_WEIGHED,
            (0, "R0", 1, 2, 4),
        ),
        ("acquire_ttl", (0, "R0", 1, 4), Opcode.ACQUIRE_TTL, (0, "R0", 1, 4)),
        ("set_latch_en", (1, 4), Opcode.SET_LATCH_EN, (1, 4)),
        ("latch_rst", (4,), Opcode.LATCH_RST, (4,)),
        ("wait", (4,), Opcode.WAIT, (4,)),
        ("wait_trigger", (1, 4), Opcode.WAIT_TRIGGER, (1, 4)),
        ("wait_sync", (4,), Opcode.WAIT_SYNC, (4,)),
    ],
)
def test_sequence_builder_adds_each_instruction(method, arguments, opcode, operands):
    builder = SequenceBuilder()

    returned = getattr(builder, method)(*arguments, comment="comment")

    assert returned is builder
    instruction = builder.q1asm_instructions[0]
    assert instruction.opcode == opcode
    assert instruction.operands == operands
    assert instruction.comment == "comment"
