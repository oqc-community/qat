# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from io import StringIO

import pytest
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.q1 import (
    Q1,
    AddressImm,
    AddRsImmRdOp,
    AddRsRsRdOp,
    AndRsImmRdOp,
    AndRsRsRdOp,
    AslRsImmRdOp,
    AslRsRsRdOp,
    AsrRsImmRdOp,
    AsrRsRsRdOp,
    BoolImm,
    DurationImm,
    FbAcqIqIdImmImmOp,
    FbAcqIqIdRsImmOp,
    FbAcqTbIdRsImmOp,
    FbAcqTbValidRsImmOp,
    FbPopDataImmRdOp,
    IllegalOp,
    JgeRsImmImmOp,
    JgeRsImmRsOp,
    JltRsImmImmOp,
    JltRsImmRsOp,
    JmpImmOp,
    JmpRsOp,
    JnzImmOp,
    JnzRsOp,
    JoImmOp,
    JoRsOp,
    JzImmOp,
    JzRsOp,
    MoveImmRdOp,
    MoveRsRdOp,
    NcoPhaseImm,
    NopOp,
    NotImmRdOp,
    NotRsRdOp,
    OrRsImmRdOp,
    OrRsRsRdOp,
    Registers,
    ResetPhOp,
    SetAwgGainImmImmOp,
    SetAwgGainRsRsOp,
    SetAwgOffsImmImmOp,
    SetAwgOffsRsRsOp,
    SetCondImmImmImmImmOp,
    SetCondRsRsRsImmOp,
    SetFreqImmOp,
    SetFreqRsOp,
    SetMrkImmOp,
    SetMrkRsOp,
    SetPhDeltaImmOp,
    SetPhImmOp,
    SetPhRsOp,
    SI16Imm,
    SI32Imm,
    StopImmOp,
    StopOp,
    StopRsOp,
    SU32Imm,
    SubRsImmRdOp,
    SubRsRsRdOp,
    UI2Imm,
    UI3Imm,
    UI4Imm,
    UI5Imm,
    UI6Imm,
    UI7Imm,
    UI8Imm,
    UI10Imm,
    UI16Imm,
    UI24Imm,
    UI32Imm,
    XorRsImmRdOp,
    XorRsRsRdOp,
    emit_program,
)
from qat.experimental.dialect.q1.ir import ops as q1_ops

_PRINT_TABLE = [
    # Nullary
    pytest.param(lambda: IllegalOp(), "illegal", (), None, id="illegal"),
    pytest.param(
        lambda: IllegalOp(comment="halt"), "illegal", (), "halt", id="illegal+comment"
    ),
    pytest.param(lambda: NopOp(), "nop", (), None, id="nop"),
    pytest.param(lambda: NopOp(comment="pad"), "nop", (), "pad", id="nop+comment"),
    pytest.param(lambda: StopOp(), "stop", (), None, id="stop_nullary"),
    pytest.param(
        lambda: StopOp(comment="end"), "stop", (), "end", id="stop_nullary+comment"
    ),
    pytest.param(lambda: ResetPhOp(), "reset_ph", (), None, id="reset_ph"),
    pytest.param(
        lambda: ResetPhOp(comment="zero"), "reset_ph", (), "zero", id="reset_ph+comment"
    ),
    # I-format
    pytest.param(lambda: StopImmOp(SI32Imm(42)), "stop", ("42",), None, id="stop_imm"),
    pytest.param(lambda: JmpImmOp(AddressImm(100)), "jmp", ("100",), None, id="jmp_imm"),
    pytest.param(lambda: JzImmOp(AddressImm(200)), "jz", ("200",), None, id="jz_imm"),
    pytest.param(lambda: JnzImmOp(AddressImm(300)), "jnz", ("300",), None, id="jnz_imm"),
    pytest.param(lambda: JoImmOp(AddressImm(400)), "jo", ("400",), None, id="jo_imm"),
    pytest.param(
        lambda: SetMrkImmOp(UI4Imm(15)), "set_mrk", ("15",), None, id="set_mrk_imm"
    ),
    pytest.param(
        lambda: SetFreqImmOp(SI32Imm(0x12345678)),
        "set_freq",
        (str(0x12345678),),
        None,
        id="set_freq_imm",
    ),
    pytest.param(
        lambda: SetPhImmOp(NcoPhaseImm(1000000000)),
        "set_ph",
        ("1000000000",),
        None,
        id="set_ph_imm",
    ),
    pytest.param(
        lambda: SetPhDeltaImmOp(NcoPhaseImm(0)),
        "set_ph_delta",
        ("0",),
        None,
        id="set_ph_delta_imm",
    ),
    pytest.param(
        lambda: SetMrkImmOp(UI4Imm(15), comment="marker at position 1000"),
        "set_mrk",
        ("15",),
        "marker at position 1000",
        id="set_mrk_imm+comment",
    ),
    # Rs-format
    pytest.param(
        lambda: StopRsOp(create_ssa_value(Registers.R0)),
        "stop",
        ("R0",),
        None,
        id="stop_rs",
    ),
    pytest.param(
        lambda: JmpRsOp(create_ssa_value(Registers.R1)), "jmp", ("R1",), None, id="jmp_rs"
    ),
    pytest.param(
        lambda: JzRsOp(create_ssa_value(Registers.R5)), "jz", ("R5",), None, id="jz_rs"
    ),
    pytest.param(
        lambda: JnzRsOp(create_ssa_value(Registers.R10)), "jnz", ("R10",), None, id="jnz_rs"
    ),
    pytest.param(
        lambda: JoRsOp(create_ssa_value(Registers.R31)), "jo", ("R31",), None, id="jo_rs"
    ),
    pytest.param(
        lambda: SetMrkRsOp(create_ssa_value(Registers.R32)),
        "set_mrk",
        ("R32",),
        None,
        id="set_mrk_rs",
    ),
    pytest.param(
        lambda: SetFreqRsOp(create_ssa_value(Registers.R48)),
        "set_freq",
        ("R48",),
        None,
        id="set_freq_rs",
    ),
    pytest.param(
        lambda: SetPhRsOp(create_ssa_value(Registers.R63)),
        "set_ph",
        ("R63",),
        None,
        id="set_ph_rs",
    ),
    # I+Rd (imm, rd)
    pytest.param(
        lambda: MoveImmRdOp(SU32Imm(42), Registers.R0),
        "move",
        ("42", "R0"),
        None,
        id="move_42_R0",
    ),
    pytest.param(
        lambda: MoveImmRdOp(SU32Imm(1000), Registers.R15),
        "move",
        ("1000", "R15"),
        None,
        id="move_1000_R15",
    ),
    pytest.param(
        lambda: NotImmRdOp(SU32Imm(0xFFFFFFFF), Registers.R31),
        "not",
        (str(0xFFFFFFFF), "R31"),
        None,
        id="not_ff_R31",
    ),
    pytest.param(
        lambda: FbPopDataImmRdOp(UI16Imm(5), Registers.R10),
        "fb_pop_data",
        ("5", "R10"),
        None,
        id="fb_pop_data",
    ),
    # Rs+Rd (rs, rd)
    pytest.param(
        lambda: MoveRsRdOp(create_ssa_value(Registers.R0), Registers.R1),
        "move",
        ("R0", "R1"),
        None,
        id="move_R0_R1",
    ),
    pytest.param(
        lambda: MoveRsRdOp(create_ssa_value(Registers.R10), Registers.R20),
        "move",
        ("R10", "R20"),
        None,
        id="move_R10_R20",
    ),
    pytest.param(
        lambda: NotRsRdOp(create_ssa_value(Registers.R32), Registers.R48),
        "not",
        ("R32", "R48"),
        None,
        id="not_R32_R48",
    ),
    pytest.param(
        lambda: NotRsRdOp(create_ssa_value(Registers.R50), Registers.R63),
        "not",
        ("R50", "R63"),
        None,
        id="not_R50_R63",
    ),
    # Rs+I (feedback)
    pytest.param(
        lambda: FbAcqTbIdRsImmOp(
            create_ssa_value(Registers.R1), DurationImm(100), comment="test feedback"
        ),
        "fb_acq_tb_id",
        ("R1", "100"),
        "test feedback",
        id="fb_acq_tb_id_ri",
    ),
    pytest.param(
        lambda: FbAcqTbValidRsImmOp(
            create_ssa_value(Registers.R1), DurationImm(100), comment="test feedback"
        ),
        "fb_acq_tb_valid",
        ("R1", "100"),
        "test feedback",
        id="fb_acq_tb_valid_ri",
    ),
    pytest.param(
        lambda: FbAcqIqIdRsImmOp(
            create_ssa_value(Registers.R1), DurationImm(100), comment="test feedback"
        ),
        "fb_acq_iq_id",
        ("R1", "100"),
        "test feedback",
        id="fb_acq_iq_id_ri",
    ),
    # Rs+I+I (legacy compare-and-jump)
    pytest.param(
        lambda: JgeRsImmImmOp(
            create_ssa_value(Registers.R0), UI32Imm(100), AddressImm(500)
        ),
        "jge",
        ("R0", "100", "500"),
        None,
        id="jge_rii",
    ),
    pytest.param(
        lambda: JltRsImmImmOp(create_ssa_value(Registers.R5), UI32Imm(50), AddressImm(200)),
        "jlt",
        ("R5", "50", "200"),
        None,
        id="jlt_rii",
    ),
    # Rs+I+Rs
    pytest.param(
        lambda: JgeRsImmRsOp(
            create_ssa_value(Registers.R0), UI32Imm(100), create_ssa_value(Registers.R5)
        ),
        "jge",
        ("R0", "100", "R5"),
        None,
        id="jge_rir",
    ),
    pytest.param(
        lambda: JltRsImmRsOp(
            create_ssa_value(Registers.R0), UI32Imm(100), create_ssa_value(Registers.R5)
        ),
        "jlt",
        ("R0", "100", "R5"),
        None,
        id="jlt_rir",
    ),
    # Rs+I→Rd (RIR arithmetic)
    *[
        pytest.param(
            lambda op=op, imm_t=imm_t, imm=imm: op(
                create_ssa_value(Registers.R1), imm_t(imm), Registers.R2
            ),
            mnemonic,
            ("R1", str(imm), "R2"),
            None,
            id=f"{mnemonic}_rir",
        )
        for op, mnemonic, imm_t, imm in [
            (AddRsImmRdOp, "add", SU32Imm, 100),
            (SubRsImmRdOp, "sub", SU32Imm, 50),
            (AndRsImmRdOp, "and", SU32Imm, 0xFF),
            (OrRsImmRdOp, "or", SU32Imm, 0xAA),
            (XorRsImmRdOp, "xor", SU32Imm, 0x55),
            (AslRsImmRdOp, "asl", UI32Imm, 5),
            (AsrRsImmRdOp, "asr", UI32Imm, 3),
        ]
    ],
    # Rs+Rs→Rd (RRR arithmetic)
    *[
        pytest.param(
            lambda op=op: op(
                create_ssa_value(Registers.R1),
                create_ssa_value(Registers.R2),
                Registers.R3,
            ),
            mnemonic,
            ("R1", "R2", "R3"),
            None,
            id=f"{mnemonic}_rrr",
        )
        for op, mnemonic in [
            (AddRsRsRdOp, "add"),
            (SubRsRsRdOp, "sub"),
            (AndRsRsRdOp, "and"),
            (OrRsRsRdOp, "or"),
            (XorRsRsRdOp, "xor"),
            (AslRsRsRdOp, "asl"),
            (AsrRsRsRdOp, "asr"),
        ]
    ],
    # I+I latched
    pytest.param(
        lambda: SetAwgGainImmImmOp(SI16Imm(0x1000), SI16Imm(0x2000)),
        "set_awg_gain",
        (str(0x1000), str(0x2000)),
        None,
        id="set_awg_gain_ii",
    ),
    pytest.param(
        lambda: SetAwgOffsImmImmOp(SI16Imm(0x0100), SI16Imm(0x0200)),
        "set_awg_offs",
        (str(0x0100), str(0x0200)),
        None,
        id="set_awg_offs_ii",
    ),
    # Rs+Rs latched
    pytest.param(
        lambda: SetAwgGainRsRsOp(
            create_ssa_value(Registers.R0), create_ssa_value(Registers.R1)
        ),
        "set_awg_gain",
        ("R0", "R1"),
        None,
        id="set_awg_gain_rr",
    ),
    pytest.param(
        lambda: SetAwgOffsRsRsOp(
            create_ssa_value(Registers.R0), create_ssa_value(Registers.R1)
        ),
        "set_awg_offs",
        ("R0", "R1"),
        None,
        id="set_awg_offs_rr",
    ),
    # I+I feedback
    pytest.param(
        lambda: FbAcqIqIdImmImmOp(UI8Imm(1), DurationImm(4)),
        "fb_acq_iq_id",
        ("1", "4"),
        None,
        id="fb_acq_iq_id_ii_small",
    ),
    pytest.param(
        lambda: FbAcqIqIdImmImmOp(UI8Imm(100), DurationImm(200)),
        "fb_acq_iq_id",
        ("100", "200"),
        None,
        id="fb_acq_iq_id_ii_large",
    ),
    # I+I+I+I (set_cond)
    pytest.param(
        lambda: SetCondImmImmImmImmOp(BoolImm(0), UI4Imm(1), UI3Imm(2), UI16Imm(3)),
        "set_cond",
        ("0", "1", "2", "3"),
        None,
        id="set_cond_iiii_small",
    ),
    pytest.param(
        lambda: SetCondImmImmImmImmOp(BoolImm(1), UI4Imm(8), UI3Imm(4), UI16Imm(400)),
        "set_cond",
        ("1", "8", "4", "400"),
        None,
        id="set_cond_iiii_large",
    ),
    # Rs+Rs+Rs+I (set_cond register)
    pytest.param(
        lambda: SetCondRsRsRsImmOp(
            create_ssa_value(Registers.R0),
            create_ssa_value(Registers.R1),
            create_ssa_value(Registers.R2),
            UI16Imm(42),
        ),
        "set_cond",
        ("R0", "R1", "R2", "42"),
        None,
        id="set_cond_rrri",
    ),
]


@pytest.mark.parametrize("factory,mnemonic,substrings,comment", _PRINT_TABLE)
def test_assembly_line_print(factory, mnemonic, substrings, comment):
    op = factory()
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(mnemonic)
    for s in substrings:
        assert s in assembly
    if comment:
        assert f"# {comment}" in assembly
    else:
        assert not assembly.rstrip().endswith("#")


def _all_concrete_q1_ops():
    r0 = create_ssa_value(Registers.R0)
    r1 = create_ssa_value(Registers.R1)
    r2 = create_ssa_value(Registers.R2)
    r3 = create_ssa_value(Registers.R3)
    r4 = create_ssa_value(Registers.R4)
    r5 = create_ssa_value(Registers.R5)
    r6 = create_ssa_value(Registers.R6)
    r7 = create_ssa_value(Registers.R7)
    r8 = create_ssa_value(Registers.R8)
    r9 = create_ssa_value(Registers.R9)
    r10 = create_ssa_value(Registers.R10)
    r11 = create_ssa_value(Registers.R11)
    r12 = create_ssa_value(Registers.R12)
    r13 = create_ssa_value(Registers.R13)
    r14 = create_ssa_value(Registers.R14)
    r15 = create_ssa_value(Registers.R15)

    return [
        # Core Instructions
        q1_ops.LabelOp("entry", comment="entry"),
        q1_ops.DefDirectiveOp("CONST", "7", comment="const"),
        q1_ops.IllegalOp(),
        q1_ops.StopImmOp(SI32Imm(1)),
        q1_ops.StopOp(),
        q1_ops.StopRsOp(r0),
        q1_ops.NopOp(),
        # Jump Instructions
        q1_ops.JmpImmOp(AddressImm(100)),
        q1_ops.JmpRsOp(r1),
        q1_ops.JzImmOp(AddressImm(101)),
        q1_ops.JzRsOp(r2),
        q1_ops.JnzImmOp(AddressImm(102)),
        q1_ops.JnzRsOp(r3),
        q1_ops.JoImmOp(AddressImm(103)),
        q1_ops.JoRsOp(r4),
        q1_ops.JnoImmOp(AddressImm(104)),
        q1_ops.JnoRsOp(r5),
        q1_ops.JsImmOp(AddressImm(105)),
        q1_ops.JsRsOp(r6),
        q1_ops.JnsImmOp(AddressImm(106)),
        q1_ops.JnsRsOp(r7),
        q1_ops.JgImmOp(AddressImm(107)),
        q1_ops.JgRsOp(r8),
        q1_ops.JlImmOp(AddressImm(108)),
        q1_ops.JlRsOp(r9),
        q1_ops.JleImmOp(AddressImm(109)),
        q1_ops.JleRsOp(r10),
        q1_ops.JaImmOp(AddressImm(110)),
        q1_ops.JaRsOp(r11),
        q1_ops.JaeImmOp(AddressImm(111)),
        q1_ops.JaeRsOp(r12),
        q1_ops.JbImmOp(AddressImm(112)),
        q1_ops.JbRsOp(r13),
        q1_ops.JbeImmOp(AddressImm(113)),
        q1_ops.JbeRsOp(r14),
        q1_ops.JgeImmOp(AddressImm(115)),
        q1_ops.JgeRsOp(r15),
        q1_ops.JgeRsImmImmOp(r0, UI32Imm(10), AddressImm(120)),
        q1_ops.JgeRsImmRsOp(r1, UI32Imm(11), r2),
        q1_ops.JltRsImmImmOp(r3, UI32Imm(12), AddressImm(121)),
        q1_ops.JltRsImmRsOp(r4, UI32Imm(13), r5),
        q1_ops.LoopRdImmOp(Registers.R16, AddressImm(122)),
        q1_ops.LoopRdRsOp(Registers.R17, r6),
        # Arithmetic Instructions
        q1_ops.MoveImmRdOp(SU32Imm(14), Registers.R18),
        q1_ops.MoveRsRdOp(r7, Registers.R19),
        q1_ops.NotImmRdOp(SU32Imm(15), Registers.R20),
        q1_ops.NotRsRdOp(r8, Registers.R21),
        q1_ops.AddRsImmRdOp(r9, SU32Imm(16), Registers.R22),
        q1_ops.AddRsRsRdOp(r10, r11, Registers.R23),
        q1_ops.SubRsImmRdOp(r12, SU32Imm(17), Registers.R24),
        q1_ops.SubRsRsRdOp(r13, r14, Registers.R25),
        q1_ops.AndRsImmRdOp(r15, SU32Imm(18), Registers.R26),
        q1_ops.AndRsRsRdOp(r0, r1, Registers.R27),
        q1_ops.OrRsImmRdOp(r2, SU32Imm(19), Registers.R28),
        q1_ops.OrRsRsRdOp(r3, r4, Registers.R29),
        q1_ops.XorRsImmRdOp(r5, SU32Imm(20), Registers.R30),
        q1_ops.XorRsRsRdOp(r6, r7, Registers.R31),
        q1_ops.AslRsImmRdOp(r8, UI32Imm(21), Registers.R32),
        q1_ops.AslRsRsRdOp(r9, r10, Registers.R33),
        q1_ops.AsrRsImmRdOp(r11, UI32Imm(22), Registers.R34),
        q1_ops.AsrRsRsRdOp(r12, r13, Registers.R35),
        # Comparison
        q1_ops.CmpRsRsOp(r0, r1),
        q1_ops.CmpRsImmOp(r2, SU32Imm(30)),
        q1_ops.CmpImmRsOp(SU32Imm(31), r3),
        q1_ops.TestRsRsOp(r4, r5),
        q1_ops.TestRsImmOp(r6, SU32Imm(32)),
        q1_ops.TestImmRsOp(SU32Imm(33), r7),
        # Logical shifts
        q1_ops.LsrRsImmRdOp(r8, UI32Imm(34), Registers.R39),
        q1_ops.LsrRsRsRdOp(r9, r10, Registers.R40),
        q1_ops.LsrImmRsRdOp(UI32Imm(35), r11, Registers.R41),
        q1_ops.LslRsImmRdOp(r12, UI32Imm(36), Registers.R42),
        q1_ops.LslRsRsRdOp(r13, r14, Registers.R43),
        q1_ops.LslImmRsRdOp(UI32Imm(37), r15, Registers.R44),
        # 16-bit multiplications
        q1_ops.Mulu16RsImmRdOp(r0, UI16Imm(40), Registers.R45),
        q1_ops.Mulu16RsRsRdOp(r1, r2, Registers.R46),
        q1_ops.Mulu16ImmRsRdOp(UI16Imm(41), r3, Registers.R47),
        q1_ops.Muls16RsImmRdOp(r4, SI16Imm(42), Registers.R48),
        q1_ops.Muls16RsRsRdOp(r5, r6, Registers.R49),
        q1_ops.Muls16ImmRsRdOp(SI16Imm(43), r7, Registers.R50),
        # 32-bit multiplications low/high
        q1_ops.Mulu32lRsImmRdOp(r8, UI32Imm(44), Registers.R51),
        q1_ops.Mulu32lRsRsRdOp(r9, r10, Registers.R52),
        q1_ops.Mulu32lImmRsRdOp(UI32Imm(45), r11, Registers.R53),
        q1_ops.Mulu32hRsImmRdOp(r12, UI32Imm(46), Registers.R54),
        q1_ops.Mulu32hRsRsRdOp(r13, r14, Registers.R55),
        q1_ops.Mulu32hImmRsRdOp(UI32Imm(47), r15, Registers.R56),
        q1_ops.Muls32lRsImmRdOp(r0, SI32Imm(48), Registers.R57),
        q1_ops.Muls32lRsRsRdOp(r1, r2, Registers.R58),
        q1_ops.Muls32lImmRsRdOp(SI32Imm(49), r3, Registers.R59),
        q1_ops.Muls32hRsImmRdOp(r4, SI32Imm(50), Registers.R60),
        q1_ops.Muls32hRsRsRdOp(r5, r6, Registers.R61),
        q1_ops.Muls32hImmRsRdOp(SI32Imm(51), r7, Registers.R62),
        # 32-bit multiplications with full 64-bit result
        q1_ops.Mulu32RsImmRdRdOp(r8, UI32Imm(60), Registers.R0, Registers.R1),
        q1_ops.Mulu32RsRsRdRdOp(r9, r10, Registers.R2, Registers.R3),
        q1_ops.Mulu32ImmRsRdRdOp(UI32Imm(61), r11, Registers.R4, Registers.R5),
        q1_ops.Muls32RsImmRdRdOp(r12, SI32Imm(62), Registers.R6, Registers.R7),
        q1_ops.Muls32RsRsRdRdOp(r13, r14, Registers.R8, Registers.R9),
        q1_ops.Muls32ImmRsRdRdOp(SI32Imm(63), r15, Registers.R10, Registers.R11),
        # Latched Instructions
        q1_ops.SetCondImmImmImmImmOp(BoolImm(1), UI4Imm(2), UI3Imm(3), UI16Imm(4)),
        q1_ops.SetCondRsRsRsImmOp(r0, r1, r2, UI16Imm(23)),
        q1_ops.SetMrkImmOp(UI4Imm(8)),
        q1_ops.SetMrkRsOp(r3),
        q1_ops.SetFreqImmOp(SI32Imm(25)),
        q1_ops.SetFreqRsOp(r4),
        q1_ops.ResetPhOp(),
        q1_ops.SetPhImmOp(NcoPhaseImm(26)),
        q1_ops.SetPhRsOp(r5),
        q1_ops.SetPhDeltaImmOp(NcoPhaseImm(27)),
        q1_ops.SetPhDeltaRsOp(r6),
        q1_ops.SetAwgGainImmImmOp(SI16Imm(28), SI16Imm(29)),
        q1_ops.SetAwgGainRsRsOp(r7, r8),
        q1_ops.SetAwgOffsImmImmOp(SI16Imm(30), SI16Imm(31)),
        q1_ops.SetAwgOffsRsRsOp(r9, r10),
        # LINQ Feedback Instructions
        q1_ops.FbAcqIqIdImmImmOp(UI8Imm(32), DurationImm(33)),
        q1_ops.FbAcqIqIdRsImmOp(r11, DurationImm(34)),
        q1_ops.FbAcqIqShiftImmImmOp(UI6Imm(35), DurationImm(36)),
        q1_ops.FbAcqTbCfgImmImmImmImmOp(
            UI2Imm(1), UI10Imm(38), UI7Imm(39), DurationImm(40)
        ),
        q1_ops.FbAcqTbExtraImmImmImmOp(BoolImm(1), UI16Imm(42), DurationImm(43)),
        q1_ops.FbAcqTbIdImmImmOp(UI8Imm(44), DurationImm(45)),
        q1_ops.FbAcqTbIdRsImmOp(r12, DurationImm(46)),
        q1_ops.FbAcqTbMockImmImmImmImmOp(
            BoolImm(1), BoolImm(1), BoolImm(1), DurationImm(50)
        ),
        q1_ops.FbAcqTbValidImmImmOp(BoolImm(1), DurationImm(52)),
        q1_ops.FbAcqTbValidRsImmOp(r13, DurationImm(53)),
        q1_ops.FbComCfgImmImmImmImmOp(UI2Imm(2), UI10Imm(55), UI7Imm(56), DurationImm(57)),
        q1_ops.FbComDataImmImmImmOp(UI8Imm(58), UI32Imm(59), DurationImm(60)),
        q1_ops.FbComDataImmRsImmOp(UI8Imm(61), r14, DurationImm(62)),
        q1_ops.FbComExtraImmImmImmOp(BoolImm(1), UI16Imm(64), DurationImm(65)),
        q1_ops.FbPopDataImmRdOp(UI16Imm(71), Registers.R36),
        q1_ops.FbPullDataRsRdOp(create_ssa_value(Registers.R37), Registers.R38),
        # Real-time Instructions
        q1_ops.WaitImmOp(DurationImm(100)),
        q1_ops.WaitRsOp(r0),
        q1_ops.WaitSyncImmOp(DurationImm(200)),
        q1_ops.WaitSyncRsOp(r1),
        q1_ops.WaitTriggerImmImmOp(UI4Imm(1), DurationImm(50)),
        q1_ops.WaitTriggerRsRsOp(r2, r3),
        q1_ops.PlayImmImmImmOp(UI10Imm(1), UI10Imm(2), DurationImm(100)),
        q1_ops.PlayRsRsImmOp(r4, r5, DurationImm(100)),
        q1_ops.AcquireImmImmImmOp(UI5Imm(0), UI24Imm(1), DurationImm(100)),
        q1_ops.AcquireImmRsImmOp(UI5Imm(0), r6, DurationImm(100)),
        q1_ops.AcquireWeightedImmImmImmImmImmOp(
            UI5Imm(0), UI24Imm(1), UI6Imm(0), UI6Imm(1), DurationImm(100)
        ),
        q1_ops.AcquireWeightedImmRsRsRsImmOp(UI5Imm(0), r7, r8, r9, DurationImm(100)),
        q1_ops.AcquireTtlImmImmImmImmOp(
            UI5Imm(0), UI24Imm(0), BoolImm(1), DurationImm(100)
        ),
        q1_ops.AcquireTtlImmRsImmImmOp(UI5Imm(0), r10, BoolImm(1), DurationImm(100)),
        q1_ops.UpdParamImmOp(DurationImm(100)),
        q1_ops.LatchRstImmOp(DurationImm(100)),
        q1_ops.LatchRstRsOp(r11),
        q1_ops.SetLatchEnImmImmOp(BoolImm(1), DurationImm(100)),
        q1_ops.SetLatchEnRsImmOp(r0, DurationImm(100)),
    ]


def test_all_ops_assembly_output_matches_q1_code():
    ops = _all_concrete_q1_ops()
    module = ModuleOp(ops)

    stream = StringIO()
    emit_program(module.body, stream)
    asm_lines = stream.getvalue().splitlines()
    expected_lines = [op.assembly_line() for op in ops]

    assert asm_lines == expected_lines
    assert {op.name for op in ops} == {op.name for op in Q1.operations}
