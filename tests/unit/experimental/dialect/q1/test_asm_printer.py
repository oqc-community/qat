# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.q1 import (
    Q1,
    AddRIROp,
    AddRRROp,
    AndRIROp,
    AndRRROp,
    AslRIROp,
    AslRRROp,
    AsrRIROp,
    AsrRRROp,
    FbAcqIqIdIIOp,
    FbAcqIqIdRIOp,
    FbAcqTbIdRIOp,
    FbAcqTbValidRIOp,
    FbPopDataIROp,
    IllegalOp,
    JgeRIIOp,
    JgeRIROp,
    JltRIIOp,
    JltRIROp,
    JmpIOp,
    JmpROp,
    JnzIOp,
    JnzROp,
    JoIOp,
    JoROp,
    JzIOp,
    JzROp,
    MoveIROp,
    MoveRROp,
    NopOp,
    NotIROp,
    NotRROp,
    OrRIROp,
    OrRRROp,
    ResetPhOp,
    SetAwgGainIIOp,
    SetAwgGainRROp,
    SetAwgOffsIIOp,
    SetAwgOffsRROp,
    SetCondIIIIOp,
    SetCondRRRIOp,
    SetFreqIOp,
    SetFreqROp,
    SetMrkIOp,
    SetMrkROp,
    SetPhDeltaIOp,
    SetPhIOp,
    SetPhROp,
    StopIOp,
    StopOp,
    StopROp,
    SubRIROp,
    SubRRROp,
    XorRIROp,
    XorRRROp,
    q1_code,
)
from qat.experimental.dialect.q1.ir import ops as q1_ops
from qat.experimental.dialect.q1.ir.reg_desc import Registers


@pytest.mark.parametrize("comment", [None, "test comment"])
@pytest.mark.parametrize(
    "op_type,expected_mnemonic",
    [
        (IllegalOp, "illegal"),
        (NopOp, "nop"),
        (StopOp, "stop"),
        (ResetPhOp, "reset_ph"),
    ],
)
def test_nullary_format_print(op_type, expected_mnemonic, comment):
    op = op_type(comment=comment)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    if comment:
        assert f"# {comment}" in assembly
    else:
        assert not assembly.rstrip().endswith("#")


@pytest.mark.parametrize("comment", [None, "marker at position 1000"])
@pytest.mark.parametrize(
    "op_type,expected_mnemonic,imm_value",
    [
        (StopIOp, "stop", 42),
        (JmpIOp, "jmp", 100),
        (JzIOp, "jz", 200),
        (JnzIOp, "jnz", 300),
        (JoIOp, "jo", 400),
        (SetMrkIOp, "set_mrk", 0xDEADBEEF),
        (SetFreqIOp, "set_freq", 0x12345678),
        (SetPhIOp, "set_ph", 0xFFFFFFFF),
        (SetPhDeltaIOp, "set_ph_delta", 0x80000000),
    ],
)
def test_imm_format_print(op_type, expected_mnemonic, imm_value, comment):
    op = op_type(imm_value, comment=comment)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert str(imm_value) in assembly
    if comment:
        assert f"# {comment}" in assembly
    else:
        assert not assembly.rstrip().endswith("#")


@pytest.mark.parametrize(
    "op_type,expected_mnemonic,register",
    [
        (StopROp, "stop", Registers.R0),
        (JmpROp, "jmp", Registers.R1),
        (JzROp, "jz", Registers.R5),
        (JnzROp, "jnz", Registers.R10),
        (JoROp, "jo", Registers.R31),
        (SetMrkROp, "set_mrk", Registers.R32),
        (SetFreqROp, "set_freq", Registers.R48),
        (SetPhROp, "set_ph", Registers.R63),
    ],
)
def test_rs_format_print(op_type, expected_mnemonic, register):
    rs_val = create_ssa_value(register)
    op = op_type(rs_val)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert register.register_name.data in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic,imm_value,register",
    [
        (MoveIROp, "move", 42, Registers.R0),
        (MoveIROp, "move", 1000, Registers.R15),
        (NotIROp, "not", 0xFFFFFFFF, Registers.R31),
        (FbPopDataIROp, "fb_pop_data", 5, Registers.R10),
    ],
)
def test_imm_rd_format_print(op_type, expected_mnemonic, imm_value, register):
    op = op_type(imm_value, register)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert str(imm_value) in assembly
    assert register.register_name.data in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic,src_register,dst_register",
    [
        (MoveRROp, "move", Registers.R0, Registers.R1),
        (MoveRROp, "move", Registers.R10, Registers.R20),
        (NotRROp, "not", Registers.R32, Registers.R48),
        (NotRROp, "not", Registers.R50, Registers.R63),
    ],
)
def test_rs_rd_format_print(op_type, expected_mnemonic, src_register, dst_register):
    src_val = create_ssa_value(src_register)
    op = op_type(src_val, dst_register)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert src_register.register_name.data in assembly
    assert dst_register.register_name.data in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic",
    [
        (FbAcqTbIdRIOp, "fb_acq_tb_id"),
        (FbAcqTbValidRIOp, "fb_acq_tb_valid"),
        (FbAcqIqIdRIOp, "fb_acq_iq_id"),
    ],
)
def test_rs_imm_format_print(op_type, expected_mnemonic):
    rs_val = create_ssa_value(Registers.R1)
    op = op_type(rs_val, 100, comment="test feedback")
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert "R1" in assembly
    assert "100" in assembly
    assert "test feedback" in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic,register,imm1,imm2",
    [
        (JgeRIIOp, "jge", Registers.R0, 100, 500),
        (JltRIIOp, "jlt", Registers.R5, 50, 200),
    ],
)
def test_rs_imm_imm_format_print(op_type, expected_mnemonic, register, imm1, imm2):
    rs_val = create_ssa_value(register)
    op = op_type(rs_val, imm1, imm2)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert register.register_name.data in assembly
    assert str(imm1) in assembly
    assert str(imm2) in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic",
    [
        (JgeRIROp, "jge"),
        (JltRIROp, "jlt"),
    ],
)
def test_rs_imm_rs_format_print(op_type, expected_mnemonic):
    rs1_val = create_ssa_value(Registers.R0)
    rs2_val = create_ssa_value(Registers.R5)
    op = op_type(rs1_val, 100, rs2_val)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert "R0" in assembly
    assert "100" in assembly
    assert "R5" in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic,imm_value",
    [
        (AddRIROp, "add", 100),
        (SubRIROp, "sub", 50),
        (AndRIROp, "and", 0xFF),
        (OrRIROp, "or", 0xAA),
        (XorRIROp, "xor", 0x55),
        (AslRIROp, "asl", 5),
        (AsrRIROp, "asr", 3),
    ],
)
def test_rs_imm_rd_arithmetic_format_print(op_type, expected_mnemonic, imm_value):
    rs_val = create_ssa_value(Registers.R1)
    op = op_type(rs_val, imm_value, Registers.R2)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert "R1" in assembly
    assert str(imm_value) in assembly
    assert "R2" in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic",
    [
        (AddRRROp, "add"),
        (SubRRROp, "sub"),
        (AndRRROp, "and"),
        (OrRRROp, "or"),
        (XorRRROp, "xor"),
        (AslRRROp, "asl"),
        (AsrRRROp, "asr"),
    ],
)
def test_rs_rs_rd_arithmetic_format_print(op_type, expected_mnemonic):
    rs1_val = create_ssa_value(Registers.R1)
    rs2_val = create_ssa_value(Registers.R2)
    op = op_type(rs1_val, rs2_val, Registers.R3)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert "R1" in assembly
    assert "R2" in assembly
    assert "R3" in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic,imm1,imm2",
    [
        (SetAwgGainIIOp, "set_awg_gain", 0x1000, 0x2000),
        (SetAwgOffsIIOp, "set_awg_offs", 0x0100, 0x0200),
    ],
)
def test_imm_imm_latched_format_print(op_type, expected_mnemonic, imm1, imm2):
    op = op_type(imm1, imm2)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert str(imm1) in assembly
    assert str(imm2) in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic",
    [
        (SetAwgGainRROp, "set_awg_gain"),
        (SetAwgOffsRROp, "set_awg_offs"),
    ],
)
def test_rs_rs_latched_format_print(op_type, expected_mnemonic):
    rs1_val = create_ssa_value(Registers.R0)
    rs2_val = create_ssa_value(Registers.R1)
    op = op_type(rs1_val, rs2_val)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert "R0" in assembly
    assert "R1" in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic,imm1,imm2",
    [
        (FbAcqIqIdIIOp, "fb_acq_iq_id", 1, 2),
        (FbAcqIqIdIIOp, "fb_acq_iq_id", 100, 200),
    ],
)
def test_imm_imm_feedback_format_print(op_type, expected_mnemonic, imm1, imm2):
    op = op_type(imm1, imm2)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert str(imm1) in assembly
    assert str(imm2) in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic,imm1,imm2,imm3,imm4",
    [
        (SetCondIIIIOp, "set_cond", 0, 1, 2, 3),
        (SetCondIIIIOp, "set_cond", 100, 200, 300, 400),
    ],
)
def test_imm_imm_imm_imm_format_print(
    op_type,
    expected_mnemonic,
    imm1,
    imm2,
    imm3,
    imm4,
):
    op = op_type(imm1, imm2, imm3, imm4)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert str(imm1) in assembly
    assert str(imm2) in assembly
    assert str(imm3) in assembly
    assert str(imm4) in assembly


@pytest.mark.parametrize(
    "op_type,expected_mnemonic",
    [
        (SetCondRRRIOp, "set_cond"),
    ],
)
def test_rs_rs_rs_imm_format_print(op_type, expected_mnemonic):
    rs1_val = create_ssa_value(Registers.R0)
    rs2_val = create_ssa_value(Registers.R1)
    rs3_val = create_ssa_value(Registers.R2)
    op = op_type(rs1_val, rs2_val, rs3_val, 42)
    assembly = op.assembly_line()
    assert assembly is not None
    assert assembly.startswith(expected_mnemonic)
    assert "R0" in assembly
    assert "R1" in assembly
    assert "R2" in assembly
    assert "42" in assembly


# region End-to-end module assembly printing


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
        q1_ops.StopIOp(1),
        q1_ops.StopOp(),
        q1_ops.StopROp(r0),
        q1_ops.NopOp(),
        # Jump Instructions
        q1_ops.JmpIOp(100),
        q1_ops.JmpROp(r1),
        q1_ops.JzIOp(101),
        q1_ops.JzROp(r2),
        q1_ops.JnzIOp(102),
        q1_ops.JnzROp(r3),
        q1_ops.JoIOp(103),
        q1_ops.JoROp(r4),
        q1_ops.JnoIOp(104),
        q1_ops.JnoROp(r5),
        q1_ops.JsIOp(105),
        q1_ops.JsROp(r6),
        q1_ops.JnsIOp(106),
        q1_ops.JnsROp(r7),
        q1_ops.JgIOp(107),
        q1_ops.JgROp(r8),
        q1_ops.JlIOp(108),
        q1_ops.JlROp(r9),
        q1_ops.JleIOp(109),
        q1_ops.JleROp(r10),
        q1_ops.JaIOp(110),
        q1_ops.JaROp(r11),
        q1_ops.JaeIOp(111),
        q1_ops.JaeROp(r12),
        q1_ops.JbIOp(112),
        q1_ops.JbROp(r13),
        q1_ops.JbeIOp(113),
        q1_ops.JbeROp(r14),
        q1_ops.JgeRIIOp(r0, 10, 120),
        q1_ops.JgeRIROp(r1, 11, r2),
        q1_ops.JltRIIOp(r3, 12, 121),
        q1_ops.JltRIROp(r4, 13, r5),
        q1_ops.LoopRIOp(Registers.R16, 122),
        q1_ops.LoopRROp(Registers.R17, r6),
        # Arithmetic Instructions
        q1_ops.MoveIROp(14, Registers.R18),
        q1_ops.MoveRROp(r7, Registers.R19),
        q1_ops.NotIROp(15, Registers.R20),
        q1_ops.NotRROp(r8, Registers.R21),
        q1_ops.AddRIROp(r9, 16, Registers.R22),
        q1_ops.AddRRROp(r10, r11, Registers.R23),
        q1_ops.SubRIROp(r12, 17, Registers.R24),
        q1_ops.SubRRROp(r13, r14, Registers.R25),
        q1_ops.AndRIROp(r15, 18, Registers.R26),
        q1_ops.AndRRROp(r0, r1, Registers.R27),
        q1_ops.OrRIROp(r2, 19, Registers.R28),
        q1_ops.OrRRROp(r3, r4, Registers.R29),
        q1_ops.XorRIROp(r5, 20, Registers.R30),
        q1_ops.XorRRROp(r6, r7, Registers.R31),
        q1_ops.AslRIROp(r8, 21, Registers.R32),
        q1_ops.AslRRROp(r9, r10, Registers.R33),
        q1_ops.AsrRIROp(r11, 22, Registers.R34),
        q1_ops.AsrRRROp(r12, r13, Registers.R35),
        # Latched Instructions
        q1_ops.SetCondIIIIOp(1, 2, 3, 4),
        q1_ops.SetCondRRRIOp(r0, r1, r2, 23),
        q1_ops.SetMrkIOp(24),
        q1_ops.SetMrkROp(r3),
        q1_ops.SetFreqIOp(25),
        q1_ops.SetFreqROp(r4),
        q1_ops.ResetPhOp(),
        q1_ops.SetPhIOp(26),
        q1_ops.SetPhROp(r5),
        q1_ops.SetPhDeltaIOp(27),
        q1_ops.SetPhDeltaROp(r6),
        q1_ops.SetAwgGainIIOp(28, 29),
        q1_ops.SetAwgGainRROp(r7, r8),
        q1_ops.SetAwgOffsIIOp(30, 31),
        q1_ops.SetAwgOffsRROp(r9, r10),
        # LINQ Feedback Instructions
        q1_ops.FbAcqIqIdIIOp(32, 33),
        q1_ops.FbAcqIqIdRIOp(r11, 34),
        q1_ops.FbAcqIqShiftIIOp(35, 36),
        q1_ops.FbAcqTbCfgIIIIOp(37, 38, 39, 40),
        q1_ops.FbAcqTbExtraIIIOp(41, 42, 43),
        q1_ops.FbAcqTbIdIIOp(44, 45),
        q1_ops.FbAcqTbIdRIOp(r12, 46),
        q1_ops.FbAcqTbMockIIIIOp(47, 48, 49, 50),
        q1_ops.FbAcqTbValidIIOp(51, 52),
        q1_ops.FbAcqTbValidRIOp(r13, 53),
        q1_ops.FbComCfgIIIIOp(54, 55, 56, 57),
        q1_ops.FbComDataIIIOp(58, 59, 60),
        q1_ops.FbComDataIRIOp(61, r14, 62),
        q1_ops.FbComExtraIIIOp(63, 64, 65),
        q1_ops.FbCmdIIIOp(66, 67, 68),
        q1_ops.FbCmdIRIOp(69, r15, 70),
        q1_ops.FbPopDataIROp(71, Registers.R36),
        q1_ops.FbPullDataRROp(Registers.R37, Registers.R38),
        # Real-time Instructions
        q1_ops.WaitIOp(100),
        q1_ops.WaitROp(r0),
        q1_ops.WaitSyncIOp(200),
        q1_ops.WaitSyncROp(r1),
        q1_ops.WaitTriggerIIOp(1, 50),
        q1_ops.WaitTriggerRROp(r2, r3),
        q1_ops.PlayIIIOp(1, 2, 100),
        q1_ops.PlayRRIOp(r4, r5, 100),
        q1_ops.AcquireIIIOp(0, 1, 100),
        q1_ops.AcquireIRIOp(0, r6, 100),
        q1_ops.AcquireWeighedIIIIIOp(0, 1, 0, 1, 100),
        q1_ops.AcquireWeighedIRRRIOp(0, r7, r8, r9, 100),
        q1_ops.AcquireTtlIIIIOp(0, 0, 1, 100),
        q1_ops.AcquireTtlIRIIOp(0, r10, 1, 100),
        q1_ops.AcquireTimetagsIIIIIOp(0, 0, 1, 0, 100),
        q1_ops.AcquireTimetagsIRIRIOp(0, r11, 1, r12, 100),
        q1_ops.AcquireDigitalIIIOp(0, 0, 100),
        q1_ops.AcquireDigitalIRIOp(0, r13, 100),
        q1_ops.UpdThresIIIOp(0, 1, 100),
        q1_ops.UpdThresIRIOp(0, r14, 100),
        q1_ops.UpdParamIOp(100),
        q1_ops.LatchRstIOp(100),
        q1_ops.LatchRstROp(r15),
        q1_ops.SetLatchEnIIOp(1, 100),
        q1_ops.SetLatchEnRIOp(r0, 100),
    ]


def test_all_ops_assembly_output_matches_q1_code():
    ops = _all_concrete_q1_ops()
    module = ModuleOp(ops)

    asm_lines = q1_code(module).splitlines()
    expected_lines = [op.assembly_line() for op in ops]

    assert asm_lines == expected_lines
    assert {op.name for op in ops} == {op.name for op in Q1.operations}


# endregion
