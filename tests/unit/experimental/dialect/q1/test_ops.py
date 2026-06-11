# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.backend.register_allocatable import (
    HasRegisterConstraintsTrait,
    RegisterConstraints,
)
from xdsl.dialects.builtin import IntAttr, IntegerAttr, StringAttr
from xdsl.traits import Commutative, IsTerminator, Pure
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.q1 import (
    AcquireDigitalIIIOp,
    AcquireDigitalIRIOp,
    AcquireIIIOp,
    AcquireIRIOp,
    AcquireTimetagsIIIIIOp,
    AcquireTimetagsIRIRIOp,
    AcquireTtlIIIIOp,
    AcquireTtlIRIIOp,
    AcquireWeighedIIIIIOp,
    AcquireWeighedIRRRIOp,
    AddRIROp,
    AddRRROp,
    AndRIROp,
    AndRRROp,
    AslRIROp,
    AslRRROp,
    AsrRIROp,
    AsrRRROp,
    DefDirectiveOp,
    FbAcqIqIdIIOp,
    FbAcqIqIdRIOp,
    FbAcqIqShiftIIOp,
    FbAcqTbCfgIIIIOp,
    FbAcqTbExtraIIIOp,
    FbAcqTbIdIIOp,
    FbAcqTbIdRIOp,
    FbAcqTbMockIIIIOp,
    FbAcqTbValidIIOp,
    FbAcqTbValidRIOp,
    FbCmdIIIOp,
    FbCmdIRIOp,
    FbComCfgIIIIOp,
    FbComDataIIIOp,
    FbComDataIRIOp,
    FbComExtraIIIOp,
    FbPopDataIROp,
    FbPullDataRROp,
    IllegalOp,
    IntRegisterType,
    JaeIOp,
    JaeROp,
    JaIOp,
    JaROp,
    JbeIOp,
    JbeROp,
    JbIOp,
    JbROp,
    JgeRIIOp,
    JgeRIROp,
    JgIOp,
    JgROp,
    JleIOp,
    JleROp,
    JlIOp,
    JlROp,
    JltRIIOp,
    JltRIROp,
    JmpIOp,
    JmpROp,
    JnoIOp,
    JnoROp,
    JnsIOp,
    JnsROp,
    JnzIOp,
    JnzROp,
    JoIOp,
    JoROp,
    JsIOp,
    JsROp,
    JzIOp,
    JzROp,
    LabelOp,
    LatchRstIOp,
    LatchRstROp,
    LoopRIOp,
    LoopRROp,
    MoveIROp,
    MoveRROp,
    NopOp,
    NotIROp,
    NotRROp,
    OrRIROp,
    OrRRROp,
    PlayIIIOp,
    PlayRRIOp,
    ResetPhOp,
    SetAwgGainIIOp,
    SetAwgGainRROp,
    SetAwgOffsIIOp,
    SetAwgOffsRROp,
    SetCondIIIIOp,
    SetCondRRRIOp,
    SetFreqIOp,
    SetFreqROp,
    SetLatchEnIIOp,
    SetLatchEnRIOp,
    SetMrkIOp,
    SetMrkROp,
    SetPhDeltaIOp,
    SetPhDeltaROp,
    SetPhIOp,
    SetPhROp,
    StopIOp,
    StopOp,
    StopROp,
    SubRIROp,
    SubRRROp,
    UpdParamIOp,
    UpdThresIIIOp,
    UpdThresIRIOp,
    WaitIOp,
    WaitROp,
    WaitSyncIOp,
    WaitSyncROp,
    WaitTriggerIIOp,
    WaitTriggerRROp,
    XorRIROp,
    XorRRROp,
)
from qat.experimental.dialect.q1.ir.attrs import LabelAttr
from qat.experimental.dialect.q1.ir.imm_desc import ui32
from qat.experimental.dialect.q1.ir.reg_desc import Registers


def test_label_op():
    op = LabelOp("loop_start", comment="branch target")

    assert op.reference == LabelAttr("loop_start")
    assert op.comment == StringAttr("branch target")
    assert op.assembly_mnemonic() == ""
    assert op.assembly_line_args() == ()

    line = op.assembly_line()
    assert line is not None
    assert line.startswith("loop_start:")
    assert line.endswith("# branch target")

    op_no_comment = LabelOp(LabelAttr("loop_end"))
    line_no_comment = op_no_comment.assembly_line()
    assert line_no_comment is not None
    assert line_no_comment.startswith("loop_end:")


@pytest.mark.parametrize(
    "alias_input,value_input,comment_input,alias_str,value_str,comment_str",
    [
        ("my_alias", "R1", "note", "my_alias", "R1", "note"),
        (
            StringAttr("loop_count"),
            StringAttr("5"),
            StringAttr("inline"),
            "loop_count",
            "5",
            "inline",
        ),
        ("threshold", "10", None, "threshold", "10", None),
    ],
)
def test_def_directive_op(
    alias_input,
    value_input,
    comment_input,
    alias_str,
    value_str,
    comment_str,
):
    op = DefDirectiveOp(alias_input, value_input, comment=comment_input)

    assert op.alias == StringAttr(alias_str)
    assert op.value == StringAttr(value_str)
    assert op.comment == (StringAttr(comment_str) if comment_str is not None else None)
    assert op.assembly_mnemonic() == ".DEF"
    assert op.assembly_line_args() == (StringAttr(alias_str), StringAttr(value_str))

    line = op.assembly_line()
    assert line is not None
    if comment_str is None:
        assert line == f".DEF {alias_str} {value_str}"
    else:
        assert line.startswith(f".DEF {alias_str} {value_str}")
        assert line.endswith(f"# {comment_str}")
    assert "," not in line


@pytest.mark.parametrize(
    "op_type,mnemonic,comment,expected_traits",
    [
        (
            LoopRIOp,
            "loop",
            "Jump to immediate address while greater or equals",
            (IsTerminator, HasRegisterConstraintsTrait),
        ),
    ],
)
def test_rd_imm_format(op_type, mnemonic, comment, expected_traits):
    a_val = 100
    op = op_type(Registers.R10, a_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert isinstance(op.rd.type, IntRegisterType)
    assert op.rd.type.is_allocated
    assert op.rd.type.index == Registers.R10.index
    assert op.assembly_line_args() == (op.rd, op.imm)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize(
    "op_type,mnemonic,comment,expected_traits",
    [
        (
            LoopRROp,
            "loop",
            "Jump to register address while greater or equals",
            (IsTerminator, HasRegisterConstraintsTrait),
        ),
    ],
)
def test_rd_rs_format(op_type, mnemonic, comment, expected_traits):
    a_val = create_ssa_value(Registers.R2)
    op = op_type(Registers.R10, a_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs.type, IntRegisterType)
    assert op.rs.type.is_allocated
    assert op.rs.type.index == a_val.type.index
    assert isinstance(op.rd.type, IntRegisterType)
    assert op.rd.type.is_allocated
    assert op.rd.type.index == Registers.R10.index
    assert op.assembly_line_args() == (op.rd, op.rs)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (JgeRIIOp, "jge", (IsTerminator, HasRegisterConstraintsTrait)),
        (JltRIIOp, "jlt", (IsTerminator, HasRegisterConstraintsTrait)),
    ],
)
def test_rs_imm_imm_format(op_type, mnemonic, comment, expected_traits):
    a_val = create_ssa_value(Registers.R1)
    b_val = 1000
    d_val = 100

    op = op_type(a_val, b_val, d_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=b_val), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=d_val), value_type=ui32)
    assert isinstance(op.rs.type, IntRegisterType)
    assert op.assembly_line_args() == (op.rs, op.imm1, op.imm2)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)
    op2 = op_type(
        a_val, IntegerAttr(b_val, ui32), IntegerAttr(d_val, ui32), comment=comment
    )
    assert op2.imm1 == IntegerAttr(b_val, ui32)
    assert op2.imm2 == IntegerAttr(d_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (JgeRIROp, "jge", (IsTerminator, HasRegisterConstraintsTrait)),
        (JltRIROp, "jlt", (IsTerminator, HasRegisterConstraintsTrait)),
    ],
)
def test_rs_imm_rs_format(op_type, mnemonic, comment, expected_traits):
    a_val = create_ssa_value(Registers.R1)
    b_val = 1000
    c_val = create_ssa_value(Registers.R2)
    op = op_type(a_val, b_val, c_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=b_val), value_type=ui32)
    assert isinstance(op.rs1.type, IntRegisterType)
    assert isinstance(op.rs2.type, IntRegisterType)
    assert op.assembly_line_args() == (op.rs1, op.imm, op.rs2)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)
    op2 = op_type(a_val, IntegerAttr(b_val, ui32), c_val, comment=comment)
    assert op2.imm == IntegerAttr(b_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (IllegalOp, "illegal", (IsTerminator, HasRegisterConstraintsTrait)),
        (StopOp, "stop", (IsTerminator, HasRegisterConstraintsTrait)),
        (NopOp, "nop", (Pure, HasRegisterConstraintsTrait)),
        (ResetPhOp, "reset_ph", (HasRegisterConstraintsTrait,)),
    ],
)
def test_nullary_format(op_type, mnemonic, comment, expected_traits):
    op = op_type(comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.assembly_line_args() == ()
    assert op.assembly_line() is not None
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (StopIOp, "stop", (IsTerminator, HasRegisterConstraintsTrait)),
        (JmpIOp, "jmp", (IsTerminator, HasRegisterConstraintsTrait)),
        (JzIOp, "jz", (IsTerminator, HasRegisterConstraintsTrait)),
        (JnzIOp, "jnz", (IsTerminator, HasRegisterConstraintsTrait)),
        (JoIOp, "jo", (IsTerminator, HasRegisterConstraintsTrait)),
        (JnoIOp, "jno", (IsTerminator, HasRegisterConstraintsTrait)),
        (JsIOp, "js", (IsTerminator, HasRegisterConstraintsTrait)),
        (JnsIOp, "jns", (IsTerminator, HasRegisterConstraintsTrait)),
        (JgIOp, "jg", (IsTerminator, HasRegisterConstraintsTrait)),
        (JlIOp, "jl", (IsTerminator, HasRegisterConstraintsTrait)),
        (JleIOp, "jle", (IsTerminator, HasRegisterConstraintsTrait)),
        (JaIOp, "ja", (IsTerminator, HasRegisterConstraintsTrait)),
        (JaeIOp, "jae", (IsTerminator, HasRegisterConstraintsTrait)),
        (JbIOp, "jb", (IsTerminator, HasRegisterConstraintsTrait)),
        (JbeIOp, "jbe", (IsTerminator, HasRegisterConstraintsTrait)),
        (SetMrkIOp, "set_mrk", (HasRegisterConstraintsTrait,)),
        (SetFreqIOp, "set_freq", (HasRegisterConstraintsTrait,)),
        (SetPhIOp, "set_ph", (HasRegisterConstraintsTrait,)),
        (SetPhDeltaIOp, "set_ph_delta", (HasRegisterConstraintsTrait,)),
    ],
)
def test_imm_format(op_type, mnemonic, comment, expected_traits):
    a_val = 1024
    op = op_type(a_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert op.assembly_line_args() == (op.imm,)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)
    op2 = op_type(IntegerAttr(a_val, ui32), comment=comment)
    assert op2.imm == IntegerAttr(a_val, ui32)


@pytest.mark.parametrize(
    "op_type,mnemonic,comment,expected_traits",
    [
        (StopROp, "stop", "Stop with status", (IsTerminator, HasRegisterConstraintsTrait)),
        (JmpROp, "jmp", "Unconditional jump", (IsTerminator, HasRegisterConstraintsTrait)),
        (JzROp, "jz", "Jump if zero", (IsTerminator, HasRegisterConstraintsTrait)),
        (JnzROp, "jnz", "Jump if not zero", (IsTerminator, HasRegisterConstraintsTrait)),
        (JoROp, "jo", "Jump if overflow", (IsTerminator, HasRegisterConstraintsTrait)),
        (
            JnoROp,
            "jno",
            "Jump if not overflow",
            (IsTerminator, HasRegisterConstraintsTrait),
        ),
        (JsROp, "js", "Jump if negative", (IsTerminator, HasRegisterConstraintsTrait)),
        (
            JnsROp,
            "jns",
            "Jump if not negative",
            (IsTerminator, HasRegisterConstraintsTrait),
        ),
        (JgROp, "jg", "Jump if greater than", (IsTerminator, HasRegisterConstraintsTrait)),
        (JlROp, "jl", "Jump if less than", (IsTerminator, HasRegisterConstraintsTrait)),
        (
            JleROp,
            "jle",
            "Jump if less than or equal",
            (IsTerminator, HasRegisterConstraintsTrait),
        ),
        (JaROp, "ja", "Jump if above", (IsTerminator, HasRegisterConstraintsTrait)),
        (
            JaeROp,
            "jae",
            "Jump if above or equal",
            (IsTerminator, HasRegisterConstraintsTrait),
        ),
        (JbROp, "jb", "Jump if below", (IsTerminator, HasRegisterConstraintsTrait)),
        (
            JbeROp,
            "jbe",
            "Jump if below or equal",
            (IsTerminator, HasRegisterConstraintsTrait),
        ),
        (
            SetMrkROp,
            "set_mrk",
            "Set marker output channels",
            (HasRegisterConstraintsTrait,),
        ),
        (SetFreqROp, "set_freq", "Set frequency", (HasRegisterConstraintsTrait,)),
        (SetPhROp, "set_ph", "Set phase", (HasRegisterConstraintsTrait,)),
        (SetPhDeltaROp, "set_ph_delta", "Set phase offset", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rs_format(op_type, mnemonic, comment, expected_traits):
    a_val = create_ssa_value(Registers.R32)
    op = op_type(a_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs.type, IntRegisterType)
    assert op.assembly_line_args() == (op.rs,)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (MoveIROp, "move", (Pure, HasRegisterConstraintsTrait)),
        (NotIROp, "not", (Pure, HasRegisterConstraintsTrait)),
        (FbPopDataIROp, "fb_pop_data", (HasRegisterConstraintsTrait,)),
    ],
)
def test_imm_rd_format(op_type, mnemonic, comment, expected_traits):
    a_val = 101

    if op_type is FbPopDataIROp:
        op = op_type(a_val, destination=Registers.R0)
    else:
        op = op_type(a_val, rd=Registers.R0)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.imm, IntegerAttr)
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert isinstance(op.rd.type, IntRegisterType)
    assert op.rd.type.index == Registers.R0.index
    assert op.assembly_line_args() == (op.imm, op.rd)
    if op_type is FbPopDataIROp:
        assert op.id == op.imm
        assert op.destination == op.rd
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)
    if op_type is FbPopDataIROp:
        op2 = op_type(IntegerAttr(a_val, ui32), destination=Registers.R0, comment=comment)
    else:
        op2 = op_type(IntegerAttr(a_val, ui32), rd=Registers.R0, comment=comment)
    assert op2.imm == IntegerAttr(a_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize("expected_traits", [(HasRegisterConstraintsTrait,)])
def test_fb_pull_data_rd_rd_format(comment, expected_traits):
    op = FbPullDataRROp(id=Registers.R0, destination=Registers.R1, comment=comment)
    assert op.assembly_mnemonic() == "fb_pull_data"
    assert isinstance(op.rd1.type, IntRegisterType)
    assert isinstance(op.rd2.type, IntRegisterType)
    assert op.rd1.type.index == Registers.R0.index
    assert op.rd2.type.index == Registers.R1.index
    assert op.assembly_line_args() == (op.rd1, op.rd2)
    assert op.destination_id == op.rd1
    assert op.destination == op.rd2
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "opt_type,mnemonic,expected_traits",
    [
        (MoveRROp, "move", (Pure, HasRegisterConstraintsTrait)),
        (NotRROp, "not", (Pure, HasRegisterConstraintsTrait)),
    ],
)
def test_rs_rd_format(opt_type, mnemonic, comment, expected_traits):
    a_val = create_ssa_value(Registers.R62)

    op = opt_type(a_val, rd=Registers.R63)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs.type, IntRegisterType)
    assert op.rs.type.index == Registers.R62.index
    assert isinstance(op.rd.type, IntRegisterType)
    assert op.rd.type.index == Registers.R63.index
    assert op.assembly_line_args() == (op.rs, op.rd)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)
    op2 = opt_type(create_ssa_value(Registers.R62), rd=Registers.R63, comment=comment)
    assert op2.rd.type.index == Registers.R63.index


@pytest.mark.parametrize(
    "opt_type,mnemonic,expected_traits",
    [
        (AddRRROp, "add", (Pure, Commutative, HasRegisterConstraintsTrait)),
        (SubRRROp, "sub", (Pure, HasRegisterConstraintsTrait)),
        (AndRRROp, "and", (Pure, Commutative, HasRegisterConstraintsTrait)),
        (OrRRROp, "or", (Pure, Commutative, HasRegisterConstraintsTrait)),
        (XorRRROp, "xor", (Pure, Commutative, HasRegisterConstraintsTrait)),
        (AslRRROp, "asl", (Pure, HasRegisterConstraintsTrait)),
        (AsrRRROp, "asr", (Pure, HasRegisterConstraintsTrait)),
    ],
)
def test_rs_rs_rd_format(opt_type, mnemonic, expected_traits):
    r1 = create_ssa_value(Registers.R1)
    r2 = create_ssa_value(Registers.R2)

    op = opt_type(r1, r2, rd=Registers.R3)
    assert op.assembly_mnemonic() == mnemonic

    assert r1.type is op.rs1.type
    assert r2.type is op.rs2.type

    assert isinstance(r1.type, IntRegisterType)
    assert r1.type.register_name.data == "R1"
    assert r1.type.index == Registers.R1.index

    assert isinstance(r2.type, IntRegisterType)
    assert r2.type.register_name.data == "R2"
    assert r2.type.index == Registers.R2.index

    assert isinstance(op.rd.type, IntRegisterType)
    assert op.rd.type.register_name.data == "R3"
    assert op.rd.type.index == Registers.R3.index
    assert op.assembly_line_args() == (op.rs1, op.rs2, op.rd)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (AddRIROp, "add", (Pure, HasRegisterConstraintsTrait)),
        (SubRIROp, "sub", (Pure, HasRegisterConstraintsTrait)),
        (AndRIROp, "and", (Pure, HasRegisterConstraintsTrait)),
        (OrRIROp, "or", (Pure, HasRegisterConstraintsTrait)),
        (XorRIROp, "xor", (Pure, HasRegisterConstraintsTrait)),
        (AslRIROp, "asl", (Pure, HasRegisterConstraintsTrait)),
        (AsrRIROp, "asr", (Pure, HasRegisterConstraintsTrait)),
    ],
)
def test_rs_imm_rd_format(op_type, mnemonic, comment, expected_traits):
    r1 = create_ssa_value(Registers.R1)
    assert isinstance(r1.type, IntRegisterType)
    assert r1.type.register_name.data == "R1"
    assert r1.type.index == Registers.R1.index

    a_val = 101

    op = op_type(r1, a_val, rd=Registers.R3)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.imm, IntegerAttr)
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert isinstance(op.rd.type, IntRegisterType)
    assert op.rd.type.register_name.data == "R3"
    assert op.rd.type.index == Registers.R3.index
    assert op.assembly_line_args() == (op.rs, op.imm, op.rd)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)
    op2 = op_type(r1, IntegerAttr(a_val, ui32), rd=Registers.R3, comment=comment)
    assert op2.imm == IntegerAttr(a_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (FbAcqTbIdRIOp, "fb_acq_tb_id", (HasRegisterConstraintsTrait,)),
        (FbAcqTbValidRIOp, "fb_acq_tb_valid", (HasRegisterConstraintsTrait,)),
        (FbAcqIqIdRIOp, "fb_acq_iq_id", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rs_imm_format(op_type, mnemonic, comment, expected_traits):
    r1 = create_ssa_value(Registers.R1)
    a_val = 101

    op = op_type(r1, IntegerAttr(a_val, ui32), comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs.type, IntRegisterType)
    assert isinstance(op.properties["imm"], IntegerAttr)
    assert op.properties["imm"] == IntegerAttr(a_val, ui32)
    assert op.operands[0] is op.rs
    assert op.assembly_line_args() == (op.rs, op.imm)
    if op_type in (FbAcqTbIdRIOp, FbAcqIqIdRIOp):
        assert op.id == op.rs
    if op_type is FbAcqTbValidRIOp:
        assert op.tb_valid == op.rs
    assert op.duration == op.imm
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)

    op2 = op_type(r1, IntegerAttr(a_val, ui32), comment=comment)
    assert op2.properties["imm"] == IntegerAttr(a_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (SetCondIIIIOp, "set_cond", (HasRegisterConstraintsTrait,)),
    ],
)
def test_imm_imm_imm_imm_format(op_type, mnemonic, comment, expected_traits):
    a_val = 1
    b_val = 1
    c_val = 1
    d_val = 100

    op = op_type(a_val, b_val, c_val, d_val)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.imm1, IntegerAttr)
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)

    assert isinstance(op.imm2, IntegerAttr)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=b_val), value_type=ui32)

    assert isinstance(op.imm3, IntegerAttr)
    assert op.properties["imm3"] == IntegerAttr(value=IntAttr(data=c_val), value_type=ui32)

    assert isinstance(op.imm4, IntegerAttr)
    assert op.properties["imm4"] == IntegerAttr(value=IntAttr(data=d_val), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.imm2, op.imm3, op.imm4)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)
    op2 = op_type(
        IntegerAttr(a_val, ui32),
        IntegerAttr(b_val, ui32),
        IntegerAttr(c_val, ui32),
        IntegerAttr(d_val, ui32),
        comment=comment,
    )
    assert op2.imm4 == IntegerAttr(d_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (FbComDataIIIOp, "fb_com_data", (HasRegisterConstraintsTrait,)),
        (FbCmdIIIOp, "fb_cmd", (HasRegisterConstraintsTrait,)),
        (FbComExtraIIIOp, "fb_com_extra", (HasRegisterConstraintsTrait,)),
        (FbAcqTbExtraIIIOp, "fb_acq_tb_extra", (HasRegisterConstraintsTrait,)),
    ],
)
def test_imm_imm_imm_format(op_type, mnemonic, comment, expected_traits):
    a_val = 1
    b_val = 2
    c_val = 100

    op = op_type(a_val, b_val, c_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.imm1, IntegerAttr)
    assert isinstance(op.imm2, IntegerAttr)
    assert isinstance(op.imm3, IntegerAttr)
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=b_val), value_type=ui32)
    assert op.properties["imm3"] == IntegerAttr(value=IntAttr(data=c_val), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.imm2, op.imm3)
    if op_type in (FbComDataIIIOp, FbCmdIIIOp):
        assert op.id == op.imm1
        assert op.duration == op.imm3
    if op_type is FbComDataIIIOp:
        assert op.value == op.imm2
    if op_type is FbCmdIIIOp:
        assert op.value == op.imm2
    if op_type in (FbComExtraIIIOp, FbAcqTbExtraIIIOp):
        assert op.extra_vld == op.imm1
        assert op.extra == op.imm2
        assert op.duration == op.imm3
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)

    op2 = op_type(
        IntegerAttr(a_val, ui32),
        IntegerAttr(b_val, ui32),
        IntegerAttr(c_val, ui32),
        comment=comment,
    )
    assert op2.imm3 == IntegerAttr(c_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize("expected_traits", [(HasRegisterConstraintsTrait,)])
def test_imm_rs_imm_format(comment, expected_traits):
    r1 = create_ssa_value(Registers.R1)
    a_val = 1
    c_val = 100

    op = FbComDataIRIOp(
        IntegerAttr(a_val, ui32), r1, IntegerAttr(c_val, ui32), comment=comment
    )
    assert op.assembly_mnemonic() == "fb_com_data"
    assert isinstance(op.rs.type, IntRegisterType)
    assert isinstance(op.imm1, IntegerAttr)
    assert isinstance(op.imm2, IntegerAttr)
    assert op.properties["imm1"] == IntegerAttr(a_val, ui32)
    assert op.properties["imm2"] == IntegerAttr(c_val, ui32)
    assert op.assembly_line_args() == (op.imm1, op.rs, op.imm2)
    assert op.id == op.imm1
    assert op.value == op.rs
    assert op.duration == op.imm2
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)

    op2 = FbComDataIRIOp(
        IntegerAttr(a_val, ui32), r1, IntegerAttr(c_val, ui32), comment=comment
    )
    assert op2.imm1 == IntegerAttr(a_val, ui32)
    assert op2.imm2 == IntegerAttr(c_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize("expected_traits", [(HasRegisterConstraintsTrait,)])
def test_fb_cmd_imm_rs_imm_format(comment, expected_traits):
    r1 = create_ssa_value(Registers.R1)
    a_val = 1
    c_val = 100

    op = FbCmdIRIOp(IntegerAttr(a_val, ui32), r1, IntegerAttr(c_val, ui32), comment=comment)
    assert op.assembly_mnemonic() == "fb_cmd"
    assert isinstance(op.rs.type, IntRegisterType)
    assert isinstance(op.imm1, IntegerAttr)
    assert isinstance(op.imm2, IntegerAttr)
    assert op.properties["imm1"] == IntegerAttr(a_val, ui32)
    assert op.properties["imm2"] == IntegerAttr(c_val, ui32)
    assert op.assembly_line_args() == (op.imm1, op.rs, op.imm2)
    assert op.id == op.imm1
    assert op.value == op.rs
    assert op.duration == op.imm2
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)

    op2 = FbCmdIRIOp(
        IntegerAttr(a_val, ui32), r1, IntegerAttr(c_val, ui32), comment=comment
    )
    assert op2.imm1 == IntegerAttr(a_val, ui32)
    assert op2.imm2 == IntegerAttr(c_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (FbComCfgIIIIOp, "fb_com_cfg", (HasRegisterConstraintsTrait,)),
        (FbAcqTbCfgIIIIOp, "fb_acq_tb_cfg", (HasRegisterConstraintsTrait,)),
        (FbAcqTbMockIIIIOp, "fb_acq_tb_mock", (HasRegisterConstraintsTrait,)),
    ],
)
def test_fb_imm_imm_imm_imm_format(op_type, mnemonic, comment, expected_traits):
    a_val = 1
    b_val = 2
    c_val = 3
    d_val = 100

    op = op_type(a_val, b_val, c_val, d_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.imm1, IntegerAttr)
    assert isinstance(op.imm2, IntegerAttr)
    assert isinstance(op.imm3, IntegerAttr)
    assert isinstance(op.imm4, IntegerAttr)
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=b_val), value_type=ui32)
    assert op.properties["imm3"] == IntegerAttr(value=IntAttr(data=c_val), value_type=ui32)
    assert op.properties["imm4"] == IntegerAttr(value=IntAttr(data=d_val), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.imm2, op.imm3, op.imm4)
    if op_type in (FbComCfgIIIIOp, FbAcqTbCfgIIIIOp):
        assert op.wc == op.imm1
        assert op.shift == op.imm2
        assert op.length == op.imm3
        assert op.duration == op.imm4
    if op_type is FbAcqTbMockIIIIOp:
        assert op.mock_en == op.imm1
        assert op.mock_vld == op.imm2
        assert op.mock_data == op.imm3
        assert op.duration == op.imm4
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)

    op2 = op_type(
        IntegerAttr(a_val, ui32),
        IntegerAttr(b_val, ui32),
        IntegerAttr(c_val, ui32),
        IntegerAttr(d_val, ui32),
        comment=comment,
    )
    assert op2.imm4 == IntegerAttr(d_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (FbAcqTbIdIIOp, "fb_acq_tb_id", (HasRegisterConstraintsTrait,)),
        (FbAcqTbValidIIOp, "fb_acq_tb_valid", (HasRegisterConstraintsTrait,)),
        (FbAcqIqIdIIOp, "fb_acq_iq_id", (HasRegisterConstraintsTrait,)),
        (FbAcqIqShiftIIOp, "fb_acq_iq_shift", (HasRegisterConstraintsTrait,)),
    ],
)
def test_fb_imm_imm_format(op_type, mnemonic, comment, expected_traits):
    a_val = 100
    b_val = 200

    op = op_type(a_val, b_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.imm1, IntegerAttr)
    assert isinstance(op.imm2, IntegerAttr)
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=b_val), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.imm2)
    if op_type in (FbAcqTbIdIIOp, FbAcqIqIdIIOp):
        assert op.id == op.imm1
        assert op.duration == op.imm2
    if op_type is FbAcqTbValidIIOp:
        assert op.tb_valid == op.imm1
        assert op.duration == op.imm2
    if op_type is FbAcqIqShiftIIOp:
        assert op.shift == op.imm1
        assert op.duration == op.imm2
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)

    op2 = op_type(IntegerAttr(a_val, ui32), IntegerAttr(b_val, ui32), comment=comment)
    assert op2.imm1 == IntegerAttr(a_val, ui32)
    assert op2.imm2 == IntegerAttr(b_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (
            SetCondRRRIOp,
            "set_cond",
            (HasRegisterConstraintsTrait,),
        ),
    ],
)
def test_rs_rs_rs_imm_format(op_type, mnemonic, comment, expected_traits):
    r1 = create_ssa_value(Registers.R1)
    r2 = create_ssa_value(Registers.R2)
    r3 = create_ssa_value(Registers.R3)
    d_val = 100

    op = op_type(r1, r2, r3, d_val)
    assert op.assembly_mnemonic() == mnemonic

    assert isinstance(op.rs1.type, IntRegisterType)
    assert r1.type.register_name.data == "R1"
    assert r1.type.index == Registers.R1.index

    assert isinstance(op.rs2.type, IntRegisterType)
    assert r2.type.register_name.data == "R2"
    assert r2.type.index == Registers.R2.index

    assert isinstance(op.rs3.type, IntRegisterType)
    assert r3.type.register_name.data == "R3"
    assert r3.type.index == Registers.R3.index

    assert isinstance(op.imm, IntegerAttr)
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=d_val), value_type=ui32)
    assert op.assembly_line_args() == (op.rs1, op.rs2, op.rs3, op.imm)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)
    op2 = op_type(r1, r2, r3, IntegerAttr(d_val, ui32), comment=comment)
    assert op2.imm == IntegerAttr(d_val, ui32)
    assert isinstance(op2.get_register_constraints(), RegisterConstraints)


@pytest.mark.parametrize(
    "op_type,mnemonic,comment,expected_traits",
    [
        (SetAwgGainRROp, "set_awg_gain", "Set AWG gain", (HasRegisterConstraintsTrait,)),
        (SetAwgOffsRROp, "set_awg_offs", "Set AWG offset", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rs_rs_format(op_type, mnemonic, comment, expected_traits):
    r1 = create_ssa_value(Registers.R1)
    r2 = create_ssa_value(Registers.R2)

    op = op_type(r1, r2, comment=comment)
    assert op.assembly_mnemonic() == mnemonic

    assert r1.type is op.rs1.type
    assert isinstance(r1.type, IntRegisterType)

    assert r2.type is op.rs2.type
    assert isinstance(r2.type, IntRegisterType)
    assert op.assembly_line_args() == (op.rs1, op.rs2)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (SetAwgGainIIOp, "set_awg_gain", (HasRegisterConstraintsTrait,)),
        (SetAwgOffsIIOp, "set_awg_offs", (HasRegisterConstraintsTrait,)),
    ],
)
def test_imm_imm_format(op_type, mnemonic, comment, expected_traits):
    a_val = 100
    b_val = 200

    op = op_type(a_val, b_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.imm1, IntegerAttr)
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert isinstance(op.imm2, IntegerAttr)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=b_val), value_type=ui32)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)
    op2 = op_type(IntegerAttr(a_val, ui32), IntegerAttr(b_val, ui32), comment=comment)
    assert op2.imm1 == IntegerAttr(a_val, ui32)
    assert op2.imm2 == IntegerAttr(b_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (LatchRstIOp, "latch_rst", (HasRegisterConstraintsTrait,)),
        (
            WaitIOp,
            "wait",
            (
                Pure(),
                HasRegisterConstraintsTrait,
            ),
        ),
        (
            WaitSyncIOp,
            "wait_sync",
            (
                Pure(),
                HasRegisterConstraintsTrait,
            ),
        ),
        (UpdParamIOp, "upd_param", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rt_imm_format(op_type, mnemonic, comment, expected_traits):
    imm1 = 300

    op = op_type(imm1, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=imm1), value_type=ui32)
    assert op.assembly_line_args() == (op.imm,)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (SetLatchEnIIOp, "set_latch_en", (HasRegisterConstraintsTrait,)),
        (
            WaitTriggerIIOp,
            "wait_trigger",
            (
                Pure(),
                HasRegisterConstraintsTrait,
            ),
        ),
    ],
)
def test_rt_imm_imm_format(op_type, mnemonic, comment, expected_traits):
    imm1 = 300
    imm2 = 400

    op = op_type(imm1, imm2, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=imm1), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=imm2), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.imm2)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (PlayIIIOp, "play", (HasRegisterConstraintsTrait,)),
        (AcquireIIIOp, "acquire", (HasRegisterConstraintsTrait,)),
        (AcquireDigitalIIIOp, "acquire_digital", (HasRegisterConstraintsTrait,)),
        (UpdThresIIIOp, "upd_thres", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rt_imm_imm_imm_format(op_type, mnemonic, comment, expected_traits):
    imm1 = 300
    imm2 = 500
    imm3 = 700

    op = op_type(imm1, imm2, imm3, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=imm1), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=imm2), value_type=ui32)
    assert op.properties["imm3"] == IntegerAttr(value=IntAttr(data=imm3), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.imm2, op.imm3)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (AcquireTtlIIIIOp, "acquire_ttl", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rt_imm_imm_imm_imm_format(op_type, mnemonic, comment, expected_traits):
    imm1 = 300
    imm2 = 400
    imm3 = 500
    imm4 = 600

    op = op_type(imm1, imm2, imm3, imm4, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=imm1), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=imm2), value_type=ui32)
    assert op.properties["imm3"] == IntegerAttr(value=IntAttr(data=imm3), value_type=ui32)
    assert op.properties["imm4"] == IntegerAttr(value=IntAttr(data=imm4), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.imm2, op.imm3, op.imm4)
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (AcquireWeighedIIIIIOp, "acquire_weighed", (HasRegisterConstraintsTrait,)),
        (AcquireTimetagsIIIIIOp, "acquire_timetags", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rt_imm_imm_imm_imm_imm_format(op_type, mnemonic, comment, expected_traits):
    imm1 = 300
    imm2 = 400
    imm3 = 500
    imm4 = 600
    imm5 = 700

    op = op_type(imm1, imm2, imm3, imm4, imm5, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=imm1), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=imm2), value_type=ui32)
    assert op.properties["imm3"] == IntegerAttr(value=IntAttr(data=imm3), value_type=ui32)
    assert op.properties["imm4"] == IntegerAttr(value=IntAttr(data=imm4), value_type=ui32)
    assert op.properties["imm5"] == IntegerAttr(value=IntAttr(data=imm5), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.imm2, op.imm3, op.imm4, op.imm5)

    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (LatchRstROp, "latch_rst", (HasRegisterConstraintsTrait,)),
        (
            WaitROp,
            "wait",
            (
                Pure(),
                HasRegisterConstraintsTrait,
            ),
        ),
        (
            WaitSyncROp,
            "wait_sync",
            (
                Pure(),
                HasRegisterConstraintsTrait,
            ),
        ),
    ],
)
def test_rt_rs_format(op_type, mnemonic, comment, expected_traits):
    r1 = create_ssa_value(Registers.R1)

    op = op_type(r1, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs.type, IntRegisterType)
    assert r1.type.register_name.data == "R1"
    assert r1.type.index == Registers.R1.index
    assert op.assembly_line_args() == (op.rs,)
    assert len(expected_traits) == len(op.traits.traits)

    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (
            WaitTriggerRROp,
            "wait_trigger",
            (
                Pure(),
                HasRegisterConstraintsTrait,
            ),
        ),
    ],
)
def test_rt_rs_rs_format(op_type, mnemonic, comment, expected_traits):
    r1 = create_ssa_value(Registers.R1)
    r2 = create_ssa_value(Registers.R2)

    op = op_type(r1, r2, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs1.type, IntRegisterType)
    assert r1.type.register_name.data == "R1"
    assert r1.type.index == Registers.R1.index
    assert isinstance(op.rs2.type, IntRegisterType)
    assert r2.type.register_name.data == "R2"
    assert r2.type.index == Registers.R2.index
    assert op.assembly_line_args() == (op.rs1, op.rs2)
    assert len(expected_traits) == len(op.traits.traits)

    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (SetLatchEnRIOp, "set_latch_en", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rt_rs_imm_format(op_type, mnemonic, comment, expected_traits):
    r1 = create_ssa_value(Registers.R1)
    imm1 = 300

    op = op_type(r1, imm1, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs.type, IntRegisterType)
    assert r1.type.register_name.data == "R1"
    assert r1.type.index == Registers.R1.index
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=imm1), value_type=ui32)
    assert op.assembly_line_args() == (op.rs, op.imm)
    assert len(expected_traits) == len(op.traits.traits)

    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (PlayRRIOp, "play", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rt_rs_rs_imm_format(op_type, mnemonic, comment, expected_traits):
    r1 = create_ssa_value(Registers.R1)
    r2 = create_ssa_value(Registers.R2)
    imm1 = 300

    op = op_type(r1, r2, imm1, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs1.type, IntRegisterType)
    assert r1.type.register_name.data == "R1"
    assert r1.type.index == Registers.R1.index
    assert isinstance(op.rs2.type, IntRegisterType)
    assert r2.type.register_name.data == "R2"
    assert r2.type.index == Registers.R2.index
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=imm1), value_type=ui32)
    assert op.assembly_line_args() == (op.rs1, op.rs2, op.imm)
    assert len(expected_traits) == len(op.traits.traits)

    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (AcquireTtlIRIIOp, "acquire_ttl", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rt_imm_rs_imm_imm_format(op_type, mnemonic, comment, expected_traits):
    imm1 = 100
    r1 = create_ssa_value(Registers.R1)
    imm2 = 300
    imm3 = 500

    op = op_type(imm1, r1, imm2, imm3, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs.type, IntRegisterType)
    assert r1.type.register_name.data == "R1"
    assert r1.type.index == Registers.R1.index
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=imm1), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=imm2), value_type=ui32)
    assert op.properties["imm3"] == IntegerAttr(value=IntAttr(data=imm3), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.rs, op.imm2, op.imm3)
    assert len(expected_traits) == len(op.traits.traits)

    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (AcquireIRIOp, "acquire", (HasRegisterConstraintsTrait,)),
        (AcquireDigitalIRIOp, "acquire_digital", (HasRegisterConstraintsTrait,)),
        (UpdThresIRIOp, "upd_thres", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rt_imm_rs_imm_format(op_type, mnemonic, comment, expected_traits):
    imm1 = 100
    r1 = create_ssa_value(Registers.R1)
    imm2 = 300

    op = op_type(imm1, r1, imm2, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs.type, IntRegisterType)
    assert r1.type.register_name.data == "R1"
    assert r1.type.index == Registers.R1.index
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=imm1), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=imm2), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.rs, op.imm2)
    assert len(expected_traits) == len(op.traits.traits)

    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (AcquireTimetagsIRIRIOp, "acquire_timetags", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rt_imm_rs_imm_rs_imm_format(op_type, mnemonic, comment, expected_traits):
    imm1 = 100
    r1 = create_ssa_value(Registers.R1)
    imm2 = 300
    r2 = create_ssa_value(Registers.R2)
    imm3 = 500

    op = op_type(imm1, r1, imm2, r2, imm3, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs1.type, IntRegisterType)
    assert r1.type.register_name.data == "R1"
    assert r1.type.index == Registers.R1.index
    assert isinstance(op.rs2.type, IntRegisterType)
    assert r2.type.register_name.data == "R2"
    assert r2.type.index == Registers.R2.index
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=imm1), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=imm2), value_type=ui32)
    assert op.properties["imm3"] == IntegerAttr(value=IntAttr(data=imm3), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.rs1, op.imm2, op.rs2, op.imm3)
    assert len(expected_traits) == len(op.traits.traits)

    for trait in expected_traits:
        assert op.has_trait(trait)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic,expected_traits",
    [
        (AcquireWeighedIRRRIOp, "acquire_weighed", (HasRegisterConstraintsTrait,)),
    ],
)
def test_rt_imm_rs_rs_rs_imm_format(op_type, mnemonic, comment, expected_traits):
    imm1 = 100
    r1 = create_ssa_value(Registers.R1)
    r2 = create_ssa_value(Registers.R2)
    r3 = create_ssa_value(Registers.R3)
    imm2 = 300

    op = op_type(imm1, r1, r2, r3, imm2, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs1.type, IntRegisterType)
    assert r1.type.register_name.data == "R1"
    assert r1.type.index == Registers.R1.index
    assert isinstance(op.rs2.type, IntRegisterType)
    assert r2.type.register_name.data == "R2"
    assert r2.type.index == Registers.R2.index
    assert isinstance(op.rs3.type, IntRegisterType)
    assert r3.type.register_name.data == "R3"
    assert r3.type.index == Registers.R3.index
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=imm1), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=imm2), value_type=ui32)
    assert op.assembly_line_args() == (op.imm1, op.rs1, op.rs2, op.rs3, op.imm2)
    assert len(expected_traits) == len(op.traits.traits)

    for trait in expected_traits:
        assert op.has_trait(trait)


class TestSemAliasing:
    """Tests for semantic aliasing of operation operands."""

    @pytest.mark.parametrize(
        "imm_op_variant,rs_op_variant,reg_idx,imm_addr",
        [
            (JmpIOp, JmpROp, Registers.R5, 42),
            (JzIOp, JzROp, Registers.R1, 100),
            (JoIOp, JoROp, Registers.R2, 200),
            (JnzIOp, JnzROp, Registers.R22, 321),
            (JnoIOp, JnoROp, Registers.R23, 322),
            (JsIOp, JsROp, Registers.R24, 323),
            (JnsIOp, JnsROp, Registers.R25, 324),
            (JgIOp, JgROp, Registers.R26, 325),
            (JlIOp, JlROp, Registers.R27, 326),
            (JleIOp, JleROp, Registers.R28, 327),
            (JaIOp, JaROp, Registers.R29, 328),
            (JaeIOp, JaeROp, Registers.R30, 329),
            (JbIOp, JbROp, Registers.R31, 330),
            (JbeIOp, JbeROp, Registers.R32, 331),
        ],
    )
    def test_flag_based_jumping(
        self, imm_op_variant, rs_op_variant, reg_idx, imm_addr
    ) -> None:
        imm_variant = imm_op_variant(address=imm_addr)
        reg_variant = rs_op_variant(address=create_ssa_value(reg_idx))

        assert imm_variant.address == imm_variant.imm
        assert reg_variant.address == reg_variant.rs

    def test_comparison_based_jumping(self) -> None:
        jge_rii = JgeRIIOp(a=create_ssa_value(Registers.R22), b=1000, address=100)
        assert jge_rii.a == jge_rii.rs
        assert jge_rii.b == jge_rii.imm1
        assert jge_rii.address == jge_rii.imm2

        jge_rir = JgeRIROp(
            a=create_ssa_value(Registers.R23),
            b=2000,
            address=create_ssa_value(Registers.R24),
        )
        assert jge_rir.a == jge_rir.rs1
        assert jge_rir.b == jge_rir.imm
        assert jge_rir.address == jge_rir.rs2

        jlt_rii = JltRIIOp(a=create_ssa_value(Registers.R25), b=3000, address=300)
        assert jlt_rii.a == jlt_rii.rs
        assert jlt_rii.b == jlt_rii.imm1
        assert jlt_rii.address == jlt_rii.imm2

        jlt_rir = JltRIROp(
            a=create_ssa_value(Registers.R26),
            b=4000,
            address=create_ssa_value(Registers.R27),
        )
        assert jlt_rir.a == jlt_rir.rs1
        assert jlt_rir.b == jlt_rir.imm
        assert jlt_rir.address == jlt_rir.rs2

    def test_stop(self) -> None:
        stop_imm = StopIOp(status=10)
        stop_reg = StopROp(status=create_ssa_value(Registers.R3))

        assert stop_imm.status == stop_imm.imm
        assert stop_reg.status == stop_reg.rs

    def test_move(self) -> None:
        move_imm = MoveIROp(source=999, rd=Registers.R0)
        move_reg = MoveRROp(source=create_ssa_value(Registers.R4), rd=Registers.R5)

        assert move_imm.source == move_imm.imm
        assert move_reg.source == move_reg.rs

    def test_not(self) -> None:
        not_imm = NotIROp(source=555, rd=Registers.R10)
        not_reg = NotRROp(source=create_ssa_value(Registers.R11), rd=Registers.R12)

        assert not_imm.source == not_imm.imm
        assert not_reg.source == not_reg.rs

    def test_add(self) -> None:
        add_rir = AddRIROp(a=create_ssa_value(Registers.R1), b=50, rd=Registers.R2)
        assert add_rir.a == add_rir.rs
        assert add_rir.b == add_rir.imm

        add_rrr = AddRRROp(
            a=create_ssa_value(Registers.R3),
            b=create_ssa_value(Registers.R4),
            rd=Registers.R5,
        )
        assert add_rrr.a == add_rrr.rs1
        assert add_rrr.b == add_rrr.rs2

    def test_sub(self) -> None:
        sub_rir = SubRIROp(a=create_ssa_value(Registers.R1), b=75, rd=Registers.R2)
        assert sub_rir.a == sub_rir.rs
        assert sub_rir.b == sub_rir.imm

        sub_rrr = SubRRROp(
            a=create_ssa_value(Registers.R3),
            b=create_ssa_value(Registers.R4),
            rd=Registers.R5,
        )
        assert sub_rrr.a == sub_rrr.rs1
        assert sub_rrr.b == sub_rrr.rs2

    def test_bitwise(self) -> None:
        # AND
        and_rir = AndRIROp(a=create_ssa_value(Registers.R1), b=0xFF, rd=Registers.R2)
        assert and_rir.a == and_rir.rs
        assert and_rir.b == and_rir.imm

        and_rrr = AndRRROp(
            a=create_ssa_value(Registers.R3),
            b=create_ssa_value(Registers.R4),
            rd=Registers.R5,
        )
        assert and_rrr.a == and_rrr.rs1
        assert and_rrr.b == and_rrr.rs2

        # OR
        or_rir = OrRIROp(a=create_ssa_value(Registers.R3), b=0xAA, rd=Registers.R4)
        assert or_rir.a == or_rir.rs
        assert or_rir.b == or_rir.imm

        or_rrr = OrRRROp(
            a=create_ssa_value(Registers.R3),
            b=create_ssa_value(Registers.R4),
            rd=Registers.R5,
        )
        assert or_rrr.a == or_rrr.rs1
        assert or_rrr.b == or_rrr.rs2

        # XOR
        xor_rir = XorRIROp(a=create_ssa_value(Registers.R6), b=0x55, rd=Registers.R7)
        assert xor_rir.a == xor_rir.rs
        assert xor_rir.b == xor_rir.imm

        xor_rrr = XorRRROp(
            a=create_ssa_value(Registers.R8),
            b=create_ssa_value(Registers.R9),
            rd=Registers.R10,
        )
        assert xor_rrr.a == xor_rrr.rs1
        assert xor_rrr.b == xor_rrr.rs2

    def test_shift(self) -> None:
        # Arithmetic left shift
        asl_rir = AslRIROp(a=create_ssa_value(Registers.R1), b=3, rd=Registers.R2)
        assert asl_rir.a == asl_rir.rs
        assert asl_rir.b == asl_rir.imm

        asl_rrr = AslRRROp(
            a=create_ssa_value(Registers.R3),
            b=create_ssa_value(Registers.R4),
            rd=Registers.R5,
        )
        assert asl_rrr.a == asl_rrr.rs1
        assert asl_rrr.b == asl_rrr.rs2

        # Arithmetic right shift
        asr_rir = AsrRIROp(a=create_ssa_value(Registers.R6), b=2, rd=Registers.R7)
        assert asr_rir.a == asr_rir.rs
        assert asr_rir.b == asr_rir.imm

        asr_rrr = AsrRRROp(
            a=create_ssa_value(Registers.R11),
            b=create_ssa_value(Registers.R12),
            rd=Registers.R13,
        )
        assert asr_rrr.a == asr_rrr.rs1
        assert asr_rrr.b == asr_rrr.rs2

    def test_parameter_setting(self) -> None:
        # SetMrk
        mrk_imm = SetMrkIOp(0x1234)
        assert mrk_imm.mrk == mrk_imm.imm

        mrk_reg = SetMrkROp(create_ssa_value(Registers.R8))
        assert mrk_reg.mrk == mrk_reg.rs

        # SetFreq
        freq_imm = SetFreqIOp(5000000)
        assert freq_imm.nco_freq == freq_imm.imm

        freq_reg = SetFreqROp(create_ssa_value(Registers.R9))
        assert freq_reg.nco_freq == freq_reg.rs

        # SetPh
        ph_imm = SetPhIOp(1000)
        assert ph_imm.nco_po == ph_imm.imm

        ph_reg = SetPhROp(create_ssa_value(Registers.R10))
        assert ph_reg.nco_po == ph_reg.rs

        # SetPhDelta
        phd_imm = SetPhDeltaIOp(500)
        assert phd_imm.nco_delta_po == phd_imm.imm

        phd_reg = SetPhDeltaROp(create_ssa_value(Registers.R11))
        assert phd_reg.nco_delta_po == phd_reg.rs

    def test_awg(self) -> None:
        # SetAwgGain
        gain_imm = SetAwgGainIIOp(gain0=100, gain1=150)
        assert gain_imm.gain0 == gain_imm.imm1
        assert gain_imm.gain1 == gain_imm.imm2

        gain_reg = SetAwgGainRROp(
            gain0=create_ssa_value(Registers.R12),
            gain1=create_ssa_value(Registers.R13),
        )
        assert gain_reg.gain0 == gain_reg.rs1
        assert gain_reg.gain1 == gain_reg.rs2

        # SetAwgOffs
        offs_imm = SetAwgOffsIIOp(offs0=200, offs1=250)
        assert offs_imm.offs0 == offs_imm.imm1
        assert offs_imm.offs1 == offs_imm.imm2

        offs_reg = SetAwgOffsRROp(
            offs0=create_ssa_value(Registers.R14),
            offs1=create_ssa_value(Registers.R15),
        )
        assert offs_reg.offs0 == offs_reg.rs1
        assert offs_reg.offs1 == offs_reg.rs2

    def test_set_cond(self) -> None:
        # Immediate version
        cond_imm = SetCondIIIIOp(cond_en=1, mask=2, op=3, else_cnt=4)
        assert cond_imm.cond_en == cond_imm.imm1
        assert cond_imm.mask == cond_imm.imm2
        assert cond_imm.op == cond_imm.imm3
        assert cond_imm.else_cnt == cond_imm.imm4

        # Register version
        cond_reg = SetCondRRRIOp(
            cond_en=create_ssa_value(Registers.R16),
            mask=create_ssa_value(Registers.R17),
            op=create_ssa_value(Registers.R18),
            else_cnt=100,
        )
        assert cond_reg.cond_en == cond_reg.rs1
        assert cond_reg.mask == cond_reg.rs2
        assert cond_reg.op == cond_reg.rs3
        assert cond_reg.else_cnt == cond_reg.imm

    def test_loop(self) -> None:
        loop_ri = LoopRIOp(Registers.R19, 500)
        assert loop_ri.source == loop_ri.rd
        assert loop_ri.address == loop_ri.imm

        loop_rr = LoopRROp(Registers.R20, create_ssa_value(Registers.R21))
        assert loop_rr.source == loop_rr.rd
        assert loop_rr.address == loop_rr.rs

    def test_rt_wait_sem_alias(self) -> None:
        op = WaitIOp(duration=100)
        assert op.duration == op.imm

        op = WaitROp(duration=create_ssa_value(Registers.R14))
        assert op.duration == op.rs

    def test_rt_wait_sync_sem_alias(self) -> None:
        op = WaitSyncIOp(duration=200)
        assert op.duration == op.imm

        op = WaitSyncROp(duration=create_ssa_value(Registers.R12))
        assert op.duration == op.rs

    def test_rt_wait_trigger_sem_alias(self) -> None:
        op = WaitTriggerIIOp(trig_addr=2, duration=50)
        assert op.trig_addr == op.imm1
        assert op.duration == op.imm2

        op = WaitTriggerRROp(
            trig_addr=create_ssa_value(Registers.R13),
            duration=create_ssa_value(Registers.R14),
        )
        assert op.trig_addr == op.rs1
        assert op.duration == op.rs2

    def test_rt_play_sem_alias(self) -> None:
        op = PlayRRIOp(
            wave0=create_ssa_value(Registers.R12),
            wave1=create_ssa_value(Registers.R13),
            duration=100,
        )
        assert op.wave0 == op.rs1
        assert op.wave1 == op.rs2
        assert op.duration == op.imm

        op = PlayIIIOp(wave0=3, wave1=2, duration=100)
        assert op.wave0 == op.imm1
        assert op.wave1 == op.imm2
        assert op.duration == op.imm3

    def test_rt_acquire_sem_alias(self) -> None:
        op = AcquireIRIOp(acq_idx=44, bin_idx=create_ssa_value(Registers.R13), duration=128)
        assert op.acq_idx == op.imm1
        assert op.bin_idx == op.rs
        assert op.duration == op.imm2

        op = AcquireIIIOp(acq_idx=100, bin_idx=2, duration=50)
        assert op.acq_idx == op.imm1
        assert op.bin_idx == op.imm2
        assert op.duration == op.imm3

    def test_rt_acquire_weighted_sem_alias(self) -> None:
        op = AcquireWeighedIRRRIOp(
            acq_idx=10,
            bin_idx=create_ssa_value(Registers.R13),
            weight_idx0=create_ssa_value(Registers.R14),
            weight_idx1=create_ssa_value(Registers.R15),
            duration=100,
        )
        assert op.acq_idx == op.imm1
        assert op.bin_idx == op.rs1
        assert op.weight_idx0 == op.rs2
        assert op.weight_idx1 == op.rs3
        assert op.duration == op.imm2

        op = AcquireWeighedIIIIIOp(
            acq_idx=20, bin_idx=3, weight_idx0=2, weight_idx1=1, duration=350
        )
        assert op.acq_idx == op.imm1
        assert op.bin_idx == op.imm2
        assert op.weight_idx0 == op.imm3
        assert op.weight_idx1 == op.imm4
        assert op.duration == op.imm5

    def test_rt_acquire_ttl_sem_alias(self) -> None:
        op = AcquireTtlIRIIOp(
            acq_idx=5, bin_idx=create_ssa_value(Registers.R13), ttl_en=1, duration=20
        )
        assert op.acq_idx == op.imm1
        assert op.bin_idx == op.rs
        assert op.ttl_en == op.imm2
        assert op.duration == op.imm3

        op = AcquireTtlIIIIOp(acq_idx=10, bin_idx=5, ttl_en=0, duration=400)
        assert op.acq_idx == op.imm1
        assert op.bin_idx == op.imm2
        assert op.ttl_en == op.imm3
        assert op.duration == op.imm4

    def test_rt_acquire_timetags_sem_alias(self) -> None:
        op = AcquireTimetagsIRIRIOp(
            acq_idx=3,
            bin_idx=create_ssa_value(Registers.R13),
            window_en=1,
            fine_acq_delay=create_ssa_value(Registers.R14),
            duration=100,
        )
        assert op.acq_idx == op.imm1
        assert op.bin_idx == op.rs1
        assert op.window_en == op.imm2
        assert op.fine_acq_delay == op.rs2
        assert op.duration == op.imm3

        op = AcquireTimetagsIIIIIOp(
            acq_idx=4, bin_idx=1, window_en=0, fine_acq_delay=40, duration=200
        )
        assert op.acq_idx == op.imm1
        assert op.bin_idx == op.imm2
        assert op.window_en == op.imm3
        assert op.fine_acq_delay == op.imm4
        assert op.duration == op.imm5

    def test_rt_acquire_digital_sem_alias(self) -> None:
        op = AcquireDigitalIIIOp(acq_idx=7, bin_idx=4, duration=30)
        assert op.acq_idx == op.imm1
        assert op.bin_idx == op.imm2
        assert op.duration == op.imm3

        op = AcquireDigitalIRIOp(
            acq_idx=1, bin_idx=create_ssa_value(Registers.R13), duration=28
        )
        assert op.acq_idx == op.imm1
        assert op.bin_idx == op.rs
        assert op.duration == op.imm2

    def test_rt_upd_thres_sem_alias(self) -> None:
        op = UpdThresIIIOp(dio_thres_idx=12, value=40, duration=250)
        assert op.dio_thres_idx == op.imm1
        assert op.value == op.imm2
        assert op.duration == op.imm3

        op = UpdThresIRIOp(
            dio_thres_idx=3, value=create_ssa_value(Registers.R13), duration=150
        )
        assert op.dio_thres_idx == op.imm1
        assert op.value == op.rs
        assert op.duration == op.imm2

    def test_rt_upd_param_sem_alias(self) -> None:
        op = UpdParamIOp(duration=100)
        assert op.duration == op.imm

    def test_rt_latch_rst_sem_alias(self) -> None:
        op = LatchRstIOp(duration=300)
        assert op.duration == op.imm

        op = LatchRstROp(duration=create_ssa_value(Registers.R12))
        assert op.duration == op.rs

    def test_rt_latch_en_sem_alias(self) -> None:
        op = SetLatchEnIIOp(latch_en=1, duration=300)
        assert op.latch_en == op.imm1
        assert op.duration == op.imm2

        op = SetLatchEnRIOp(latch_en=create_ssa_value(Registers.R13), duration=300)
        assert op.latch_en == op.rs
        assert op.duration == op.imm
