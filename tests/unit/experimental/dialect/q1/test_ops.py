# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.backend.register_allocatable import RegisterConstraints
from xdsl.dialects.builtin import IntAttr, IntegerAttr, StringAttr
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.q1 import (
    AddRIROp,
    AddRRROp,
    AndRIROp,
    AndRRROp,
    AslRIROp,
    AslRRROp,
    AsrRIROp,
    AsrRRROp,
    DefDirectiveOp,
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
    LoopRIOp,
    LoopRROp,
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
    SetPhDeltaROp,
    SetPhIOp,
    SetPhROp,
    StopIOp,
    StopOp,
    StopROp,
    SubRIROp,
    SubRRROp,
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
    "op_type,mnemonic,comment",
    [
        (LoopRIOp, "loop", "Jump to immediate address while greater or equals"),
    ],
)
def test_rd_imm_format(op_type, mnemonic, comment):
    a_val = 100
    op = op_type(Registers.R10, a_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert isinstance(op.rd.type, IntRegisterType)
    assert op.rd.type.is_allocated
    assert op.rd.type.index == Registers.R10.index
    assert op.assembly_line_args() == (op.rd, op.imm)


@pytest.mark.parametrize(
    "op_type,mnemonic,comment",
    [
        (LoopRROp, "loop", "Jump to register address while greater or equals"),
    ],
)
def test_rd_rs_format(op_type, mnemonic, comment):
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


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic",
    [
        (JgeRIIOp, "jge"),
        (JltRIIOp, "jlt"),
    ],
)
def test_rs_imm_imm_format(op_type, mnemonic, comment):
    a_val = create_ssa_value(Registers.R1)
    b_val = 1000
    d_val = 100

    op = op_type(a_val, b_val, d_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=b_val), value_type=ui32)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=d_val), value_type=ui32)
    assert isinstance(op.rs.type, IntRegisterType)
    assert op.assembly_line_args() == (op.rs, op.imm1, op.imm2)
    op2 = op_type(
        a_val, IntegerAttr(b_val, ui32), IntegerAttr(d_val, ui32), comment=comment
    )
    assert op2.imm1 == IntegerAttr(b_val, ui32)
    assert op2.imm2 == IntegerAttr(d_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic",
    [
        (JgeRIROp, "jge"),
        (JltRIROp, "jlt"),
    ],
)
def test_rs_imm_rs_format(op_type, mnemonic, comment):
    a_val = create_ssa_value(Registers.R1)
    b_val = 1000
    c_val = create_ssa_value(Registers.R2)
    op = op_type(a_val, b_val, c_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=b_val), value_type=ui32)
    assert isinstance(op.rs1.type, IntRegisterType)
    assert isinstance(op.rs2.type, IntRegisterType)
    assert op.assembly_line_args() == (op.rs1, op.imm, op.rs2)
    op2 = op_type(a_val, IntegerAttr(b_val, ui32), c_val, comment=comment)
    assert op2.imm == IntegerAttr(b_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic",
    [
        (IllegalOp, "illegal"),
        (StopOp, "stop"),
        (NopOp, "nop"),
        (ResetPhOp, "reset_ph"),
    ],
)
def test_nullary_format(op_type, mnemonic, comment):
    op = op_type(comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.assembly_line_args() == ()
    assert op.assembly_line() is not None


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic",
    [
        (StopIOp, "stop"),
        (JmpIOp, "jmp"),
        (JzIOp, "jz"),
        (JnzIOp, "jnz"),
        (JoIOp, "jo"),
        (JnoIOp, "jno"),
        (JsIOp, "js"),
        (JnsIOp, "jns"),
        (JgIOp, "jg"),
        (JlIOp, "jl"),
        (JleIOp, "jle"),
        (JaIOp, "ja"),
        (JaeIOp, "jae"),
        (JbIOp, "jb"),
        (JbeIOp, "jbe"),
        (SetMrkIOp, "set_mrk"),
        (SetFreqIOp, "set_freq"),
        (SetPhIOp, "set_ph"),
        (SetPhDeltaIOp, "set_ph_delta"),
    ],
)
def test_imm_format(op_type, mnemonic, comment):
    a_val = 1024
    op = op_type(a_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert op.assembly_line_args() == (op.imm,)
    op2 = op_type(IntegerAttr(a_val, ui32), comment=comment)
    assert op2.imm == IntegerAttr(a_val, ui32)


@pytest.mark.parametrize(
    "op_type,mnemonic,comment",
    [
        (StopROp, "stop", "Stop with status"),
        (JmpROp, "jmp", "Unconditional jump"),
        (JzROp, "jz", "Jump if zero"),
        (JnzROp, "jnz", "Jump if not zero"),
        (JoROp, "jo", "Jump if overflow"),
        (JnoROp, "jno", "Jump if not overflow"),
        (JsROp, "js", "Jump if negative"),
        (JnsROp, "jns", "Jump if not negative"),
        (JgROp, "jg", "Jump if greater than"),
        (JlROp, "jl", "Jump if less than"),
        (JleROp, "jle", "Jump if less than or equal"),
        (JaROp, "ja", "Jump if above"),
        (JaeROp, "jae", "Jump if above or equal"),
        (JbROp, "jb", "Jump if below"),
        (JbeROp, "jbe", "Jump if below or equal"),
        (SetMrkROp, "set_mrk", "Set marker output channels"),
        (SetFreqROp, "set_freq", "Set frequency"),
        (SetPhROp, "set_ph", "Set phase"),
        (SetPhDeltaROp, "set_ph_delta", "Set phase offset"),
    ],
)
def test_rs_format(op_type, mnemonic, comment):
    a_val = create_ssa_value(Registers.R32)
    op = op_type(a_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs.type, IntRegisterType)
    assert op.assembly_line_args() == (op.rs,)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic",
    [
        (MoveIROp, "move"),
        (NotIROp, "not"),
    ],
)
def test_imm_rd_format(op_type, mnemonic, comment):
    a_val = 101

    op = op_type(a_val, rd=Registers.R0)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.imm, IntegerAttr)
    assert op.properties["imm"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert isinstance(op.rd.type, IntRegisterType)
    assert op.rd.type.index == Registers.R0.index
    assert op.assembly_line_args() == (op.imm, op.rd)
    op2 = op_type(IntegerAttr(a_val, ui32), rd=Registers.R0, comment=comment)
    assert op2.imm == IntegerAttr(a_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "opt_type,mnemonic",
    [
        (MoveRROp, "move"),
        (NotRROp, "not"),
    ],
)
def test_rs_rd_format(opt_type, mnemonic, comment):
    a_val = create_ssa_value(Registers.R62)

    op = opt_type(a_val, rd=Registers.R63)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.rs.type, IntRegisterType)
    assert op.rs.type.index == Registers.R62.index
    assert isinstance(op.rd.type, IntRegisterType)
    assert op.rd.type.index == Registers.R63.index
    assert op.assembly_line_args() == (op.rs, op.rd)
    op2 = opt_type(create_ssa_value(Registers.R62), rd=Registers.R63, comment=comment)
    assert op2.rd.type.index == Registers.R63.index


@pytest.mark.parametrize(
    "opt_type,mnemonic",
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
def test_rs_rs_rd_format(opt_type, mnemonic):
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


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic",
    [
        (AddRIROp, "add"),
        (SubRIROp, "sub"),
        (AndRIROp, "and"),
        (OrRIROp, "or"),
        (XorRIROp, "xor"),
        (AslRIROp, "asl"),
        (AsrRIROp, "asr"),
    ],
)
def test_rs_imm_rd_format(op_type, mnemonic, comment):
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
    op2 = op_type(r1, IntegerAttr(a_val, ui32), rd=Registers.R3, comment=comment)
    assert op2.imm == IntegerAttr(a_val, ui32)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic",
    [
        (
            SetCondIIIIOp,
            "set_cond",
        ),
    ],
)
def test_imm_imm_imm_imm_format(op_type, mnemonic, comment):
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
    "op_type,mnemonic",
    [
        (
            SetCondRRRIOp,
            "set_cond",
        ),
    ],
)
def test_rs_rs_rs_imm_format(op_type, mnemonic, comment):
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
    op2 = op_type(r1, r2, r3, IntegerAttr(d_val, ui32), comment=comment)
    assert op2.imm == IntegerAttr(d_val, ui32)
    assert isinstance(op2.get_register_constraints(), RegisterConstraints)


@pytest.mark.parametrize(
    "op_type,mnemonic,comment",
    [
        (SetAwgGainRROp, "set_awg_gain", "Set AWG gain"),
        (SetAwgOffsRROp, "set_awg_offs", "Set AWG offset"),
    ],
)
def test_rs_rs_format(op_type, mnemonic, comment):
    r1 = create_ssa_value(Registers.R1)
    r2 = create_ssa_value(Registers.R2)

    op = op_type(r1, r2, comment=comment)
    assert op.assembly_mnemonic() == mnemonic

    assert r1.type is op.rs1.type
    assert isinstance(r1.type, IntRegisterType)

    assert r2.type is op.rs2.type
    assert isinstance(r2.type, IntRegisterType)
    assert op.assembly_line_args() == (op.rs1, op.rs2)


@pytest.mark.parametrize("comment", ["test-comment", StringAttr("test-comment")])
@pytest.mark.parametrize(
    "op_type,mnemonic",
    [
        (SetAwgGainIIOp, "set_awg_gain"),
        (SetAwgOffsIIOp, "set_awg_offs"),
    ],
)
def test_imm_imm_format(op_type, mnemonic, comment):
    a_val = 100
    b_val = 200

    op = op_type(a_val, b_val, comment=comment)
    assert op.assembly_mnemonic() == mnemonic
    assert isinstance(op.imm1, IntegerAttr)
    assert op.properties["imm1"] == IntegerAttr(value=IntAttr(data=a_val), value_type=ui32)
    assert isinstance(op.imm2, IntegerAttr)
    assert op.properties["imm2"] == IntegerAttr(value=IntAttr(data=b_val), value_type=ui32)
    op2 = op_type(IntegerAttr(a_val, ui32), IntegerAttr(b_val, ui32), comment=comment)
    assert op2.imm1 == IntegerAttr(a_val, ui32)
    assert op2.imm2 == IntegerAttr(b_val, ui32)


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
        assert mrk_imm.mask == mrk_imm.imm

        mrk_reg = SetMrkROp(create_ssa_value(Registers.R8))
        assert mrk_reg.mask == mrk_reg.rs

        # SetFreq
        freq_imm = SetFreqIOp(5000000)
        assert freq_imm.frequency == freq_imm.imm

        freq_reg = SetFreqROp(create_ssa_value(Registers.R9))
        assert freq_reg.frequency == freq_reg.rs

        # SetPh
        ph_imm = SetPhIOp(1000)
        assert ph_imm.phase_offset == ph_imm.imm

        ph_reg = SetPhROp(create_ssa_value(Registers.R10))
        assert ph_reg.phase_offset == ph_reg.rs

        # SetPhDelta
        phd_imm = SetPhDeltaIOp(500)
        assert phd_imm.phase_delta == phd_imm.imm

        phd_reg = SetPhDeltaROp(create_ssa_value(Registers.R11))
        assert phd_reg.phase_delta == phd_reg.rs

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
        offs_imm = SetAwgOffsIIOp(offset0=200, offset1=250)
        assert offs_imm.offset0 == offs_imm.imm1
        assert offs_imm.offset1 == offs_imm.imm2

        offs_reg = SetAwgOffsRROp(
            offset0=create_ssa_value(Registers.R14),
            offset1=create_ssa_value(Registers.R15),
        )
        assert offs_reg.offset0 == offs_reg.rs1
        assert offs_reg.offset1 == offs_reg.rs2

    def test_set_cond(self) -> None:
        # Immediate version
        cond_imm = SetCondIIIIOp(enable=1, mask=2, operator=3, else_duration=4)
        assert cond_imm.enable == cond_imm.imm1
        assert cond_imm.mask == cond_imm.imm2
        assert cond_imm.operator == cond_imm.imm3
        assert cond_imm.else_duration == cond_imm.imm4

        # Register version
        cond_reg = SetCondRRRIOp(
            enable=create_ssa_value(Registers.R16),
            mask=create_ssa_value(Registers.R17),
            operator=create_ssa_value(Registers.R18),
            else_duration=100,
        )
        assert cond_reg.enable == cond_reg.rs1
        assert cond_reg.mask == cond_reg.rs2
        assert cond_reg.operator == cond_reg.rs3
        assert cond_reg.else_duration == cond_reg.imm

    def test_loop(self) -> None:
        loop_ri = LoopRIOp(Registers.R19, 500)
        assert loop_ri.source == loop_ri.rd
        assert loop_ri.address == loop_ri.imm

        loop_rr = LoopRROp(Registers.R20, create_ssa_value(Registers.R21))
        assert loop_rr.source == loop_rr.rd
        assert loop_rr.address == loop_rr.rs
