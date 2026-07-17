# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.parser import Parser

from qat.experimental.dialect.q1 import (
    AcquireImmImmImmOp,
    AcquireImmRsImmOp,
    AcquireTtlImmImmImmImmOp,
    AcquireTtlImmRsImmImmOp,
    AcquireWeightedImmImmImmImmImmOp,
    AcquireWeightedImmRsRsRsImmOp,
    AddRsImmRdOp,
    AddRsRsRdOp,
    AndRsImmRdOp,
    AndRsRsRdOp,
    AslRsImmRdOp,
    AslRsRsRdOp,
    AsrRsImmRdOp,
    AsrRsRsRdOp,
    CmpImmRsOp,
    CmpRsImmOp,
    CmpRsRsOp,
    DefDirectiveOp,
    FbAcqIqIdImmImmOp,
    FbAcqIqIdRsImmOp,
    FbAcqIqShiftImmImmOp,
    FbAcqTbCfgImmImmImmImmOp,
    FbAcqTbExtraImmImmImmOp,
    FbAcqTbIdImmImmOp,
    FbAcqTbIdRsImmOp,
    FbAcqTbMockImmImmImmImmOp,
    FbAcqTbValidImmImmOp,
    FbAcqTbValidRsImmOp,
    FbComCfgImmImmImmImmOp,
    FbComDataImmImmImmOp,
    FbComDataImmRsImmOp,
    FbComExtraImmImmImmOp,
    FbPopDataImmRdOp,
    FbPullDataRsRdOp,
    IllegalOp,
    JaeImmOp,
    JaeRsOp,
    JaImmOp,
    JaRsOp,
    JbeImmOp,
    JbeRsOp,
    JbImmOp,
    JbRsOp,
    JgeImmOp,
    JgeRsImmImmOp,
    JgeRsImmRsOp,
    JgeRsOp,
    JgImmOp,
    JgRsOp,
    JleImmOp,
    JleRsOp,
    JlImmOp,
    JlRsOp,
    JltRsImmImmOp,
    JltRsImmRsOp,
    JmpImmOp,
    JmpRsOp,
    JnoImmOp,
    JnoRsOp,
    JnsImmOp,
    JnsRsOp,
    JnzImmOp,
    JnzRsOp,
    JoImmOp,
    JoRsOp,
    JsImmOp,
    JsRsOp,
    JzImmOp,
    JzRsOp,
    LatchRstImmOp,
    LatchRstRsOp,
    LoopRdImmOp,
    LoopRdRsOp,
    LslImmRsRdOp,
    LslRsImmRdOp,
    LslRsRsRdOp,
    LsrImmRsRdOp,
    LsrRsImmRdOp,
    LsrRsRsRdOp,
    MoveImmRdOp,
    MoveRsRdOp,
    Muls16ImmRsRdOp,
    Muls16RsImmRdOp,
    Muls16RsRsRdOp,
    Muls32hImmRsRdOp,
    Muls32hRsImmRdOp,
    Muls32hRsRsRdOp,
    Muls32ImmRsRdRdOp,
    Muls32lImmRsRdOp,
    Muls32lRsImmRdOp,
    Muls32lRsRsRdOp,
    Muls32RsImmRdRdOp,
    Muls32RsRsRdRdOp,
    Mulu16ImmRsRdOp,
    Mulu16RsImmRdOp,
    Mulu16RsRsRdOp,
    Mulu32hImmRsRdOp,
    Mulu32hRsImmRdOp,
    Mulu32hRsRsRdOp,
    Mulu32ImmRsRdRdOp,
    Mulu32lImmRsRdOp,
    Mulu32lRsImmRdOp,
    Mulu32lRsRsRdOp,
    Mulu32RsImmRdRdOp,
    Mulu32RsRsRdRdOp,
    NopOp,
    NotImmRdOp,
    NotRsRdOp,
    OrRsImmRdOp,
    OrRsRsRdOp,
    PlayImmImmImmOp,
    PlayRsRsImmOp,
    ResetPhOp,
    SetAwgGainImmImmOp,
    SetAwgGainRsRsOp,
    SetAwgOffsImmImmOp,
    SetAwgOffsRsRsOp,
    SetCondImmImmImmImmOp,
    SetCondRsRsRsImmOp,
    SetFreqImmOp,
    SetFreqRsOp,
    SetLatchEnImmImmOp,
    SetLatchEnRsImmOp,
    SetMrkImmOp,
    SetMrkRsOp,
    SetPhDeltaImmOp,
    SetPhDeltaRsOp,
    SetPhImmOp,
    SetPhRsOp,
    StopImmOp,
    StopOp,
    StopRsOp,
    SubRsImmRdOp,
    SubRsRsRdOp,
    TestImmRsOp,
    TestRsImmOp,
    TestRsRsOp,
    UpdParamImmOp,
    WaitImmOp,
    WaitRsOp,
    WaitSyncImmOp,
    WaitSyncRsOp,
    WaitTriggerImmImmOp,
    WaitTriggerRsRsOp,
    XorRsImmRdOp,
    XorRsRsRdOp,
)

NULLARY_OPS = (IllegalOp, NopOp, ResetPhOp, StopOp)
IMM_OPS = (
    JaImmOp,
    JaeImmOp,
    JbImmOp,
    JbeImmOp,
    JgImmOp,
    JgeImmOp,
    JlImmOp,
    JleImmOp,
    JmpImmOp,
    JnoImmOp,
    JnsImmOp,
    JnzImmOp,
    JoImmOp,
    JsImmOp,
    JzImmOp,
    SetFreqImmOp,
    SetMrkImmOp,
    SetPhDeltaImmOp,
    SetPhImmOp,
    WaitImmOp,
    WaitSyncImmOp,
    UpdParamImmOp,
    LatchRstImmOp,
    StopImmOp,
)
RS_OPS = (
    JaRsOp,
    JaeRsOp,
    JbRsOp,
    JbeRsOp,
    JgRsOp,
    JgeRsOp,
    JlRsOp,
    JleRsOp,
    JmpRsOp,
    JnoRsOp,
    JnsRsOp,
    JnzRsOp,
    JoRsOp,
    JsRsOp,
    JzRsOp,
    SetFreqRsOp,
    SetMrkRsOp,
    SetPhDeltaRsOp,
    SetPhRsOp,
    WaitRsOp,
    WaitSyncRsOp,
    LatchRstRsOp,
    StopRsOp,
)
II_OPS = (
    FbAcqIqIdImmImmOp,
    FbAcqIqShiftImmImmOp,
    FbAcqTbIdImmImmOp,
    FbAcqTbValidImmImmOp,
    SetAwgGainImmImmOp,
    SetAwgOffsImmImmOp,
    WaitTriggerImmImmOp,
    SetLatchEnImmImmOp,
)
IMM_RD_OPS = (FbPopDataImmRdOp, MoveImmRdOp, NotImmRdOp)
RS_RD_OPS = (MoveRsRdOp, NotRsRdOp, FbPullDataRsRdOp)
RS_IMM_OPS = (
    CmpRsImmOp,
    FbAcqIqIdRsImmOp,
    FbAcqTbIdRsImmOp,
    FbAcqTbValidRsImmOp,
    SetLatchEnRsImmOp,
    TestRsImmOp,
)
IMM_RS_OPS = (CmpImmRsOp, TestImmRsOp)
RS_IMM_IMM_OPS = (JgeRsImmImmOp, JltRsImmImmOp)
RS_IMM_RS_OPS = (JgeRsImmRsOp, JltRsImmRsOp)
RI_ARITHMETIC_OPS = (
    AddRsImmRdOp,
    AndRsImmRdOp,
    AslRsImmRdOp,
    AsrRsImmRdOp,
    LslRsImmRdOp,
    LsrRsImmRdOp,
    Mulu16RsImmRdOp,
    Muls16RsImmRdOp,
    Mulu32lRsImmRdOp,
    Mulu32hRsImmRdOp,
    Muls32lRsImmRdOp,
    Muls32hRsImmRdOp,
    OrRsImmRdOp,
    SubRsImmRdOp,
    XorRsImmRdOp,
)
RR_ARITHMETIC_OPS = (
    AddRsRsRdOp,
    AndRsRsRdOp,
    AslRsRsRdOp,
    AsrRsRsRdOp,
    LslRsRsRdOp,
    LsrRsRsRdOp,
    Mulu16RsRsRdOp,
    Muls16RsRsRdOp,
    Mulu32lRsRsRdOp,
    Mulu32hRsRsRdOp,
    Muls32lRsRsRdOp,
    Muls32hRsRsRdOp,
    OrRsRsRdOp,
    SubRsRsRdOp,
    XorRsRsRdOp,
)
IR_ARITHMETIC_OPS = (
    LslImmRsRdOp,
    LsrImmRsRdOp,
    Mulu16ImmRsRdOp,
    Muls16ImmRsRdOp,
    Mulu32lImmRsRdOp,
    Mulu32hImmRsRdOp,
    Muls32lImmRsRdOp,
    Muls32hImmRsRdOp,
)
RR_RR_ARITHMETIC_OPS = (Mulu32RsRsRdRdOp, Muls32RsRsRdRdOp)
RI_RR_ARITHMETIC_OPS = (Mulu32RsImmRdRdOp, Muls32RsImmRdRdOp)
IR_RR_ARITHMETIC_OPS = (Mulu32ImmRsRdRdOp, Muls32ImmRsRdRdOp)
IMM_IMM_IMM_OPS = (
    AcquireImmImmImmOp,
    FbAcqTbExtraImmImmImmOp,
    FbComDataImmImmImmOp,
    FbComExtraImmImmImmOp,
    PlayImmImmImmOp,
)
IMM_RS_IMM_OPS = (
    FbComDataImmRsImmOp,
    AcquireImmRsImmOp,
)
IMM_IMM_IMM_IMM_OPS = (
    AcquireTtlImmImmImmImmOp,
    FbAcqTbCfgImmImmImmImmOp,
    FbAcqTbMockImmImmImmImmOp,
    FbComCfgImmImmImmImmOp,
    SetCondImmImmImmImmOp,
)
IMM_IMM_IMM_IMM_IMM_OPS = (AcquireWeightedImmImmImmImmImmOp,)
RS_RS_RS_IMM_OPS = (SetCondRsRsRsImmOp,)
RS_RS_OPS = (CmpRsRsOp, SetAwgGainRsRsOp, SetAwgOffsRsRsOp, TestRsRsOp, WaitTriggerRsRsOp)
RS_RS_IMM_OPS = (PlayRsRsImmOp,)
RS_IMM_IMM_IMM_OPS = (AcquireTtlImmRsImmImmOp,)
RS_RS_RS_IMM_IMM_OPS = (AcquireWeightedImmRsRsRsImmOp,)
RD_IMM_OPS = (LoopRdImmOp,)
RD_RS_OPS = (LoopRdRsOp,)


def _parse_op_type(op_type, signature: str) -> tuple[int, int]:
    parser = Parser(_ctx(), signature)
    operand_types, result_types = op_type.parse_op_type(parser)
    return len(operand_types), len(result_types)


def _parse_op(op_type, signature: str):
    return op_type.parse(Parser(_ctx(), signature))


def _ctx():
    """A Context with Builtin + Q1 loaded so dialect attrs (incl.

    Q1Imm) parse.
    """
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin

    from qat.experimental.dialect.q1 import Q1

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Q1)
    return ctx


def _imm_props(op_type) -> str:
    """Build the textual `<{imm = #q1.<name>_imm<v>, ...}>` block for ``op_type``.

    Introspects ``op_type.__orig_bases__`` to find the bound Q1Imm subclasses and
    pairs them with the shape's slot names (``imm`` for one slot, ``imm1``..``immN``
    for many). Returns ``""`` if the op has no immediate properties.
    """
    imm_types = _bound_imm_types(op_type)
    if not imm_types:
        return ""
    slots = (
        ["imm"] if len(imm_types) == 1 else [f"imm{i + 1}" for i in range(len(imm_types))]
    )
    body = ", ".join(
        f"{s} = #{t.name}<{max(t._MIN, min(t._MAX, i + 1))}>"
        for i, (s, t) in enumerate(zip(slots, imm_types, strict=True))
    )
    return f"<{{{body}}}>"


def _bound_imm_types(op_type) -> list[type]:
    """Return the Q1Imm subclasses bound to ``op_type``'s shape, in slot order."""
    import typing

    from qat.experimental.dialect.q1 import Q1Imm

    orig_bases = getattr(op_type, "__orig_bases__", ())
    if not orig_bases:
        return []
    return [
        a
        for a in typing.get_args(orig_bases[0])
        if isinstance(a, type) and issubclass(a, Q1Imm)
    ]


def _assert_props_are_q1imm(op, op_type) -> None:
    imm_types = _bound_imm_types(op_type)
    if not imm_types:
        return
    slots = (
        ["imm"] if len(imm_types) == 1 else [f"imm{i + 1}" for i in range(len(imm_types))]
    )
    for slot, expected in zip(slots, imm_types, strict=True):
        actual = op.properties[slot]
        assert isinstance(actual, expected), (
            f"{op_type.__name__}.{slot}: expected {expected.__name__}, got"
            f" {type(actual).__name__}"
        )


_FORMAT_TABLE = [
    (NULLARY_OPS, "()", "()"),
    (IMM_OPS, "()", "()"),
    (RS_OPS, "(i32)", "()"),
    (II_OPS, "()", "()"),
    (IMM_RD_OPS, "()", "(i32)"),
    (RS_RD_OPS, "(i32)", "(i32)"),
    (RS_IMM_OPS, "(i32)", "()"),
    (RS_IMM_IMM_OPS, "(i32)", "()"),
    (RS_IMM_RS_OPS, "(i32, i32)", "()"),
    (IMM_RS_OPS, "(i32)", "()"),
    (RI_ARITHMETIC_OPS, "(i32)", "(i32)"),
    (IR_ARITHMETIC_OPS, "(i32)", "(i32)"),
    (RR_ARITHMETIC_OPS, "(i32, i32)", "(i32)"),
    (RR_RR_ARITHMETIC_OPS, "(i32, i32)", "(i32, i32)"),
    (RI_RR_ARITHMETIC_OPS, "(i32)", "(i32, i32)"),
    (IR_RR_ARITHMETIC_OPS, "(i32)", "(i32, i32)"),
    (IMM_IMM_IMM_OPS, "()", "()"),
    (IMM_RS_IMM_OPS, "(i32)", "()"),
    (IMM_IMM_IMM_IMM_OPS, "()", "()"),
    (IMM_IMM_IMM_IMM_IMM_OPS, "()", "()"),
    (RS_RS_RS_IMM_OPS, "(i32, i32, i32)", "()"),
    (RS_RS_IMM_OPS, "(i32, i32)", "()"),
    (RS_IMM_IMM_IMM_OPS, "(i32)", "()"),
    (RS_RS_RS_IMM_IMM_OPS, "(i32, i32, i32)", "()"),
    (RD_IMM_OPS, "()", "(i32)"),
    (RD_RS_OPS, "(i32)", "(i32)"),
    (RS_RS_OPS, "(i32, i32)", "()"),
]


def _operand_placeholders(operand_type_str: str) -> str:
    if operand_type_str == "()":
        return "()"
    return "(" + ", ".join(f"%{i}" for i in range(operand_type_str.count("i32"))) + ")"


@pytest.mark.parametrize(
    "op_type,operand_type_str,result_type_str",
    [
        (op_type, op_types, res_types)
        for ops, op_types, res_types in _FORMAT_TABLE
        for op_type in ops
    ],
)
def test_format_parse(op_type, operand_type_str, result_type_str):
    expected_operands = operand_type_str.count("i32")
    expected_results = result_type_str.count("i32")

    operand_count, result_count = _parse_op_type(
        op_type, f": {operand_type_str} -> {result_type_str}"
    )
    assert operand_count == expected_operands
    assert result_count == expected_results

    operands = _operand_placeholders(operand_type_str)
    props_str = _imm_props(op_type)
    props = f" {props_str}" if props_str else ""
    signature = f"{operands}{props} : {operand_type_str} -> {result_type_str}"
    op = _parse_op(op_type, signature)
    assert isinstance(op, op_type)
    assert len(op.operands) == expected_operands
    assert len(op.results) == expected_results
    # Confirm each parsed property is the bound Q1Imm subtype, not a generic IntegerAttr.
    _assert_props_are_q1imm(op, op_type)


def test_parse_rejects_trailing_comma_in_operand_list():
    with pytest.raises(Exception, match="operand expected"):
        _parse_op(JmpRsOp, "(%0, ) : (i32) -> ()")


def test_parse_accepts_properties_in_mlir_position_without_operands():
    op = _parse_op(
        DefDirectiveOp,
        '() <{alias = "CONST", value = "7", comment = "const"}> : () -> ()',
    )

    assert op.alias == StringAttr("CONST")
    assert op.value == StringAttr("7")
    assert op.comment == StringAttr("const")


def test_parse_accepts_properties_and_attributes_in_mlir_position():
    op = _parse_op(
        JmpRsOp,
        '(%0) <{comment = "branch"}> {tag = "meta"} : (i32) -> ()',
    )

    assert len(op.operands) == 1
    assert op.comment == StringAttr("branch")
    assert op.attributes["tag"] == StringAttr("meta")


def test_parse_accepts_attributes_before_type_with_results():
    op = _parse_op(
        AddRsRsRdOp,
        '(%0, %1) {tag = "meta"} : (i32, i32) -> (i32)',
    )

    assert len(op.operands) == 2
    assert len(op.results) == 1
    assert op.attributes["tag"] == StringAttr("meta")


def test_parse_accepts_attribute_dictionary_without_operands():
    op = _parse_op(
        DefDirectiveOp,
        '() {alias = "CONST", value = "7", comment = "const", tag = "meta"} : () -> ()',
    )

    assert op.alias == StringAttr("CONST")
    assert op.value == StringAttr("7")
    assert op.comment == StringAttr("const")
    assert op.attributes["tag"] == StringAttr("meta")


def test_parse_accepts_attribute_dictionary_with_operands():
    op = _parse_op(
        JmpRsOp,
        '(%0) {comment = "branch", tag = "meta"} : (i32) -> ()',
    )

    assert len(op.operands) == 1
    assert op.comment == StringAttr("branch")
    assert op.attributes["tag"] == StringAttr("meta")


def test_parse_accepts_attributes_with_operands_and_results():
    op = _parse_op(
        AddRsRsRdOp,
        '(%0, %1) {tag = "meta"} : (i32, i32) -> (i32)',
    )

    assert len(op.operands) == 2
    assert len(op.results) == 1
    assert op.attributes["tag"] == StringAttr("meta")
