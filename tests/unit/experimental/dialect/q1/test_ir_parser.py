# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.parser import Parser

from qat.experimental.dialect.q1.ir.ops import (
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

NULLARY_OPS = (IllegalOp, NopOp, ResetPhOp, StopOp)
IMM_OPS = (
    JaIOp,
    JaeIOp,
    JbIOp,
    JbeIOp,
    JgIOp,
    JlIOp,
    JleIOp,
    JmpIOp,
    JnoIOp,
    JnsIOp,
    JnzIOp,
    JoIOp,
    JsIOp,
    JzIOp,
    SetFreqIOp,
    SetMrkIOp,
    SetPhDeltaIOp,
    SetPhIOp,
    WaitIOp,
    WaitSyncIOp,
    UpdParamIOp,
    LatchRstIOp,
    StopIOp,
)
RS_OPS = (
    JaROp,
    JaeROp,
    JbROp,
    JbeROp,
    JgROp,
    JlROp,
    JleROp,
    JmpROp,
    JnoROp,
    JnsROp,
    JnzROp,
    JoROp,
    JsROp,
    JzROp,
    SetFreqROp,
    SetMrkROp,
    SetPhDeltaROp,
    SetPhROp,
    WaitROp,
    WaitSyncROp,
    LatchRstROp,
    StopROp,
)
II_OPS = (
    FbAcqIqIdIIOp,
    FbAcqIqShiftIIOp,
    FbAcqTbIdIIOp,
    FbAcqTbValidIIOp,
    SetAwgGainIIOp,
    SetAwgOffsIIOp,
    WaitTriggerIIOp,
    SetLatchEnIIOp,
)
IMM_RD_OPS = (FbPopDataIROp, MoveIROp, NotIROp)
RS_RD_OPS = (MoveRROp, NotRROp)
RS_IMM_OPS = (FbAcqIqIdRIOp, FbAcqTbIdRIOp, FbAcqTbValidRIOp, SetLatchEnRIOp)
RS_IMM_IMM_OPS = (JgeRIIOp, JltRIIOp)
RS_IMM_RS_OPS = (JgeRIROp, JltRIROp)
RI_ARITHMETIC_OPS = (
    AddRIROp,
    AndRIROp,
    AslRIROp,
    AsrRIROp,
    OrRIROp,
    SubRIROp,
    XorRIROp,
)
RR_ARITHMETIC_OPS = (
    AddRRROp,
    AndRRROp,
    AslRRROp,
    AsrRRROp,
    OrRRROp,
    SubRRROp,
    XorRRROp,
)
IMM_IMM_IMM_OPS = (
    AcquireIIIOp,
    AcquireDigitalIIIOp,
    FbAcqTbExtraIIIOp,
    FbCmdIIIOp,
    FbComDataIIIOp,
    FbComExtraIIIOp,
    PlayIIIOp,
    UpdThresIIIOp,
)
IMM_RS_IMM_OPS = (
    FbCmdIRIOp,
    FbComDataIRIOp,
    AcquireIRIOp,
    AcquireDigitalIRIOp,
    UpdThresIRIOp,
)
IMM_IMM_IMM_IMM_OPS = (
    AcquireTtlIIIIOp,
    FbAcqTbCfgIIIIOp,
    FbAcqTbMockIIIIOp,
    FbComCfgIIIIOp,
    SetCondIIIIOp,
)
IMM_IMM_IMM_IMM_IMM_OPS = (AcquireWeighedIIIIIOp, AcquireTimetagsIIIIIOp)
RS_RS_RS_IMM_OPS = (SetCondRRRIOp,)
RS_RS_OPS = (SetAwgGainRROp, SetAwgOffsRROp, WaitTriggerRROp)
RS_RS_IMM_OPS = (PlayRRIOp,)
RS_IMM_IMM_IMM_OPS = (AcquireTtlIRIIOp,)
RS_RS_IMM_IMM_IMM_OPS = (AcquireTimetagsIRIRIOp,)
RS_RS_RS_IMM_IMM_OPS = (AcquireWeighedIRRRIOp,)
RD_IMM_OPS = (LoopRIOp,)
RD_RS_OPS = (LoopRROp,)
RD_RD_OPS = (FbPullDataRROp,)


def _parse_op_type(op_type, signature: str) -> tuple[int, int]:
    parser = Parser(None, signature)
    operand_types, result_types = op_type.parse_op_type(parser)
    return len(operand_types), len(result_types)


def _parse_op(op_type, signature: str):
    return op_type.parse(Parser(None, signature))


I_PROPS = "<{imm = 1 : ui32}>"
II_PROPS = "<{imm1 = 1 : ui32, imm2 = 2 : ui32}>"
III_PROPS = "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 3 : ui32}>"
IIII_PROPS = "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 3 : ui32, imm4 = 4 : ui32}>"
IIIII_PROPS = "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 3 : ui32, imm4 = 4 : ui32, imm5 = 5 : ui32}>"


@pytest.mark.parametrize("op_type", NULLARY_OPS)
def test_nullary_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": () -> ()")
    assert operand_count == 0
    assert result_count == 0

    op = _parse_op(op_type, "() : () -> ()")
    assert isinstance(op, op_type)


@pytest.mark.parametrize("op_type", IMM_OPS)
def test_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": () -> ()")
    assert operand_count == 0
    assert result_count == 0

    op = _parse_op(op_type, f"() {I_PROPS} : () -> ()")
    assert isinstance(op, op_type)


@pytest.mark.parametrize("op_type", RS_OPS)
def test_rs_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32) -> ()")
    assert operand_count == 1
    assert result_count == 0

    op = _parse_op(op_type, "(%0) : (i32) -> ()")
    assert isinstance(op, op_type)
    assert len(op.operands) == 1


@pytest.mark.parametrize("op_type", II_OPS)
def test_ii_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": () -> ()")
    assert operand_count == 0
    assert result_count == 0

    op = _parse_op(op_type, f"() {II_PROPS} : () -> ()")
    assert isinstance(op, op_type)


@pytest.mark.parametrize("op_type", IMM_RD_OPS)
def test_imm_rd_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": () -> (i32)")
    assert operand_count == 0
    assert result_count == 1

    op = _parse_op(op_type, f"() {I_PROPS} : () -> (i32)")
    assert isinstance(op, op_type)
    assert len(op.results) == 1


@pytest.mark.parametrize("op_type", RS_RD_OPS)
def test_rs_rd_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32) -> (i32)")
    assert operand_count == 1
    assert result_count == 1

    op = _parse_op(op_type, "(%0) : (i32) -> (i32)")
    assert isinstance(op, op_type)
    assert len(op.operands) == 1
    assert len(op.results) == 1


@pytest.mark.parametrize("op_type", RS_IMM_OPS)
def test_rs_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32) -> ()")
    assert operand_count == 1
    assert result_count == 0

    op = _parse_op(op_type, f"(%0) {I_PROPS} : (i32) -> ()")
    assert isinstance(op, op_type)
    assert len(op.operands) == 1


@pytest.mark.parametrize("op_type", RS_IMM_IMM_OPS)
def test_rs_imm_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32) -> ()")
    assert operand_count == 1
    assert result_count == 0

    op = _parse_op(op_type, f"(%0) {II_PROPS} : (i32) -> ()")
    assert isinstance(op, op_type)
    assert len(op.operands) == 1


@pytest.mark.parametrize("op_type", RS_IMM_RS_OPS)
def test_rs_imm_rs_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32, i32) -> ()")
    assert operand_count == 2
    assert result_count == 0

    op = _parse_op(op_type, f"(%0, %1) {I_PROPS} : (i32, i32) -> ()")
    assert isinstance(op, op_type)
    assert len(op.operands) == 2


@pytest.mark.parametrize("op_type", RI_ARITHMETIC_OPS)
def test_ri_arithmetic_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32) -> (i32)")
    assert operand_count == 1
    assert result_count == 1

    op = _parse_op(op_type, f"(%0) {I_PROPS} : (i32) -> (i32)")
    assert isinstance(op, op_type)
    assert len(op.operands) == 1
    assert len(op.results) == 1


@pytest.mark.parametrize("op_type", RR_ARITHMETIC_OPS)
def test_rr_arithmetic_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32, i32) -> (i32)")
    assert operand_count == 2
    assert result_count == 1

    op = _parse_op(op_type, "(%0, %1) : (i32, i32) -> (i32)")
    assert isinstance(op, op_type)
    assert len(op.operands) == 2
    assert len(op.results) == 1


@pytest.mark.parametrize("op_type", IMM_IMM_IMM_OPS)
def test_imm_imm_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": () -> ()")
    assert operand_count == 0
    assert result_count == 0

    op = _parse_op(op_type, f"() {III_PROPS} : () -> ()")
    assert isinstance(op, op_type)


@pytest.mark.parametrize("op_type", IMM_RS_IMM_OPS)
def test_imm_rs_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32) -> ()")
    assert operand_count == 1
    assert result_count == 0

    op = _parse_op(op_type, f"(%0) {II_PROPS} : (i32) -> ()")
    assert isinstance(op, op_type)
    assert len(op.operands) == 1


@pytest.mark.parametrize("op_type", IMM_IMM_IMM_IMM_OPS)
def test_imm_imm_imm_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(
        op_type,
        ": () -> ()",
    )
    assert operand_count == 0
    assert result_count == 0

    op = _parse_op(op_type, f"() {IIII_PROPS} : () -> ()")
    assert isinstance(op, op_type)


@pytest.mark.parametrize("op_type", IMM_IMM_IMM_IMM_IMM_OPS)
def test_imm_imm_imm_imm_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(
        op_type,
        ": () -> ()",
    )
    assert operand_count == 0
    assert result_count == 0

    op = _parse_op(op_type, f"() {IIIII_PROPS} : () -> ()")
    assert isinstance(op, op_type)


@pytest.mark.parametrize("op_type", RS_RS_RS_IMM_OPS)
def test_rs_rs_rs_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32, i32, i32) -> ()")
    assert operand_count == 3
    assert result_count == 0

    op = _parse_op(op_type, f"(%0, %1, %2) {I_PROPS} : (i32, i32, i32) -> ()")
    assert isinstance(op, op_type)
    assert len(op.operands) == 3


@pytest.mark.parametrize("op_type", RS_RS_IMM_OPS)
def test_rs_rs_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32, i32) -> ()")
    assert operand_count == 2
    assert result_count == 0

    op = _parse_op(op_type, f"(%0, %1) {I_PROPS} : (i32, i32) -> ()")
    assert isinstance(op, op_type)
    assert len(op.operands) == 2


@pytest.mark.parametrize("op_type", RS_IMM_IMM_IMM_OPS)
def test_rs_imm_imm_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32) -> ()")
    assert operand_count == 1
    assert result_count == 0

    op = _parse_op(op_type, f"(%0) {III_PROPS} : (i32) -> ()")
    assert isinstance(op, op_type)
    assert len(op.operands) == 1


@pytest.mark.parametrize("op_type", RS_RS_IMM_IMM_IMM_OPS)
def test_rs_rs_imm_imm_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32, i32) -> ()")
    assert operand_count == 2
    assert result_count == 0

    op = _parse_op(op_type, f"(%0, %1) {III_PROPS} : (i32, i32) -> ()")
    assert isinstance(op, op_type)
    assert len(op.operands) == 2


@pytest.mark.parametrize("op_type", RS_RS_RS_IMM_IMM_OPS)
def test_rs_rs_rs_imm_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32, i32, i32) -> ()")
    assert operand_count == 3
    assert result_count == 0

    op = _parse_op(op_type, f"(%0, %1, %2) {II_PROPS} : (i32, i32, i32) -> ()")
    assert isinstance(op, op_type)
    assert len(op.operands) == 3


@pytest.mark.parametrize("op_type", RD_IMM_OPS)
def test_rd_imm_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": () -> (i32)")
    assert operand_count == 0
    assert result_count == 1

    op = _parse_op(op_type, f"() {I_PROPS} : () -> (i32)")
    assert isinstance(op, op_type)
    assert len(op.results) == 1


@pytest.mark.parametrize("op_type", RD_RS_OPS)
def test_rd_rs_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32) -> (i32)")
    assert operand_count == 1
    assert result_count == 1

    op = _parse_op(op_type, "(%0) : (i32) -> (i32)")
    assert isinstance(op, op_type)
    assert len(op.operands) == 1
    assert len(op.results) == 1


@pytest.mark.parametrize("op_type", RS_RS_OPS)
def test_rs_rs_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": (i32, i32) -> ()")
    assert operand_count == 2
    assert result_count == 0

    op = _parse_op(op_type, "(%0, %1) : (i32, i32) -> ()")
    assert isinstance(op, op_type)
    assert len(op.operands) == 2


@pytest.mark.parametrize("op_type", RD_RD_OPS)
def test_rd_rd_format_parse(op_type):
    operand_count, result_count = _parse_op_type(op_type, ": () -> (i32, i32)")
    assert operand_count == 0
    assert result_count == 2

    op = _parse_op(op_type, "() : () -> (i32, i32)")
    assert isinstance(op, op_type)
    assert len(op.results) == 2


def test_parse_rejects_trailing_comma_in_operand_list():
    with pytest.raises(Exception, match="operand expected"):
        _parse_op(JmpROp, "(%0, ) : (i32) -> ()")


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
        JmpROp,
        '(%0) <{comment = "branch"}> {tag = "meta"} : (i32) -> ()',
    )

    assert len(op.operands) == 1
    assert op.comment == StringAttr("branch")
    assert op.attributes["tag"] == StringAttr("meta")


def test_parse_accepts_attributes_before_type_with_results():
    op = _parse_op(
        AddRRROp,
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
        JmpROp,
        '(%0) {comment = "branch", tag = "meta"} : (i32) -> ()',
    )

    assert len(op.operands) == 1
    assert op.comment == StringAttr("branch")
    assert op.attributes["tag"] == StringAttr("meta")


def test_parse_accepts_attributes_with_operands_and_results():
    op = _parse_op(
        AddRRROp,
        '(%0, %1) {tag = "meta"} : (i32, i32) -> (i32)',
    )

    assert len(op.operands) == 2
    assert len(op.results) == 1
    assert op.attributes["tag"] == StringAttr("meta")
