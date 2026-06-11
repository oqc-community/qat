# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from __future__ import annotations

from io import StringIO

import pytest
from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr
from xdsl.irdl import irdl_op_definition
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.q1.ir.abstract_ops import (
    IIIIIOperation,
    IIIOperation,
    IIOperation,
    IRsIIOperation,
    IRsIOperation,
    IRsIRsIOperation,
    IRsRsRsIOperation,
    Q1Instruction,
    RdIOperation,
    RsRsIOperation,
    RsRsRdOperation,
    _assembly_arg_str,
)
from qat.experimental.dialect.q1.ir.imm_desc import ui32
from qat.experimental.dialect.q1.ir.reg_desc import IntRegisterType, Registers


@irdl_op_definition
class BadOp(Q1Instruction):
    name = "nonsense"


@irdl_op_definition
class DummyIIOp(IIOperation):
    name = "q1.ii.dummy"


@irdl_op_definition
class DummyRdIOp(RdIOperation[IntRegisterType]):
    name = "q1.ri.dummy"


@irdl_op_definition
class DummyRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    name = "q1.rrd.dummy"


@irdl_op_definition
class DummyIIIOp(IIIOperation):
    name = "q1.iii.dummy"


@irdl_op_definition
class DummyIRsIIOp(IRsIIOperation[IntRegisterType]):
    name = "q1.irii.dummy"


@irdl_op_definition
class DummyRsRsIOp(RsRsIOperation[IntRegisterType]):
    name = "q1.rri.dummy"


@irdl_op_definition
class DummyIRsRsRsIOp(IRsRsRsIOperation[IntRegisterType]):
    name = "q1.irrri.dummy"


@irdl_op_definition
class DummyIIIIIOp(IIIIIOperation):
    name = "q1.iiiii.dummy"


@irdl_op_definition
class DummyIRsIRsIOp(IRsIRsIOperation[IntRegisterType]):
    name = "q1.iriri.dummy"


@irdl_op_definition
class DummyIRsIOp(IRsIOperation[IntRegisterType]):
    name = "q1.iri.dummy"


def test_assembly_arg_str():
    assert _assembly_arg_str("plain") == "plain"
    assert _assembly_arg_str(Registers.R4) == "R4"
    assert _assembly_arg_str(IntegerAttr(11, ui32)) == "11"
    assert _assembly_arg_str(create_ssa_value(Registers.R5)) == "R5"
    assert _assembly_arg_str(StringAttr("text")) == "text"

    bad_ssa = create_ssa_value(IntegerType(32))
    with pytest.raises(ValueError, match="Unexpected register type"):
        _assembly_arg_str(bad_ssa)


def test_assembly_line_args_raises():
    with pytest.raises(NotImplementedError):
        BadOp().assembly_line_args()


def test_assembly_mnemonic_raises():
    with pytest.raises(ValueError, match="Invalid operation name"):
        BadOp().assembly_mnemonic()


def test_ii_operation_assembly_line_args():
    op = DummyIIOp(11, 22)
    assert tuple(_assembly_arg_str(arg) for arg in op.assembly_line_args()) == (
        "11",
        "22",
    )


def test_rdi_operation_with_integer_attr_imm():
    op = DummyRdIOp(Registers.R3, IntegerAttr(7, ui32))
    assert tuple(_assembly_arg_str(arg) for arg in op.assembly_line_args()) == (
        "R3",
        "7",
    )


def test_rrd_operation_comment_branch():
    op = DummyRsRsRdOp(
        create_ssa_value(Registers.R1),
        create_ssa_value(Registers.R2),
        Registers.R4,
        comment="hello",
    )
    assert tuple(_assembly_arg_str(arg) for arg in op.assembly_line_args()) == (
        "R1",
        "R2",
        "R4",
    )


def test_iii_operation_init_and_assembly_line_args():
    op = DummyIIIOp(1, 2, 3, comment="hello")
    assert tuple(_assembly_arg_str(arg) for arg in op.assembly_line_args()) == (
        "1",
        "2",
        "3",
    )


def test_iiiii_operation_init_and_assembly_line_args():
    op = DummyIIIIIOp(1, 2, 3, 4, 5, comment="hello")
    assert tuple(_assembly_arg_str(arg) for arg in op.assembly_line_args()) == (
        "1",
        "2",
        "3",
        "4",
        "5",
    )


def test_iriri_operation_init_and_assembly_line_args():
    op = DummyIRsIRsIOp(
        1,
        create_ssa_value(Registers.R2),
        3,
        create_ssa_value(Registers.R4),
        5,
        comment="hello",
    )
    assert tuple(_assembly_arg_str(arg) for arg in op.assembly_line_args()) == (
        "1",
        "R2",
        "3",
        "R4",
        "5",
    )


def test_rri_operation_init_and_assembly_line_args():
    op = DummyRsRsIOp(
        create_ssa_value(Registers.R1),
        create_ssa_value(Registers.R2),
        3,
        comment="hello",
    )
    assert tuple(_assembly_arg_str(arg) for arg in op.assembly_line_args()) == (
        "R1",
        "R2",
        "3",
    )


def test_irii_operation_init_and_assembly_line_args():
    op = DummyIRsIIOp(
        1,
        create_ssa_value(Registers.R2),
        3,
        4,
        comment="hello",
    )
    assert tuple(_assembly_arg_str(arg) for arg in op.assembly_line_args()) == (
        "1",
        "R2",
        "3",
        "4",
    )


def test_irrri_operation_init_and_assembly_line_args():
    op = DummyIRsRsRsIOp(
        1,
        create_ssa_value(Registers.R2),
        create_ssa_value(Registers.R3),
        create_ssa_value(Registers.R4),
        5,
        comment="hello",
    )
    assert tuple(_assembly_arg_str(arg) for arg in op.assembly_line_args()) == (
        "1",
        "R2",
        "R3",
        "R4",
        "5",
    )


def test_parse_with_multiple_operands():
    op = DummyRsRsRdOp.parse(Parser(None, "(%0, %1) : (i32, i32) -> (i32)"))
    assert len(op.operands) == 2
    assert len(op.results) == 1


def test_parse_accepts_attributes_in_mlir_position():
    op = BadOp.parse(Parser(None, '() {tag = "x"} : () -> ()'))
    assert op.attributes["tag"] == StringAttr("x")


def test_print_emits_attributes_when_present_in_memory():
    op = BadOp.create(attributes={"tag": StringAttr("x")})
    output = StringIO()
    op.print(Printer(output))
    assert '{tag = "x"}' in output.getvalue()


def test_irsi_operation_converts_integer_immediates():
    op = DummyIRsIOp(1, create_ssa_value(Registers.R2), 3)
    assert tuple(_assembly_arg_str(arg) for arg in op.assembly_line_args()) == (
        "1",
        "R2",
        "3",
    )
