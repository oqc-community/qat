# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.irdl import irdl_op_definition
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.q1.ir.abstract_ops import (
    IIIOperation,
    IIOperation,
    Q1Instruction,
    RdIOperation,
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


def test_assembly_arg_str():
    assert _assembly_arg_str("plain") == "plain"
    assert _assembly_arg_str(Registers.R4) == "R4"
    assert _assembly_arg_str(IntegerAttr(11, ui32)) == "11"
    assert _assembly_arg_str(create_ssa_value(Registers.R5)) == "R5"

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
