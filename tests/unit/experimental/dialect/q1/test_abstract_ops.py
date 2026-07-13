# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from __future__ import annotations

from io import StringIO

import pytest
from xdsl.dialects.builtin import IntegerType, StringAttr
from xdsl.irdl import irdl_op_definition
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.q1 import IntRegisterType, Registers, UI32Imm
from qat.experimental.dialect.q1.ir.abstract_ops import (
    ImmImmImmImmImmOperation,
    ImmImmImmOperation,
    ImmImmOperation,
    ImmRsImmImmOperation,
    ImmRsImmOperation,
    ImmRsOperation,
    ImmRsRdOperation,
    ImmRsRdRdOperation,
    ImmRsRsRsImmOperation,
    Q1Instruction,
    RdImmOperation,
    RsImmRdRdOperation,
    RsRsImmOperation,
    RsRsRdOperation,
    RsRsRdRdOperation,
    _assembly_arg_str,
)


@irdl_op_definition
class BadOp(Q1Instruction):
    name = "nonsense"


@irdl_op_definition
class DummyIIOp(ImmImmOperation[UI32Imm, UI32Imm]):
    name = "q1.ii.dummy"


@irdl_op_definition
class DummyRdIOp(RdImmOperation[IntRegisterType, UI32Imm]):
    name = "q1.ri.dummy"


@irdl_op_definition
class DummyRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    name = "q1.rrd.dummy"


@irdl_op_definition
class DummyIIIOp(ImmImmImmOperation[UI32Imm, UI32Imm, UI32Imm]):
    name = "q1.iii.dummy"


@irdl_op_definition
class DummyIRsIIOp(ImmRsImmImmOperation[IntRegisterType, UI32Imm, UI32Imm, UI32Imm]):
    name = "q1.irii.dummy"


@irdl_op_definition
class DummyRsRsIOp(RsRsImmOperation[IntRegisterType, UI32Imm]):
    name = "q1.rri.dummy"


@irdl_op_definition
class DummyIRsRsRsIOp(ImmRsRsRsImmOperation[IntRegisterType, UI32Imm, UI32Imm]):
    name = "q1.irrri.dummy"


@irdl_op_definition
class DummyIIIIIOp(ImmImmImmImmImmOperation[UI32Imm, UI32Imm, UI32Imm, UI32Imm, UI32Imm]):
    name = "q1.iiiii.dummy"


@irdl_op_definition
class DummyIRsIOp(ImmRsImmOperation[IntRegisterType, UI32Imm, UI32Imm]):
    name = "q1.iri.dummy"


@irdl_op_definition
class DummyIRsOp(ImmRsOperation[IntRegisterType, UI32Imm]):
    name = "q1.ir.dummy"


@irdl_op_definition
class DummyIRsRdOp(ImmRsRdOperation[IntRegisterType, UI32Imm]):
    name = "q1.irr.dummy"


@irdl_op_definition
class DummyRsRsRdRdOp(RsRsRdRdOperation[IntRegisterType]):
    name = "q1.rrrr.dummy"


@irdl_op_definition
class DummyRsIRdRdOp(RsImmRdRdOperation[IntRegisterType, UI32Imm]):
    name = "q1.rirr.dummy"


@irdl_op_definition
class DummyIRsRdRdOp(ImmRsRdRdOperation[IntRegisterType, UI32Imm]):
    name = "q1.irrr.dummy"


def test_assembly_arg_str():
    assert _assembly_arg_str("plain") == "plain"
    assert _assembly_arg_str(Registers.R4) == "R4"
    assert _assembly_arg_str(UI32Imm(11)) == "11"
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


@pytest.mark.parametrize(
    "op_factory,expected_args",
    [
        pytest.param(
            lambda: DummyIIOp(UI32Imm(11), UI32Imm(22)),
            ("11", "22"),
            id="ii",
        ),
        pytest.param(
            lambda: DummyRdIOp(Registers.R3, UI32Imm(7)),
            ("R3", "7"),
            id="rd_i",
        ),
        pytest.param(
            lambda: DummyRsRsRdOp(
                create_ssa_value(Registers.R1),
                create_ssa_value(Registers.R2),
                Registers.R4,
                comment="hello",
            ),
            ("R1", "R2", "R4"),
            id="rs_rs_rd",
        ),
        pytest.param(
            lambda: DummyIIIOp(UI32Imm(1), UI32Imm(2), UI32Imm(3), comment="hello"),
            ("1", "2", "3"),
            id="iii",
        ),
        pytest.param(
            lambda: DummyIIIIIOp(
                UI32Imm(1),
                UI32Imm(2),
                UI32Imm(3),
                UI32Imm(4),
                UI32Imm(5),
                comment="hello",
            ),
            ("1", "2", "3", "4", "5"),
            id="iiiii",
        ),
        pytest.param(
            lambda: DummyRsRsIOp(
                create_ssa_value(Registers.R1),
                create_ssa_value(Registers.R2),
                UI32Imm(3),
                comment="hello",
            ),
            ("R1", "R2", "3"),
            id="rs_rs_i",
        ),
        pytest.param(
            lambda: DummyIRsIIOp(
                UI32Imm(1),
                create_ssa_value(Registers.R2),
                UI32Imm(3),
                UI32Imm(4),
                comment="hello",
            ),
            ("1", "R2", "3", "4"),
            id="i_rs_i_i",
        ),
        pytest.param(
            lambda: DummyIRsRsRsIOp(
                UI32Imm(1),
                create_ssa_value(Registers.R2),
                create_ssa_value(Registers.R3),
                create_ssa_value(Registers.R4),
                UI32Imm(5),
                comment="hello",
            ),
            ("1", "R2", "R3", "R4", "5"),
            id="i_rs_rs_rs_i",
        ),
        pytest.param(
            lambda: DummyIRsIOp(UI32Imm(1), create_ssa_value(Registers.R2), UI32Imm(3)),
            ("1", "R2", "3"),
            id="i_rs_i",
        ),
        pytest.param(
            lambda: DummyIRsOp(UI32Imm(1), create_ssa_value(Registers.R2), comment="hello"),
            ("1", "R2"),
            id="i_rs",
        ),
        pytest.param(
            lambda: DummyIRsRdOp(
                UI32Imm(1),
                create_ssa_value(Registers.R2),
                Registers.R3,
                comment="hello",
            ),
            ("1", "R2", "R3"),
            id="i_rs_rd",
        ),
        pytest.param(
            lambda: DummyRsRsRdRdOp(
                create_ssa_value(Registers.R1),
                create_ssa_value(Registers.R2),
                Registers.R3,
                Registers.R4,
                comment="hello",
            ),
            ("R1", "R2", "R3", "R4"),
            id="rs_rs_rd_rd",
        ),
        pytest.param(
            lambda: DummyRsIRdRdOp(
                create_ssa_value(Registers.R1),
                UI32Imm(2),
                Registers.R3,
                Registers.R4,
                comment="hello",
            ),
            ("R1", "2", "R3", "R4"),
            id="rs_i_rd_rd",
        ),
        pytest.param(
            lambda: DummyIRsRdRdOp(
                UI32Imm(1),
                create_ssa_value(Registers.R2),
                Registers.R3,
                Registers.R4,
                comment="hello",
            ),
            ("1", "R2", "R3", "R4"),
            id="i_rs_rd_rd",
        ),
    ],
)
def test_shape_assembly_line_args(op_factory, expected_args) -> None:
    """``assembly_line_args()`` stringifies each shape's args via ``_assembly_arg_str``."""

    op = op_factory()
    actual = tuple(_assembly_arg_str(arg) for arg in op.assembly_line_args())
    assert actual == expected_args


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
