# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from io import StringIO

import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.printer import Printer
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
    DefDirectiveOp,
    FbAcqIqIdIIOp,
    FbAcqTbValidIIOp,
    FbCmdIIIOp,
    FbCmdIRIOp,
    FbComDataIIIOp,
    FbComDataIRIOp,
    FbPopDataIROp,
    FbPullDataRROp,
    IllegalOp,
    JaIOp,
    JaROp,
    JmpIOp,
    JmpROp,
    LabelOp,
    LatchRstIOp,
    LatchRstROp,
    LoopRIOp,
    LoopRROp,
    MoveIROp,
    MoveRROp,
    NopOp,
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
    UpdParamIOp,
    UpdThresIIIOp,
    UpdThresIRIOp,
    WaitIOp,
    WaitROp,
    WaitSyncIOp,
    WaitSyncROp,
    WaitTriggerIIOp,
    WaitTriggerRROp,
)
from qat.experimental.dialect.q1.ir.attrs import LabelAttr
from qat.experimental.dialect.q1.ir.reg_desc import Registers


def _print_ir(op) -> str:
    stream = StringIO()
    Printer(stream=stream).print_op(op)
    return stream.getvalue()


@pytest.mark.parametrize(
    "op,operands,properties,attributes",
    [
        # --- Nullary ops (no operands, no properties) ---
        (IllegalOp(), None, None, None),
        (NopOp(), None, None, None),
        (StopOp(), None, None, None),
        (ResetPhOp(), None, None, None),
        # --- Immediate ops: no SSA operands, one or more properties ---
        (StopIOp(42), None, "<{imm = 42 : ui32}>", None),
        (JaIOp(100), None, "<{imm = 100 : ui32}>", None),
        (JmpIOp(50), None, "<{imm = 50 : ui32}>", None),
        (MoveIROp(source=999, rd=Registers.R0), None, "<{imm = 999 : ui32}>", None),
        (SetMrkIOp(0x1234), None, "<{imm = 4660 : ui32}>", None),
        (SetFreqIOp(5000000), None, "<{imm = 5000000 : ui32}>", None),
        (SetPhIOp(1000), None, "<{imm = 1000 : ui32}>", None),
        (SetPhDeltaIOp(500), None, "<{imm = 500 : ui32}>", None),
        (
            SetAwgGainIIOp(gain0=100, gain1=150),
            None,
            "<{imm1 = 100 : ui32, imm2 = 150 : ui32}>",
            None,
        ),
        (
            SetAwgOffsIIOp(offs0=200, offs1=250),
            None,
            "<{imm1 = 200 : ui32, imm2 = 250 : ui32}>",
            None,
        ),
        (
            SetCondIIIIOp(cond_en=1, mask=2, op=3, else_cnt=4),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 3 : ui32, imm4 = 4 : ui32}>",
            None,
        ),
        (LoopRIOp(Registers.R19, 500), None, "<{imm = 500 : ui32}>", None),
        (FbPopDataIROp(1, destination=Registers.R0), None, "<{imm = 1 : ui32}>", None),
        (
            FbComDataIIIOp(1, 2, 100),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 100 : ui32}>",
            None,
        ),
        (
            FbCmdIIIOp(1, 2, 100),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 100 : ui32}>",
            None,
        ),
        (
            FbAcqIqIdIIOp(1, 2),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32}>",
            None,
        ),
        (
            FbAcqTbValidIIOp(1, 2),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32}>",
            None,
        ),
        (WaitIOp(10), None, "<{imm = 10 : ui32}>", None),
        (WaitSyncIOp(20), None, "<{imm = 20 : ui32}>", None),
        (WaitTriggerIIOp(1, 50), None, "<{imm1 = 1 : ui32, imm2 = 50 : ui32}>", None),
        (
            PlayIIIOp(1, 2, 3),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 3 : ui32}>",
            None,
        ),
        (
            AcquireIIIOp(1, 2, 3),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 3 : ui32}>",
            None,
        ),
        (
            AcquireWeighedIIIIIOp(1, 2, 3, 4, 5),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 3 : ui32, imm4 = 4 : ui32, imm5 = 5 : ui32}>",
            None,
        ),
        (
            AcquireTtlIIIIOp(1, 2, 3, 4),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 3 : ui32, imm4 = 4 : ui32}>",
            None,
        ),
        (
            AcquireTimetagsIIIIIOp(1, 2, 3, 4, 5),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 3 : ui32, imm4 = 4 : ui32, imm5 = 5 : ui32}>",
            None,
        ),
        (
            AcquireDigitalIIIOp(1, 2, 3),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 3 : ui32}>",
            None,
        ),
        (
            UpdThresIIIOp(1, 2, 3),
            None,
            "<{imm1 = 1 : ui32, imm2 = 2 : ui32, imm3 = 3 : ui32}>",
            None,
        ),
        (UpdParamIOp(10), None, "<{imm = 10 : ui32}>", None),
        (LatchRstIOp(10), None, "<{imm = 10 : ui32}>", None),
        (SetLatchEnIIOp(1, 2), None, "<{imm1 = 1 : ui32, imm2 = 2 : ui32}>", None),
        # --- Register ops: SSA operands, no properties ---
        (JaROp(create_ssa_value(Registers.R0)), "(%0)", None, None),
        (JmpROp(create_ssa_value(Registers.R0)), "(%0)", None, None),
        # Result (%0) is R5; the source SSA value is printed as %1.
        (
            MoveRROp(source=create_ssa_value(Registers.R4), rd=Registers.R5),
            "(%1)",
            None,
            None,
        ),
        (SetMrkROp(create_ssa_value(Registers.R8)), "(%0)", None, None),
        (SetFreqROp(create_ssa_value(Registers.R9)), "(%0)", None, None),
        (SetPhROp(create_ssa_value(Registers.R10)), "(%0)", None, None),
        (SetPhDeltaROp(create_ssa_value(Registers.R11)), "(%0)", None, None),
        (
            SetAwgGainRROp(
                gain0=create_ssa_value(Registers.R12),
                gain1=create_ssa_value(Registers.R13),
            ),
            "(%0, %1)",
            None,
            None,
        ),
        (
            SetAwgOffsRROp(
                offs0=create_ssa_value(Registers.R14),
                offs1=create_ssa_value(Registers.R15),
            ),
            "(%0, %1)",
            None,
            None,
        ),
        # Result (%0) is R20; the address SSA value is printed as %1.
        (LoopRROp(Registers.R20, create_ssa_value(Registers.R21)), "(%1)", None, None),
        # FbPullDataRROp has two destination results and no SSA operands
        (FbPullDataRROp(id=Registers.R0, destination=Registers.R1), None, None, None),
        (
            # Result (%0) is R3; the two source SSA values are printed as %1, %2.
            AddRRROp(
                create_ssa_value(Registers.R1),
                create_ssa_value(Registers.R2),
                Registers.R3,
            ),
            "(%1, %2)",
            None,
            None,
        ),
        (WaitROp(create_ssa_value(Registers.R1)), "(%0)", None, None),
        (WaitSyncROp(create_ssa_value(Registers.R2)), "(%0)", None, None),
        (
            WaitTriggerRROp(create_ssa_value(Registers.R1), create_ssa_value(Registers.R2)),
            "(%0, %1)",
            None,
            None,
        ),
        (LatchRstROp(create_ssa_value(Registers.R1)), "(%0)", None, None),
        # --- Mixed: SSA operands + properties ---
        (
            # Result (%0) is R3; the source SSA value is printed as %1.
            AddRIROp(create_ssa_value(Registers.R1), 5, rd=Registers.R3),
            "(%1)",
            "<{imm = 5 : ui32}>",
            None,
        ),
        (
            SetCondRRRIOp(
                cond_en=create_ssa_value(Registers.R16),
                mask=create_ssa_value(Registers.R17),
                op=create_ssa_value(Registers.R18),
                else_cnt=100,
            ),
            "(%0, %1, %2)",
            "<{imm = 100 : ui32}>",
            None,
        ),
        (
            FbComDataIRIOp(1, create_ssa_value(Registers.R1), 100),
            "(%0)",
            "<{imm1 = 1 : ui32, imm2 = 100 : ui32}>",
            None,
        ),
        (
            FbCmdIRIOp(1, create_ssa_value(Registers.R1), 100),
            "(%0)",
            "<{imm1 = 1 : ui32, imm2 = 100 : ui32}>",
            None,
        ),
        (
            PlayRRIOp(create_ssa_value(Registers.R1), create_ssa_value(Registers.R2), 3),
            "(%0, %1)",
            "<{imm = 3 : ui32}>",
            None,
        ),
        (
            AcquireIRIOp(1, create_ssa_value(Registers.R1), 3),
            "(%0)",
            "<{imm1 = 1 : ui32, imm2 = 3 : ui32}>",
            None,
        ),
        (
            AcquireWeighedIRRRIOp(
                1,
                create_ssa_value(Registers.R1),
                create_ssa_value(Registers.R2),
                create_ssa_value(Registers.R3),
                5,
            ),
            "(%0, %1, %2)",
            "<{imm1 = 1 : ui32, imm2 = 5 : ui32}>",
            None,
        ),
        (
            AcquireTtlIRIIOp(1, create_ssa_value(Registers.R1), 3, 4),
            "(%0)",
            "<{imm1 = 1 : ui32, imm2 = 3 : ui32, imm3 = 4 : ui32}>",
            None,
        ),
        (
            AcquireTimetagsIRIRIOp(
                1,
                create_ssa_value(Registers.R1),
                3,
                create_ssa_value(Registers.R2),
                5,
            ),
            "(%0, %1)",
            "<{imm1 = 1 : ui32, imm2 = 3 : ui32, imm3 = 5 : ui32}>",
            None,
        ),
        (
            AcquireDigitalIRIOp(1, create_ssa_value(Registers.R1), 3),
            "(%0)",
            "<{imm1 = 1 : ui32, imm2 = 3 : ui32}>",
            None,
        ),
        (
            UpdThresIRIOp(1, create_ssa_value(Registers.R1), 3),
            "(%0)",
            "<{imm1 = 1 : ui32, imm2 = 3 : ui32}>",
            None,
        ),
        (
            SetLatchEnRIOp(create_ssa_value(Registers.R1), 2),
            "(%0)",
            "<{imm = 2 : ui32}>",
            None,
        ),
        # --- String-typed properties (directive and label ops) ---
        (
            DefDirectiveOp("CONST", "7", comment="const"),
            None,
            '<{alias = "CONST", value = "7", comment = "const"}>',
            None,
        ),
        (LabelOp("entry", comment="entry label"), None, "reference =", None),
        # --- Attributes only ---
        (
            LabelOp.create(attributes={"tag": StringAttr("x")}),
            None,
            None,
            '{tag = "x"}',
        ),
        # --- Both properties and attributes: verifies properties before attributes ---
        (
            LabelOp.create(
                properties={"reference": LabelAttr("entry")},
                attributes={"tag": StringAttr("x")},
            ),
            None,
            "reference =",
            '{tag = "x"}',
        ),
    ],
)
def test_ir_printer_emits_generic_operation_syntax(op, operands, properties, attributes):
    """Verify each op prints the MLIR generic operation syntax in the correct order.

    The expected textual form for Q1 ops (no successors or regions) is:   opname (operands)
    <{properties}> {attributes} : (input_types) -> (result_types)
    """

    text = _print_ir(op)
    # text.rindex(" : ") finds the type-signature delimiter even when " : " also
    # appears inside property value strings such as "<{imm = 1 : ui32}>".
    type_index = text.rindex(" : ")

    assert op.name in text

    if operands is not None:
        assert operands in text
        assert text.index(operands) < type_index

    if properties is not None:
        assert properties in text
        assert text.index(properties) < type_index

    if attributes is not None:
        assert attributes in text
        assert text.index(attributes) < type_index

    # Relative ordering: operands < properties < attributes < type
    if operands is not None and properties is not None:
        assert text.index(operands) < text.index(properties)
    if operands is not None and attributes is not None:
        assert text.index(operands) < text.index(attributes)
    if properties is not None and attributes is not None:
        assert text.index(properties) < text.index(attributes)
