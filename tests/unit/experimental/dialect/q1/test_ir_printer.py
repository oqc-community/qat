# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from io import StringIO

import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.printer import Printer
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.q1 import (
    AcquireImmImmImmOp,
    AcquireImmRsImmOp,
    AcquireTtlImmImmImmImmOp,
    AcquireTtlImmRsImmImmOp,
    AcquireWeightedImmImmImmImmImmOp,
    AcquireWeightedImmRsRsRsImmOp,
    AddressImm,
    AddRsImmRdOp,
    AddRsRsRdOp,
    BoolImm,
    CmpImmRsOp,
    CmpRsImmOp,
    CmpRsRsOp,
    DefDirectiveOp,
    DurationImm,
    FbAcqIqIdImmImmOp,
    FbAcqTbValidImmImmOp,
    FbComDataImmImmImmOp,
    FbComDataImmRsImmOp,
    FbPopDataImmRdOp,
    FbPullDataRsRdOp,
    IllegalOp,
    JaImmOp,
    JaRsOp,
    JgeImmOp,
    JgeRsImmImmOp,
    JgeRsImmRsOp,
    JgeRsOp,
    JmpImmOp,
    JmpRsOp,
    LabelAttr,
    LabelOp,
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
    Muls32ImmRsRdRdOp,
    Muls32RsImmRdRdOp,
    Muls32RsRsRdRdOp,
    Mulu16ImmRsRdOp,
    Mulu16RsImmRdOp,
    Mulu16RsRsRdOp,
    Mulu32ImmRsRdRdOp,
    Mulu32RsImmRdRdOp,
    Mulu32RsRsRdRdOp,
    NcoPhaseImm,
    NopOp,
    PlayImmImmImmOp,
    PlayRsRsImmOp,
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
    SetLatchEnImmImmOp,
    SetLatchEnRsImmOp,
    SetMrkImmOp,
    SetMrkRsOp,
    SetPhDeltaImmOp,
    SetPhDeltaRsOp,
    SetPhImmOp,
    SetPhRsOp,
    SI16Imm,
    SI32Imm,
    StopImmOp,
    StopOp,
    SU32Imm,
    TestImmRsOp,
    TestRsImmOp,
    TestRsRsOp,
    UI3Imm,
    UI4Imm,
    UI5Imm,
    UI6Imm,
    UI8Imm,
    UI10Imm,
    UI16Imm,
    UI24Imm,
    UI32Imm,
    UpdParamImmOp,
    WaitImmOp,
    WaitRsOp,
    WaitSyncImmOp,
    WaitSyncRsOp,
    WaitTriggerImmImmOp,
    WaitTriggerRsRsOp,
)


def _print_ir(op) -> str:
    stream = StringIO()
    Printer(stream=stream).print_op(op)
    return stream.getvalue()


@pytest.mark.parametrize(
    "op,operands,properties,attributes",
    [
        # Nullary ops (no operands, no properties)
        (IllegalOp(), None, None, None),
        (NopOp(), None, None, None),
        (StopOp(), None, None, None),
        (ResetPhOp(), None, None, None),
        # Immediate ops
        (StopImmOp(SI32Imm(42)), None, "<{imm = #q1.si32_imm<42>}>", None),
        (JaImmOp(AddressImm(100)), None, "<{imm = #q1.ui14_imm<100>}>", None),
        (JmpImmOp(AddressImm(50)), None, "<{imm = #q1.ui14_imm<50>}>", None),
        (
            MoveImmRdOp(SU32Imm(999), Registers.R0),
            None,
            "<{imm = #q1.su32_imm<999>}>",
            None,
        ),
        (SetMrkImmOp(UI4Imm(0x4)), None, "<{imm = #q1.ui4_imm<4>}>", None),
        (SetFreqImmOp(SI32Imm(5000000)), None, "<{imm = #q1.si32_imm<5000000>}>", None),
        (SetPhImmOp(NcoPhaseImm(1000)), None, "<{imm = #q1.nco_phase_imm<1000>}>", None),
        (SetPhDeltaImmOp(NcoPhaseImm(500)), None, "<{imm = #q1.nco_phase_imm<500>}>", None),
        (
            SetAwgGainImmImmOp(SI16Imm(100), SI16Imm(150)),
            None,
            "<{imm1 = #q1.si16_imm<100>, imm2 = #q1.si16_imm<150>}>",
            None,
        ),
        (
            SetAwgOffsImmImmOp(SI16Imm(200), SI16Imm(250)),
            None,
            "<{imm1 = #q1.si16_imm<200>, imm2 = #q1.si16_imm<250>}>",
            None,
        ),
        (
            SetCondImmImmImmImmOp(BoolImm(1), UI4Imm(2), UI3Imm(3), UI16Imm(4)),
            None,
            "<{imm1 = #q1.bool_imm<1>, imm2 = #q1.ui4_imm<2>, imm3 = #q1.ui3_imm<3>, imm4 = #q1.ui16_imm<4>}>",
            None,
        ),
        (
            LoopRdImmOp(Registers.R19, AddressImm(500)),
            None,
            "<{imm = #q1.ui14_imm<500>}>",
            None,
        ),
        (
            FbPopDataImmRdOp(UI16Imm(1), Registers.R0),
            None,
            "<{imm = #q1.ui16_imm<1>}>",
            None,
        ),
        (
            FbComDataImmImmImmOp(UI8Imm(1), UI32Imm(2), DurationImm(100)),
            None,
            "<{imm1 = #q1.ui8_imm<1>, imm2 = #q1.ui32_imm<2>, imm3 = #q1.duration_imm<100>}>",
            None,
        ),
        (
            FbAcqIqIdImmImmOp(UI8Imm(1), DurationImm(4)),
            None,
            "<{imm1 = #q1.ui8_imm<1>, imm2 = #q1.duration_imm<4>}>",
            None,
        ),
        (
            FbAcqTbValidImmImmOp(BoolImm(1), DurationImm(4)),
            None,
            "<{imm1 = #q1.bool_imm<1>, imm2 = #q1.duration_imm<4>}>",
            None,
        ),
        (WaitImmOp(DurationImm(10)), None, "<{imm = #q1.duration_imm<10>}>", None),
        (WaitSyncImmOp(DurationImm(20)), None, "<{imm = #q1.duration_imm<20>}>", None),
        (
            WaitTriggerImmImmOp(UI4Imm(1), DurationImm(50)),
            None,
            "<{imm1 = #q1.ui4_imm<1>, imm2 = #q1.duration_imm<50>}>",
            None,
        ),
        (
            PlayImmImmImmOp(UI10Imm(1), UI10Imm(2), DurationImm(4)),
            None,
            "<{imm1 = #q1.ui10_imm<1>, imm2 = #q1.ui10_imm<2>, imm3 = #q1.duration_imm<4>}>",
            None,
        ),
        (
            AcquireImmImmImmOp(UI5Imm(1), UI24Imm(2), DurationImm(4)),
            None,
            "<{imm1 = #q1.ui5_imm<1>, imm2 = #q1.ui24_imm<2>, imm3 = #q1.duration_imm<4>}>",
            None,
        ),
        (
            AcquireWeightedImmImmImmImmImmOp(
                UI5Imm(1), UI24Imm(2), UI6Imm(3), UI6Imm(4), DurationImm(5)
            ),
            None,
            "<{imm1 = #q1.ui5_imm<1>, imm2 = #q1.ui24_imm<2>, imm3 = #q1.ui6_imm<3>, imm4 = #q1.ui6_imm<4>, imm5 = #q1.duration_imm<5>}>",
            None,
        ),
        (
            AcquireTtlImmImmImmImmOp(UI5Imm(1), UI24Imm(2), BoolImm(1), DurationImm(4)),
            None,
            "<{imm1 = #q1.ui5_imm<1>, imm2 = #q1.ui24_imm<2>, imm3 = #q1.bool_imm<1>, imm4 = #q1.duration_imm<4>}>",
            None,
        ),
        (
            UpdParamImmOp(DurationImm(10)),
            None,
            "<{imm = #q1.duration_imm<10>}>",
            None,
        ),
        (LatchRstImmOp(DurationImm(10)), None, "<{imm = #q1.duration_imm<10>}>", None),
        (
            SetLatchEnImmImmOp(BoolImm(1), DurationImm(4)),
            None,
            "<{imm1 = #q1.bool_imm<1>, imm2 = #q1.duration_imm<4>}>",
            None,
        ),
        # Register ops
        (JaRsOp(create_ssa_value(Registers.R0)), "(%0)", None, None),
        (JmpRsOp(create_ssa_value(Registers.R0)), "(%0)", None, None),
        (
            MoveRsRdOp(create_ssa_value(Registers.R4), Registers.R5),
            "(%1)",
            None,
            None,
        ),
        (SetMrkRsOp(create_ssa_value(Registers.R8)), "(%0)", None, None),
        (SetFreqRsOp(create_ssa_value(Registers.R9)), "(%0)", None, None),
        (SetPhRsOp(create_ssa_value(Registers.R10)), "(%0)", None, None),
        (SetPhDeltaRsOp(create_ssa_value(Registers.R11)), "(%0)", None, None),
        (
            SetAwgGainRsRsOp(
                create_ssa_value(Registers.R12), create_ssa_value(Registers.R13)
            ),
            "(%0, %1)",
            None,
            None,
        ),
        (
            SetAwgOffsRsRsOp(
                create_ssa_value(Registers.R14), create_ssa_value(Registers.R15)
            ),
            "(%0, %1)",
            None,
            None,
        ),
        (LoopRdRsOp(Registers.R20, create_ssa_value(Registers.R21)), "(%1)", None, None),
        (
            FbPullDataRsRdOp(create_ssa_value(Registers.R0), Registers.R1),
            "(%1)",
            None,
            None,
        ),
        (
            AddRsRsRdOp(
                create_ssa_value(Registers.R1), create_ssa_value(Registers.R2), Registers.R3
            ),
            "(%1, %2)",
            None,
            None,
        ),
        (WaitRsOp(create_ssa_value(Registers.R1)), "(%0)", None, None),
        (WaitSyncRsOp(create_ssa_value(Registers.R2)), "(%0)", None, None),
        (
            WaitTriggerRsRsOp(
                create_ssa_value(Registers.R1), create_ssa_value(Registers.R2)
            ),
            "(%0, %1)",
            None,
            None,
        ),
        (LatchRstRsOp(create_ssa_value(Registers.R1)), "(%0)", None, None),
        # Mixed
        (
            AddRsImmRdOp(create_ssa_value(Registers.R1), SU32Imm(5), Registers.R3),
            "(%1)",
            "<{imm = #q1.su32_imm<5>}>",
            None,
        ),
        (
            JgeRsImmImmOp(create_ssa_value(Registers.R1), UI32Imm(5), AddressImm(100)),
            "(%0)",
            "<{imm1 = #q1.ui32_imm<5>, imm2 = #q1.ui14_imm<100>}>",
            None,
        ),
        (
            JgeRsImmRsOp(
                create_ssa_value(Registers.R1), UI32Imm(5), create_ssa_value(Registers.R2)
            ),
            "(%0, %1)",
            "<{imm = #q1.ui32_imm<5>}>",
            None,
        ),
        (
            SetCondRsRsRsImmOp(
                create_ssa_value(Registers.R16),
                create_ssa_value(Registers.R17),
                create_ssa_value(Registers.R18),
                UI16Imm(100),
            ),
            "(%0, %1, %2)",
            "<{imm = #q1.ui16_imm<100>}>",
            None,
        ),
        (
            FbComDataImmRsImmOp(
                UI8Imm(1), create_ssa_value(Registers.R1), DurationImm(100)
            ),
            "(%0)",
            "<{imm1 = #q1.ui8_imm<1>, imm2 = #q1.duration_imm<100>}>",
            None,
        ),
        (
            PlayRsRsImmOp(
                create_ssa_value(Registers.R1),
                create_ssa_value(Registers.R2),
                DurationImm(4),
            ),
            "(%0, %1)",
            "<{imm = #q1.duration_imm<4>}>",
            None,
        ),
        (
            AcquireImmRsImmOp(UI5Imm(1), create_ssa_value(Registers.R1), DurationImm(4)),
            "(%0)",
            "<{imm1 = #q1.ui5_imm<1>, imm2 = #q1.duration_imm<4>}>",
            None,
        ),
        (
            AcquireWeightedImmRsRsRsImmOp(
                UI5Imm(1),
                create_ssa_value(Registers.R1),
                create_ssa_value(Registers.R2),
                create_ssa_value(Registers.R3),
                DurationImm(5),
            ),
            "(%0, %1, %2)",
            "<{imm1 = #q1.ui5_imm<1>, imm2 = #q1.duration_imm<5>}>",
            None,
        ),
        (
            AcquireTtlImmRsImmImmOp(
                UI5Imm(1), create_ssa_value(Registers.R1), BoolImm(1), DurationImm(4)
            ),
            "(%0)",
            "<{imm1 = #q1.ui5_imm<1>, imm2 = #q1.bool_imm<1>, imm3 = #q1.duration_imm<4>}>",
            None,
        ),
        (
            SetLatchEnRsImmOp(create_ssa_value(Registers.R1), DurationImm(4)),
            "(%0)",
            "<{imm = #q1.duration_imm<4>}>",
            None,
        ),
        (JgeImmOp(AddressImm(115)), None, "<{imm = #q1.ui14_imm<115>}>", None),
        (JgeRsOp(create_ssa_value(Registers.R5)), "(%0)", None, None),
        (
            CmpRsRsOp(create_ssa_value(Registers.R1), create_ssa_value(Registers.R2)),
            "(%0, %1)",
            None,
            None,
        ),
        (
            CmpRsImmOp(create_ssa_value(Registers.R3), SU32Imm(30)),
            "(%0)",
            "<{imm = #q1.su32_imm<30>}>",
            None,
        ),
        (
            CmpImmRsOp(SU32Imm(31), create_ssa_value(Registers.R4)),
            "(%0)",
            "<{imm = #q1.su32_imm<31>}>",
            None,
        ),
        (
            TestRsRsOp(create_ssa_value(Registers.R5), create_ssa_value(Registers.R6)),
            "(%0, %1)",
            None,
            None,
        ),
        (
            TestRsImmOp(create_ssa_value(Registers.R7), SU32Imm(32)),
            "(%0)",
            "<{imm = #q1.su32_imm<32>}>",
            None,
        ),
        (
            TestImmRsOp(SU32Imm(33), create_ssa_value(Registers.R8)),
            "(%0)",
            "<{imm = #q1.su32_imm<33>}>",
            None,
        ),
        (
            LsrRsImmRdOp(create_ssa_value(Registers.R1), UI32Imm(34), Registers.R0),
            "(%1)",
            "<{imm = #q1.ui32_imm<34>}>",
            None,
        ),
        (
            LsrRsRsRdOp(
                create_ssa_value(Registers.R2), create_ssa_value(Registers.R3), Registers.R4
            ),
            "(%1, %2)",
            None,
            None,
        ),
        (
            LsrImmRsRdOp(UI32Imm(35), create_ssa_value(Registers.R5), Registers.R6),
            "(%1)",
            "<{imm = #q1.ui32_imm<35>}>",
            None,
        ),
        (
            LslRsImmRdOp(create_ssa_value(Registers.R7), UI32Imm(36), Registers.R8),
            "(%1)",
            "<{imm = #q1.ui32_imm<36>}>",
            None,
        ),
        (
            LslRsRsRdOp(
                create_ssa_value(Registers.R9),
                create_ssa_value(Registers.R10),
                Registers.R11,
            ),
            "(%1, %2)",
            None,
            None,
        ),
        (
            LslImmRsRdOp(UI32Imm(37), create_ssa_value(Registers.R12), Registers.R13),
            "(%1)",
            "<{imm = #q1.ui32_imm<37>}>",
            None,
        ),
        (
            Mulu16RsImmRdOp(create_ssa_value(Registers.R1), UI16Imm(40), Registers.R0),
            "(%1)",
            "<{imm = #q1.ui16_imm<40>}>",
            None,
        ),
        (
            Mulu16RsRsRdOp(
                create_ssa_value(Registers.R2), create_ssa_value(Registers.R3), Registers.R4
            ),
            "(%1, %2)",
            None,
            None,
        ),
        (
            Mulu16ImmRsRdOp(UI16Imm(41), create_ssa_value(Registers.R5), Registers.R6),
            "(%1)",
            "<{imm = #q1.ui16_imm<41>}>",
            None,
        ),
        (
            Muls16RsImmRdOp(create_ssa_value(Registers.R1), SI16Imm(42), Registers.R0),
            "(%1)",
            "<{imm = #q1.si16_imm<42>}>",
            None,
        ),
        (
            Muls16RsRsRdOp(
                create_ssa_value(Registers.R2), create_ssa_value(Registers.R3), Registers.R4
            ),
            "(%1, %2)",
            None,
            None,
        ),
        (
            Muls16ImmRsRdOp(SI16Imm(43), create_ssa_value(Registers.R5), Registers.R6),
            "(%1)",
            "<{imm = #q1.si16_imm<43>}>",
            None,
        ),
        (
            Mulu32RsImmRdRdOp(
                create_ssa_value(Registers.R1), UI32Imm(60), Registers.R0, Registers.R2
            ),
            "(%2)",
            "<{imm = #q1.ui32_imm<60>}>",
            None,
        ),
        (
            Mulu32RsRsRdRdOp(
                create_ssa_value(Registers.R3),
                create_ssa_value(Registers.R4),
                Registers.R5,
                Registers.R6,
            ),
            "(%2, %3)",
            None,
            None,
        ),
        (
            Mulu32ImmRsRdRdOp(
                UI32Imm(61), create_ssa_value(Registers.R7), Registers.R8, Registers.R9
            ),
            "(%2)",
            "<{imm = #q1.ui32_imm<61>}>",
            None,
        ),
        (
            Muls32RsImmRdRdOp(
                create_ssa_value(Registers.R10), SI32Imm(62), Registers.R11, Registers.R12
            ),
            "(%2)",
            "<{imm = #q1.si32_imm<62>}>",
            None,
        ),
        (
            Muls32RsRsRdRdOp(
                create_ssa_value(Registers.R13),
                create_ssa_value(Registers.R14),
                Registers.R15,
                Registers.R16,
            ),
            "(%2, %3)",
            None,
            None,
        ),
        (
            Muls32ImmRsRdRdOp(
                SI32Imm(63), create_ssa_value(Registers.R17), Registers.R18, Registers.R19
            ),
            "(%2)",
            "<{imm = #q1.si32_imm<63>}>",
            None,
        ),
        # String-typed properties (directive and label ops)
        (
            DefDirectiveOp("CONST", "7", comment="const"),
            None,
            '<{alias = "CONST", value = "7", comment = "const"}>',
            None,
        ),
        (LabelOp("entry", comment="entry label"), None, "reference =", None),
        # Attributes only
        (
            LabelOp.create(attributes={"tag": StringAttr("x")}),
            None,
            None,
            '{tag = "x"}',
        ),
        # Both properties and attributes
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

    if operands is not None and properties is not None:
        assert text.index(operands) < text.index(properties)
    if operands is not None and attributes is not None:
        assert text.index(operands) < text.index(attributes)
    if properties is not None and attributes is not None:
        assert text.index(properties) < text.index(attributes)
