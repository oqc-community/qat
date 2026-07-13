# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for QBlox Q1 dialect concrete operations.

Tests are organised by operand-shape: one class per abstract op format from
:mod:`qat.experimental.dialect.q1.ir.abstract_ops`. Each class registers its concrete
ops in an ``_OPS_TABLE`` and parametrises a shared set of structural tests.

Adding a new op of a known shape normally only requires appending one entry to the
relevant ``_OPS_TABLE``. Ops that don't share a shape (``LabelOp``, ``DefDirectiveOp``)
get their own dedicated test class.
"""

from __future__ import annotations

from typing import Any

import pytest
from xdsl.backend.register_allocatable import (
    HasRegisterConstraintsTrait,
    RegisterConstraints,
)
from xdsl.dialects.builtin import StringAttr
from xdsl.traits import Commutative, IsTerminator, Pure
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
    FbPullDataRdRdOp,
    IllegalOp,
    IntRegisterType,
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
    Q1Imm,
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

COMMENT_INPUTS = ["test-comment", StringAttr("test-comment")]


def _imm_types(op_class) -> tuple[type[Q1Imm], ...]:
    """Return the Q1Imm types bound to each imm slot of an op class (in slot order)."""
    import typing

    return tuple(
        a
        for a in typing.get_args(op_class.__orig_bases__[0])
        if isinstance(a, type) and issubclass(a, Q1Imm)
    )


def _imm(op_class, *values) -> tuple[Q1Imm, ...]:
    """Wrap the given int(s) in the op class's bound Q1Imm subtypes.

    Values are clamped into the bound type's valid range so that generic test bodies don't
    need to know each op's per-slot constraints.
    """
    types_ = _imm_types(op_class)
    assert len(values) == len(types_), (
        f"{op_class.__name__} expects {len(types_)} imm(s), got {len(values)}"
    )
    return tuple(
        t(max(t._MIN, min(t._MAX, v))) for t, v in zip(types_, values, strict=False)
    )


def _assert_traits(op: Any, expected_traits: tuple) -> None:
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


def _assert_aliases(op: Any, aliases: dict[str, str]) -> None:
    for alias, target in aliases.items():
        assert getattr(op, alias) is getattr(op, target)


class TestLabelOp:
    def test_construction_with_comment(self) -> None:
        op = LabelOp("loop_start", comment="branch target")

        assert op.reference == LabelAttr("loop_start")
        assert op.comment == StringAttr("branch target")
        assert op.assembly_mnemonic() == ""
        assert op.assembly_line_args() == ()

        line = op.assembly_line()
        assert line is not None
        assert line.startswith("loop_start:")
        assert line.endswith("# branch target")

    def test_construction_without_comment(self) -> None:
        op = LabelOp(LabelAttr("loop_end"))
        line = op.assembly_line()
        assert line is not None
        assert line.startswith("loop_end:")


class TestDefDirectiveOp:
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
    def test_def_directive(
        self,
        alias_input,
        value_input,
        comment_input,
        alias_str,
        value_str,
        comment_str,
    ) -> None:
        op = DefDirectiveOp(alias_input, value_input, comment=comment_input)

        assert op.alias == StringAttr(alias_str)
        assert op.value == StringAttr(value_str)
        assert op.comment == (StringAttr(comment_str) if comment_str is not None else None)
        assert op.assembly_mnemonic() == ".DEF"
        assert op.assembly_line_args() == (
            StringAttr(alias_str),
            StringAttr(value_str),
        )

        line = op.assembly_line()
        assert line is not None
        if comment_str is None:
            assert line == f".DEF {alias_str} {value_str}"
        else:
            assert line.startswith(f".DEF {alias_str} {value_str}")
            assert line.endswith(f"# {comment_str}")
        assert "," not in line


class TestNullaryFormat:
    _OPS_TABLE = [
        (IllegalOp, "illegal", (IsTerminator, HasRegisterConstraintsTrait)),
        (StopOp, "stop", (IsTerminator, HasRegisterConstraintsTrait)),
        (NopOp, "nop", (Pure, HasRegisterConstraintsTrait)),
        (ResetPhOp, "reset_ph", (HasRegisterConstraintsTrait,)),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,traits", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, traits, comment) -> None:
        op = op_type(comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.assembly_line_args() == ()
        assert op.assembly_line() is not None
        _assert_traits(op, traits)


class TestIFormat:
    _STOP_TRAITS = (IsTerminator, HasRegisterConstraintsTrait)
    _BRANCH_TRAITS = (HasRegisterConstraintsTrait,)
    _PARAM_TRAITS = (HasRegisterConstraintsTrait,)
    _PURE_RT_TRAITS = (Pure(), HasRegisterConstraintsTrait)
    _OPS_TABLE = [
        (StopImmOp, "stop", _STOP_TRAITS, {"status": "imm"}),
        (JmpImmOp, "jmp", _BRANCH_TRAITS, {"address": "imm"}),
        (JzImmOp, "jz", _BRANCH_TRAITS, {"address": "imm"}),
        (JnzImmOp, "jnz", _BRANCH_TRAITS, {"address": "imm"}),
        (JoImmOp, "jo", _BRANCH_TRAITS, {"address": "imm"}),
        (JnoImmOp, "jno", _BRANCH_TRAITS, {"address": "imm"}),
        (JsImmOp, "js", _BRANCH_TRAITS, {"address": "imm"}),
        (JnsImmOp, "jns", _BRANCH_TRAITS, {"address": "imm"}),
        (JgImmOp, "jg", _BRANCH_TRAITS, {"address": "imm"}),
        (JlImmOp, "jl", _BRANCH_TRAITS, {"address": "imm"}),
        (JleImmOp, "jle", _BRANCH_TRAITS, {"address": "imm"}),
        (JaImmOp, "ja", _BRANCH_TRAITS, {"address": "imm"}),
        (JaeImmOp, "jae", _BRANCH_TRAITS, {"address": "imm"}),
        (JbImmOp, "jb", _BRANCH_TRAITS, {"address": "imm"}),
        (JbeImmOp, "jbe", _BRANCH_TRAITS, {"address": "imm"}),
        (JgeImmOp, "jge", _BRANCH_TRAITS, {"address": "imm"}),
        (SetMrkImmOp, "set_mrk", _PARAM_TRAITS, {"mrk": "imm"}),
        (SetFreqImmOp, "set_freq", _PARAM_TRAITS, {"nco_freq": "imm"}),
        (SetPhImmOp, "set_ph", _PARAM_TRAITS, {"nco_po": "imm"}),
        (SetPhDeltaImmOp, "set_ph_delta", _PARAM_TRAITS, {"nco_delta_po": "imm"}),
        (WaitImmOp, "wait", _PURE_RT_TRAITS, {"duration": "imm"}),
        (WaitSyncImmOp, "wait_sync", _PURE_RT_TRAITS, {"duration": "imm"}),
        (UpdParamImmOp, "upd_param", _PARAM_TRAITS, {"duration": "imm"}),
        (LatchRstImmOp, "latch_rst", _PARAM_TRAITS, {"duration": "imm"}),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, traits, aliases, comment) -> None:
        (imm,) = _imm(op_type, 1)
        op = op_type(imm, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm"] == imm
        assert op.assembly_line_args() == (op.imm,)
        _assert_traits(op, traits)

    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, traits, aliases) -> None:
        (imm,) = _imm(op_type, 1)
        _assert_aliases(op_type(imm), aliases)


class TestRsFormat:
    _STOP_TRAITS = (IsTerminator, HasRegisterConstraintsTrait)
    _BRANCH_TRAITS = (HasRegisterConstraintsTrait,)
    _PARAM_TRAITS = (HasRegisterConstraintsTrait,)
    _PURE_RT_TRAITS = (Pure(), HasRegisterConstraintsTrait)
    _OPS_TABLE = [
        (StopRsOp, "stop", _STOP_TRAITS, {"status": "rs"}),
        (JmpRsOp, "jmp", _BRANCH_TRAITS, {"address": "rs"}),
        (JzRsOp, "jz", _BRANCH_TRAITS, {"address": "rs"}),
        (JnzRsOp, "jnz", _BRANCH_TRAITS, {"address": "rs"}),
        (JoRsOp, "jo", _BRANCH_TRAITS, {"address": "rs"}),
        (JnoRsOp, "jno", _BRANCH_TRAITS, {"address": "rs"}),
        (JsRsOp, "js", _BRANCH_TRAITS, {"address": "rs"}),
        (JnsRsOp, "jns", _BRANCH_TRAITS, {"address": "rs"}),
        (JgRsOp, "jg", _BRANCH_TRAITS, {"address": "rs"}),
        (JlRsOp, "jl", _BRANCH_TRAITS, {"address": "rs"}),
        (JleRsOp, "jle", _BRANCH_TRAITS, {"address": "rs"}),
        (JaRsOp, "ja", _BRANCH_TRAITS, {"address": "rs"}),
        (JaeRsOp, "jae", _BRANCH_TRAITS, {"address": "rs"}),
        (JbRsOp, "jb", _BRANCH_TRAITS, {"address": "rs"}),
        (JbeRsOp, "jbe", _BRANCH_TRAITS, {"address": "rs"}),
        (JgeRsOp, "jge", _BRANCH_TRAITS, {"address": "rs"}),
        (SetMrkRsOp, "set_mrk", _PARAM_TRAITS, {"mrk": "rs"}),
        (SetFreqRsOp, "set_freq", _PARAM_TRAITS, {"nco_freq": "rs"}),
        (SetPhRsOp, "set_ph", _PARAM_TRAITS, {"nco_po": "rs"}),
        (SetPhDeltaRsOp, "set_ph_delta", _PARAM_TRAITS, {"nco_delta_po": "rs"}),
        (WaitRsOp, "wait", _PURE_RT_TRAITS, {"duration": "rs"}),
        (WaitSyncRsOp, "wait_sync", _PURE_RT_TRAITS, {"duration": "rs"}),
        (LatchRstRsOp, "latch_rst", _PARAM_TRAITS, {"duration": "rs"}),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, traits, aliases, comment) -> None:
        rs = create_ssa_value(Registers.R32)
        op = op_type(rs, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert isinstance(op.rs.type, IntRegisterType)
        assert op.assembly_line_args() == (op.rs,)
        _assert_traits(op, traits)

    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, traits, aliases) -> None:
        rs = create_ssa_value(Registers.R1)
        _assert_aliases(op_type(rs), aliases)


class TestIIFormat:
    _OPS_TABLE = [
        (SetAwgGainImmImmOp, "set_awg_gain", {"gain0": "imm1", "gain1": "imm2"}),
        (SetAwgOffsImmImmOp, "set_awg_offs", {"offs0": "imm1", "offs1": "imm2"}),
        (SetLatchEnImmImmOp, "set_latch_en", {"latch_en": "imm1", "duration": "imm2"}),
        (WaitTriggerImmImmOp, "wait_trigger", {"trig_addr": "imm1", "duration": "imm2"}),
        (FbAcqIqIdImmImmOp, "fb_acq_iq_id", {"id": "imm1", "duration": "imm2"}),
        (FbAcqIqShiftImmImmOp, "fb_acq_iq_shift", {"shift": "imm1", "duration": "imm2"}),
        (FbAcqTbIdImmImmOp, "fb_acq_tb_id", {"id": "imm1", "duration": "imm2"}),
        (FbAcqTbValidImmImmOp, "fb_acq_tb_valid", {"tb_valid": "imm1", "duration": "imm2"}),
    ]
    _TRAITS_BY_OP = {
        WaitTriggerImmImmOp: (Pure(), HasRegisterConstraintsTrait),
    }

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        imm1, imm2 = _imm(op_type, 1, 4)
        op = op_type(imm1, imm2, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm1"] == imm1
        assert op.properties["imm2"] == imm2
        assert op.assembly_line_args() == (op.imm1, op.imm2)
        _assert_traits(op, self._TRAITS_BY_OP.get(op_type, (HasRegisterConstraintsTrait,)))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        imm1, imm2 = _imm(op_type, 1, 4)
        _assert_aliases(op_type(imm1, imm2), aliases)


class TestRsIFormat:
    _OPS_TABLE = [
        (FbAcqIqIdRsImmOp, "fb_acq_iq_id", {"id": "rs", "duration": "imm"}),
        (FbAcqTbIdRsImmOp, "fb_acq_tb_id", {"id": "rs", "duration": "imm"}),
        (FbAcqTbValidRsImmOp, "fb_acq_tb_valid", {"tb_valid": "rs", "duration": "imm"}),
        (SetLatchEnRsImmOp, "set_latch_en", {"latch_en": "rs", "duration": "imm"}),
        (CmpRsImmOp, "cmp", {"a": "rs", "b": "imm"}),
        (TestRsImmOp, "test", {"a": "rs", "b": "imm"}),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 4)
        op = op_type(rs, imm, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert isinstance(op.rs.type, IntRegisterType)
        assert op.properties["imm"] == imm
        assert op.assembly_line_args() == (op.rs, op.imm)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 4)
        _assert_aliases(op_type(rs, imm), aliases)


class TestIRsFormat:
    _OPS_TABLE = [
        (CmpImmRsOp, "cmp", {"a": "rs", "b": "imm"}),
        (TestImmRsOp, "test", {"a": "rs", "b": "imm"}),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 4)
        op = op_type(imm, rs, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert isinstance(op.rs.type, IntRegisterType)
        assert op.properties["imm"] == imm
        assert op.assembly_line_args() == (op.imm, op.rs)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 4)
        _assert_aliases(op_type(imm, rs), aliases)


class TestRsRsFormat:
    _OPS_TABLE = [
        (SetAwgGainRsRsOp, "set_awg_gain", {"gain0": "rs1", "gain1": "rs2"}),
        (SetAwgOffsRsRsOp, "set_awg_offs", {"offs0": "rs1", "offs1": "rs2"}),
        (CmpRsRsOp, "cmp", {"a": "rs1", "b": "rs2"}),
        (TestRsRsOp, "test", {"a": "rs1", "b": "rs2"}),
        (WaitTriggerRsRsOp, "wait_trigger", {"trig_addr": "rs1", "duration": "rs2"}),
    ]
    _TRAITS_BY_OP = {
        WaitTriggerRsRsOp: (Pure(), HasRegisterConstraintsTrait),
    }

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        op = op_type(r1, r2, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert isinstance(op.rs1.type, IntRegisterType)
        assert isinstance(op.rs2.type, IntRegisterType)
        assert op.assembly_line_args() == (op.rs1, op.rs2)
        _assert_traits(op, self._TRAITS_BY_OP.get(op_type, (HasRegisterConstraintsTrait,)))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        _assert_aliases(op_type(r1, r2), aliases)


class TestIRdFormat:
    _OPS_TABLE = [
        (MoveImmRdOp, "move", (Pure, HasRegisterConstraintsTrait), {"source": "imm"}),
        (NotImmRdOp, "not", (Pure, HasRegisterConstraintsTrait), {"source": "imm"}),
        (
            FbPopDataImmRdOp,
            "fb_pop_data",
            (HasRegisterConstraintsTrait,),
            {"id": "imm", "destination": "rd"},
        ),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, traits, aliases, comment) -> None:
        (imm,) = _imm(op_type, 4)
        op = op_type(imm, Registers.R0, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm"] == imm
        assert isinstance(op.rd.type, IntRegisterType)
        assert op.rd.type.index == Registers.R0.index
        assert op.assembly_line_args() == (op.imm, op.rd)
        _assert_traits(op, traits)

    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, traits, aliases) -> None:
        (imm,) = _imm(op_type, 4)
        op = op_type(imm, Registers.R0)
        _assert_aliases(op, aliases)


class TestRsRdFormat:
    _OPS_TABLE = [
        (MoveRsRdOp, "move", {"source": "rs"}),
        (NotRsRdOp, "not", {"source": "rs"}),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        rs = create_ssa_value(Registers.R62)
        op = op_type(rs, rd=Registers.R63, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert isinstance(op.rs.type, IntRegisterType)
        assert op.rs.type.index == Registers.R62.index
        assert isinstance(op.rd.type, IntRegisterType)
        assert op.rd.type.index == Registers.R63.index
        assert op.assembly_line_args() == (op.rs, op.rd)
        _assert_traits(op, (Pure, HasRegisterConstraintsTrait))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        rs = create_ssa_value(Registers.R1)
        _assert_aliases(op_type(rs, rd=Registers.R2), aliases)


class TestRdRdFormat:
    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    def test_fb_pull_data(self, comment) -> None:
        op = FbPullDataRdRdOp(Registers.R0, Registers.R1, comment=comment)
        assert op.assembly_mnemonic() == "fb_pull_data"
        assert isinstance(op.rd1.type, IntRegisterType)
        assert isinstance(op.rd2.type, IntRegisterType)
        assert op.rd1.type.index == Registers.R0.index
        assert op.rd2.type.index == Registers.R1.index
        assert op.assembly_line_args() == (op.rd1, op.rd2)
        assert op.destination_id == op.rd1
        assert op.destination == op.rd2
        _assert_traits(op, (HasRegisterConstraintsTrait,))


class TestRdIFormat:
    @pytest.mark.parametrize(
        "comment",
        ["Jump to immediate address while greater or equals"],
    )
    def test_loop_ri(self, comment) -> None:
        imm = AddressImm(100)
        op = LoopRdImmOp(Registers.R10, imm, comment=comment)
        assert op.assembly_mnemonic() == "loop"
        assert op.properties["imm"] == imm
        assert isinstance(op.rd.type, IntRegisterType)
        assert op.rd.type.is_allocated
        assert op.rd.type.index == Registers.R10.index
        assert op.assembly_line_args() == (op.rd, op.imm)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    def test_loop_ri_semantic_aliases(self) -> None:
        op = LoopRdImmOp(Registers.R19, AddressImm(500))
        _assert_aliases(op, {"source": "rd", "address": "imm"})


class TestRdRsFormat:
    @pytest.mark.parametrize(
        "comment",
        ["Jump to register address while greater or equals"],
    )
    def test_loop_rr(self, comment) -> None:
        rs = create_ssa_value(Registers.R2)
        op = LoopRdRsOp(Registers.R10, rs, comment=comment)
        assert op.assembly_mnemonic() == "loop"
        assert isinstance(op.rs.type, IntRegisterType)
        assert op.rs.type.index == rs.type.index
        assert isinstance(op.rd.type, IntRegisterType)
        assert op.rd.type.is_allocated
        assert op.rd.type.index == Registers.R10.index
        assert op.assembly_line_args() == (op.rd, op.rs)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    def test_loop_rr_semantic_aliases(self) -> None:
        op = LoopRdRsOp(Registers.R20, create_ssa_value(Registers.R21))
        _assert_aliases(op, {"source": "rd", "address": "rs"})


class TestRsIIFormat:
    _OPS_TABLE = [
        (JgeRsImmImmOp, "jge", {"a": "rs", "b": "imm1", "address": "imm2"}),
        (JltRsImmImmOp, "jlt", {"a": "rs", "b": "imm1", "address": "imm2"}),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        rs = create_ssa_value(Registers.R1)
        imm1, imm2 = _imm(op_type, 1000, 100)
        op = op_type(rs, imm1, imm2, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm1"] == imm1
        assert op.properties["imm2"] == imm2
        assert isinstance(op.rs.type, IntRegisterType)
        assert op.assembly_line_args() == (op.rs, op.imm1, op.imm2)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        rs = create_ssa_value(Registers.R1)
        imm1, imm2 = _imm(op_type, 1, 2)
        _assert_aliases(op_type(rs, imm1, imm2), aliases)


class TestRsIRsFormat:
    _OPS_TABLE = [
        (JgeRsImmRsOp, "jge", {"a": "rs1", "b": "imm", "address": "rs2"}),
        (JltRsImmRsOp, "jlt", {"a": "rs1", "b": "imm", "address": "rs2"}),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        a = create_ssa_value(Registers.R1)
        c = create_ssa_value(Registers.R2)
        (imm,) = _imm(op_type, 1000)
        op = op_type(a, imm, c, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm"] == imm
        assert isinstance(op.rs1.type, IntRegisterType)
        assert isinstance(op.rs2.type, IntRegisterType)
        assert op.assembly_line_args() == (op.rs1, op.imm, op.rs2)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        a = create_ssa_value(Registers.R1)
        c = create_ssa_value(Registers.R2)
        (imm,) = _imm(op_type, 1)
        _assert_aliases(op_type(a, imm, c), aliases)


class TestRsIRdFormat:
    _ARITH_TRAITS = (Pure, HasRegisterConstraintsTrait)
    _AB_RD = {"a": "rs", "b": "imm", "dst_low": "rd"}
    _AB_RD_HIGH = {"a": "rs", "b": "imm", "dst_high": "rd"}
    _OPS_TABLE = [
        (AddRsImmRdOp, "add", _ARITH_TRAITS, {"a": "rs", "b": "imm"}),
        (SubRsImmRdOp, "sub", _ARITH_TRAITS, {"a": "rs", "b": "imm"}),
        (AndRsImmRdOp, "and", _ARITH_TRAITS, {"a": "rs", "b": "imm"}),
        (OrRsImmRdOp, "or", _ARITH_TRAITS, {"a": "rs", "b": "imm"}),
        (XorRsImmRdOp, "xor", _ARITH_TRAITS, {"a": "rs", "b": "imm"}),
        (AslRsImmRdOp, "asl", _ARITH_TRAITS, {"a": "rs", "b": "imm"}),
        (AsrRsImmRdOp, "asr", _ARITH_TRAITS, {"a": "rs", "b": "imm"}),
        (LslRsImmRdOp, "lsl", _ARITH_TRAITS, _AB_RD),
        (LsrRsImmRdOp, "lsr", _ARITH_TRAITS, _AB_RD),
        (Mulu16RsImmRdOp, "mulu16", _ARITH_TRAITS, _AB_RD),
        (Muls16RsImmRdOp, "muls16", _ARITH_TRAITS, _AB_RD),
        (Mulu32lRsImmRdOp, "mulu32l", _ARITH_TRAITS, _AB_RD),
        (Muls32lRsImmRdOp, "muls32l", _ARITH_TRAITS, _AB_RD),
        (Mulu32hRsImmRdOp, "mulu32h", _ARITH_TRAITS, _AB_RD_HIGH),
        (Muls32hRsImmRdOp, "muls32h", _ARITH_TRAITS, _AB_RD_HIGH),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, traits, aliases, comment) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 4)
        op = op_type(rs, imm, rd=Registers.R3, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm"] == imm
        assert isinstance(op.rd.type, IntRegisterType)
        assert op.rd.type.index == Registers.R3.index
        assert op.assembly_line_args() == (op.rs, op.imm, op.rd)
        _assert_traits(op, traits)

    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, traits, aliases) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 4)
        _assert_aliases(op_type(rs, imm, rd=Registers.R3), aliases)


class TestRsRsRdFormat:
    _TRAITS_C = (Pure, Commutative, HasRegisterConstraintsTrait)
    _TRAITS_NC = (Pure, HasRegisterConstraintsTrait)
    _AB_RD = {"a": "rs1", "b": "rs2", "dst_low": "rd"}
    _AB_RD_HIGH = {"a": "rs1", "b": "rs2", "dst_high": "rd"}
    _OPS_TABLE = [
        (AddRsRsRdOp, "add", _TRAITS_C, {"a": "rs1", "b": "rs2"}),
        (SubRsRsRdOp, "sub", _TRAITS_NC, {"a": "rs1", "b": "rs2"}),
        (AndRsRsRdOp, "and", _TRAITS_C, {"a": "rs1", "b": "rs2"}),
        (OrRsRsRdOp, "or", _TRAITS_C, {"a": "rs1", "b": "rs2"}),
        (XorRsRsRdOp, "xor", _TRAITS_C, {"a": "rs1", "b": "rs2"}),
        (AslRsRsRdOp, "asl", _TRAITS_NC, {"a": "rs1", "b": "rs2"}),
        (AsrRsRsRdOp, "asr", _TRAITS_NC, {"a": "rs1", "b": "rs2"}),
        (LslRsRsRdOp, "lsl", _TRAITS_NC, _AB_RD),
        (LsrRsRsRdOp, "lsr", _TRAITS_NC, _AB_RD),
        (Mulu16RsRsRdOp, "mulu16", _TRAITS_C, _AB_RD),
        (Muls16RsRsRdOp, "muls16", _TRAITS_C, _AB_RD),
        (Mulu32lRsRsRdOp, "mulu32l", _TRAITS_C, _AB_RD),
        (Muls32lRsRsRdOp, "muls32l", _TRAITS_C, _AB_RD),
        (Mulu32hRsRsRdOp, "mulu32h", _TRAITS_C, _AB_RD_HIGH),
        (Muls32hRsRsRdOp, "muls32h", _TRAITS_C, _AB_RD_HIGH),
    ]

    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, traits, aliases) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        op = op_type(r1, r2, rd=Registers.R3)
        assert op.assembly_mnemonic() == mnemonic
        assert r1.type is op.rs1.type
        assert r2.type is op.rs2.type
        assert isinstance(op.rd.type, IntRegisterType)
        assert op.rd.type.index == Registers.R3.index
        assert op.assembly_line_args() == (op.rs1, op.rs2, op.rd)
        _assert_traits(op, traits)

    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, traits, aliases) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        _assert_aliases(op_type(r1, r2, rd=Registers.R3), aliases)


class TestIRsRdFormat:
    _TRAITS = (Pure, HasRegisterConstraintsTrait)
    _AB_RD_LOW = {"a": "rs", "b": "imm", "dst_low": "rd"}
    _AB_RD_HIGH = {"a": "rs", "b": "imm", "dst_high": "rd"}
    _OPS_TABLE = [
        (LslImmRsRdOp, "lsl", _TRAITS, _AB_RD_LOW),
        (LsrImmRsRdOp, "lsr", _TRAITS, _AB_RD_LOW),
        (Mulu16ImmRsRdOp, "mulu16", _TRAITS, _AB_RD_LOW),
        (Muls16ImmRsRdOp, "muls16", _TRAITS, _AB_RD_LOW),
        (Mulu32lImmRsRdOp, "mulu32l", _TRAITS, _AB_RD_LOW),
        (Muls32lImmRsRdOp, "muls32l", _TRAITS, _AB_RD_LOW),
        (Mulu32hImmRsRdOp, "mulu32h", _TRAITS, _AB_RD_HIGH),
        (Muls32hImmRsRdOp, "muls32h", _TRAITS, _AB_RD_HIGH),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, traits, aliases, comment) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 4)
        op = op_type(imm, rs, rd=Registers.R3, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm"] == imm
        assert isinstance(op.rd.type, IntRegisterType)
        assert op.rd.type.index == Registers.R3.index
        assert op.assembly_line_args() == (op.imm, op.rs, op.rd)
        _assert_traits(op, traits)

    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, traits, aliases) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 4)
        _assert_aliases(op_type(imm, rs, rd=Registers.R3), aliases)


class TestIIIFormat:
    _OPS_TABLE = [
        (PlayImmImmImmOp, "play", {"wave0": "imm1", "wave1": "imm2", "duration": "imm3"}),
        (
            AcquireImmImmImmOp,
            "acquire",
            {"acq_idx": "imm1", "bin_idx": "imm2", "duration": "imm3"},
        ),
        (
            FbComDataImmImmImmOp,
            "fb_com_data",
            {"id": "imm1", "value": "imm2", "duration": "imm3"},
        ),
        (
            FbComExtraImmImmImmOp,
            "fb_com_extra",
            {"extra_vld": "imm1", "extra": "imm2", "duration": "imm3"},
        ),
        (
            FbAcqTbExtraImmImmImmOp,
            "fb_acq_tb_extra",
            {"extra_vld": "imm1", "extra": "imm2", "duration": "imm3"},
        ),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        imm1, imm2, imm3 = _imm(op_type, 1, 2, 100)
        op = op_type(imm1, imm2, imm3, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm1"] == imm1
        assert op.properties["imm2"] == imm2
        assert op.properties["imm3"] == imm3
        assert op.assembly_line_args() == (op.imm1, op.imm2, op.imm3)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        imm1, imm2, imm3 = _imm(op_type, 1, 1, 100)
        _assert_aliases(op_type(imm1, imm2, imm3), aliases)


class TestIRsIFormat:
    _OPS_TABLE = [
        (
            AcquireImmRsImmOp,
            "acquire",
            {"acq_idx": "imm1", "bin_idx": "rs", "duration": "imm2"},
        ),
        (
            FbComDataImmRsImmOp,
            "fb_com_data",
            {"id": "imm1", "value": "rs", "duration": "imm2"},
        ),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        rs = create_ssa_value(Registers.R1)
        imm1, imm2 = _imm(op_type, 1, 100)
        op = op_type(imm1, rs, imm2, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert isinstance(op.rs.type, IntRegisterType)
        assert op.properties["imm1"] == imm1
        assert op.properties["imm2"] == imm2
        assert op.assembly_line_args() == (op.imm1, op.rs, op.imm2)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        rs = create_ssa_value(Registers.R1)
        imm1, imm2 = _imm(op_type, 1, 100)
        _assert_aliases(op_type(imm1, rs, imm2), aliases)


class TestRsRsIFormat:
    _OPS_TABLE = [
        (PlayRsRsImmOp, "play", {"wave0": "rs1", "wave1": "rs2", "duration": "imm"}),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        (imm,) = _imm(op_type, 300)
        op = op_type(r1, r2, imm, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert isinstance(op.rs1.type, IntRegisterType)
        assert isinstance(op.rs2.type, IntRegisterType)
        assert op.properties["imm"] == imm
        assert op.assembly_line_args() == (op.rs1, op.rs2, op.imm)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        (imm,) = _imm(op_type, 100)
        _assert_aliases(op_type(r1, r2, imm), aliases)


class TestRsRsRsIFormat:
    _OPS_TABLE = [
        (
            SetCondRsRsRsImmOp,
            "set_cond",
            {"cond_en": "rs1", "mask": "rs2", "op": "rs3", "else_cnt": "imm"},
        ),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        r3 = create_ssa_value(Registers.R3)
        (imm,) = _imm(op_type, 100)
        op = op_type(r1, r2, r3, imm, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        for rs in (op.rs1, op.rs2, op.rs3):
            assert isinstance(rs.type, IntRegisterType)
        assert op.properties["imm"] == imm
        assert op.assembly_line_args() == (op.rs1, op.rs2, op.rs3, op.imm)
        _assert_traits(op, (HasRegisterConstraintsTrait,))
        assert isinstance(op.get_register_constraints(), RegisterConstraints)

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        r3 = create_ssa_value(Registers.R3)
        (imm,) = _imm(op_type, 100)
        _assert_aliases(op_type(r1, r2, r3, imm), aliases)


class TestIIIIFormat:
    _OPS_TABLE = [
        (
            SetCondImmImmImmImmOp,
            "set_cond",
            {"cond_en": "imm1", "mask": "imm2", "op": "imm3", "else_cnt": "imm4"},
        ),
        (
            FbComCfgImmImmImmImmOp,
            "fb_com_cfg",
            {"wc": "imm1", "shift": "imm2", "length": "imm3", "duration": "imm4"},
        ),
        (
            FbAcqTbCfgImmImmImmImmOp,
            "fb_acq_tb_cfg",
            {"wc": "imm1", "shift": "imm2", "length": "imm3", "duration": "imm4"},
        ),
        (
            FbAcqTbMockImmImmImmImmOp,
            "fb_acq_tb_mock",
            {
                "mock_en": "imm1",
                "mock_vld": "imm2",
                "mock_data": "imm3",
                "duration": "imm4",
            },
        ),
        (
            AcquireTtlImmImmImmImmOp,
            "acquire_ttl",
            {
                "acq_idx": "imm1",
                "bin_idx": "imm2",
                "ttl_en": "imm3",
                "duration": "imm4",
            },
        ),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        imm1, imm2, imm3, imm4 = _imm(op_type, 1, 1, 1, 100)
        op = op_type(imm1, imm2, imm3, imm4, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm1"] == imm1
        assert op.properties["imm2"] == imm2
        assert op.properties["imm3"] == imm3
        assert op.properties["imm4"] == imm4
        assert op.assembly_line_args() == (op.imm1, op.imm2, op.imm3, op.imm4)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        imm1, imm2, imm3, imm4 = _imm(op_type, 1, 1, 1, 100)
        _assert_aliases(op_type(imm1, imm2, imm3, imm4), aliases)


class TestIRsIIFormat:
    _OPS_TABLE = [
        (
            AcquireTtlImmRsImmImmOp,
            "acquire_ttl",
            {
                "acq_idx": "imm1",
                "bin_idx": "rs",
                "ttl_en": "imm2",
                "duration": "imm3",
            },
        ),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        rs = create_ssa_value(Registers.R1)
        imm1, imm2, imm3 = _imm(op_type, 1, 1, 100)
        op = op_type(imm1, rs, imm2, imm3, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert isinstance(op.rs.type, IntRegisterType)
        assert op.properties["imm1"] == imm1
        assert op.properties["imm2"] == imm2
        assert op.properties["imm3"] == imm3
        assert op.assembly_line_args() == (op.imm1, op.rs, op.imm2, op.imm3)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        rs = create_ssa_value(Registers.R1)
        imm1, imm2, imm3 = _imm(op_type, 1, 1, 100)
        _assert_aliases(op_type(imm1, rs, imm2, imm3), aliases)


class TestIIIIIFormat:
    _ALIASES = {
        "acq_idx": "imm1",
        "bin_idx": "imm2",
        "weight_idx0": "imm3",
        "weight_idx1": "imm4",
        "duration": "imm5",
    }
    _OPS_TABLE = [
        (AcquireWeightedImmImmImmImmImmOp, "acquire_weighted", _ALIASES),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        imm1, imm2, imm3, imm4, imm5 = _imm(op_type, 1, 1, 1, 1, 100)
        op = op_type(imm1, imm2, imm3, imm4, imm5, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm1"] == imm1
        assert op.properties["imm2"] == imm2
        assert op.properties["imm3"] == imm3
        assert op.properties["imm4"] == imm4
        assert op.properties["imm5"] == imm5
        assert op.assembly_line_args() == (op.imm1, op.imm2, op.imm3, op.imm4, op.imm5)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        imm1, imm2, imm3, imm4, imm5 = _imm(op_type, 1, 1, 1, 1, 100)
        _assert_aliases(op_type(imm1, imm2, imm3, imm4, imm5), aliases)


class TestIRsRsRsIFormat:
    _ALIASES = {
        "acq_idx": "imm1",
        "bin_idx": "rs1",
        "weight_idx0": "rs2",
        "weight_idx1": "rs3",
        "duration": "imm2",
    }
    _OPS_TABLE = [
        (AcquireWeightedImmRsRsRsImmOp, "acquire_weighted", _ALIASES),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, aliases, comment) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        r3 = create_ssa_value(Registers.R3)
        imm1, imm2 = _imm(op_type, 1, 100)
        op = op_type(imm1, r1, r2, r3, imm2, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        for rs in (op.rs1, op.rs2, op.rs3):
            assert isinstance(rs.type, IntRegisterType)
        assert op.properties["imm1"] == imm1
        assert op.properties["imm2"] == imm2
        assert op.assembly_line_args() == (op.imm1, op.rs1, op.rs2, op.rs3, op.imm2)
        _assert_traits(op, (HasRegisterConstraintsTrait,))

    @pytest.mark.parametrize("op_type,mnemonic,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, aliases) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        r3 = create_ssa_value(Registers.R3)
        imm1, imm2 = _imm(op_type, 1, 100)
        _assert_aliases(op_type(imm1, r1, r2, r3, imm2), aliases)


class TestRsIRdRdFormat:
    _TRAITS = (Pure, HasRegisterConstraintsTrait)
    _ALIASES = {"a": "rs", "b": "imm", "dst_low": "rd1", "dst_high": "rd2"}
    _OPS_TABLE = [
        (Mulu32RsImmRdRdOp, "mulu32", _TRAITS, _ALIASES),
        (Muls32RsImmRdRdOp, "muls32", _TRAITS, _ALIASES),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, traits, aliases, comment) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 99)
        op = op_type(rs, imm, rd1=Registers.R3, rd2=Registers.R4, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm"] == imm
        assert op.assembly_line_args() == (op.rs, op.imm, op.rd1, op.rd2)
        _assert_traits(op, traits)

    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, traits, aliases) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 7)
        op = op_type(rs, imm, rd1=Registers.R3, rd2=Registers.R4)
        _assert_aliases(op, aliases)


class TestRsRsRdRdFormat:
    _TRAITS = (Pure, Commutative, HasRegisterConstraintsTrait)
    _ALIASES = {"a": "rs1", "b": "rs2", "dst_low": "rd1", "dst_high": "rd2"}
    _OPS_TABLE = [
        (Mulu32RsRsRdRdOp, "mulu32", _TRAITS, _ALIASES),
        (Muls32RsRsRdRdOp, "muls32", _TRAITS, _ALIASES),
    ]

    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, traits, aliases) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        op = op_type(r1, r2, rd1=Registers.R3, rd2=Registers.R4)
        assert op.assembly_mnemonic() == mnemonic
        assert op.assembly_line_args() == (op.rs1, op.rs2, op.rd1, op.rd2)
        _assert_traits(op, traits)

    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, traits, aliases) -> None:
        r1 = create_ssa_value(Registers.R1)
        r2 = create_ssa_value(Registers.R2)
        op = op_type(r1, r2, rd1=Registers.R3, rd2=Registers.R4)
        _assert_aliases(op, aliases)


class TestIRsRdRdFormat:
    _TRAITS = (Pure, HasRegisterConstraintsTrait)
    _ALIASES = {"a": "rs", "b": "imm", "dst_low": "rd1", "dst_high": "rd2"}
    _OPS_TABLE = [
        (Mulu32ImmRsRdRdOp, "mulu32", _TRAITS, _ALIASES),
        (Muls32ImmRsRdRdOp, "muls32", _TRAITS, _ALIASES),
    ]

    @pytest.mark.parametrize("comment", COMMENT_INPUTS)
    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_construction(self, op_type, mnemonic, traits, aliases, comment) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 88)
        op = op_type(imm, rs, rd1=Registers.R3, rd2=Registers.R4, comment=comment)
        assert op.assembly_mnemonic() == mnemonic
        assert op.properties["imm"] == imm
        assert op.assembly_line_args() == (op.imm, op.rs, op.rd1, op.rd2)
        _assert_traits(op, traits)

    @pytest.mark.parametrize("op_type,mnemonic,traits,aliases", _OPS_TABLE)
    def test_semantic_aliases(self, op_type, mnemonic, traits, aliases) -> None:
        rs = create_ssa_value(Registers.R1)
        (imm,) = _imm(op_type, 7)
        op = op_type(imm, rs, rd1=Registers.R3, rd2=Registers.R4)
        _assert_aliases(op, aliases)
