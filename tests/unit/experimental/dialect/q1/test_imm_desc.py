# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Boundary tests for every :class:`Q1Imm` subclass."""

from io import StringIO

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1 import Q1
from qat.experimental.dialect.q1.ir.imm_desc import (
    AddressImm,
    BoolImm,
    DurationImm,
    NcoPhaseImm,
    Q1Imm,
    SI16Imm,
    SI32Imm,
    SU32Imm,
    UI2Imm,
    UI3Imm,
    UI4Imm,
    UI5Imm,
    UI6Imm,
    UI7Imm,
    UI8Imm,
    UI10Imm,
    UI14Imm,
    UI16Imm,
    UI24Imm,
    UI32Imm,
)

_FIXED_RANGE_TYPES = [
    (BoolImm, 0, 1),
    (UI2Imm, 0, 3),
    (UI3Imm, 0, 7),
    (UI4Imm, 0, 15),
    (UI5Imm, 0, 31),
    (UI6Imm, 0, 63),
    (UI7Imm, 0, 127),
    (UI8Imm, 0, 255),
    (UI10Imm, 0, 1023),
    (UI14Imm, 0, 16383),
    (UI16Imm, 0, 65535),
    (SI16Imm, -32768, 32767),
    (DurationImm, 4, 65535),
    (UI24Imm, 0, 16_777_215),
    (UI32Imm, 0, 2**32 - 1),
    (SI32Imm, -(2**31), 2**31 - 1),
    (SU32Imm, -(2**31), 2**32 - 1),
    (NcoPhaseImm, 0, 1_000_000_000),
]


@pytest.mark.parametrize("attr_cls,lo,hi", _FIXED_RANGE_TYPES)
class TestBoundary:
    def test_valid_min(self, attr_cls, lo, hi):
        assert attr_cls(lo).data == lo

    def test_valid_max(self, attr_cls, lo, hi):
        assert attr_cls(hi).data == hi

    def test_invalid_below_min(self, attr_cls, lo, hi):
        with pytest.raises(VerifyException, match=attr_cls.__name__):
            attr_cls(lo - 1)

    def test_invalid_above_max(self, attr_cls, lo, hi):
        with pytest.raises(VerifyException, match=attr_cls.__name__):
            attr_cls(hi + 1)


def test_address_imm_is_ui14_alias():
    assert AddressImm is UI14Imm


def test_verify_runs_validation_post_construction():
    op = UI4Imm(7)
    object.__setattr__(op, "data", 99)
    with pytest.raises(VerifyException, match="UI4Imm"):
        op.verify()


def test_diagnostic_includes_type_name_value_and_range():
    with pytest.raises(VerifyException, match=r"UI4Imm.*\[0, 15\].*42"):
        UI4Imm(42)


@pytest.mark.parametrize(
    "attr_cls,value",
    [
        (BoolImm, 1),
        (UI4Imm, 12),
        (SI16Imm, -7),
        (DurationImm, 100),
        (SU32Imm, -42),
        (NcoPhaseImm, 1_000_000_000),
    ],
)
def test_print_parse_roundtrip(attr_cls, value):
    attr = attr_cls(value)
    out = StringIO()
    Printer(stream=out).print_attribute(attr)
    encoded = out.getvalue()
    assert encoded == f"#{attr_cls.name}<{value}>"

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Q1)
    parsed = Parser(ctx, encoded).parse_attribute()
    assert parsed == attr


def test_q1_imm_subclass_without_range_class_vars_fails():
    class BadImm(Q1Imm):
        name = "q1.bad_imm"

    with pytest.raises(AttributeError):
        BadImm(0)
