# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
"""Boundary tests for every :class:`Q1Imm` subclass in the q1_sequence dialect."""

from io import StringIO

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1 import Q1
from qat.experimental.dialect.q1_sequence import (
    AcqTableIndex,
    BinCountImm,
    Q1_sequence,
    WaveformTableIndex,
    WeightTableIndex,
)

_FIXED_RANGE_TYPES = [
    (WaveformTableIndex, 0, 1023),
    (WeightTableIndex, 0, 31),
    (AcqTableIndex, 0, 31),
    (BinCountImm, 0, 7_000_000),
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


@pytest.mark.parametrize(
    "attr_cls,value",
    [
        (WaveformTableIndex, 42),
        (WeightTableIndex, 7),
        (AcqTableIndex, 31),
        (BinCountImm, 1_000_000),
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
    ctx.load_dialect(Q1_sequence)
    parsed = Parser(ctx, encoded).parse_attribute()
    assert parsed == attr
