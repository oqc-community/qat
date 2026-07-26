# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for the q1_cf predicate attributes.

Coverage:
* The two predicate enumerations expose exactly the intended members.
* Each :class:`~xdsl.ir.EnumAttribute` wrapper preserves its enum member and
  round-trips through the textual attribute syntax.
"""

from __future__ import annotations

from io import StringIO

import pytest
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.printer import Printer

from qat.experimental.dialect.q1_cf import (
    BinaryPredicate,
    BinaryPredicateAttr,
    Q1_cf,
    UnaryPredicate,
    UnaryPredicateAttr,
)

_FLAG_MEMBERS = {"eqz", "nez", "ltz", "gez"}
_COMPARISON_MEMBERS = {
    "eq",
    "ne",
    "slt",
    "sle",
    "sgt",
    "sge",
    "ult",
    "ule",
    "ugt",
    "uge",
}


def _round_trip(attr):
    """Print an attribute and parse it back through a q1_cf-loaded context."""

    sio = StringIO()
    Printer(sio).print_attribute(attr)
    printed = sio.getvalue()
    ctx = Context()
    ctx.load_dialect(Q1_cf)
    return printed, Parser(ctx, printed).parse_attribute()


class TestPredicateEnums:
    def test_flag_members(self):
        assert {member.value for member in UnaryPredicate} == _FLAG_MEMBERS

    def test_comparison_members(self):
        assert {member.value for member in BinaryPredicate} == _COMPARISON_MEMBERS

    @pytest.mark.parametrize("member", list(UnaryPredicate), ids=lambda m: m.value)
    def test_flag_member_is_str(self, member: UnaryPredicate):
        # StrEnum members are plain strings equal to their spelling.
        assert member == member.value

    @pytest.mark.parametrize("member", list(BinaryPredicate), ids=lambda m: m.value)
    def test_comparison_member_is_str(self, member: BinaryPredicate):
        assert member == member.value


class TestPredicateAttributes:
    @pytest.mark.parametrize("member", list(UnaryPredicate), ids=lambda m: m.value)
    def test_flag_attr_wraps_member(self, member: UnaryPredicate):
        attr = UnaryPredicateAttr(member)
        assert attr.data is member

    @pytest.mark.parametrize("member", list(BinaryPredicate), ids=lambda m: m.value)
    def test_comparison_attr_wraps_member(self, member: BinaryPredicate):
        attr = BinaryPredicateAttr(member)
        assert attr.data is member

    def test_flag_attr_name(self):
        assert UnaryPredicateAttr.name == "q1_cf.unary_predicate"

    def test_comparison_attr_name(self):
        assert BinaryPredicateAttr.name == "q1_cf.binary_predicate"

    @pytest.mark.parametrize("member", list(UnaryPredicate), ids=lambda m: m.value)
    def test_flag_attr_round_trips(self, member: UnaryPredicate):
        attr = UnaryPredicateAttr(member)
        printed, parsed = _round_trip(attr)
        assert printed == f"#q1_cf<unary_predicate {member.value}>"
        assert parsed == attr
        assert parsed.data is member

    @pytest.mark.parametrize("member", list(BinaryPredicate), ids=lambda m: m.value)
    def test_comparison_attr_round_trips(self, member: BinaryPredicate):
        attr = BinaryPredicateAttr(member)
        printed, parsed = _round_trip(attr)
        assert printed == f"#q1_cf<binary_predicate {member.value}>"
        assert parsed == attr
        assert parsed.data is member
