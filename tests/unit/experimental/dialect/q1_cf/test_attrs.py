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
    ComparisonPredicate,
    ComparisonPredicateAttr,
    FlagPredicate,
    FlagPredicateAttr,
    Q1_cf,
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
        assert {member.value for member in FlagPredicate} == _FLAG_MEMBERS

    def test_comparison_members(self):
        assert {member.value for member in ComparisonPredicate} == _COMPARISON_MEMBERS

    @pytest.mark.parametrize("member", list(FlagPredicate), ids=lambda m: m.value)
    def test_flag_member_is_str(self, member: FlagPredicate):
        # StrEnum members are plain strings equal to their spelling.
        assert member == member.value

    @pytest.mark.parametrize("member", list(ComparisonPredicate), ids=lambda m: m.value)
    def test_comparison_member_is_str(self, member: ComparisonPredicate):
        assert member == member.value


class TestPredicateAttributes:
    @pytest.mark.parametrize("member", list(FlagPredicate), ids=lambda m: m.value)
    def test_flag_attr_wraps_member(self, member: FlagPredicate):
        attr = FlagPredicateAttr(member)
        assert attr.data is member

    @pytest.mark.parametrize("member", list(ComparisonPredicate), ids=lambda m: m.value)
    def test_comparison_attr_wraps_member(self, member: ComparisonPredicate):
        attr = ComparisonPredicateAttr(member)
        assert attr.data is member

    def test_flag_attr_name(self):
        assert FlagPredicateAttr.name == "q1_cf.flag_predicate"

    def test_comparison_attr_name(self):
        assert ComparisonPredicateAttr.name == "q1_cf.comparison_predicate"

    @pytest.mark.parametrize("member", list(FlagPredicate), ids=lambda m: m.value)
    def test_flag_attr_round_trips(self, member: FlagPredicate):
        attr = FlagPredicateAttr(member)
        printed, parsed = _round_trip(attr)
        assert printed == f"#q1_cf<flag_predicate {member.value}>"
        assert parsed == attr
        assert parsed.data is member

    @pytest.mark.parametrize("member", list(ComparisonPredicate), ids=lambda m: m.value)
    def test_comparison_attr_round_trips(self, member: ComparisonPredicate):
        attr = ComparisonPredicateAttr(member)
        printed, parsed = _round_trip(attr)
        assert printed == f"#q1_cf<comparison_predicate {member.value}>"
        assert parsed == attr
        assert parsed.data is member
