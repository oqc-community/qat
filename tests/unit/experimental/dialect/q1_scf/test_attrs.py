# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for the q1_scf attributes.

Covers :class:`IterParameter`, :class:`IterParameterAttr` and
:class:`IterDomainAttr`: member/field exposure, textual round-trips and the
:meth:`IterDomainAttr.verify` step and count constraints.
"""

from __future__ import annotations

from io import StringIO

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1_scf import (
    IterDomainAttr,
    IterParameter,
    IterParameterAttr,
    Q1_scf,
)

_ITER_MEMBERS = {"frequency", "phase", "gain", "duration"}


def _round_trip_attribute(attr):
    """Print an attribute and parse it back through a q1_scf-loaded context."""
    stream = StringIO()
    Printer(stream).print_attribute(attr)
    printed = stream.getvalue()
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Q1_scf)
    return printed, Parser(ctx, printed).parse_attribute()


class TestIterParameter:
    def test_members(self):
        assert {member.value for member in IterParameter} == _ITER_MEMBERS

    @pytest.mark.parametrize("member", list(IterParameter), ids=lambda m: m.value)
    def test_member_is_str(self, member: IterParameter):
        # StrEnum members are plain strings equal to their spelling.
        assert member == member.value

    def test_attr_name(self):
        assert IterParameterAttr.name == "q1_scf.iter_parameter"

    @pytest.mark.parametrize("member", list(IterParameter), ids=lambda m: m.value)
    def test_attr_wraps_member(self, member: IterParameter):
        assert IterParameterAttr(member).data is member

    @pytest.mark.parametrize("member", list(IterParameter), ids=lambda m: m.value)
    def test_attr_round_trips(self, member: IterParameter):
        attr = IterParameterAttr(member)
        printed, parsed = _round_trip_attribute(attr)
        assert printed == f"#q1_scf<iter_parameter {member.value}>"
        assert parsed == attr
        assert parsed.data is member


class TestIterDomain:
    def test_attr_name(self):
        assert IterDomainAttr.name == "q1_scf.iter_domain"

    def test_attr_construction_fields(self):
        attr = IterDomainAttr(0.0, 10.0, 2.0, 5, IterParameter.frequency)
        assert attr.start.value.data == 0.0
        assert attr.stop.value.data == 10.0
        assert attr.step.value.data == 2.0
        assert attr.count.data == 5
        assert attr.parameter.data is IterParameter.frequency

    @pytest.mark.parametrize("member", list(IterParameter), ids=lambda m: m.value)
    def test_attr_round_trips(self, member: IterParameter):
        attr = IterDomainAttr(0.0, 8.0, 2.0, 4, member)
        _printed, parsed = _round_trip_attribute(attr)
        assert parsed == attr

    @pytest.mark.parametrize(
        "start,stop,step,count",
        [
            (0.0, 10.0, 2.0, 5),  # exact division
            (0.0, 9.0, 2.0, 5),  # non-integer division rounds up
            (10.0, 0.0, -2.0, 5),  # negative step, descending iteration
            (3.0, 3.0, 1.0, 0),  # empty domain
            (0.0, 1.0, 1.0, 1),  # single point
        ],
    )
    def test_attr_verify_accepts_linear_domains(self, start, stop, step, count):
        # The constructor verifies eagerly. No exception means success.
        IterDomainAttr(start, stop, step, count, IterParameter.gain)

    def test_attr_verify_rejects_zero_step(self):
        # The attribute constructor verifies eagerly.
        with pytest.raises(VerifyException, match="step must be non-zero"):
            IterDomainAttr(0.0, 10.0, 0.0, 5, IterParameter.gain)

    @pytest.mark.parametrize("count", [3, 4, 6])
    def test_attr_verify_rejects_count_mismatch(self, count):
        with pytest.raises(VerifyException, match="does not equal"):
            IterDomainAttr(0.0, 10.0, 2.0, count, IterParameter.gain)

    def test_attr_verify_rejects_negative_count(self):
        # A step with the wrong sign yields a negative inferred count, which a
        # matching negative count must not satisfy.
        with pytest.raises(VerifyException, match="must be non-negative"):
            IterDomainAttr(0.0, 10.0, -2.0, -5, IterParameter.gain)
