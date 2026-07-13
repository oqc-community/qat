# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Predicate attributes for the q1_cf conditional branches.

Two enumerations classify the conditions a branch may test:

- ``FlagPredicate`` — a condition-code test (zero/sign) on a single register.
- ``ComparisonPredicate`` — a signed/unsigned comparison between two operands,
  matching the ``arith.cmpi`` predicate set.

Each enumeration is wrapped in an :class:`~xdsl.ir.EnumAttribute` so it can be
attached to an operation as a property.

Reference: https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html
"""

from __future__ import annotations

from enum import auto

from xdsl.ir import EnumAttribute, SpacedOpaqueSyntaxAttribute, StrEnum
from xdsl.irdl import irdl_attr_definition


class FlagPredicate(StrEnum):
    """Condition-code test applied to a single register value."""

    eqz = auto()  # value == 0
    nez = auto()  # value != 0
    ltz = auto()  # value < 0  (signed)
    gez = auto()  # value >= 0 (signed)


class ComparisonPredicate(StrEnum):
    """Comparison applied between two operand values."""

    eq = auto()  # ==
    ne = auto()  # !=
    slt = auto()  # <  (signed)
    sle = auto()  # <= (signed)
    sgt = auto()  # >  (signed)
    sge = auto()  # >= (signed)
    ult = auto()  # <  (unsigned)
    ule = auto()  # <= (unsigned)
    ugt = auto()  # >  (unsigned)
    uge = auto()  # >= (unsigned)


@irdl_attr_definition
class FlagPredicateAttr(EnumAttribute[FlagPredicate], SpacedOpaqueSyntaxAttribute):
    """Attribute wrapper carrying a :class:`FlagPredicate`."""

    name = "q1_cf.flag_predicate"


@irdl_attr_definition
class ComparisonPredicateAttr(
    EnumAttribute[ComparisonPredicate], SpacedOpaqueSyntaxAttribute
):
    """Attribute wrapper carrying a :class:`ComparisonPredicate`."""

    name = "q1_cf.comparison_predicate"
