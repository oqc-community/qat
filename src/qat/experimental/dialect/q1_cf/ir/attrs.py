# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Predicate attributes for the q1_cf conditional branches.

Two enumerations classify the conditions a branch may test:

- ``UnaryPredicate`` — a condition-code test (zero/sign) on a single register.
- ``BinaryPredicate`` — a signed/unsigned comparison between two operands,
  matching the ``arith.cmpi`` predicate set.

Each enumeration is wrapped in an :class:`~xdsl.ir.EnumAttribute` so it can be
attached to an operation as a property.

Reference: https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html
"""

from __future__ import annotations

from enum import auto

from xdsl.ir import EnumAttribute, SpacedOpaqueSyntaxAttribute, StrEnum
from xdsl.irdl import irdl_attr_definition


class UnaryPredicate(StrEnum):
    """Condition-code test applied to a single register value."""

    eqz = auto()  # value == 0
    nez = auto()  # value != 0
    ltz = auto()  # value < 0  (signed)
    gez = auto()  # value >= 0 (signed)


class BinaryPredicate(StrEnum):
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
class UnaryPredicateAttr(EnumAttribute[UnaryPredicate], SpacedOpaqueSyntaxAttribute):
    """Attribute wrapper carrying a :class:`UnaryPredicate`."""

    name = "q1_cf.unary_predicate"


@irdl_attr_definition
class BinaryPredicateAttr(EnumAttribute[BinaryPredicate], SpacedOpaqueSyntaxAttribute):
    """Attribute wrapper carrying a :class:`BinaryPredicate`."""

    name = "q1_cf.binary_predicate"
