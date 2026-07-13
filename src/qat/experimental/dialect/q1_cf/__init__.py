# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""The ``q1_cf`` dialect.

``q1_cf`` specifies target-specific CFG operations for QBlox Q1. It sits above
flat ``q1`` (ISA mnemonics). Every operation is a block terminator with named
successor blocks, per-successor SSA operand groups, and explicit register-type
condition operands. Conditional branches carry a predicate attribute selecting
the test applied to their operands.

Reference: https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html
"""

from xdsl.ir import Dialect

from qat.experimental.dialect.q1_cf.ir.attrs import (
    ComparisonPredicate,
    ComparisonPredicateAttr,
    FlagPredicate,
    FlagPredicateAttr,
)
from qat.experimental.dialect.q1_cf.ir.ops import (
    ComparisonBranchOp,
    FlagBranchOp,
    JmpBranchOp,
    LoopBranchOp,
)

Q1_cf = Dialect(
    "q1_cf",
    [
        JmpBranchOp,
        FlagBranchOp,
        ComparisonBranchOp,
        LoopBranchOp,
    ],
    [
        FlagPredicateAttr,
        ComparisonPredicateAttr,
    ],
)

__all__ = [
    "ComparisonBranchOp",
    "ComparisonPredicate",
    "ComparisonPredicateAttr",
    "FlagBranchOp",
    "FlagPredicate",
    "FlagPredicateAttr",
    "JmpBranchOp",
    "LoopBranchOp",
    "Q1_cf",
]
