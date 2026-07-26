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
    BinaryPredicate,
    BinaryPredicateAttr,
    UnaryPredicate,
    UnaryPredicateAttr,
)
from qat.experimental.dialect.q1_cf.ir.ops import (
    BinaryPredicateBranchOp,
    JmpBranchOp,
    LoopBranchOp,
    UnaryPredicateBranchOp,
)

Q1_cf = Dialect(
    "q1_cf",
    [
        JmpBranchOp,
        UnaryPredicateBranchOp,
        BinaryPredicateBranchOp,
        LoopBranchOp,
    ],
    [
        UnaryPredicateAttr,
        BinaryPredicateAttr,
    ],
)

__all__ = [
    "BinaryPredicateBranchOp",
    "BinaryPredicate",
    "BinaryPredicateAttr",
    "UnaryPredicateBranchOp",
    "UnaryPredicate",
    "UnaryPredicateAttr",
    "JmpBranchOp",
    "LoopBranchOp",
    "Q1_cf",
]
