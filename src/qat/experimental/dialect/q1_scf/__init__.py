# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""The ``q1_scf`` dialect.

``q1_scf`` is the structured control flow dialect for the QBlox Q1 sequence
processor. It sits above ``q1_cf``, the target-specific block CFG, and below the
pulse level, and provides an ``if`` conditional and ``while``/``for`` loop
containers with the ``condition`` and ``yield`` terminators. The dialect neither
lowers control flow nor performs register allocation. It is the representation on
which a subsequent register-allocation pass operates.

Structured control flow branches on register values alone. A measurement outcome
reaches ``q1_scf`` only as a ``q1.reg`` dequeued through ``q1_linq``; an outcome
consumed through ``q1_trigger`` drives predicated execution without exposing its
value and never reaches this dialect.

Reference: https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html
"""

from xdsl.ir import Dialect

from qat.experimental.dialect.q1_cf.ir.attrs import (
    ComparisonPredicate,
    ComparisonPredicateAttr,
    FlagPredicate,
    FlagPredicateAttr,
)
from qat.experimental.dialect.q1_scf.ir.attrs import (
    IterDomainAttr,
    IterParameter,
    IterParameterAttr,
)
from qat.experimental.dialect.q1_scf.ir.ops import (
    ConditionOp,
    ForOp,
    IfOp,
    WhileOp,
    YieldOp,
)

Q1_scf = Dialect(
    "q1_scf",
    [
        IfOp,
        WhileOp,
        ForOp,
        ConditionOp,
        YieldOp,
    ],
    [
        IterDomainAttr,
        IterParameterAttr,
    ],
)

__all__ = [
    "ComparisonPredicate",
    "ComparisonPredicateAttr",
    "ConditionOp",
    "FlagPredicate",
    "FlagPredicateAttr",
    "ForOp",
    "IfOp",
    "IterDomainAttr",
    "IterParameter",
    "IterParameterAttr",
    "Q1_scf",
    "WhileOp",
    "YieldOp",
]
