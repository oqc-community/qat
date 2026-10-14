# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""The Gate dialect models quantum programs at the gate level.

It is the entry point of the compiler: gate-level programs are optimised, placed
(algorithmic qubits mapped to physical qubits) and then lowered to pulses.
"""

from xdsl.ir import Dialect

from .ops import (
    AllocateAlgorithmicQubitOp,
    AllocatePhysicalQubitOp,
    AllocateQubitOp,
    BarrierOp,
    CXOp,
    CZOp,
    DeallocateQubitOp,
    HOp,
    IdentityOp,
    MeasureOp,
    ResetOp,
    RZOp,
    U3Op,
    XOp,
)
from .traits import CliffordTrait
from .types import AlgorithmicQubitType, PhysicalQubitType, QubitType

_ops = [
    AllocateQubitOp,
    AllocateAlgorithmicQubitOp,
    AllocatePhysicalQubitOp,
    DeallocateQubitOp,
    MeasureOp,
    HOp,
    XOp,
    CZOp,
    CXOp,
    RZOp,
    U3Op,
    ResetOp,
    IdentityOp,
    BarrierOp,
]

_traits = [CliffordTrait]

# QubitType is abstract, so it is exported for type checks but not registered.
_types = [AlgorithmicQubitType, PhysicalQubitType]

Gate = Dialect("gate", _ops, _types, _traits)

_all_classes = _ops + _traits + _types

__all__ = ["Gate", "QubitType"] + [cls_.__name__ for cls_ in _all_classes]
