# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Structural visitor interface for experimental IR operations."""

from typing import Protocol, TypeVar

from xdsl.ir import Operation

_OperationT_contra = TypeVar(
    "_OperationT_contra",
    bound=Operation,
    contravariant=True,
)
_VisitResultT_co = TypeVar("_VisitResultT_co", covariant=True)


class OperationVisitor(Protocol[_OperationT_contra, _VisitResultT_co]):
    """A visitor that accepts an IR operation and returns a typed result."""

    def visit(self, operation: _OperationT_contra) -> _VisitResultT_co:
        """Visit an operation."""
        ...
