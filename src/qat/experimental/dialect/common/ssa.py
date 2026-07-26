# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""SSA value resolution helpers."""

from __future__ import annotations

from xdsl.ir import BlockArgument, SSAValue


def resolve_block_argument(
    value: SSAValue, representative: dict[BlockArgument, SSAValue]
) -> SSAValue:
    """Follow a block-argument substitution map to a non-argument value.

    ``representative`` maps a block argument to the value that replaces it. The
    walk chases chained substitutions and stops at the first value that is not a
    mapped block argument. A cycle terminates the walk rather than looping.
    """
    seen: set[SSAValue] = set()
    while (
        isinstance(value, BlockArgument) and value in representative and value not in seen
    ):
        seen.add(value)
        value = representative[value]
    return value
