# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Block and region surgery helpers."""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.ir import Block, Operation, Region


def detach_non_terminator_ops(block: Block) -> list[Operation]:
    """Detach and return every operation of ``block`` except its terminator.

    The block's last operation is assumed to be its terminator and is left in place. An
    empty or terminator-only block yields an empty list.
    """
    ops = list(block.ops)[:-1]
    for op in ops:
        op.detach()
    return ops


def install_single_block(region: Region, ops: Sequence[Operation]) -> None:
    """Replace every block of ``region`` with one block holding ``ops``."""
    block = Block()
    block.add_ops(list(ops))
    while region.blocks:
        region.erase_block(region.blocks[0])
    region.add_block(block)
