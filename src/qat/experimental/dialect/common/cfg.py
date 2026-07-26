# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Control-flow analysis over block CFGs.

The algorithms here traverse a region's blocks through their terminators without
knowing any concrete operation type. A terminator advertises its outgoing edges
through the :class:`SuccessorOperandsTrait`, the local counterpart to MLIR's
``BranchOpInterface``. xDSL tracks a block's successors but not the operand group
forwarded along each edge, so the trait supplies that missing half and lets
reachability and predecessor analysis stay dialect-neutral.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

from xdsl.ir import Block, Operation, SSAValue
from xdsl.traits import OpTrait


class SuccessorOperandsTrait(OpTrait, abc.ABC):
    """Trait exposing a terminator's successor edges.

    Each edge pairs a successor block with the operand group forwarded to its block
    arguments. A terminator that ends a path, a halt or return, does not carry the trait and
    is treated as having no successors.
    """

    @classmethod
    @abc.abstractmethod
    def successor_edges(cls, op: Operation) -> list[tuple[Block, list[SSAValue]]]:
        """Return the ``(successor, forwarded operands)`` edges of ``op``."""
        raise NotImplementedError


def successor_edges(terminator: Operation) -> list[tuple[Block, list[SSAValue]]]:
    """Return the ``(successor, forwarded operands)`` edges of ``terminator``.

    A terminator that does not carry :class:`SuccessorOperandsTrait` ends a path
    and reports no successors.
    """
    trait = terminator.get_trait(SuccessorOperandsTrait)
    if trait is None:
        return []
    return trait.successor_edges(terminator)


def reachable_blocks(entry: Block) -> set[Block]:
    """Return the set of blocks reachable from ``entry`` along successor edges."""
    seen: set[Block] = set()
    worklist = [entry]
    while worklist:
        block = worklist.pop()
        if block in seen:
            continue
        seen.add(block)
        terminator = block.last_op
        if terminator is not None:
            worklist.extend(succ for succ, _ in successor_edges(terminator))
    return seen


def block_predecessors(
    ordered: Sequence[Block],
) -> dict[Block, list[tuple[Block, list[SSAValue]]]]:
    """Map each block to the ``(predecessor, forwarded operands)`` edges into it.

    Every block in ``ordered`` is keyed, so a block with no predecessor maps to
    an empty list.
    """
    predecessors: dict[Block, list[tuple[Block, list[SSAValue]]]] = {
        block: [] for block in ordered
    }
    for block in ordered:
        terminator = block.last_op
        if terminator is None:
            continue
        for successor, forwarded in successor_edges(terminator):
            predecessors[successor].append((block, list(forwarded)))
    return predecessors
