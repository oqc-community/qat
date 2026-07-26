# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for control-flow analysis in :mod:`qat.experimental.dialect.common.cfg`.

The graph is built from throwaway terminators defined here rather than any real
dialect, so the analyses are exercised purely through the
:class:`SuccessorOperandsTrait` contract.
"""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import i32
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    successor_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import IsTerminator
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.common.cfg import (
    SuccessorOperandsTrait,
    block_predecessors,
    reachable_blocks,
    successor_edges,
)


class _ForwardingJumpSuccessors(SuccessorOperandsTrait):
    @classmethod
    def successor_edges(cls, op: Operation) -> list[tuple[Block, list[SSAValue]]]:
        assert isinstance(op, _ForwardingJumpOp)
        return [(op.dest, list(op.arguments))]


class _TwoWaySuccessors(SuccessorOperandsTrait):
    @classmethod
    def successor_edges(cls, op: Operation) -> list[tuple[Block, list[SSAValue]]]:
        assert isinstance(op, _TwoWayOp)
        return [(op.left, []), (op.right, [])]


@irdl_op_definition
class _ForwardingJumpOp(IRDLOperation):
    name = "test.fwd_jump"
    arguments = var_operand_def(i32)
    dest = successor_def()
    traits = traits_def(IsTerminator(), _ForwardingJumpSuccessors())

    def __init__(self, arguments, dest):
        super().__init__(operands=[arguments], successors=(dest,))


@irdl_op_definition
class _TwoWayOp(IRDLOperation):
    name = "test.two_way"
    left = successor_def()
    right = successor_def()
    traits = traits_def(IsTerminator(), _TwoWaySuccessors())

    def __init__(self, left, right):
        super().__init__(successors=(left, right))


@irdl_op_definition
class _LeafOp(IRDLOperation):
    """A terminator that ends a path and carries no successor-operands trait."""

    name = "test.leaf"
    traits = traits_def(IsTerminator())


def _region(*blocks: Block) -> Region:
    return Region(list(blocks))


def test_successor_edges_empty_for_terminator_without_trait():
    assert successor_edges(_LeafOp()) == []


def test_reachable_blocks_follows_successor_edges():
    body, left, right, sink, orphan = Block(), Block(), Block(), Block(), Block()
    body.add_op(_TwoWayOp(left, right))
    left.add_op(_ForwardingJumpOp([], sink))
    right.add_op(_LeafOp())
    sink.add_op(_LeafOp())
    orphan.add_op(_LeafOp())
    _region(body, left, right, sink, orphan)

    assert reachable_blocks(body) == {body, left, right, sink}


def test_block_predecessors_keys_every_block_including_sourceless():
    head, tail = Block(), Block()
    head.add_op(_ForwardingJumpOp([], tail))
    tail.add_op(_LeafOp())
    ordered = [head, tail]
    _region(*ordered)

    predecessors = block_predecessors(ordered)

    assert predecessors[head] == []
    assert predecessors[tail] == [(head, [])]


def test_block_predecessors_records_forwarded_operands():
    head, tail = Block(), Block()
    forwarded = create_ssa_value(i32)
    head.add_op(_ForwardingJumpOp([forwarded], tail))
    tail.add_op(_LeafOp())
    ordered = [head, tail]
    _region(*ordered)

    assert block_predecessors(ordered)[tail] == [(head, [forwarded])]


@pytest.mark.parametrize("trait_op", [_ForwardingJumpOp, _TwoWayOp])
def test_each_terminator_carries_the_trait(trait_op):
    assert trait_op.get_trait(SuccessorOperandsTrait) is not None
