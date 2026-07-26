# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :mod:`qat.experimental.dialect.common.region`."""

from __future__ import annotations

from xdsl.dialects.test import TestOp as _FillerOp
from xdsl.ir import Block, Region

from qat.experimental.dialect.common.region import (
    detach_non_terminator_ops,
    install_single_block,
)


def test_detach_non_terminator_ops_leaves_the_terminator_in_place():
    lead, middle, terminator = _FillerOp(), _FillerOp(), _FillerOp()
    block = Block([lead, middle, terminator])

    detached = detach_non_terminator_ops(block)

    assert detached == [lead, middle]
    assert all(op.parent is None for op in detached)
    assert list(block.ops) == [terminator]


def test_detach_non_terminator_ops_on_a_terminator_only_block_returns_empty():
    terminator = _FillerOp()
    block = Block([terminator])

    assert detach_non_terminator_ops(block) == []
    assert list(block.ops) == [terminator]


def test_install_single_block_replaces_every_block():
    region = Region([Block([_FillerOp()]), Block([_FillerOp()])])
    first, second = _FillerOp(), _FillerOp()

    install_single_block(region, [first, second])

    assert len(region.blocks) == 1
    assert list(region.blocks[0].ops) == [first, second]
