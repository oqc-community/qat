# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :mod:`qat.experimental.dialect.common.ssa`."""

from __future__ import annotations

from xdsl.dialects.builtin import i32
from xdsl.ir import Block
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.common.ssa import resolve_block_argument


def test_non_argument_value_is_returned_unchanged():
    value = create_ssa_value(i32)
    assert resolve_block_argument(value, {}) is value


def test_chained_substitutions_resolve_to_the_root_value():
    block = Block(arg_types=[i32, i32])
    first, second = block.args
    root = create_ssa_value(i32)
    representative = {first: second, second: root}
    assert resolve_block_argument(first, representative) is root


def test_unmapped_argument_is_returned_unchanged():
    block = Block(arg_types=[i32])
    (arg,) = block.args
    assert resolve_block_argument(arg, {}) is arg


def test_a_cycle_terminates_the_walk():
    block = Block(arg_types=[i32, i32])
    first, second = block.args
    representative = {first: second, second: first}
    assert resolve_block_argument(first, representative) in (first, second)
