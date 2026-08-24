# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :mod:`qat.experimental.passes.pass_ordering`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass, PassPipeline
from xdsl.utils.exceptions import VerifyException

from qat.experimental.passes.pass_ordering import (
    OrderedPass,
    OrderedPassPipeline,
    validate_pass_ordering,
)


@dataclass(frozen=True)
class _PlainPass(ModulePass):
    """A pass without ordering constraints; not an :class:`OrderedPass`."""

    name = "plain"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        pass


@dataclass(frozen=True)
class _FoldPass(OrderedPass, ModulePass):
    name = "fold"

    _runs_before: ClassVar[frozenset[type[ModulePass]] | None] = None

    def runs_before(self) -> frozenset[type[ModulePass]]:
        if _FoldPass._runs_before is None:
            _FoldPass._runs_before = frozenset({_EvaluatePass})
        return _FoldPass._runs_before

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        pass


@dataclass(frozen=True)
class _EvaluatePass(ModulePass):
    name = "evaluate"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        pass


@dataclass(frozen=True)
class _RequireFoldPass(OrderedPass, ModulePass):
    name = "require-fold"

    _required_predecessors: ClassVar[frozenset[type[ModulePass]] | None] = None

    def required_predecessors(self) -> frozenset[type[ModulePass]]:
        if _RequireFoldPass._required_predecessors is None:
            _RequireFoldPass._required_predecessors = frozenset({_FoldPass})
        return _RequireFoldPass._required_predecessors

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        pass


class TestValidatePassOrdering:
    def test_correct_runs_before_order_is_accepted(self):
        validate_pass_ordering([_FoldPass(), _EvaluatePass()])

    def test_runs_before_ignored_when_target_absent(self):
        # 'fold' declares runs_before 'evaluate'; with no evaluate present there is
        # nothing to order against, so this must not raise.
        validate_pass_ordering([_FoldPass(), _PlainPass()])

    def test_runs_before_wrong_order_is_rejected(self):
        with pytest.raises(VerifyException, match="must run before 'evaluate'"):
            validate_pass_ordering([_EvaluatePass(), _FoldPass()])

    def test_duplicate_pass_after_target_is_accepted_as_cleanup(self):
        # 'fold' runs before 'evaluate' but may reappear afterwards as a clean-up stage;
        # only its first occurrence is checked, so the trailing copy is not a violation.
        validate_pass_ordering([_FoldPass(), _EvaluatePass(), _FoldPass()])

    def test_correct_required_predecessor_order_is_accepted(self):
        validate_pass_ordering([_FoldPass(), _RequireFoldPass()])

    def test_missing_required_predecessor_is_rejected(self):
        with pytest.raises(VerifyException, match="requires 'fold'"):
            validate_pass_ordering([_RequireFoldPass()])

    def test_required_predecessor_after_dependent_is_rejected(self):
        with pytest.raises(VerifyException, match="not before 'require-fold'"):
            validate_pass_ordering([_RequireFoldPass(), _FoldPass()])

    def test_passes_without_constraints_are_skipped(self):
        # Neither pass is an OrderedPass, so validation is a no-op.
        validate_pass_ordering([_PlainPass(), _EvaluatePass()])


class TestOrderedPassPipeline:
    def test_valid_pipeline_is_constructed_and_applies(self):
        pipeline = OrderedPassPipeline((_FoldPass(), _EvaluatePass()))
        assert isinstance(pipeline, PassPipeline)
        pipeline.apply(Context(), ModuleOp([]))

    def test_invalid_pipeline_raises_on_construction(self):
        with pytest.raises(VerifyException, match="must run before 'evaluate'"):
            OrderedPassPipeline((_EvaluatePass(), _FoldPass()))

    def test_missing_required_predecessor_raises_on_construction(self):
        with pytest.raises(VerifyException, match="requires 'fold'"):
            OrderedPassPipeline((_RequireFoldPass(),))
