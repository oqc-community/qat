# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Contiguous pulse-frame optimizations for phase and delay instructions.

Optimize contiguous phases and delays within a frame, leveraging that phase instructions
commute with wait instructions. This allows for folding sequences of phase shifts
interleaved with waits into a single phase operation. Likewise, sequences of wait
instructions interleaved with phase shifts can also be folded into a single wait. This
reduces the number of quantum instructions.
"""

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from qat.experimental.dialect.pulse.ir import AddOp, PhaseShiftOp, PhaseType, TimeType
from qat.experimental.dialect.pulse.ir.ops import PhaseSetOp, WaitOp


class CommuteWaitBeforePhaseShift(RewritePattern):
    """Commute phase shift operations past wait operations to enable better folding.

    **Scenario 1: PhaseShift, Wait -> Wait, PhaseShift**

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %phase = pulse.constant<0.5> : !pulse.phase
        %duration = pulse.constant<1.0e-9> : !pulse.time
        %shift = pulse.phase_shift(%frame, %phase) : !pulse.frame
        %wait = pulse.wait(%shift, %duration) : !pulse.frame
        %acquire = pulse.acquire(%wait, %duration) : !pulse.frame

    becomes:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %phase = pulse.constant<0.5> : !pulse.phase
        %duration = pulse.constant<1.0e-9> : !pulse.time
        %wait = pulse.wait(%frame, %duration) : !pulse.frame
        %shift = pulse.phase_shift(%wait, %phase) : !pulse.frame
        %acquire = pulse.acquire(%shift, %duration) : !pulse.frame

    This enables subsequent folding of multiple waits or phases that were initially
    interleaved. Repeated application via `GreedyRewritePatternApplier` bubbles all
    waits to one side, after which fold patterns eliminate redundant operations.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: WaitOp, rewriter: PatternRewriter) -> None:
        anchor_block = op.parent
        frame_owner = op.frame.owner
        if not isinstance(frame_owner, PhaseShiftOp):
            return
        if (
            frame_owner.parent is not anchor_block
            or frame_owner.result.uses.get_length() != 1
        ):
            return

        new_wait = WaitOp(frame_owner.frame, op.duration)
        new_phase = PhaseShiftOp(new_wait.result, frame_owner.phase)

        rewriter.replace_op(op, [new_wait, new_phase], new_results=[new_phase.result])
        rewriter.erase_op(frame_owner)


class FoldContiguousPhases(RewritePattern):
    """Fold contiguous phase shifts on the same frame into a single phase shift or set.

    **Scenario 1: PhaseShift + PhaseShift**

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %p1 = pulse.constant<0.5> : !pulse.phase
        %p2 = pulse.constant<1.0> : !pulse.phase
        %shift1 = pulse.phase_shift(%frame, %p1) : !pulse.frame
        %shift2 = pulse.phase_shift(%shift1, %p2) : !pulse.frame
        %acquire = pulse.acquire(%shift2, %duration) : !pulse.frame

    becomes:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %p1 = pulse.constant<0.5> : !pulse.phase
        %p2 = pulse.constant<1.0> : !pulse.phase
        %total = pulse.add(%p1, %p2) : !pulse.phase
        %merged = pulse.phase_shift(%frame, %total) : !pulse.frame
        %acquire = pulse.acquire(%merged, %duration) : !pulse.frame

    **Scenario 2: PhaseShift + PhaseSet**

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %p1 = pulse.constant<0.5> : !pulse.phase
        %p2 = pulse.constant<1.0> : !pulse.phase
        %shift = pulse.phase_shift(%frame, %p1) : !pulse.frame
        %set = pulse.phase_set(%shift, %p2) : !pulse.frame
        %acquire = pulse.acquire(%set, %duration) : !pulse.frame

    becomes:

    .. code-block:: mlir
        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %p1 = pulse.constant<0.5> : !pulse.phase
        %p2 = pulse.constant<1.0> : !pulse.phase
        %merged = pulse.phase_set(%frame, %p2) : !pulse.frame
        %acquire = pulse.acquire(%merged, %duration) : !pulse.frame

    **Scenario 3: PhaseSet + PhaseShift**

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %p1 = pulse.constant<0.5> : !pulse.phase
        %p2 = pulse.constant<1.0> : !pulse.phase
        %set = pulse.phase_set(%frame, %p1) : !pulse.frame
        %shift = pulse.phase_shift(%set, %p2) : !pulse.frame
        %acquire = pulse.acquire(%shift, %duration) : !pulse.frame

    becomes:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %p1 = pulse.constant<0.5> : !pulse.phase
        %p2 = pulse.constant<1.0> : !pulse.phase
        %total = pulse.add(%p1, %p2) : !pulse.phase
        %merged = pulse.phase_set(%frame, %total) : !pulse.frame
        %acquire = pulse.acquire(%merged, %duration) : !pulse.frame

    **Scenario 4: PhaseSet + PhaseSet**

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %p1 = pulse.constant<0.5> : !pulse.phase
        %p2 = pulse.constant<1.0> : !pulse.phase
        %set1 = pulse.phase_set(%frame, %p1) : !pulse.frame
        %set2 = pulse.phase_set(%set1, %p2) : !pulse.frame
        %acquire = pulse.acquire(%set2, %duration) : !pulse.frame

    becomes:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %p2 = pulse.constant<1.0> : !pulse.phase
        %merged = pulse.phase_set(%frame, %p2) : !pulse.frame
        %acquire = pulse.acquire(%merged, %duration) : !pulse.frame
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: PhaseSetOp | PhaseShiftOp, rewriter: PatternRewriter
    ) -> None:
        anchor_block = op.parent
        frame_owner = op.frame.owner
        if not isinstance(frame_owner, PhaseShiftOp | PhaseSetOp):
            return
        if (
            frame_owner.parent is not anchor_block
            or frame_owner.result.uses.get_length() != 1
        ):
            return

        if isinstance(op, PhaseShiftOp):
            total_phase = AddOp(op.phase, frame_owner.phase, PhaseType())
            rewriter.insert_op(total_phase, InsertPoint.before(op))
            if isinstance(frame_owner, PhaseShiftOp):
                replacement = PhaseShiftOp(frame_owner.frame, total_phase.result)
            else:
                replacement = PhaseSetOp(frame_owner.frame, total_phase.result)
        else:
            replacement = PhaseSetOp(frame_owner.frame, op.phase)

        rewriter.replace_op(op, replacement)
        rewriter.erase_op(frame_owner)


class FoldContiguousWaits(RewritePattern):
    """Fold contiguous wait operations on the same frame into a single wait.

    **Scenario 1: Wait + Wait**

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %duration1 = pulse.constant<1.0e-9> : !pulse.time
        %duration2 = pulse.constant<2.0e-9> : !pulse.time
        %wait1 = pulse.wait(%frame, %duration1) : !pulse.frame
        %wait2 = pulse.wait(%wait1, %duration2) : !pulse.frame
        %acquire = pulse.acquire(%wait2, %duration) : !pulse.frame

    becomes:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame
        %duration1 = pulse.constant<1.0e-9> : !pulse.time
        %duration2 = pulse.constant<2.0e-9> : !pulse.time
        %total = pulse.add(%duration1, %duration2) : !pulse.time
        %merged = pulse.wait(%frame, %total) : !pulse.frame
        %acquire = pulse.acquire(%merged, %duration) : !pulse.frame
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: WaitOp, rewriter: PatternRewriter) -> None:
        anchor_block = op.parent
        frame_owner = op.frame.owner
        if not isinstance(frame_owner, WaitOp):
            return
        if (
            frame_owner.parent is not anchor_block
            or frame_owner.result.uses.get_length() != 1
        ):
            return

        total_duration = AddOp(op.duration, frame_owner.duration, TimeType())
        rewriter.insert_op(total_duration, InsertPoint.before(op))
        replacement = WaitOp(frame_owner.frame, total_duration.result)
        rewriter.replace_op(op, replacement)
        rewriter.erase_op(frame_owner)


class ApplySquashContiguousOptimizations(ModulePass):
    """Apply commuting + contiguous fold optimizations."""

    name = "apply-squash-contiguous-optimizations"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    CommuteWaitBeforePhaseShift(),
                    FoldContiguousWaits(),
                    FoldContiguousPhases(),
                ]
            )
        )
        walker.rewrite_module(op)
