# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Rewrite patterns for Pulse-to-Q1 instruction lowering."""

from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern

from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SynchronizeOp,
    WaitOp,
)


class RewriteSynchronizeOp(RewritePattern):
    """Skeleton for COMPILER-1344 synchronize macro-expansion."""

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SynchronizeOp, _rewriter: PatternRewriter) -> None:
        # TODO(COMPILER-1343): Replace pulse.sync with Q1 macro-expansion.
        return


class RewriteWaitOp(RewritePattern):
    """Skeleton for COMPILER-1344 wait macro-expansion."""

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: WaitOp, _rewriter: PatternRewriter) -> None:
        # TODO(COMPILER-1343): Replace pulse.wait with Q1 macro-expansion.
        return


class RewritePhaseSetOp(RewritePattern):
    """Skeleton for COMPILER-1345 phase-set macro-expansion."""

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PhaseSetOp, _rewriter: PatternRewriter) -> None:
        # TODO(COMPILER-1344): Replace pulse.phase_set with Q1 macro-expansion.
        return


class RewritePhaseShiftOp(RewritePattern):
    """Skeleton for COMPILER-1345 phase-shift macro-expansion."""

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PhaseShiftOp, _rewriter: PatternRewriter) -> None:
        # TODO(COMPILER-1344): Replace pulse.phase_shift with Q1 macro-expansion.
        return


class RewritePulseOp(RewritePattern):
    """Skeleton for COMPILER-1343 pulse-playback macro-expansion."""

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PulseOp, _rewriter: PatternRewriter) -> None:
        # TODO(COMPILER-1345): Replace pulse.pulse with Q1 macro-expansion.
        return


class RewriteStartContinuousWaveformOp(RewritePattern):
    """Skeleton for COMPILER-1343 start-continuous-waveform macro-expansion."""

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: StartContinuousWaveformOp, _rewriter: PatternRewriter
    ) -> None:
        # TODO(COMPILER-1345): Replace pulse.start_continuous_waveform with Q1 macro.
        return


class RewriteStopContinuousWaveformOp(RewritePattern):
    """Skeleton for COMPILER-1343 stop-continuous-waveform macro-expansion."""

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: StopContinuousWaveformOp, _rewriter: PatternRewriter
    ) -> None:
        # TODO(COMPILER-1345): Replace pulse.stop_continuous_waveform with Q1 macro.
        return


class RewriteAcquireOp(RewritePattern):
    """Skeleton for COMPILER-1346 acquisition/readout macro-expansion."""

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AcquireOp, _rewriter: PatternRewriter) -> None:
        # TODO(COMPILER-1346): Replace pulse.acquire with Q1 macro-expansion.
        return


def create_pulse_to_q1_lowering_patterns(
    target_data: QbloxTargetData | None = None,
) -> tuple[RewritePattern, ...]:
    """Create the per-operation rewrite set for Pulse-to-Q1 lowering.

    The returned order reflects the present structure of the lowering work. Timing
    operations appear first, followed by phase control, waveform emission, and acquisition.
    This keeps the skeleton aligned with the intended progression of the forthcoming
    conversion tickets.
    """

    resolved_target_data = target_data or TARGET_DATA
    return (
        RewriteSynchronizeOp(resolved_target_data),
        RewriteWaitOp(resolved_target_data),
        RewritePhaseSetOp(resolved_target_data),
        RewritePhaseShiftOp(resolved_target_data),
        RewritePulseOp(resolved_target_data),
        RewriteStartContinuousWaveformOp(resolved_target_data),
        RewriteStopContinuousWaveformOp(resolved_target_data),
        RewriteAcquireOp(resolved_target_data),
    )
