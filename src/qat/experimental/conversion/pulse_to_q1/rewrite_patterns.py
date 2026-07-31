# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Rewrite patterns for the Pulse-to-Q1 phase legalisation and lowering stages."""

from collections.abc import Callable

from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern

from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.experimental.conversion.pulse_to_q1.phase import PhaseLegalisation, PhaseLowering
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    CreateFrameOp,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SynchronizeOp,
    WaitOp,
)


class RewriteCreateFrameOp(RewritePattern):
    """Skeleton for frequency initialisation from ``pulse.create_frame``.

    The intended lowering emits ``q1.set_freq`` + ``q1.upd_param`` using the
    frame's NCO intermediate frequency. ``CreateFrameOp.frequency`` carries
    the total carrier frequency. Extracting the IF requires the LO frequency,
    which is hardware-model information unavailable at this pipeline stage.
    A prior legalisation pass must decompose ``carrier = LO + IF`` and rewrite
    the operand to the IF before this pattern can fire safely.

    TODO(COMPILER-1386): Implement once the carrier-to-IF legalisation pass is in place.
    """

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CreateFrameOp, _rewriter: PatternRewriter) -> None:
        # TODO(COMPILER-1386): Emit set_freq + upd_param once IF is available.
        return


class RewriteSynchronizeOp(RewritePattern):
    """Skeleton for COMPILER-1344 synchronize macro-expansion."""

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SynchronizeOp, _rewriter: PatternRewriter) -> None:
        # TODO(COMPILER-1343): Replace pulse.sync with Q1 macro-expansion.
        return


class RewriteWaitOp(RewritePattern):
    """Skeleton for COMPILER-1344 wait macro-expansion.

    TODO(COMPILER-1343): Replace pulse.wait with Q1 macro-expansion.
    """

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: WaitOp, _rewriter: PatternRewriter) -> None:
        # TODO(COMPILER-1343): Replace pulse.wait with Q1 macro-expansion.
        return


class RewritePhaseSetOp(RewritePattern):
    """Match ``pulse.phase_set`` and delegate stage policy to ``rewrite_callable``.

    The same matcher is used in legalisation and lowering. The callable decides whether the
    rewrite remains in Pulse or emits Q1 instructions.
    """

    def __init__(
        self,
        target_data: QbloxTargetData,
        rewrite_callable: Callable,
    ) -> None:
        self.target_data = target_data
        self.rewrite_callable = rewrite_callable

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PhaseSetOp, rewriter: PatternRewriter) -> None:
        self.rewrite_callable(op, rewriter, self.target_data)


class RewritePhaseShiftOp(RewritePattern):
    """Match ``pulse.phase_shift`` and delegate stage policy to ``rewrite_callable``.

    Mirrors ``RewritePhaseSetOp`` in structure.
    """

    def __init__(
        self,
        target_data: QbloxTargetData,
        rewrite_callable: Callable,
    ) -> None:
        self.target_data = target_data
        self.rewrite_callable = rewrite_callable

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PhaseShiftOp, rewriter: PatternRewriter) -> None:
        self.rewrite_callable(op, rewriter, self.target_data)


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


def create_legalisation_patterns() -> tuple[RewritePattern, ...]:
    """Create the rewrite set used by the legalisation stage.

    Canonicalises ``pulse.phase_set`` and ``pulse.phase_shift`` operands inside
    the Pulse dialect. New legalisation behaviour can be added here without
    touching the lowering factory.

    :returns: Ordered pattern tuple for the legalisation pass.
    """
    canonicalise = PhaseLegalisation()
    return (
        RewritePhaseSetOp(
            TARGET_DATA,
            rewrite_callable=lambda op, rewriter, _target_data: canonicalise(op, rewriter),
        ),
        RewritePhaseShiftOp(
            TARGET_DATA,
            rewrite_callable=lambda op, rewriter, _target_data: canonicalise(op, rewriter),
        ),
    )


def create_pulse_to_q1_lowering_patterns(
    target_data: QbloxTargetData | None = None,
) -> tuple[RewritePattern, ...]:
    """Create the rewrite set used by the lowering stage.

    Phase entries are configured with ``PhaseLowering``. All remaining entries are
    scaffold patterns that preserve IR shape pending their dedicated lowering
    implementations.

    :param target_data: QBlox target description. When omitted, the repository
        default is used.
    :returns: Ordered pattern tuple for the lowering pass.
    """

    resolved_target_data = target_data or TARGET_DATA
    return (
        RewritePhaseSetOp(resolved_target_data, rewrite_callable=PhaseLowering()),
        RewritePhaseShiftOp(resolved_target_data, rewrite_callable=PhaseLowering()),
        RewriteCreateFrameOp(resolved_target_data),
        RewriteSynchronizeOp(resolved_target_data),
        RewriteWaitOp(resolved_target_data),
        RewritePulseOp(resolved_target_data),
        RewriteStartContinuousWaveformOp(resolved_target_data),
        RewriteStopContinuousWaveformOp(resolved_target_data),
        RewriteAcquireOp(resolved_target_data),
    )
