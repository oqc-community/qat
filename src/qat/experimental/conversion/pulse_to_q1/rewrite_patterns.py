# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Rewrite patterns for the Pulse-to-Q1 phase legalisation and lowering stages."""

import hashlib
from collections.abc import Callable

import numpy as np
from xdsl.dialects.builtin import ArrayAttr
from xdsl.ir import SSAValue
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
    WeightsAttr,
)
from qat.experimental.dialect.pulse.units import TIME_UNIT_EXPONENTS, TimeUnits
from qat.experimental.dialect.pulse.utils import require_constant_operand
from qat.experimental.dialect.q1.ir.imm_desc import DurationImm, UI5Imm, UI24Imm
from qat.experimental.dialect.q1.ir.ops import (
    AcquireImmImmImmOp,
    AcquireWeightedImmImmImmImmImmOp,
)
from qat.experimental.dialect.q1_sequence.ir.attrs import make_acquisition, make_weight
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp, find_enclosing_sequence


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
    """Lower :class:`AcquireOp` to a Q1 acquire instruction.

    Emits :class:`AcquireImmImmImmOp` when no weights are specified, or
    :class:`AcquireWeightedImmImmImmImmImmOp` when a :class:`WeightsAttr` is present.
    Acquisitions and weights are registered on the enclosing :class:`SequenceOp` and
    assigned hardware indices in encounter order.
    """

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AcquireOp, rewriter: PatternRewriter) -> None:
        sequencer = find_enclosing_sequence(op)
        acq_name = (
            op.label.data
            if op.label is not None
            else op.frame.type.port.data.replace("/", "_")
        )
        num_bins, bin_idx = self._get_bin_info(sequencer)
        if isinstance(bin_idx, SSAValue):
            raise NotImplementedError(
                "Dynamic bin indices are not supported for pulse.acquire"
            )
        duration_ns = self._get_ns_duration(op)
        acq_idx = self._register_acquisition(sequencer, acq_name, num_bins)
        # TODO(COMPILER-1349): Write ``duration_ns`` as the integration length into the
        # ``SequencerDataAttr`` on the enclosing ``SequenceOp`` once that attribute is
        # defined (pending the physical Qblox system data PR).
        if isinstance(weights_attr := op.weights, WeightsAttr):
            weights_data = weights_attr.weights.data
            weight_index_i = self._register_weight(sequencer, weights_data.real)
            weight_index_q = self._register_weight(sequencer, weights_data.imag)
            new_acquire_op = AcquireWeightedImmImmImmImmImmOp(
                UI5Imm(acq_idx),
                UI24Imm(bin_idx),
                UI5Imm(weight_index_i),
                UI5Imm(weight_index_q),
                DurationImm(duration_ns),
            )
        else:
            assert weights_attr is None
            new_acquire_op = AcquireImmImmImmOp(
                UI5Imm(acq_idx), UI24Imm(bin_idx), DurationImm(duration_ns)
            )
        # TODO: Support lowering of pulse.acquire acquisition_result
        # consumers (e.g. pulse.integrate). Once supported, replace the check below
        # with proper lowering logic. Post COMPILER-1369 work.
        if op.acquisition_result.uses:
            raise NotImplementedError(
                "pulse.acquire acquisition_result consumers are not yet supported"
            )
        rewriter.replace_op(op, new_acquire_op, new_results=[op.frame, None])
        return

    def _get_ns_duration(self, op: AcquireOp) -> int:
        """Get the duration of the acquisition in nanoseconds.

        :param op: The ``pulse.acquire`` operation to extract from.
        :returns: Duration in nanoseconds as an integer.
        :raises PassFailedException: If the duration operand is not a constant or its
            attribute is not a ``TimeAttr``.
        """
        const = require_constant_operand(op.name, "duration", op.duration)
        time_attr = const.fold()[0]
        unit = time_attr.unit.data
        value = time_attr.value.data
        ns = value * 10 ** (
            TIME_UNIT_EXPONENTS[unit] - TIME_UNIT_EXPONENTS[TimeUnits.NANOSECOND]
        )
        return round(ns)

    def _get_bin_info(self, sequence_op: SequenceOp) -> tuple[int, int | SSAValue]:
        """Get the number of bins and the next available bin index for the given sequencer.

        :returns: Tuple of (num_bins, next_bin_index).
        """
        # TODO(COMPILER-1387): Replace hard-coded num_bins / bin_idx with values derived
        # from a dedicated acquisition-binding analysis pass that counts shots per AcquireOp
        # and allocates a per-acquire bin-counter register for the repeat loop
        # (analogous to BindingPass / _legalise_bound in the legacy stack).
        num_bins = sequence_op.properties.get("num_runs", 1)
        bin_idx = sequence_op.properties.get("next_bin_index", 0)
        return num_bins, bin_idx

    def _register_weight(self, sequence_op: SequenceOp, weights_data: np.ndarray) -> int:
        """Register a new weight in the target data and return its index.

        :param sequence_op: The enclosing ``SequenceOp`` to register the weight in.
        :param weights_data: Weight coefficients to register.
        :returns: Index of the registered weight.
        """
        weight_list = weights_data.tolist()
        name = hashlib.md5(str(weight_list).encode(), usedforsecurity=False).hexdigest()[:8]
        for i, entry in enumerate(sequence_op.weights):
            if entry.weight_name.data == name:
                return i
        weight_index = len(sequence_op.weights)
        weight_attr = make_weight(name, weight_index, weight_list)
        sequence_op.properties["weights"] = ArrayAttr([*sequence_op.weights, weight_attr])
        return weight_index

    def _register_acquisition(
        self, sequence_op: SequenceOp, name: str, num_bins: int
    ) -> int:
        """Register a new acquisition in the target data and return its index.

        :param sequence_op: The enclosing ``SequenceOp`` to register the acquisition in.
        :param name: Unique name for the acquisition entry.
        :param num_bins: Number of bins to allocate for the acquisition.
        :returns: Index of the registered acquisition.
        """
        if name in (entry.acquisition_name.data for entry in sequence_op.acquisitions):
            raise ValueError(
                f"Acquisition name '{name}' already exists in sequence '{sequence_op.channel_id.data}'"
            )
        acq_index = len(sequence_op.acquisitions)
        acq_attr = make_acquisition(name, acq_index, num_bins)
        sequence_op.properties["acquisitions"] = ArrayAttr(
            [*sequence_op.acquisitions, acq_attr]
        )
        return acq_index


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

    Phase entries are configured with ``PhaseLowering``. ``RewriteAcquireOp`` fully
    lowers ``pulse.acquire`` to Q1 acquire instructions. All other entries are scaffold
    patterns that preserve IR shape pending their dedicated lowering implementations.

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
