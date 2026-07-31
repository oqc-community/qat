# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Pass and pipeline definitions for the Pulse-to-Q1 conversion."""

import math
from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass, PassPipeline
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.utils.exceptions import PassFailedException

from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.experimental.conversion.pulse_to_q1.rewrite_patterns import (
    create_legalisation_patterns,
    create_pulse_to_q1_lowering_patterns,
)
from qat.experimental.conversion.pulse_to_q1.sequence_outlining import Q1OutliningPass
from qat.experimental.dialect.pulse.ir import (
    ConstantOp,
    CreateFrameOp,
    PhaseSetOp,
    PhaseShiftOp,
    WaitOp,
)
from qat.experimental.dialect.pulse.utils import (
    extract_frequency_hz,
    extract_phase_radians,
    extract_time_seconds,
)

_TIME_ROUNDING_TOLERANCE_NS = 1e-3


@dataclass(frozen=True)
class Q1PulseValidationPass(ModulePass):
    """Validate QBlox-specific pre-conditions on constant Pulse operands.

    Enforces hardware constraints that cannot be expressed as Pulse dialect
    invariants, ahead of the legalisation and lowering stages. Only constant
    operands are checked. Dynamic operands are deferred without error.

    The following constraints are enforced:

    * ``pulse.wait`` constant duration: finite, non-negative, and an integer
      number of nanoseconds.
    * ``pulse.create_frame`` constant frequency: finite.
    * ``pulse.phase_set`` and ``pulse.phase_shift`` constant phase: finite.
    """

    name = "q1-pulse-validation"
    target_data: QbloxTargetData = field(default=TARGET_DATA)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for pulse_op in op.walk():
            if isinstance(pulse_op, WaitOp):
                self._validate_wait(pulse_op)
            elif isinstance(pulse_op, CreateFrameOp):
                self._validate_create_frame(pulse_op)
            elif isinstance(pulse_op, PhaseSetOp | PhaseShiftOp):
                self._validate_phase(pulse_op)

    def _validate_wait(self, op: WaitOp) -> None:
        if not isinstance(op.duration.owner, ConstantOp):
            return
        seconds = extract_time_seconds(op)
        if not math.isfinite(seconds):
            raise PassFailedException(f"{op.name} time must be finite. Got {seconds}.")
        if seconds < 0:
            raise PassFailedException(
                f"{op.name} time must be non-negative. Got {seconds}."
            )

        ns_float = seconds * 1e9
        if 0 < ns_float < 1:
            raise PassFailedException(
                f"pulse.wait duration smaller than one nanosecond is illegal. Got {ns_float} ns."
            )

        ns_int = round(ns_float)
        if not math.isclose(
            ns_float, ns_int, abs_tol=_TIME_ROUNDING_TOLERANCE_NS, rel_tol=0
        ):
            raise PassFailedException(
                "pulse.wait duration must map to integer nanoseconds within tolerance. "
                f"Got {ns_float} ns."
            )

    def _validate_create_frame(self, op: CreateFrameOp) -> None:
        if not isinstance(op.frequency.owner, ConstantOp):
            return
        frequency_hz = extract_frequency_hz(op)
        if not math.isfinite(frequency_hz):
            raise PassFailedException(
                f"{op.name} frequency must be finite. Got {frequency_hz}."
            )

    def _validate_phase(self, op: PhaseSetOp | PhaseShiftOp) -> None:
        if not isinstance(op.phase.owner, ConstantOp):
            return
        radians = extract_phase_radians(op)
        if not math.isfinite(radians):
            raise PassFailedException(f"{op.name} phase must be finite. Got {radians}.")


@dataclass(frozen=True)
class Q1PulseLegalisationPass(ModulePass):
    """Apply Pulse phase legalisation before Pulse-to-Q1 lowering.

    This stage applies the legalisation pattern set to Pulse-level operands after
    validation. The pattern set is expected to grow over time as more Pulse operations
    acquire legalisation support.
    """

    name = "q1-pulse-legalisation"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(create_legalisation_patterns()),
            apply_recursively=False,
        ).rewrite_module(op)


@dataclass(frozen=True)
class PulseToQ1LoweringPass(ModulePass):
    """Apply the Pulse-to-Q1 rewrite stage inside outlined sequences.

    ``Q1OutliningPass`` first isolates one logical sequence envelope for each
    frame partition. This pass then traverses those envelopes and applies the
    per-operation rewrite set that converts Pulse-level instructions into the
    flat Q1 instruction dialect.
    """

    name = "pulse-to-q1-lowering"
    target_data: QbloxTargetData = field(default=TARGET_DATA)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                create_pulse_to_q1_lowering_patterns(self.target_data)
            ),
            apply_recursively=False,
        ).rewrite_module(op)


def create_default_pulse_to_q1_pipeline(
    target_data: QbloxTargetData = TARGET_DATA,
) -> PassPipeline:
    """Create the default pass pipeline for Pulse-to-Q1 conversion.

    The pipeline has four stages. ``Q1OutliningPass`` partitions the Pulse program
    into per-frame ``q1_sequence.sequence`` envelopes. ``Q1PulseValidationPass``
    enforces QBlox-specific constant operand constraints. ``Q1PulseLegalisationPass``
    canonicalises phase operands in the Pulse dialect. ``PulseToQ1LoweringPass`` completes
    the conversion within those sequences.

    :param target_data: QBlox target description passed to outlining, validation,
        and lowering stages.
    :returns: Pass pipeline for the default Pulse-to-Q1 conversion flow.
    """

    return PassPipeline(
        (
            Q1OutliningPass(target_data=target_data),
            Q1PulseValidationPass(target_data=target_data),
            Q1PulseLegalisationPass(),
            PulseToQ1LoweringPass(target_data=target_data),
        )
    )
