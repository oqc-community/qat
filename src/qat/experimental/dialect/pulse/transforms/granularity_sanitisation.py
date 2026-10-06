# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import ConstantOp
from qat.experimental.dialect.pulse.ir.attributes import SampledWaveformAttr, TimeAttr
from qat.experimental.dialect.pulse.ir.types import TimeType, WaveformType
from qat.experimental.passes.pass_ordering import OrderedPass
from qat.experimental.system_data.pulse.constraints import PulseLevelConstraints


def _round_up_to_granularity_if_required(value: int, granularity: int) -> int:
    """Returns the value rounded up to the nearest multiple of granularity.

    :param value: Value in picoseconds.
    :param granularity: Granularity in picoseconds.
    :returns: The rounded-up value.
    """

    remainder = value % granularity
    if remainder == 0:
        return value
    return value + granularity - remainder


class GranularitySanitisation(RewritePattern):
    """Rounds the durations of quantum instructions so they are multiples of the clock
    cycle.

    This pattern ensures that ConstantOps with TimeType are rounded up to the nearest
    multiple of the specified granularity.
    ConstantOps with WaveformType are rounded up to the nearest multiple of the specified
    granularity.
    If a waveform duration is rounded up, the waveform is padded with zeros to ensure
    that the new duration is a multiple of the granularity.

    .. warning::

        This pass has the potential to invalidate the timings for sequences of instructions
        that are time-sensitive.
    """

    # Note: In comparison to the prior implementation, we currently round up to the nearest
    # granularity unit for all instructions. See COMPILER-1251 for a separate pass to handle
    # Acquire/Wait rounding-down requirements.

    def __init__(self, constraints: PulseLevelConstraints) -> None:
        """Initializes the pattern with the given constraints.

        :param constraints: Pulse level constraints containing the granularity.
        """

        self.granularity_ps = constraints.granularity_ps

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ConstantOp, rewriter: PatternRewriter) -> None:
        if not isinstance(op.result.type, TimeType | WaveformType):
            return

        # operand_value is a tuple of attributes carrying their unit via their type.
        # TimeAttr values are in picoseconds, WaveformAttr widths/sample_time in ps.
        operand_value = op.fold()
        if not operand_value:
            return

        if op.result.type == TimeType():
            time_attr = operand_value[0]

            new_duration = _round_up_to_granularity_if_required(
                time_attr.literal_value, self.granularity_ps
            )
            if new_duration == time_attr.literal_value:
                return

            new_time_op = ConstantOp(TimeAttr(new_duration), TimeType())
            rewriter.replace_op(op, new_time_op)
            return

        waveform_attr = operand_value[0]
        waveform_array = waveform_attr.literal_value

        width_attr = waveform_attr.width
        sample_time_attr = waveform_attr.sample_time
        width_ps = width_attr.literal_value
        sample_time_ps = sample_time_attr.literal_value

        if width_ps % sample_time_ps != 0:
            return

        new_width_ps = _round_up_to_granularity_if_required(width_ps, self.granularity_ps)
        if new_width_ps == width_ps:
            return

        if new_width_ps < width_ps:
            return

        width_increase_ps = new_width_ps - width_ps
        padding, remainder = divmod(width_increase_ps, sample_time_ps)
        if remainder != 0:
            raise PassFailedException(
                f"Waveform granularity rounding produces non-integral sample count: "
                f"width {width_ps} → {new_width_ps} ps (increase {width_increase_ps} ps) "
                f"with sample_time {sample_time_ps} ps. "
                f"Width must round to a multiple of sample_time."
            )
        new_waveform_array = np.pad(
            waveform_array,
            (0, padding),
            mode="constant",
            constant_values=0,
        )

        new_waveform_attr = SampledWaveformAttr(
            new_waveform_array,
            TimeAttr(new_width_ps),
            sample_time_attr,
        )
        new_waveform_op = ConstantOp(new_waveform_attr, WaveformType())
        rewriter.replace_op(op, new_waveform_op)


@dataclass(frozen=True)
class ApplyGranularitySanitisation(OrderedPass, ModulePass):
    """Apply granularity sanitisation."""

    name = "apply-granularity-sanitisation"

    constraints: PulseLevelConstraints

    _runs_before: ClassVar[frozenset[type[ModulePass]] | None] = None

    def runs_before(self) -> frozenset[type[ModulePass]]:
        # Illegal, sub-granularity times must be rounded up before waveforms are sampled
        # and before the timeline is normalised, so both passes see realisable durations.
        # Imported lazily to avoid a circular import at module load.
        if ApplyGranularitySanitisation._runs_before is None:
            from qat.experimental.dialect.pulse.transforms.timeline_normalization import (
                TimelineNormalization,
            )
            from qat.experimental.dialect.pulse.transforms.waveform_evaluation import (
                EvaluateWaveformsAsSamples,
            )

            ApplyGranularitySanitisation._runs_before = frozenset(
                {EvaluateWaveformsAsSamples, TimelineNormalization}
            )
        return ApplyGranularitySanitisation._runs_before

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GranularitySanitisation(self.constraints),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
