# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

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

from qat.experimental.dialect.pulse.ir import ConstantOp
from qat.experimental.dialect.pulse.ir.attributes import SampledWaveformAttr, TimeAttr
from qat.experimental.dialect.pulse.ir.types import TimeType, WaveformType

_COMPARISON_TOLERANCE = 1e-10


def _round_up_to_granularity_if_required(
    value: float | int, granularity: float
) -> float | None:
    """Returns the rounded-up value if sanitisation is required.

    :param value: Value in seconds.
    :param granularity: Granularity in seconds.
    :returns: The rounded-up value, or ``None`` if already aligned.
    """

    value_in_granularity_units = value / granularity
    rounded_value_in_granularity_units = int(
        np.ceil(value_in_granularity_units - _COMPARISON_TOLERANCE)
    )

    if np.isclose(
        value_in_granularity_units,
        rounded_value_in_granularity_units,
        rtol=0.0,
        atol=_COMPARISON_TOLERANCE,
    ):
        return None

    return rounded_value_in_granularity_units * granularity


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

    def __init__(self, granularity: float) -> None:
        """Initializes the pattern with the given granularity.

        :param granularity: Granularity in seconds.
        """

        self.granularity = granularity

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ConstantOp, rewriter: PatternRewriter) -> None:
        if not isinstance(op.result.type, TimeType | WaveformType):
            return

        operand_value = op.fold()
        if not operand_value:
            return

        if op.result.type == TimeType():
            time_attr = operand_value[0]

            new_duration = _round_up_to_granularity_if_required(
                time_attr.literal_value, self.granularity
            )
            if new_duration is None:
                return

            new_time_op = ConstantOp(
                TimeAttr.from_literal_value(new_duration, time_attr.unit.data), TimeType()
            )
            rewriter.replace_op(op, new_time_op)
            return

        waveform_attr = operand_value[0]
        waveform_array = waveform_attr.literal_value

        width_attr = waveform_attr.width
        sample_time_attr = waveform_attr.sample_time
        width = width_attr.literal_value
        sample_time = sample_time_attr.literal_value

        samples = width / sample_time
        if not np.isclose(
            samples,
            np.round(samples),
            rtol=0.0,
            atol=_COMPARISON_TOLERANCE,
        ):
            return

        new_width = _round_up_to_granularity_if_required(width, self.granularity)
        if new_width is None:
            return

        if new_width < width:
            return

        padding = int(np.round((new_width - width) / sample_time, 0))
        new_waveform_array = np.pad(
            waveform_array,
            (0, padding),
            mode="constant",
            constant_values=0,
        )

        new_waveform_attr = SampledWaveformAttr(
            new_waveform_array,
            TimeAttr.from_literal_value(new_width, width_attr.unit.data),
            TimeAttr.from_literal_value(sample_time, sample_time_attr.unit.data),
        )
        new_waveform_op = ConstantOp(new_waveform_attr, WaveformType())
        rewriter.replace_op(op, new_waveform_op)


class ApplyGranularitySanitisation(ModulePass):
    """Apply granularity sanitisation."""

    name = "apply-granularity-sanitisation"

    def __init__(self, granularity: TimeAttr) -> None:
        """Initialises the pass with a positive granularity.

        :param granularity: Granularity used for sanitisation.
        """

        self.granularity = granularity.literal_value

        if self.granularity <= 0.0:
            raise ValueError("granularity must be greater than zero")

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GranularitySanitisation(self.granularity),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
