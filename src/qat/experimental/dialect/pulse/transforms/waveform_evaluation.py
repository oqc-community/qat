# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Oxford Quantum Circuits Ltd
"""Pass that evaluates analytical waveform ops into constant sampled waveforms.

This is the xDSL-canonical counterpart of
:class:`~qat.middleend.passes.transform.EvaluateWaveforms`. It walks every
:class:`IsAnalyticalWaveformInterface` op in the module and, when the op has only
compile-time-constant operands, replaces it with one or more
:class:`~qat.experimental.dialect.pulse.ir.ops.ConstantOp`\\ s carrying a
:class:`~qat.experimental.dialect.pulse.ir.attributes.SampledWaveformAttr`.

The pass looks at every :class:`~qat.experimental.dialect.pulse.ir.ops.PulseOp` that
consumes the analytical waveform and groups them by the sample time associated with
their frame's port. A single :class:`ConstantOp` is emitted per distinct sample
time, and each consuming :class:`PulseOp` is rewired to the appropriate one. This makes
sharing a waveform across multiple ports safe.

Shapes that hardware supports natively are left untouched, and waveforms with any
non-constant operand (for example, sweep variables) are also skipped to be handled at
runtime.

Sampling of acquisition weights is intentionally deferred; weights are expected to move to
an attribute on :class:`~qat.experimental.dialect.pulse.ir.ops.AcquireOp` rather than a
waveform operand, at which point a dedicated pass can materialise them.

The pass runs xDSL's :class:`CanonicalizePass` before its own rewrites, so any operand
chain that folds to a constant (e.g. ``arith.mulf`` of two ``arith.constant``\\ s) is
collapsed first and then treated as a compile-time-constant operand.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir.attributes import SampledWaveformAttr, TimeAttr
from qat.experimental.dialect.pulse.ir.interfaces import IsAnalyticalWaveformInterface
from qat.experimental.dialect.pulse.ir.ops import (
    ConstantOp,
    PulseOp,
    extract_constant_scalar,
)
from qat.experimental.dialect.pulse.ir.types import FrameType, WaveformType
from qat.experimental.system_data.pulse.constraints import PulseLevelConstraints
from qat.experimental.utils.logging import get_logger
from qat.experimental.waveforms.evaluate import evaluate_waveform
from qat.experimental.waveforms.shapes.base import WaveformShape

log = get_logger(__name__)
_PICOSECONDS_PER_SECOND = 1e12


def _seconds_to_picoseconds(seconds: float, *, value_name: str) -> int:
    """Convert seconds to integer picoseconds with pass-level validation."""

    if seconds <= 0:
        raise PassFailedException(f"{value_name} must be positive.")

    picoseconds = int(round(seconds * _PICOSECONDS_PER_SECOND))
    if picoseconds <= 0:
        raise PassFailedException(f"{value_name} must be at least 1 ps after conversion.")

    if not np.isclose(
        picoseconds / _PICOSECONDS_PER_SECOND,
        seconds,
        rtol=0.0,
        atol=1e-15,
    ):
        raise PassFailedException(
            f"{value_name} {seconds} cannot be represented as integer picoseconds."
        )

    return picoseconds


def _resolve_sample_time(
    frame_operand: SSAValue,
    port_sample_times: dict[str, float],
) -> float | None:
    """Return the sample time associated with the port carried by ``frame_operand``.

    The port is encoded in the parameterised :class:`FrameType` on the frame SSA
    value, so this never needs to walk the SSA def-use chain.
    """

    frame_type = frame_operand.type
    if not isinstance(frame_type, FrameType):
        return None
    return port_sample_times.get(frame_type.port.data)


def _group_pulse_uses_by_sample_time(
    waveform_result: SSAValue,
    port_sample_times: dict[str, float],
) -> dict[float, list[tuple[PulseOp, int]]] | None:
    """Group every :class:`PulseOp` consuming ``waveform_result`` by its sample time.

    Each entry pairs the consuming :class:`PulseOp` with the operand index at which it
    holds the waveform, so the operand can be re-wired precisely. Returns ``None`` if
    any use is not a :class:`PulseOp`, or if any consuming pulse's frame has no
    configured sample time.
    """

    grouped: dict[float, list[tuple[PulseOp, int]]] = defaultdict(list)
    for use in waveform_result.uses:
        user = use.operation
        if not isinstance(user, PulseOp):
            return None
        sample_time = _resolve_sample_time(user.frame, port_sample_times)
        if sample_time is None:
            return None
        grouped[sample_time].append((user, use.index))
    return grouped


def _make_sampled_constant(
    op: IsAnalyticalWaveformInterface,
    shape: WaveformShape,
    sample_time: float,
) -> ConstantOp | None:
    """Create a constant op with sampled waveform, or return None if operands are non-
    constant.

    Extracts width, amplitude, and DRAG coefficients from the op's operands, validates
    they are compile-time constants, and samples the waveform. Returns None if any
    operand is not a constant.

    :param op: The analytical waveform operation.
    :param shape: The already-built waveform shape.
    :param sample_time: Desired sample time in seconds.
    :returns: A ``ConstantOp`` with sampled waveform, or ``None`` if sampling is skipped.
    """
    # TODO: COMPILER-1388, units should be in ps by default
    width: float = extract_constant_scalar(op.width)
    amplitude: float | complex = extract_constant_scalar(op.amplitude)
    if width is None or amplitude is None:
        return None

    drag_coefficients_operands = op.drag_coefficients
    drag_coefficients: list[float] = []
    for drag_coefficient_operand in drag_coefficients_operands:
        drag_coefficient: float = extract_constant_scalar(drag_coefficient_operand)
        if drag_coefficient is None:
            return None
        drag_coefficients.append(float(drag_coefficient))

    width_ps = _seconds_to_picoseconds(width, value_name="Width")
    sample_time_ps = _seconds_to_picoseconds(sample_time, value_name="Sample time")

    if width_ps % sample_time_ps != 0:
        raise PassFailedException(
            f"Width {width} is not an integer multiple of sample time {sample_time}."
        )

    samples = evaluate_waveform(
        width=width_ps,
        sample_time=sample_time_ps,
        shape=shape,
        amplitude=amplitude,
        drag_coefficients=drag_coefficients,
    )

    return ConstantOp(
        SampledWaveformAttr(samples, TimeAttr(width), TimeAttr(sample_time)),
        result_type=WaveformType(),
    )


class _RewriteAnalyticalWaveform(RewritePattern):
    """Replace analytical waveform ops with :class:`ConstantOp` samples per sample time."""

    def __init__(self, constraints: PulseLevelConstraints):
        super().__init__()
        self._constraints = constraints
        self.visited = 0
        self.sampled = 0

    def match_and_rewrite(self, op, rewriter: PatternRewriter) -> None:
        if not isinstance(op, IsAnalyticalWaveformInterface):
            return

        self.visited += 1
        op_type = type(op).__name__

        if self._constraints.supports_waveform_shape(type(op)):
            log.debug(
                f"waveform-evaluation: skipping {op_type}: "
                f"{op_type} is listed as a natively-supported shape "
                f"and will be played analytically by the hardware."
            )
            return

        shape = op.build_shape()
        if shape is None:
            log.debug(
                f"waveform-evaluation: skipping {op_type}: shape parameters are not "
                "compile-time constants; leaving waveform for the assembler."
            )
            return

        pulses_by_sample_time = _group_pulse_uses_by_sample_time(
            op.results[0], self._constraints.port_sample_times_seconds
        )
        if pulses_by_sample_time is None:
            log.debug(
                f"waveform-evaluation: skipping {op_type}: at least one consumer is "
                f"not a PulseOp with a resolvable sample time (known ports: "
                f"{sorted(self._constraints.ports)})."
            )
            return

        sampled_constants_by_sample_time = {}
        for sample_time in pulses_by_sample_time:
            constant_op = _make_sampled_constant(op, shape, sample_time)
            if constant_op is None:
                log.debug(
                    f"waveform-evaluation: skipping {op_type}: width, amplitude, or drag "
                    "coefficient is not a compile-time constant; leaving waveform for the "
                    "assembler."
                )
                return
            sampled_constants_by_sample_time[sample_time] = constant_op

        for sample_time, uses in pulses_by_sample_time.items():
            constant_op = sampled_constants_by_sample_time[sample_time]
            rewriter.insert_op(constant_op, InsertPoint.before(op))
            for pulse, operand_index in uses:
                pulse.operands[operand_index] = constant_op.result
            width = extract_constant_scalar(op.width)
            log.debug(
                f"waveform-evaluation: sampled {op_type} ({type(shape).__name__}) "
                f"with sample_time={sample_time:.3e}s, duration="
                f"{width:.3e}s, rewired {len(uses)} pulse(s)."
            )

        rewriter.erase_op(op)
        self.sampled += 1


@dataclass(frozen=True)
class EvaluateWaveformsAsSamples(ModulePass):
    """Evaluate analytical waveform ops into constant sampled waveforms.

    For every :class:`IsAnalyticalWaveformInterface` op with all-constant operands, the
    op is replaced with one or more :class:`ConstantOp`\\ s carrying a
    :class:`SampledWaveformAttr` of the pre-computed IQ samples — one per distinct
    sample time across the pulses that consume the waveform.

    :ivar constraints: The pulse-level constraints of the hardware, used to determine the
        sample time for each port and whether a waveform shape is supported natively.
    """

    name = "waveform-evaluation"

    constraints: PulseLevelConstraints

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        CanonicalizePass().apply(ctx, op)
        pattern = _RewriteAnalyticalWaveform(self.constraints)
        PatternRewriteWalker(pattern).rewrite_module(op)
        log.info(
            f"waveform-evaluation: sampled {pattern.sampled} of {pattern.visited} "
            f"analytical waveform op(s)."
        )
