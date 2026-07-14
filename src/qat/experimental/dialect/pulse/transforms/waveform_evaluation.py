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

Shapes that hardware supports natively (defaulting to
:class:`~qat.ir.waveforms.SquareWaveform`) are left untouched, and waveforms with any
non-constant operand (for example, sweep variables) are also skipped to be handled at
runtime.

Sampling of acquisition weights is intentionally deferred; weights are expected to move to
an attribute on :class:`~qat.experimental.dialect.pulse.ir.ops.AcquireOp` rather than a
waveform operand, at which point a dedicated pass can materialise them.

The pass runs xDSL's :class:`CanonicalizePass` before its own rewrites, so any operand
chain that folds to a constant (e.g. ``arith.mulf`` of two ``arith.constant``\\ s) is
collapsed first and then treated as a compile-time-constant operand.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalize import CanonicalizePass

from qat.experimental.dialect.pulse.ir.attributes import SampledWaveformAttr, TimeAttr
from qat.experimental.dialect.pulse.ir.interfaces import IsAnalyticalWaveformInterface
from qat.experimental.dialect.pulse.ir.ops import ConstantOp, PulseOp
from qat.experimental.dialect.pulse.ir.types import FrameType, WaveformType
from qat.ir.waveforms import SquareWaveform, sample_waveform

log = logging.getLogger(__name__)


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


def _make_sampled_constant(waveform, sample_time: float) -> ConstantOp:
    sampled = sample_waveform(waveform, sample_time)
    return ConstantOp(
        SampledWaveformAttr(
            sampled.samples,
            TimeAttr(waveform.duration),
            TimeAttr(sampled.sample_time),
        ),
        result_type=WaveformType(),
    )


class _RewriteAnalyticalWaveform(RewritePattern):
    """Replace analytical waveform ops with :class:`ConstantOp` samples per sample time."""

    def __init__(
        self,
        port_sample_times: dict[str, float],
        ignored_shapes: tuple[type, ...],
    ):
        super().__init__()
        self._port_sample_times = port_sample_times
        self._ignored_shapes = ignored_shapes
        self.visited = 0
        self.sampled = 0

    def match_and_rewrite(self, op, rewriter: PatternRewriter) -> None:
        if not isinstance(op, IsAnalyticalWaveformInterface):
            return

        self.visited += 1
        op_type = type(op).__name__

        waveform = op.build_waveform()
        if waveform is None:
            log.debug(
                f"waveform-evaluation: skipping {op_type}: at least one operand is "
                f"not a compile-time constant; leaving waveform for the assembler."
            )
            return

        if isinstance(waveform, self._ignored_shapes):
            log.debug(
                f"waveform-evaluation: skipping {op_type}: "
                f"{type(waveform).__name__} is listed as a natively-supported shape "
                f"and will be played analytically by the hardware."
            )
            return

        pulses_by_sample_time = _group_pulse_uses_by_sample_time(
            op.results[0], self._port_sample_times
        )
        if pulses_by_sample_time is None:
            log.debug(
                f"waveform-evaluation: skipping {op_type}: at least one consumer is "
                f"not a PulseOp with a resolvable sample time (known ports: "
                f"{sorted(self._port_sample_times)})."
            )
            return

        for sample_time, uses in pulses_by_sample_time.items():
            constant_op = _make_sampled_constant(waveform, sample_time)
            rewriter.insert_op(constant_op, InsertPoint.before(op))
            for pulse, operand_index in uses:
                pulse.operands[operand_index] = constant_op.result
            log.debug(
                f"waveform-evaluation: sampled {op_type} ({type(waveform).__name__}) "
                f"with sample_time={sample_time:.3e}s, duration="
                f"{waveform.duration:.3e}s, rewired {len(uses)} pulse(s)."
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

    :ivar port_sample_times: Mapping from :class:`FrameType` port token
        to the sample time in seconds used to discretise waveforms on that port.
    :ivar ignored_shapes: Waveform shapes whose analytical form is natively supported by
        the hardware and should not be sampled. Defaults to ``(SquareWaveform,)``.
    """

    name = "waveform-evaluation"

    port_sample_times: dict[str, float]
    ignored_shapes: tuple[type, ...] = field(default_factory=lambda: (SquareWaveform,))

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        CanonicalizePass().apply(ctx, op)
        pattern = _RewriteAnalyticalWaveform(self.port_sample_times, self.ignored_shapes)
        PatternRewriteWalker(pattern).rewrite_module(op)
        log.info(
            f"waveform-evaluation: sampled {pattern.sampled} of {pattern.visited} "
            f"analytical waveform op(s)."
        )
