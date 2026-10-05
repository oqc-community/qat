# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Remove dead operations that act on otherwise inactive frames.

This pass uses frame-lineage analysis to identify logical frames that are never used by
hardware-visible pulse operations. For each dead lineage, it removes only the lineage
members that directly transform or advance the frame value.

The pass intentionally stays narrow: if a frame flows through non-pulse control flow,
the lineage is conservatively preserved instead of rewriting that structure.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter

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
from qat.experimental.dialect.pulse.transforms.partition_by_frame import (
    FrameLineage,
    build_frame_lineage_analysis,
)
from qat.experimental.utils.logging import get_logger

_logger = get_logger(__name__)

_LIVE_FRAME_OP_TYPES = (
    AcquireOp,
    PulseOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SynchronizeOp,
)

_NON_LIVE_FRAME_OP_TYPES = (CreateFrameOp, PhaseSetOp, PhaseShiftOp, WaitOp)

# Type tuples are mutually exclusive. All recognized pulse frame operations must appear in
# one or the other. Operations unknown to this pass (outside the pulse dialect) are
# conservatively treated as live (fail-safe behavior), triggering a warning and preserving
# the frame.


def _is_dead_lineage(lineage: FrameLineage) -> bool:
    """Determine if a frame lineage is dead (contains no live operations).

    A lineage is dead if all of its operations are non-live operations. If any live
    operation is found, the lineage is alive and no warning is emitted. Unknown operations
    are collected across the lineage: if no live operation is found, a single warning logs
    all unsupported ops and the frame is treated as alive.

    :param lineage: Frame lineage to check.
    :returns: True if the lineage is dead, False otherwise.
    """

    unknown_ops: list[object] = []
    for op in lineage.ops:
        if isinstance(op, _LIVE_FRAME_OP_TYPES):
            return False
        if isinstance(op, _NON_LIVE_FRAME_OP_TYPES):
            continue
        unknown_ops.append(op)

    if unknown_ops:
        unknown_names = ", ".join(op.name for op in unknown_ops)
        _logger.warning(
            "Frame lineage contains unsupported operation types [%s]; treating frame as "
            "non-dead and skipping elimination.",
            unknown_names,
        )
        return False
    return True


def _erase_lineage(rewriter: Rewriter, lineage: FrameLineage) -> None:
    """Erase a dead frame lineage and all its operations.

    Erases the CreateFrameOp and all operations in the lineage that directly transform or
    advance the frame value. Operations are erased in reverse order to ensure SSA
    dependencies are respected; later operations are erased before earlier ones to avoid
    dangling uses.

    :param rewriter: xDSL rewriter for performing erasures.
    :param lineage: Lineage to erase.
    """
    for op in reversed(lineage.ops):
        rewriter.erase_op(op)


@dataclass(frozen=True)
class DeadFrameEliminationPass(ModulePass):
    """Erase frame lineages that never participate in a live pulse-level operation.

    **Operation Classification:**

    - **Live operations** (hardware-visible): AcquireOp, PulseOp,
      StartContinuousWaveformOp, StopContinuousWaveformOp, SynchronizeOp. A frame is
      preserved if its lineage contains any live operation.
    - **Non-live operations** (frame metadata/timing): CreateFrameOp, PhaseSetOp,
      PhaseShiftOp, WaitOp. These do not prevent frame elimination.
    - **Unknown operations** (outside pulse dialect): Operations unknown to this pass
      are conservatively treated as live (fail-safe behavior) to avoid incorrectly
      eliminating frames passed to external control flow or dialect operations.


    .. note::

        SynchronizeOp is classified as live because timing delays on a frame can affect
        synchronized frames even if the frame itself carries no pulse data. It is
        recommended to run this pass after synchronization elimination to avoid these
        interdependencies.

    .. note::

        Frames that thread through operations that do not belong to the pulse dialect are
        treated conservatively, and those frames are not removed. This includes frames
        created within a region-bearing operation and yielded outside the region, as the
        terminating operations will conservatively mark a frame alive.
    """

    name = "pulse.dead-frame-elimination"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Eliminate frame lineages that contain no live pulse operations.

        For each frame creation, walks its lineage to determine if it contains any hardware-
        visible pulse operations. If the lineage contains only non-live operations (phase
        sets, shifts, waits), the entire frame and its operations are erased. Unknown
        operations trigger a warning and are treated as keeping the frame alive.

        :param ctx: xDSL context.
        :param op: Module to optimize.
        """
        analysis = build_frame_lineage_analysis(op)

        dead_lineages: list[FrameLineage] = []
        for lineage in analysis.lineages:
            if _is_dead_lineage(lineage):
                dead_lineages.append(lineage)

        rewriter = Rewriter()
        for lineage in dead_lineages:
            _erase_lineage(rewriter, lineage)
