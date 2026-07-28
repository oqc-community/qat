# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Partition Pulse entry blocks by logical frame lineage.

The analysis in this module keeps logical frame identity separate from physical port
identity. This allows later passes to reason about lineages without losing the hardware-
facing port metadata attached to each frame.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import CreateFrameOp, FrameType, SynchronizeOp
from qat.experimental.dialect.pulse.utils import pulse_entry_block


@dataclass
class FrameLineage:
    """Operations associated with one logical frame lineage.

    A lineage is rooted at one `pulse.create_frame` result and collects the
    operations that remain associated with that frame through the entry block.

    :ivar frame: ``pulse.create_frame`` result representing this lineage.
    :ivar port: Physical-port token carried by ``FrameType``.
    :ivar operations: Operations associated with this lineage, in entry-block
        encounter order.
    """

    frame: SSAValue
    port: str
    operations: list[Operation] = field(default_factory=list)


@dataclass
class FrameLineageAnalysis:
    """Frame-lineage analysis for Pulse IR.

    The analysis separates logical frame identity from physical port identity. The former
    drives partitioning. The latter is preserved as metadata for subsequent lowering stages.

    :ivar frame_to_operations: Ordered operations per lineage frame. This records the
        lineage membership in the entry block, not a ready-made sequence body. A lowering
        pass may still need to clone supporting definitions before materialising a
        ``q1_sequence.sequence`` envelope.
    :ivar frame_to_port: Physical port metadata per lineage frame.
    :ivar port_to_frames: Lineage frames grouped by physical port.
    :ivar value_to_frame: Lineage frame associated with each frame SSA value.
    """

    frame_to_operations: Mapping[SSAValue, tuple[Operation, ...]]
    frame_to_port: Mapping[SSAValue, str]
    port_to_frames: Mapping[str, tuple[SSAValue, ...]]
    value_to_frame: Mapping[SSAValue, SSAValue]


@dataclass
class _LineageState:
    analysis: FrameLineageAnalysis | None = None


def build_frame_lineage_analysis(module: ModuleOp) -> FrameLineageAnalysis:
    """Analyze pulse operations by frame lineage.

    The partition key is the logical frame identity rooted at
    ``pulse.create_frame``. Physical port identity is retained as metadata so
    later lowering stages can recover the hardware-facing view without
    collapsing distinct logical frames. Region-bearing entry operations are
    rejected because the analysis currently runs on a flat instruction stream.

    :param module: Module containing pulse operations.
    :returns: Frame-lineage analysis maps.
    :raises PassFailedException: If frame lineage cannot be resolved.
    """

    lineages: dict[SSAValue, FrameLineage] = {}
    frame_of_value: dict[SSAValue, SSAValue] = {}

    for op in pulse_entry_block(module).ops:
        if op.regions:
            raise PassFailedException(
                "Frame partitioning currently requires region-free entry blocks and "
                "does not yet inspect nested regions."
            )

        frame_operands = [
            operand for operand in op.operands if isinstance(operand.type, FrameType)
        ]
        frame_results = [
            result for result in op.results if isinstance(result.type, FrameType)
        ]

        if isinstance(op, CreateFrameOp):
            frame_of_value[op.result] = op.result
            lineages[op.result] = FrameLineage(
                frame=op.result,
                port=op.port.data,
                operations=[op],
            )
            continue

        if not frame_operands:
            continue

        frames = [frame_of_value.get(operand) for operand in frame_operands]
        if any(frame is None for frame in frames):
            raise PassFailedException(
                f"Unbound frame operand encountered in operation {op.name}."
            )
        resolved_frames = [frame for frame in frames if frame is not None]

        ordered_unique_frames: list[SSAValue] = []
        for frame in resolved_frames:
            if frame not in ordered_unique_frames:
                ordered_unique_frames.append(frame)
        for frame in ordered_unique_frames:
            lineages[frame].operations.append(op)

        if not frame_results:
            continue

        if isinstance(op, SynchronizeOp):
            if len(frame_results) != len(frame_operands):
                raise PassFailedException(
                    "pulse.sync must have one frame result per frame operand."
                )
            for result, frame in zip(frame_results, resolved_frames, strict=True):
                frame_of_value[result] = frame
            continue

        if len(ordered_unique_frames) == 1:
            for result in frame_results:
                frame_of_value[result] = ordered_unique_frames[0]
            continue

        if len(frame_results) == len(resolved_frames):
            for result, frame in zip(frame_results, resolved_frames, strict=True):
                frame_of_value[result] = frame
            continue

        raise PassFailedException(
            f"Cannot map frame results for multi-frame operation {op.name}."
        )

    frame_to_port = {frame: lineage.port for frame, lineage in lineages.items()}
    port_to_frames: dict[str, list[SSAValue]] = defaultdict(list)
    for frame, port in frame_to_port.items():
        port_to_frames[port].append(frame)

    return FrameLineageAnalysis(
        frame_to_operations={
            frame: tuple(lineage.operations) for frame, lineage in lineages.items()
        },
        frame_to_port=frame_to_port,
        port_to_frames={port: tuple(frames) for port, frames in port_to_frames.items()},
        value_to_frame=dict(frame_of_value),
    )


@dataclass(frozen=True)
class FrameLineagePass(ModulePass):
    """Compute frame-lineage analysis for pulse-level lowering."""

    name = "pulse.compute-frame-lineage"
    state: _LineageState = field(default_factory=_LineageState, init=False)

    @property
    def analysis(self) -> FrameLineageAnalysis | None:
        """Return the most recent frame-lineage analysis computed by this pass."""

        return self.state.analysis

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        self.state.analysis = build_frame_lineage_analysis(op)
