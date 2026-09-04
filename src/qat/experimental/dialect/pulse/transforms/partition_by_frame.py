# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Partition Pulse entry blocks by logical frame lineage.

The analysis in this module keeps logical frame identity separate from physical port
identity. This allows later passes to reason about lineages without losing the hardware-
facing port metadata attached to each frame.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import CreateFrameOp, FrameType
from qat.experimental.dialect.pulse.utils import pulse_entry_block


class FrameNode:
    """One operation in a frame-lineage chain.

    :ivar op: The operation at this position in the chain.
    :ivar parent: The predecessor node; ``None`` for the root.
    """

    def __init__(self, op: Operation, parent: FrameNode | None) -> None:
        self.op = op
        self.parent = parent

    def chain(self) -> Iterator[FrameNode]:
        """Yield this node and each ancestor in order, ending at the root."""
        node: FrameNode | None = self
        while node is not None:
            yield node
            node = node.parent

    @property
    def root(self) -> FrameNode:
        """The root node (the ``CreateFrameOp`` node)."""
        *_, last = self.chain()
        return last


class FrameLineage:
    """Full lineage of a single ``pulse.create_frame``, including all related operations.

    :ivar create_frame: Root ``CreateFrameOp``.
    :ivar port: Physical-port token from ``CreateFrameOp``.
    :ivar related_ops: Ordered chain of :class:`FrameNode` objects, one per entry-block
        op that interacts with this lineage. Each node's parent is the preceding node,
        so walking the chain yields the full lineage history from any point.
    """

    def __init__(
        self,
        create_frame: CreateFrameOp,
        port: str,
        related_ops: list[FrameNode] | None = None,
    ) -> None:
        self.create_frame = create_frame
        self.port = port
        self.related_ops = related_ops if related_ops is not None else []

    @property
    def frame(self) -> SSAValue:
        """Shorthand for ``create_frame.result``."""
        return self.create_frame.result

    @property
    def root_node(self) -> FrameNode | None:
        """The root :class:`FrameNode`; its ``op`` is ``create_frame``.

        ``None`` if
        ``related_ops`` is empty.
        """
        if not self.related_ops:
            return None
        return self.related_ops[0].root

    @property
    def ops(self) -> tuple[Operation, ...]:
        """Ops from :attr:`related_ops` in encounter order."""
        return tuple(n.op for n in self.related_ops)

    def add_node(self, op: Operation) -> None:
        """Append a FrameNode for op unless the tail already records it."""
        if not self.related_ops:
            self.related_ops.append(FrameNode(op=op, parent=None))
            return
        if self.related_ops[-1].op is not op:
            self.related_ops.append(FrameNode(op=op, parent=self.related_ops[-1]))


class FrameLineageAnalysis:
    """Module-level frame-lineage analysis; one :class:`FrameLineage` per ``CreateFrameOp``.

    Value ownership is tracked centrally in this object, so resolving a value to its
    owning lineage and attaching a new usage are both O(1) operations rather than scans
    over ``lineages``.

    :ivar lineages: All discovered lineages in encounter order.
    """

    def __init__(self, lineages: list[FrameLineage] | None = None) -> None:
        self.lineages = lineages if lineages is not None else []
        self._owner: dict[SSAValue, FrameLineage] = {}

    @property
    def port_counts(self) -> dict[str, int]:
        """Return the number of lineages sharing each port token."""
        counts: dict[str, int] = {}
        for lineage in self.lineages:
            counts[lineage.port] = counts.get(lineage.port, 0) + 1
        return counts

    def lineage_for_frame(self, frame: SSAValue) -> FrameLineage | None:
        """Return the :class:`FrameLineage` whose root value is ``frame``, or ``None``."""
        lineage = self._owner.get(frame)
        return lineage if lineage is not None and lineage.frame is frame else None

    def lineage_for_result(self, value: SSAValue) -> FrameLineage | None:
        """Return the :class:`FrameLineage` that owns ``value``, or ``None``."""
        return self._owner.get(value)

    def begin_lineage(self, create_frame: CreateFrameOp, node: FrameNode) -> FrameLineage:
        """Start a new lineage rooted at ``create_frame``, seeded with ``node``."""
        lineage = FrameLineage(
            create_frame=create_frame, port=create_frame.port.data, related_ops=[node]
        )
        self.lineages.append(lineage)
        self._owner[create_frame.result] = lineage
        return lineage

    def attach(
        self, entry_op: Operation, lineage: FrameLineage, result: SSAValue | None = None
    ) -> None:
        """Record ``entry_op``'s use of ``lineage``, optionally claiming ``result``."""
        lineage.add_node(entry_op)
        if result is not None:
            self._owner[result] = lineage


class _LineageState:
    def __init__(self) -> None:
        self.analysis: FrameLineageAnalysis | None = None


def build_frame_lineage_analysis(module: ModuleOp) -> FrameLineageAnalysis:
    """Analyze pulse operations by frame lineage.

    The partition key is the logical frame identity rooted at
    ``pulse.create_frame``. Physical port identity is retained as metadata so
    later lowering stages can recover the hardware-facing view without
    collapsing distinct logical frames.

    Region-bearing entry operations are traversed recursively: a ``CreateFrameOp``
    inside a nested region starts a new lineage with the enclosing entry-block op
    as its representative; references to outer frame values append a new node to
    those lineages.

    :param module: Module containing pulse operations.
    :returns: Frame-lineage analysis.
    :raises PassFailedException: If frame lineage cannot be resolved.
    """
    analysis = FrameLineageAnalysis()

    def _visit_op(op: Operation, representative: Operation | None) -> None:
        entry_op = representative or op

        if isinstance(op, CreateFrameOp):
            root = FrameNode(op=op, parent=None)
            node = root if entry_op is op else FrameNode(op=entry_op, parent=root)
            analysis.begin_lineage(op, node)
            return

        if op.regions:
            for region in op.regions:
                for block in region.blocks:
                    for inner_op in block.ops:
                        _visit_op(inner_op, representative=entry_op)
            return

        frame_operands = [o for o in op.operands if isinstance(o.type, FrameType)]
        if not frame_operands:
            return

        # Resolve every operand before mutating anything; bail out (or raise) if any
        # operand's lineage is not yet known.
        resolved: list[FrameLineage] = []
        for operand in frame_operands:
            lineage = analysis.lineage_for_result(operand)
            if lineage is None:
                if representative is None:
                    raise PassFailedException(
                        f"Unbound frame operand encountered in operation {op.name}."
                    )
                return
            resolved.append(lineage)

        frame_results = [r for r in op.results if isinstance(r.type, FrameType)]
        if frame_results and len(frame_results) != len(frame_operands):
            raise PassFailedException(
                f"Cannot map frame results for multi-frame operation {op.name}."
            )

        # All validation is complete; attach usage and claim results one value at a time.
        if frame_results:
            for lineage, result in zip(resolved, frame_results, strict=True):
                analysis.attach(entry_op, lineage, result)
        else:
            for lineage in resolved:
                analysis.attach(entry_op, lineage)

    entry_block = pulse_entry_block(module)
    for op in entry_block.ops:
        _visit_op(op, representative=None)

    return analysis


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
