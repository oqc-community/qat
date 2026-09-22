# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Partition Pulse entry blocks by logical frame lineage.

The analysis in this module keeps logical frame identity separate from physical port
identity. This allows later passes to reason about lineages without losing the hardware-
facing port metadata attached to each frame.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import CreateFrameOp, FrameType
from qat.experimental.dialect.pulse.utils import pulse_entry_block


class FrameNode:
    """One operation in a frame lineage, positioned in the IR nesting hierarchy.

    :ivar op: The operation this node records.
    :ivar parent: Node of the region-bearing operation enclosing :attr:`op`; ``None``
        when :attr:`op` sits directly in the entry block.
    """

    def __init__(self, op: Operation, parent: FrameNode | None) -> None:
        self.op = op
        self.parent = parent

    def chain(self) -> Iterator[FrameNode]:
        """Yield this node and each enclosing operation, outwards to the entry block."""
        node: FrameNode | None = self
        while node is not None:
            yield node
            node = node.parent

    @property
    def root(self) -> FrameNode:
        """The outermost node, whose ``op`` is an entry-block operation."""
        *_, last = self.chain()
        return last


class FrameLineage:
    """Full lineage of a single ``pulse.create_frame``, including all related operations.

    :ivar create_frame: Root ``CreateFrameOp``.
    :ivar port: Physical-port token from ``CreateFrameOp``.
    :ivar related_ops: Ordered list of :class:`FrameNode` objects, one per operation
        that interacts with this lineage, recorded at its own nesting depth. Each
        node's parent is the region-bearing op enclosing it, so walking the chain
        yields the path out to the entry block.
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
    def entry_ops(self) -> tuple[Operation, ...]:
        """Entry-block operations this lineage passes through, in encounter order.

        Each node's :attr:`FrameNode.root` is its outermost enclosing operation, which
        is the entry-block op that has to be carried along to bring this lineage's work
        with it. Deduplicated, since several lineage ops commonly share one enclosing op.
        """
        entry_ops: dict[Operation, None] = {}
        for node in self.related_ops:
            entry_ops.setdefault(node.root.op, None)
        return tuple(entry_ops)

    @property
    def ops(self) -> tuple[Operation, ...]:
        """Ops from :attr:`related_ops` in encounter order."""
        return tuple(n.op for n in self.related_ops)

    def add_node(self, op: Operation, enclosing: FrameNode | None = None) -> None:
        """Append a FrameNode for ``op`` unless the tail already records it.

        ``enclosing`` is the node of the region-bearing op containing ``op``, or
        ``None`` when ``op`` sits in the entry block. A single walk attaches ``op``
        once per frame operand resolving to this lineage, so repeats are always
        adjacent and checking the tail is enough to deduplicate them.
        """
        if self.related_ops and self.related_ops[-1].op is op:
            return
        self.related_ops.append(FrameNode(op=op, parent=enclosing))


class FrameLineageAnalysis:
    """Module-level frame-lineage analysis; one :class:`FrameLineage` per ``CreateFrameOp``.

    Value ownership and operation membership are tracked centrally in this object.
    Operations may intentionally belong to multiple lineages, for example when a
    synchronization coordinates several frames. Consumers that require independent
    partitions can inspect :attr:`shared_ops` without reconstructing membership from the
    lineages.

    :ivar lineages: All discovered lineages in encounter order.
    """

    def __init__(self) -> None:
        self.lineages: list[FrameLineage] = []
        self._owner: dict[SSAValue, FrameLineage] = {}
        self._lineages_by_op: dict[Operation, list[FrameLineage]] = {}

    @classmethod
    def from_lineages(cls, lineages: Iterable[FrameLineage]) -> FrameLineageAnalysis:
        """Construct an analysis from previously assembled lineages.

        Normal analysis construction uses :meth:`begin_lineage` and :meth:`attach`.
        This factory is useful when a caller already owns complete lineage records.

        :param lineages: Complete lineages to register in encounter order.
        :returns: An analysis containing ``lineages`` and their derived indexes.
        """

        analysis = cls()
        for lineage in lineages:
            analysis._register_lineage(lineage)
        return analysis

    @property
    def port_counts(self) -> dict[str, int]:
        """Return the number of lineages sharing each port token."""
        counts: dict[str, int] = {}
        for lineage in self.lineages:
            counts[lineage.port] = counts.get(lineage.port, 0) + 1
        return counts

    @property
    def shared_ops(self) -> tuple[Operation, ...]:
        """Return operations attached to more than one lineage, in encounter order."""
        return tuple(
            op for op, lineages in self._lineages_by_op.items() if len(lineages) > 1
        )

    def lineages_for_op(self, op: Operation) -> tuple[FrameLineage, ...]:
        """Return every lineage ``op`` is attached to, in encounter order.

        Empty for operations the analysis does not track, such as constants and the region-
        bearing operations that merely enclose a lineage.
        """
        return tuple(self._lineages_by_op.get(op, ()))

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
        self._register_lineage(lineage)
        return lineage

    def attach(
        self,
        op: Operation,
        lineage: FrameLineage,
        result: SSAValue | None = None,
        enclosing: FrameNode | None = None,
    ) -> None:
        """Record ``op``'s use of ``lineage``, optionally claiming ``result``."""
        lineage.add_node(op, enclosing)
        self._record_op_membership(op, lineage)
        if result is not None:
            self._owner[result] = lineage

    def _record_op_membership(self, op: Operation, lineage: FrameLineage) -> None:
        lineages = self._lineages_by_op.setdefault(op, [])
        if lineage not in lineages:
            lineages.append(lineage)

    def _register_lineage(self, lineage: FrameLineage) -> None:
        self.lineages.append(lineage)
        self._owner[lineage.frame] = lineage
        for op in lineage.ops:
            self._record_op_membership(op, lineage)


class _LineageState:
    def __init__(self) -> None:
        self.analysis: FrameLineageAnalysis | None = None


def build_frame_lineage_analysis(module: ModuleOp) -> FrameLineageAnalysis:
    """Analyze pulse operations by frame lineage.

    The partition key is the logical frame identity rooted at
    ``pulse.create_frame``. Physical port identity is retained as metadata so
    later lowering stages can recover the hardware-facing view without
    collapsing distinct logical frames.

    Region-bearing operations are traversed recursively. Membership records the pulse
    operations themselves at whatever depth they occur, so ownership stays decidable
    per operation. Each node's parent is the enclosing region-bearing op, so
    :attr:`FrameLineage.entry_ops` recovers the entry-block ops a lineage passes
    through.

    :param module: Module containing pulse operations.
    :returns: Frame-lineage analysis.
    :raises PassFailedException: If frame lineage cannot be resolved.
    """
    analysis = FrameLineageAnalysis()

    def _visit_op(op: Operation, enclosing: FrameNode | None) -> None:
        if isinstance(op, CreateFrameOp):
            analysis.begin_lineage(op, FrameNode(op=op, parent=enclosing))
            return

        if op.regions:
            # One node per region-bearing op, shared by everything nested inside it, so
            # that each contained lineage op walks the same path back to the entry block.
            node = FrameNode(op=op, parent=enclosing)
            for region in op.regions:
                for block in region.blocks:
                    for inner_op in block.ops:
                        _visit_op(inner_op, enclosing=node)
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
                if enclosing is None:
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
                analysis.attach(op, lineage, result, enclosing=enclosing)
        else:
            for lineage in resolved:
                analysis.attach(op, lineage, enclosing=enclosing)

    entry_block = pulse_entry_block(module)
    for op in entry_block.ops:
        _visit_op(op, enclosing=None)

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
