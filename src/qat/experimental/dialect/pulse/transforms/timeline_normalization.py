# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements a pass that operates on pulse level dialect IR, and resolves Synchronize
operations into Wait operations where it can.

A synchronization can be resolved into waits using the following algorithm:

1. We add up the accumulated times for each frame in the synchronization up to that point.
2. Take the maximum over those accumulated times.
3. Subtract the elapsed time from the maximum to determine the wait time.
4. Wait (do nothing) on that frame for the wait time, bringing all frames in line.

The pass works by walking through the IR and tracking the temporal relationships between
frames, and when it encounters a Synchronize operation, it resolves them using the above
algorithm if the relationships are known.

Operations might be encountered that affect frames and the scheduling semantics of frames
but in a way that is unknown. In that case, we lose all temporal relationships for the
affected frames, which results in subsequent synchronization operation(s) not being
resolvable. We do this conservatively; any unknown operation with a body (region(s))
invalidate temporal relationships for all frames, as we do not have context to understand
the meaning of the region (and if it applies to all frames). For operations without a body
that consume frames, we just invalidate the relationships of those frames.

In practice, the way we track temporal relationships is through "domains". Domains contain
frames as members. They are relative to a temporal "anchor point" that frames have an
accumulated time relative to. Frames are added to and removed from domains as needed.
Frame membership has the following rules:

* Frames in the same domain have a known temporal relationship to each other, and can be
  synchronized explicitly with padding and known time expressions.
* Frames in different domains have no known temporal relationship to each other, and cannot
  have synchronizations resolved between them. Instead, the frames in the synchronize are
  moved into a shared, new domain, and the synchronize is left unresolved.
* Operations with unknown semantics move respective frames into new domains, one for each
  frame.

The temporal relationships are tracked via expressions over time SSA values. During
analysis, we build up symbolic expressions for the accumulated time for each frame, but
do not create operations immediately. This is because, in some cases, those expressions
need not be materialised (e.g. if there are no subsequent synchronizations, or if
synchronizations destroy the temporal relationships). When we do need to materialise those
expressions, we insert them into the IR, at the position they're calculated from (e.g.
addition of times due to a pulse is inserted next to the pulse op).

When any expressions are need to be materialised, if not done already, we add the operations
that represent those expressions to the IR at the position where the analysis calculated
the expression. Any synchronizations that can be resolved are replaced with Wait operations,
with operands that point to those expressions.

This pass isn't yet robust to multiple blocks, as we need a more comprehensive analysis
method that does fixed point analysis between strongly connected components to understand
the temporal relationships between frames across blocks. For now, we just raise an error if
we encounter that situation, but eventually, we need to handle that too. This works fine
in a world where control flow is represented entirely in high-level representations, e.g.,
structured control flow, but not in an unstructured regime. COMPILER-1224

Additionally, as support for control flow (COMPILER-1225) and function calls (COMPILER-1226)
on pulse-level programs is added, we'll want to add specific support for those constructs in
this pass, as they have well defined semantics that we can leverage.
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import singledispatchmethod
from typing import TypeAlias

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.traits import IsolatedFromAbove
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AddOp,
    ConstantOp,
    CreateFrameOp,
    FrameType,
    IsAnalyticalWaveformInterface,
    MaxTimeOp,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    SubOp,
    SynchronizeOp,
    TimeAttr,
    TimeType,
    WaitOp,
)


class _TimeOperationType(Enum):
    """The type of a time operation, which is used to determine how to evaluate the time
    expression.

    :cvar CONSTANT: A constant time value, used when the time comes from an attribute and
        has no SSA representation.
    :cvar ADD: The addition of two time expressions or values, used when time is accumulated
        from evolution of frames.
    :cvar SUB: The subtraction of two time expressions or values, used when calculating the
        time difference between frames and the maximum time during synchronization.
    :cvar MAX: The maximum of multiple time expressions or values, used when calculating the
        maximum accumulated time during synchronization.
    """

    CONSTANT = 0
    ADD = 1
    SUB = 2
    MAX = 3


@dataclass(frozen=True, eq=True)
class _TimeExpression:
    """Models an expression for a time, which is manifested as an operation that produces a
    time SSA value.

    This is only to be used to store analysis on expressions we need to build for this pass.
    It does not model expressions currently in the IR. It does store an insertion point,
    which is used if the expression is materialized into the IR.

    :ivar operation_type: The type of the time operation, which is used to determine how to
        evaluate the time expression.
    :ivar operands: The operands of the time expression, which can be SSA values or other
        time expressions.
    :ivar insertion_point: The insertion point where the time expression should be
        materialized in the IR if needed.
    """

    operation_type: _TimeOperationType
    operands: tuple["SSAValue | TimeAttr | _TimeExpression", ...]
    insertion_point: InsertPoint

    @classmethod
    def constant(cls, value: TimeAttr, insertion_point: InsertPoint) -> "_TimeExpression":
        """Creates a time expression that represents a constant time value."""
        return cls(_TimeOperationType.CONSTANT, (value,), insertion_point)

    @classmethod
    def add(
        cls, operand1: "_TIME_TYPE", operand2: "_TIME_TYPE", insertion_point: InsertPoint
    ) -> "_TimeExpression":
        """Creates a time expression that represents the addition of two time expressions or
        values."""

        if operand1 is None:
            return operand2
        if operand2 is None:
            return operand1
        return cls(_TimeOperationType.ADD, (operand1, operand2), insertion_point)

    @classmethod
    def subtract(
        cls, operand1: "_TIME_TYPE", operand2: "_TIME_TYPE", insertion_point: InsertPoint
    ) -> "_TimeExpression":
        """Creates a time expression that represents the subtraction of two time expressions
        or values."""

        if operand2 is None:
            return operand1
        if operand1 == operand2:
            return None
        return cls(_TimeOperationType.SUB, (operand1, operand2), insertion_point)

    @classmethod
    def max(
        cls, *operands: "_TIME_TYPE", insertion_point: InsertPoint
    ) -> "_TimeExpression":
        """Creates a time expression that represents the maximum of multiple time
        expressions or values."""

        non_null_operands = tuple(
            dict.fromkeys(operand for operand in operands if operand is not None)
        )
        if len(non_null_operands) == 0:
            return None
        if len(non_null_operands) == 1:
            return non_null_operands[0]
        return cls(_TimeOperationType.MAX, non_null_operands, insertion_point)


_TIME_TYPE: TypeAlias = SSAValue | _TimeExpression | None


class _TimeExpressionMaterializer:
    """Materializes time expressions into operations.

    When a resolvable synchronize operation is found, we replace that synchronize operation
    with Wait operations. The operands of the Wait operations is calculated from nested
    expressions that add up the elapsed time for each frames, finds the maximum elapsed time
    over all frames and subtracts the elapsed time for each frame from that maximum to find
    operand for the Wait operation.

    This class is responsible for materializing those expressions from analysis into
    operations and inserting them into the IR if those expressions are needed. The
    expressions keep track of their insertion point, so the materializer can insert them as
    soon as that information is available. For example, when encountering a Pulse, Acquire
    or Wait operation, we insert the expression to add to the elapsed time next to those
    operations. When resolving a Synchronize, we insert the max and subtract operations next
    to the Synchronize operation.

    This approach helps minimize the number of operations we need to insert, as we can keep
    track of expressions that we have already materialized.
    """

    def __init__(self):
        self._expression_value_cache: dict[_TimeExpression, SSAValue] = {}

    def materialize(self, value: _TIME_TYPE) -> SSAValue | None:
        """Materializes a time value/expression to an SSA result, reusing duplicates."""
        if value is None:
            return None

        if isinstance(value, SSAValue):
            return value

        if isinstance(value, _TimeExpression):
            if value in self._expression_value_cache:
                return self._expression_value_cache[value]

            operands = value.operands

            match value.operation_type:
                case _TimeOperationType.CONSTANT:
                    op = ConstantOp(value=operands[0], result_type=TimeType())
                case _TimeOperationType.ADD:
                    op = AddOp(
                        self.materialize(operands[0]),
                        self.materialize(operands[1]),
                        result_type=TimeType(),
                    )
                case _TimeOperationType.SUB:
                    op = SubOp(
                        self.materialize(operands[0]),
                        self.materialize(operands[1]),
                        result_type=TimeType(),
                    )
                case _TimeOperationType.MAX:
                    op = MaxTimeOp(*(self.materialize(operand) for operand in operands))
                case _:
                    raise PassFailedException(
                        f"Invalid time operation type: {value.operation_type}"
                    )

            Rewriter.insert_op(op, value.insertion_point)
            self._expression_value_cache[value] = op.results[0]
            return op.results[0]

        raise PassFailedException(f"Invalid operand type: {type(value)}")


@dataclass
class _SynchronizeCandidate:
    """Models a candidate for synchronization resolution, which is a set of frames that are
    synchronized to each other, and the time expression that relates them.

    These are created during an analysis phase, which walks the IR and conservatively finds
    :class:`SynchronizeOp` operations that can be resolved to concrete time expressions.

    :ivar sync_op: The synchronization operator to be resolved.
    :ivar expressions: A list of symbolic expressions that are used to create arithmetic
        operator chains.
    """

    sync_op: SynchronizeOp
    expressions: list[_TIME_TYPE]


@dataclass(eq=False)
class _TimeDomain:
    """Models a time domain that multiple frames can belong to, in which timing
    relationships are known and differences can be resolved to an SSA value.

    This is used as bookkeeping to track the temporal relationships between frames, and is
    used to determine when synchronizes can be resolved.

    :ivar absolute_anchor: Models if the time of the domain is known absolutely relative to
        environment. Once set to ``False``, this cannot be restored, which happens when
        a synchronize is encountered that cannot be resolved.
    :ivar members: The currently tracked frame facts that belong to this domain.
    """

    absolute_anchor: bool = False
    members: set["_FrameFact"] = field(default_factory=set)

    def remove_frame(self, frame_fact: "_FrameFact") -> None:
        """Removes a frame fact from this domain."""
        self.members.discard(frame_fact)

    def add_frame(self, frame_fact: "_FrameFact") -> None:
        """Adds a frame fact to this domain."""
        self.members.add(frame_fact)


@dataclass(eq=False)
class _FrameFact:
    """Contains the factual information known about a frame at a given point in the program,
    which is used to resolve Synchronize operations.

    :ivar current: We track the current SSA value of the frame, which evolves with every
        frame-consuming operations.
    :ivar domain: The time domain to which this frame belongs.
    :ivar sync_base: The last known synchronization time for this frame, which can be used
        as a reference points in subsequent synchronizations to simplify time expressions.
    :ivar offset: The accumulated time offset for this frame relative to the synchronization
        base, which is built up over multiple operations.
    """

    current: SSAValue
    domain: _TimeDomain
    sync_base: _TIME_TYPE | None = None
    offset: _TIME_TYPE | None = None

    def replace(self, new_frame: SSAValue) -> None:
        """Replaces the current SSA value of this frame."""
        self.current = new_frame

    def add_offset(self, offset: _TIME_TYPE, insertion_point: InsertPoint) -> None:
        """Adds a time offset to this frame, which is used when the frame is advanced in
        time."""
        self.offset = _TimeExpression.add(
            self.offset, offset, insertion_point=insertion_point
        )

    def rebase(self, sync_base: _TIME_TYPE = None) -> None:
        """Moves frame time tracking to a new synchronize base with no relative offset."""
        self.sync_base = sync_base
        self.offset = None


class _FrameFactTracker:
    """Tracks the facts about frames at different points in the program, which is used to
    resolve Synchronize operations.

    Provides methods to evolve those frames, managing domain membership and facts
    surrounding frames through methods like ``replace_frame``, which is used when a frame is
    replaced by another SSA value but maintains its temporal relationships,
    ``synchronize_frames`` which deals with frame synchronization accounting for domain
    membership, and ``add_time_offset``, which is used when a frame is advanced in time by a
    known amount.
    """

    def __init__(self, _frame_facts: dict[SSAValue, _FrameFact] | None = None):
        self._frame_facts: dict[SSAValue, _FrameFact] = (
            _frame_facts if _frame_facts is not None else {}
        )

    def clone(self) -> "_FrameFactTracker":
        """Clones the tracker state, preserving intra-domain relationships.

        This is used for entering regions so child analysis can update frame state without
        mutating the parent tracker.
        """

        domain_clones: dict[_TimeDomain, _TimeDomain] = {}
        frame_facts: dict[SSAValue, _FrameFact] = {}

        for frame, fact in self._frame_facts.items():
            domain_key = fact.domain
            if (domain := domain_clones.get(domain_key)) is None:
                domain = _TimeDomain(
                    absolute_anchor=fact.domain.absolute_anchor,
                )
                domain_clones[domain_key] = domain

            cloned_frame = _FrameFact(
                current=fact.current,
                domain=domain,
                sync_base=fact.sync_base,
                offset=fact.offset,
            )
            domain.add_frame(cloned_frame)
            frame_facts[frame] = cloned_frame

        return _FrameFactTracker(frame_facts)

    def replace_frame(
        self, old_frame: SSAValue, new_frame: SSAValue, maintain_relative_time: bool = True
    ):
        """Updates the fact for a given frame, which is used when frames are merged or
        synchronized.

        There is flag to state if relative time is maintained, which is used to determine if
        the new frame should be in the same domain as the old frame, or if it should be in a
        new domain with no temporal relationships to other frames.

        This is expected to be used in a number of situations:

        * Encountering a known frame operation that does not advance time, but changes its
          state.
        * Encountering an operation with unknown scheduling semantics, where we want to be
          conservative and break temporal relationships with other frames.
        """

        fact = self._remove_frame_fact(old_frame)
        if fact is None:
            return

        if maintain_relative_time:
            fact.replace(new_frame)
        else:
            self._reset_fact_temporal_state(fact)
            fact.current = new_frame
        self._add_frame_fact(new_frame, fact)

    def add_frames(self, *frames: SSAValue, anchored: bool = False):
        """Adds new frames to the tracker, which is used when new frames are created.

        If the frames are anchored, then we add them to the same domain with a shared offset
        of None, which means they are all anchored to the same absolute time. If the
        frames are not anchored, then we add them to separate domains, as there is no
        semantic guarantee that they are temporally related to each other in any way.

        This is expected to be used for:

        * CreateFrame operations, where new frames are created, where ``anchored = True``
          as the frames are guaranteed to be synchronized to each other at time zero.
        * Block arguments, where new frames are created, where ``anchored = False`` as the
          source of the frames is unknown, and there is no guarantee they are synchronized
          to each other or to any other frame.
        * Results of operations with unknown scheduling semantics, where new frames are
          created, where ``anchored = False`` as there is no guarantee they are synchronized
          to each other or to any other frame.
        """

        if anchored:
            domain = _TimeDomain(absolute_anchor=True)
            for frame in frames:
                self._add_frame_fact(frame, _FrameFact(current=frame, domain=domain))
        else:
            for frame in frames:
                domain = _TimeDomain()
                self._add_frame_fact(frame, _FrameFact(current=frame, domain=domain))

    def remove_frames(self, *frames: SSAValue):
        """Removes frames from the tracker, which is used when frames go out of scope.

        This is expected to be used for operations that consume frames but aren't
        semantically understood, especially in regards to scheduling. When we encounter such
        an operation, we want to be conservative and remove the frames from the tracker, and
        add a new fact for any resulting frames.
        """
        for frame in frames:
            self._remove_frame_fact(frame)

    def invalidate_all_frames(self):
        """Invalidates all currently tracked frames."""
        for fact in self._frame_facts.values():
            self._reset_fact_temporal_state(fact)

    def add_time_offset(
        self,
        frame: SSAValue,
        new_frame: SSAValue,
        offset: _TIME_TYPE,
        insertion_point: InsertPoint,
    ):
        """Adds a time offset to a frame, which is used when a frame is advanced in time.

        Expected to be used by operations that evolve a frame in time in an understood way,
        such as Pulse, Acquire and Wait operations.
        """

        if isinstance(offset, TimeAttr):
            offset = _TimeExpression.constant(offset, insertion_point)

        fact = self._frame_facts[frame]
        fact.add_offset(offset, insertion_point=insertion_point)
        self.replace_frame(frame, new_frame)

    def synchronize_frames(
        self, frames: list[SSAValue], insertion_point: InsertPoint
    ) -> list[_TIME_TYPE | None] | None:
        """Synchronizes a list of frames, which is used when a Synchronize operation is
        encountered.

        Returns the time expressions for each frame, which are used to determine the time
        offsets that need to be added to each frame to synchronize them.
        """

        facts = [self._frame_facts[frame] for frame in frames]
        domains = {fact.domain for fact in facts}
        if len(domains) == 1:
            return self._synchronize_frames_in_domain(frames, insertion_point)

        absolute_anchor = all(domain.absolute_anchor for domain in domains)
        if absolute_anchor:
            self._merge_domains(domains)
            return self._synchronize_frames_in_domain(frames, insertion_point)

        self._add_to_new_domain(frames)
        return None

    def _reset_fact_temporal_state(self, fact: _FrameFact) -> None:
        """Resets a frame fact to a new domain with no temporal relationships."""
        fact.domain.remove_frame(fact)
        fact.domain = _TimeDomain(members={fact})
        fact.rebase()

    def _add_frame_fact(self, frame: SSAValue, fact: _FrameFact) -> None:
        """Adds a frame fact and indexes it by domain membership."""
        self._frame_facts[frame] = fact
        fact.domain.members.add(fact)

    def _remove_frame_fact(self, frame: SSAValue) -> _FrameFact | None:
        """Removes a frame fact and updates domain membership indexes."""
        if frame not in self._frame_facts:
            return None

        fact = self._frame_facts.pop(frame)
        fact.domain.members.discard(fact)
        return fact

    def _add_to_new_domain(self, frames: list[SSAValue]):
        """Creates a new domain for a list of frames, which is used when synchronizing
        frames that belong to different domains with no absolute anchor."""

        domain = _TimeDomain()
        for frame in frames:
            fact = self._frame_facts[frame]
            fact.domain.remove_frame(fact)
            fact.domain = domain
            domain.add_frame(fact)
            fact.rebase()

    def _synchronize_frames_in_domain(
        self, frames: list[SSAValue], insertion_point: InsertPoint
    ) -> list[_TIME_TYPE | None]:
        """Synchronizes tracked frames that are known to be in the same domain."""

        frame_facts = [self._frame_facts[frame] for frame in frames]
        bases = {fact.sync_base for fact in frame_facts}

        if len(bases) == 1:
            # CASE: All frames are already synchronized to the same base, just with some
            # known offsets relative to that. We can just do a maximum over the offsets.
            offsets = [frame_fact.offset for frame_fact in frame_facts]
            max_offset = _TimeExpression.max(*offsets, insertion_point=insertion_point)
            subtract_exprs = [
                _TimeExpression.subtract(
                    max_offset, offset, insertion_point=insertion_point
                )
                for offset in offsets
            ]
            total_time = _TimeExpression.add(
                next(iter(bases)), max_offset, insertion_point=insertion_point
            )
        else:
            # CASE: Frames are synchronized to different bases
            total_fact_times = [
                _TimeExpression.add(
                    fact.sync_base, fact.offset, insertion_point=insertion_point
                )
                for fact in frame_facts
            ]
            total_time = _TimeExpression.max(
                *total_fact_times, insertion_point=insertion_point
            )
            subtract_exprs = [
                _TimeExpression.subtract(
                    total_time, fact_time, insertion_point=insertion_point
                )
                for fact_time in total_fact_times
            ]

        for fact in frame_facts:
            fact.rebase(total_time)
        return subtract_exprs

    def _merge_domains(self, domains: set[_TimeDomain]):
        """Merges multiple time domains into a single domain when each frame has an absolute
        anchor.

        This is used when synchronizing frames that belong to different domains.
        """

        new_domain = _TimeDomain(absolute_anchor=True)
        for domain in domains:
            for fact in tuple(domain.members):
                fact.domain.remove_frame(fact)
                fact.domain = new_domain
                new_domain.add_frame(fact)


class TimelineNormalization(ModulePass):
    """Resolves Synchronize operations into Wait operations where possible by analyzing
    temporal relationships in program control flow.

    This is a best effort pass, and might not be able to resolve all Synchronize operations
    but can help reduce the number of Synchronize operations in the IR. Resolving
    synchronizations is important for reducing latency overhead, and not all hardware
    supports synchronizations.

    The following considerations are taken during timeline normalization:

    * Frames defined by a CreateFrameOp are treated as time zero within the scope they're
      defined.
    * Frames that enter a block via block arguments have no assumptions on time, and the
      timing of the frames are treated as unknown.
    * Operations with regions are conservatively handled. For non-isolated operations we
      invalidate tracked frame relationships before visiting region bodies.
    * Isolated region operations are analyzed with a fresh tracker so they do not affect
      outer frame facts.
    * Unknown operations that consume frames are assumed to have unknown scheduling
      semantics, and all consumed frames are treated as having unknown timing after that
      operation.

    .. warning::

        Region scheduling semantics are handled conservatively. This pass currently does
        not model detailed interleaving costs for general control-flow constructs.
    """

    name = "pulse.timeline-normalization"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        sync_candidates = self._get_synchronize_candidates(op)
        expression_materializer = _TimeExpressionMaterializer()
        for candidate in sync_candidates:
            self._replace_synchronize_candidate(candidate, expression_materializer)

    def _get_synchronize_candidates(
        self, operation: ModuleOp
    ) -> list[_SynchronizeCandidate]:
        """Walks a region, and returns a list of Synchronize operations that can be
        resolved, along with the time expressions for each frame in the synchronize."""
        candidates: list[_SynchronizeCandidate] = []
        region = operation.body
        candidates.extend(self._walk_blocks_in_region(region, _FrameFactTracker()))
        return candidates

    def _walk_blocks_in_region(
        self, region: Region, frame_fact_tracker: _FrameFactTracker
    ) -> list[_SynchronizeCandidate]:
        """Walks a region and accumulates synchronize candidates.

        The tracker is updated in-place as operations are visited.
        """
        candidates: list[_SynchronizeCandidate] = []

        if len(region.blocks) > 1:
            # TODO: COMPILER-1224
            raise NotImplementedError(
                f"Encountered region with multiple blocks, which is not currently "
                f"supported by the timeline normalization pass: {region}."
            )
        for block in region.blocks:
            candidates.extend(self._walk_block(block, frame_fact_tracker))
        return candidates

    def _walk_block(
        self, block: Block, frame_fact_tracker: _FrameFactTracker
    ) -> list[_SynchronizeCandidate]:
        """Walks a block, updating the frame facts as it goes, and resolving any Synchronize
        operations it encounters.

        This walks the block in order.
        """
        frame_block_args = [arg for arg in block.args if isinstance(arg.type, FrameType)]
        if frame_block_args:
            frame_fact_tracker.add_frames(*frame_block_args, anchored=False)

        candidates: list[_SynchronizeCandidate] = []
        for op in block.ops:
            candidates.extend(self._visit_operation(op, frame_fact_tracker))
        return candidates

    @singledispatchmethod
    def _visit_operation(
        self, op: Operation, frame_fact_tracker: _FrameFactTracker
    ) -> list[_SynchronizeCandidate]:
        """Visits an operation, updating the frame facts as it goes, and resolving any
        Synchronize operations it encounters.

        If the operation is not specifically handled, this implementation is used.

        This is the conservative implementation that is used for operations that are not
        specifically handled. It conservatively treats frame operands as consumed, removes
        them from tracking, and treats frame results as fresh unknown frames.

        For region operations:

        * If the operation is isolated-from-above, region bodies are analyzed with a fresh
          tracker.
        * Otherwise, currently tracked frame relationships are invalidated before analyzing
          region bodies.
        """

        consumed_frames = [
            operand for operand in op.operands if isinstance(operand.type, FrameType)
        ]
        frame_fact_tracker.remove_frames(*consumed_frames)

        candidates: list[_SynchronizeCandidate] = []
        if op.regions:
            if op.has_trait(IsolatedFromAbove):
                # If the operation is isolated from above, we don't need to worry about
                # invalidating frames across it as they cannot interact with operations
                # outside the region
                child_tracker = _FrameFactTracker()
            else:
                # If not, we need to be conservative and break temporal relationships across
                # the region, as we have no guarantees about how it schedules with respect
                # to pulse-level operations. We do this by invalidating all currently
                # tracked frames
                frame_fact_tracker.invalidate_all_frames()
                child_tracker = frame_fact_tracker

            for region in op.regions:
                candidates.extend(
                    self._walk_blocks_in_region(region, child_tracker.clone())
                )

        frame_results = [
            result for result in op.results if isinstance(result.type, FrameType)
        ]
        frame_fact_tracker.add_frames(*frame_results, anchored=False)
        return candidates

    @_visit_operation.register(SynchronizeOp)
    def _visit_synchronize_op(
        self, op: SynchronizeOp, frame_fact_tracker: _FrameFactTracker
    ) -> list[_SynchronizeCandidate]:
        """Resolves a Synchronize operation into a Wait operation if possible."""

        insertion_point = InsertPoint.before(op)
        expressions = frame_fact_tracker.synchronize_frames(op.frames, insertion_point)
        for i, frame in enumerate(op.frames):
            new_frame = op.results[i]
            frame_fact_tracker.replace_frame(frame, new_frame)

        if expressions is None:
            return []
        return [_SynchronizeCandidate(op, expressions)]

    @_visit_operation.register(WaitOp)
    def _visit_wait_op(
        self, op: WaitOp, frame_fact_tracker: _FrameFactTracker
    ) -> list[_SynchronizeCandidate]:
        """Updates the frame facts for a Wait operation, by adding the time offset to the
        frame."""
        insertion_point = InsertPoint.before(op)
        frame_fact_tracker.add_time_offset(
            op.frame, op.result, op.duration, insertion_point
        )
        return []

    @_visit_operation.register(AcquireOp)
    def _visit_acquire_op(
        self, op: AcquireOp, frame_fact_tracker: _FrameFactTracker
    ) -> list[_SynchronizeCandidate]:
        """Updates the frame facts for an Acquire operation, by adding the time offset to
        the frame."""
        insertion_point = InsertPoint.before(op)
        frame_fact_tracker.add_time_offset(
            op.frame, op.frame_result, op.duration, insertion_point
        )
        return []

    @_visit_operation.register(PulseOp)
    def _visit_pulse_op(
        self, op: PulseOp, frame_fact_tracker: _FrameFactTracker
    ) -> list[_SynchronizeCandidate]:
        """Updates the frame facts for a Pulse operation, by adding the time offset to the
        frame."""

        match op.waveform.owner:
            case IsAnalyticalWaveformInterface():
                duration = op.waveform.owner.width
            case ConstantOp():
                duration = op.waveform.owner.value.width
            case _:
                # In the future, we could add an operation to get the duration of a waveform
                # which could collapse down with canonicalisation. For now, conservative
                # treat it as unknown and move the frame to a new domain
                frame_fact_tracker.replace_frame(
                    op.frame, op.result, maintain_relative_time=False
                )
                return []

        insertion_point = InsertPoint.before(op)
        frame_fact_tracker.add_time_offset(op.frame, op.result, duration, insertion_point)
        return []

    @_visit_operation.register(PhaseShiftOp)
    @_visit_operation.register(PhaseSetOp)
    def _visit_phase_op(
        self, op: PhaseShiftOp | PhaseSetOp, frame_fact_tracker: _FrameFactTracker
    ) -> list[_SynchronizeCandidate]:
        """Updates the frame facts for a PhaseShift or PhaseSet operation, by replacing the
        frame with the result frame, as these operations are assumed to not advance time,
        but just change the state of the frame.

        This is an important part of maintaining correct frame facts.
        """

        frame_fact_tracker.replace_frame(op.frame, op.result)
        return []

    @_visit_operation.register(CreateFrameOp)
    def _visit_create_frame_op(
        self, op: CreateFrameOp, frame_fact_tracker: _FrameFactTracker
    ) -> list[_SynchronizeCandidate]:
        """Adds the newly created frame to the tracker."""

        frame_fact_tracker.add_frames(op.result, anchored=True)
        return []

    def _replace_synchronize_candidate(
        self, candidate: _SynchronizeCandidate, materializer: _TimeExpressionMaterializer
    ) -> None:
        """Replaces a Synchronize operation with Wait operations according to the given
        candidate, by inserting the operations returned by the time expressions into the IR,
        and replacing the synchronize with waits with the appropriate time offsets."""

        sync_op = candidate.sync_op
        expressions = candidate.expressions

        wait_ops = []
        ssa_results = []

        for i, expression in enumerate(expressions):
            if expression is None:
                ssa_results.append(sync_op.frames[i])
                continue

            wait_time = materializer.materialize(expression)
            wait_op = WaitOp(sync_op.frames[i], wait_time)
            wait_ops.append(wait_op)
            ssa_results.append(wait_op.results[0])

        Rewriter.replace_op(sync_op, wait_ops, ssa_results)
