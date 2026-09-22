# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Fission a shot loop into one copy per frame lineage.

A shot loop drives every frame in the program, so it belongs to every lineage at once and
cannot be attributed to a single outlined sequence. Qblox runs one program per sequencer,
each with its own loop, so the loop is replicated.

Only ``scf.for`` is handled. It is the sole region-bearing operation the importer puts
in a Pulse entry block, so it is the only shape that needs splitting today. Anything else
enclosing more than one lineage is rejected rather than guessed at, since splitting it
would need rules for its regions and block arguments that nothing yet exercises.

The copy carries no values between iterations. An ``scf.for`` can pass a value from one
iteration to the next through ``iter_args``, and the importer uses that to accumulate
acquisitions into a results array. A sequencer has no equivalent: acquisitions land in
hardware bins, and the bin index is recomputed from the loop counter when the acquire is
lowered. The array is bookkeeping with no hardware counterpart, so the copy simply does
not have it.
"""

# TODO(COMPILER-1466): Accepted as M1 technical debt
# This module only rebuilds a single top-level ``scf.for`` and rejects everything else.
# A more general solution is required to enable structure to be partitioned per frame.

from __future__ import annotations

from xdsl.dialects.scf import ForOp, YieldOp
from xdsl.ir import Block, Operation, SSAValue
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import AcquireOp, FrameType, IntegrateOp
from qat.experimental.dialect.pulse.transforms.partition_by_frame import (
    FrameLineage,
    FrameLineageAnalysis,
)
from qat.experimental.dialect.results.ir import Results


def is_results_op(op: Operation) -> bool:
    """Return whether ``op`` is a results-dialect operation.

    Results operations are bookkeeping with no hardware counterpart, so they are dropped
    from emitted sequences. Membership is tested against the dialect's registered
    operations.

    :param op: Operation to classify.
    :returns: ``True`` if ``op`` belongs to the results dialect.
    """
    return type(op) in Results.operations


def fission_for_lineage(
    op: Operation,
    lineage: FrameLineage,
    analysis: FrameLineageAnalysis,
    value_mapper: dict[SSAValue, SSAValue],
) -> ForOp:
    """Build a copy of an ``scf.for`` with only the contents belonging to ``lineage``.

    :param op: Shot loop enclosing the work of one or more lineages.
    :param lineage: Lineage whose work the returned copy carries.
    :param analysis: Analysis providing per-operation lineage ownership.
    :param value_mapper: Clone mapping for the partition, used to resolve operands
        defined by entry-block operations already copied into this sequence.
    :returns: A newly built ``scf.for`` carrying only ``lineage``'s operations, with no
        loop-carried values.
    :raises PassFailedException: If ``op`` is not an ``scf.for``, if the loop has an
        unsupported shape, or if the copy would have to carry a loop-carried value.
    """
    if not isinstance(op, ForOp):
        raise PassFailedException(
            f"{op.name} encloses more than one frame lineage. Only scf.for shot loops "
            "are split into per-sequencer copies; splitting this operation would need "
            "rules for its regions and block arguments that nothing yet exercises."
        )
    _reject_unsupported_shape(op)
    return _build_loop(op, _retained_ops(op, lineage, analysis), lineage, value_mapper)


def _reject_unsupported_shape(op: ForOp) -> None:
    """Reject shot loops whose shape cannot be split into one copy per frame.

    Three shapes are rejected:

    * a frame carried between iterations as a loop-carried value;
    * a region-bearing operation nested in the loop body;
    * an acquisition consumed by anything other than ``pulse.integrate`` or results
      bookkeeping.

    Each is rejected rather than handled because splitting it needs a rule nothing
    currently exercises, and because getting it wrong is silent: the copy would simply not
    contain the work, leaving valid IR that describes a different program.

    :param op: Shot loop whose shape is checked.
    :raises PassFailedException: If ``op`` has any of the shapes listed above.
    """
    block = op.body.block

    # Frames carried between iterations. The analysis cannot resolve a frame arriving as
    # a block argument, so ownership inside the body would be unknown, and a frame-valued
    # scf.yield would collide with the terminator the copy builds.
    for index, argument in enumerate(block.args[1:], start=1):
        if isinstance(argument.type, FrameType):
            raise PassFailedException(
                f"{op.name} carries a frame as loop-carried value {index}; frames "
                "crossing iterations cannot be split per sequencer"
            )

    # Nested regions. Retained operations are copied from the body block only, so work
    # inside a nested region would be dropped rather than copied.
    for inner in block.ops:
        if inner.regions:
            raise PassFailedException(
                f"{inner.name} nested inside a shot loop is region-bearing; fission "
                "copies the loop body only and would silently drop its contents"
            )

    # Acquisition consumers. pulse.integrate is carried into the copy and results
    # bookkeeping is dropped deliberately, but any other consumer would be dropped
    # silently and the acquisition would come to mean something different.
    for inner in block.ops:
        if not isinstance(inner, AcquireOp):
            continue
        for use in inner.acquisition_result.uses:
            consumer = use.operation
            if isinstance(consumer, IntegrateOp) or is_results_op(consumer):
                continue
            raise PassFailedException(
                f"{inner.name} result is consumed by {consumer.name}; an outlined "
                "sequence carries only pulse.integrate, so this consumer would be lost"
            )


def _retained_ops(
    op: Operation, lineage: FrameLineage, analysis: FrameLineageAnalysis
) -> set[Operation]:
    """Return the operations inside ``op`` that ``lineage``'s copy must contain.

    Seeded with the operations the analysis attributes to ``lineage``, then closed over
    operands so the copy is self-contained. ``pulse.integrate`` is pulled in explicitly:
    it is a downstream marker for its acquisition rather than an operand dependency, and
    nothing else would reach it.
    """
    retained: set[Operation] = set()
    pending = [inner for inner in op.walk() if lineage in analysis.lineages_for_op(inner)]
    while pending:
        current = pending.pop()
        if current in retained:
            continue
        retained.add(current)
        if isinstance(current, AcquireOp):
            pending.extend(
                use.operation
                for use in current.acquisition_result.uses
                if isinstance(use.operation, IntegrateOp)
            )
        for operand in current.operands:
            owner = operand.owner
            if isinstance(owner, Operation) and owner is not op and op.is_ancestor(owner):
                pending.append(owner)
    return retained


def _build_loop(
    op: ForOp,
    retained: set[Operation],
    lineage: FrameLineage,
    value_mapper: dict[SSAValue, SSAValue],
) -> ForOp:
    """Build a loop over ``retained``, carrying no values between iterations."""
    block = op.body.block
    for index, argument in enumerate(block.args[1:], start=1):
        if any(use.operation in retained for use in argument.uses):
            raise PassFailedException(
                f"{op.name} carries value {index} into operations retained for "
                f"{lineage.port}, which a hardware sequence cannot express"
            )

    body = Block(arg_types=[block.args[0].type])
    mapper = dict(value_mapper)
    mapper[block.args[0]] = body.args[0]
    body.add_ops([inner.clone(mapper) for inner in block.ops if inner in retained])
    body.add_op(YieldOp())

    bounds = [value_mapper.get(bound, bound) for bound in (op.lb, op.ub, op.step)]
    for result in op.results:
        value_mapper.pop(result, None)
    return ForOp(*bounds, [], body)
