# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Linearise a multi-block ``q1_cf`` CFG to a single flat ``q1`` block.

``q1_cf`` expresses control flow as a graph: branch terminators carry named
successor blocks and per-successor SSA block arguments, with conditions held as
register operands plus a predicate attribute. The Q1 processor executes a flat
instruction stream whose only block structure is fall-through and whose control
transfer is by label. This pass collapses the graph into that flat stream inside
each :class:`SequenceOp` through the following stages:

1. Statically decided branches fold through :meth:`const_evaluate` to an
   unconditional branch, and blocks left unreachable are pruned.
 2. Blocks are kept in producer order. For conditional branches, ``else`` is the next
    block in the layout (fall-through). Other successors are reached via explicit jumps.
    Each block is assigned a label.
3. Block arguments are erased by register coalescing. ``q1_cf`` requires every
   forwarded operand to share the register of the successor argument it feeds, so
   an argument and its incoming values already occupy one register. Erasure is a
   rename: uses of the argument are redirected to the value from its nearest
   forward predecessor, and the register itself carries the value across edges.
4. Each terminator lowers to flat jumps that target labels: an unconditional
   jump, a ``cmp``/``test`` plus the conditional jump selected by the predicate,
   or a counted ``loop``.
5. All blocks inline into one, referenced heads carrying a ``q1.x.label``. A jump
   to the immediately following label is dropped as a fall-through, and any label
   that no remaining jump targets is removed as dead. Halts of one terminal state
   converge to a single ``Stop*``, a clean ``stop`` and an ``illegal`` error trap
   being distinct states that cannot. The result is a single block of ``q1``
   instructions ending in that halt with no ``q1_cf`` residue.

Reference: https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, BlockArgument, Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.common.cfg import block_predecessors, reachable_blocks
from qat.experimental.dialect.common.naming import assign_unique_name, assign_unique_names
from qat.experimental.dialect.common.region import (
    detach_non_terminator_ops,
    install_single_block,
)
from qat.experimental.dialect.common.ssa import resolve_block_argument
from qat.experimental.dialect.q1 import (
    INT_REGISTER_BIT_WIDTH,
    INT_REGISTER_VALUE_MASK,
    CmpRsRsOp,
    IllegalOp,
    JaeImmOp,
    JaImmOp,
    JbeImmOp,
    JbImmOp,
    JgeImmOp,
    JgImmOp,
    JleImmOp,
    JlImmOp,
    JmpImmOp,
    JnsImmOp,
    JnzImmOp,
    JsImmOp,
    JzImmOp,
    LabelOp,
    LoopRdImmOp,
    MoveImmRdOp,
    StopImmOp,
    StopOp,
    StopRsOp,
    TestRsRsOp,
)
from qat.experimental.dialect.q1.ir.abstract_ops import JumpImmOperation, LoopImmOperation
from qat.experimental.dialect.q1.ir.attrs import LabelAttr
from qat.experimental.dialect.q1_cf import (
    BinaryPredicate,
    BinaryPredicateBranchOp,
    JmpBranchOp,
    LoopBranchOp,
    UnaryPredicate,
    UnaryPredicateBranchOp,
)
from qat.experimental.dialect.q1_sequence import SequenceOp

# Taken-edge conditional jump for each unary predicate. Emitted after a
# ``test rs, rs`` that sets the zero and sign flags from the tested register.
_UNARY_PREDICATE_JUMP = {
    UnaryPredicate.eqz: JzImmOp,
    UnaryPredicate.nez: JnzImmOp,
    UnaryPredicate.ltz: JsImmOp,
    UnaryPredicate.gez: JnsImmOp,
}

# Taken-edge conditional jump for each binary predicate. Emitted after a
# ``cmp lhs, rhs``. Signed predicates read the sign and overflow flags, unsigned
# predicates read the carry flag, which ``cmp`` sets faithfully.
_BINARY_PREDICATE_JUMP = {
    BinaryPredicate.eq: JzImmOp,
    BinaryPredicate.ne: JnzImmOp,
    BinaryPredicate.slt: JlImmOp,
    BinaryPredicate.sle: JleImmOp,
    BinaryPredicate.sgt: JgImmOp,
    BinaryPredicate.sge: JgeImmOp,
    BinaryPredicate.ult: JbImmOp,
    BinaryPredicate.ule: JbeImmOp,
    BinaryPredicate.ugt: JaImmOp,
    BinaryPredicate.uge: JaeImmOp,
}

# The flat q1 terminators that halt a path: the clean ``Stop*`` family and the
# ``illegal`` error trap. Both end a path and are carried through unchanged,
# unlike the q1_cf branch terminators this pass lowers. The two are distinct
# terminal states, an illegal raising an error flag a stop does not, so they
# converge only within a kind (see _pick_sink).
_HALT_OPS = (IllegalOp, StopImmOp, StopRsOp, StopOp)

# Prefix for pass-generated block labels. Deconflicted against labels already
# present in a body by _fresh_labels.
_BLOCK_LABEL_PREFIX = "bb"


def _referenced_label(op: Operation) -> str | None:
    """Return the label an op jumps to, or ``None`` when it references none.

    A jump or loop op holds its destination in ``imm`` as either an address
    immediate or a :class:`LabelAttr`. Only the label form names a label, so an
    address target and every non-jump op reference none.
    """
    if isinstance(op, JumpImmOperation | LoopImmOperation) and isinstance(
        op.imm, LabelAttr
    ):
        return op.imm.data
    return None


def _move_immediate_value(value: SSAValue) -> int | None:
    """Return the signless register value of an operand materialised by ``move``.

    The immediate is masked to the integer register width. Only a direct
    ``q1.i.move`` of an immediate is treated as constant. This is the case that
    lets a branch on a compile-time condition fold to an unconditional edge.
    """
    if isinstance(value, OpResult) and isinstance(value.op, MoveImmRdOp):
        return value.op.imm.data & INT_REGISTER_VALUE_MASK
    return None


def _fold_branch(term: Operation) -> JmpBranchOp | None:
    """Fold a constant-decided conditional branch to an unconditional branch.

    :returns: The replacement :class:`JmpBranchOp` on the statically taken edge,
        or ``None`` when the branch is not statically decidable.
    """
    if isinstance(term, UnaryPredicateBranchOp):
        rs = _move_immediate_value(term.rs)
        if rs is None:
            return None
        taken = term.const_evaluate(rs, INT_REGISTER_BIT_WIDTH)
        if taken:
            return JmpBranchOp(list(term.then_arguments), term.then_block)
        return JmpBranchOp(list(term.else_arguments), term.else_block)
    if isinstance(term, BinaryPredicateBranchOp):
        lhs, rhs = _move_immediate_value(term.lhs), _move_immediate_value(term.rhs)
        if lhs is None or rhs is None:
            return None
        taken = term.const_evaluate(lhs, rhs, INT_REGISTER_BIT_WIDTH)
        if taken:
            return JmpBranchOp(list(term.then_arguments), term.then_block)
        return JmpBranchOp(list(term.else_arguments), term.else_block)
    return None


def _fold_and_prune(entry: Block) -> None:
    """Fold constant branches and drop unreachable blocks to a fixpoint."""
    parent = entry.parent
    if parent is None:
        raise AssertionError("entry block is detached from its region")
    changed = True
    while changed:
        changed = False
        for block in list(parent.blocks):
            term = block.last_op
            if term is None:
                continue
            folded = _fold_branch(term)
            if folded is not None:
                Rewriter.replace_op(term, folded)
                changed = True
        reachable = reachable_blocks(entry)
        for block in list(parent.blocks):
            if block not in reachable:
                parent.erase_block(block)
                changed = True


def _elide_redundant_control(ops: list[Operation]) -> list[Operation]:
    """Drop jumps to the next instruction and the labels left unreferenced.

    A ``jmp`` whose target is the immediately following label is a fall-through
    and is removed. A label that no remaining jump targets is dead and is
    removed. Both run to a fixpoint so a chain of such redundancies collapses.
    """
    changed = True
    while changed:
        changed = False
        pruned: list[Operation] = []
        for i, op in enumerate(ops):
            following = ops[i + 1] if i + 1 < len(ops) else None
            jump_target = _referenced_label(op)
            if (
                isinstance(op, JmpImmOp)
                and jump_target is not None
                and isinstance(following, LabelOp)
                and jump_target == following.reference.data
            ):
                changed = True
                continue
            pruned.append(op)
        targets = {label for op in pruned if (label := _referenced_label(op)) is not None}
        ops = [
            op
            for op in pruned
            if not isinstance(op, LabelOp) or op.reference.data in targets
        ]
        changed = changed or len(ops) != len(pruned)
    return ops


@dataclass(frozen=True)
class _Layout:
    """Immutable control-flow facts for one folded, pruned sequence body.

    Produced by :func:`_prepare_layout` and consumed unchanged by argument erasure
    and code generation, so the two mutating phases share a single read-only view
    of the graph rather than rediscovering it.

    :param ordered: Blocks in the post-fold/post-prune producer order that drives
        linearisation.
    :param index_of: Position of each block in ``ordered`` for constant-time
        order comparisons.
    :param label_of: Generated q1 label name for each block head in the flat
        instruction stream.
    :param predecessors: Incoming CFG edges per block as
        ``(predecessor, forwarded_operands)`` pairs.
    :param sink: Terminal block whose halt is retained as the single program
        terminator in the flattened body.
    """

    ordered: list[Block]
    index_of: dict[Block, int]
    label_of: dict[Block, str]
    predecessors: dict[Block, list[tuple[Block, Sequence[SSAValue]]]]
    sink: Block


def _pick_sink(seq: SequenceOp, ordered: list[Block]) -> Block:
    """Return the block whose halt ends the linearised program.

    Halts partition into two terminal states. A ``stop`` halts cleanly, its code
    carrying no control-flow semantics the ISA acts on, so every ``stop`` is
    interchangeable. An ``illegal`` halts by raising an error flag that drives the
    sequencer status to error, so it is distinct from a clean stop. Within one
    state the last halt in producer order, the program footer of a structured
    body, is the sink and any earlier halt redirects to a shared halt point,
    which also satisfies the single-block form that admits one terminal
    instruction.

    :raises PassFailedException: If no block halts, leaving the flat block without
        a terminator to anchor it, or if the body can exit through both a clean
        stop and an illegal trap, two terminal states a single block cannot host
        and neither redirects to the other losslessly.
    """
    halts = [block for block in ordered if isinstance(block.last_op, _HALT_OPS)]
    if not halts:
        raise PassFailedException(
            f"Sequence '{seq.channel_id.data}': no halt terminator (Stop* or illegal) "
            "to end the linearised block"
        )
    if any(isinstance(block.last_op, IllegalOp) for block in halts) and not all(
        isinstance(block.last_op, IllegalOp) for block in halts
    ):
        raise PassFailedException(
            f"Sequence '{seq.channel_id.data}': body exits through both a"
            f" clean stop and an illegal trap. The flat block admits one"
            f" terminal halt and an illegal raises an error flag a stop does"
            f" not, so the two states cannot converge"
        )
    return halts[-1]


def _fresh_labels(ordered: list[Block]) -> dict[Block, str]:
    """Assign each block a label kept distinct from labels already present.

    The pass owns label generation, but a block body may already define a
    ``q1.x.label`` or hold a jump that references one. A generated name colliding
    with such a label would emit a duplicate definition or silently retarget an
    existing jump. Both label definitions and jump targets are collected first,
    then handed to the allocator as the reserved set.
    """
    reserved: set[str] = set()
    for block in ordered:
        for op in block.ops:
            if isinstance(op, LabelOp):
                reserved.add(op.reference.data)
            referenced = _referenced_label(op)
            if referenced is not None:
                reserved.add(referenced)
    return assign_unique_names(ordered, reserved, _BLOCK_LABEL_PREFIX)


def _prepare_layout(seq: SequenceOp) -> _Layout:
    """Fold, prune, then snapshot the sequence body into immutable layout facts.

    Folding and pruning mutate the body first so the snapshot describes the graph the later
    phases act on. The facts are gathered once here and consumed read-only, sparing argument
    erasure and code generation a rediscovery of block order, indices, labels, predecessors,
    and the halt sink.
    """
    region = seq.body
    _fold_and_prune(region.blocks[0])
    ordered = list(region.blocks)
    return _Layout(
        ordered=ordered,
        index_of={block: i for i, block in enumerate(ordered)},
        label_of=_fresh_labels(ordered),
        predecessors=block_predecessors(ordered),
        sink=_pick_sink(seq, ordered),
    )


def _primary_incoming(layout: _Layout, block: Block, position: int) -> SSAValue:
    """Return the incoming value that defines a block argument.

    The nearest forward predecessor is preferred so that the substitute value precedes the
    argument's uses in the linear layout. A block reachable from the entry always has such a
    predecessor.
    """
    incoming = layout.predecessors[block]
    forward = [
        (pred, forwarded[position])
        for pred, forwarded in incoming
        if layout.index_of[pred] < layout.index_of[block]
    ]
    if forward:
        return max(forward, key=lambda edge: layout.index_of[edge[0]])[1]
    return incoming[0][1][position]


def _coalescing_rename_block_arguments(seq: SequenceOp, layout: _Layout) -> None:
    """Erase every block argument by a coalescing rename to its incoming value.

    ``q1_cf`` requires each forwarded operand to share the register of the
    successor argument it feeds, so a block argument and its incoming values
    already occupy one register. The rename coalesces that class: every use of the
    argument is redirected to the value from its nearest forward predecessor, and
    the shared register carries the value across the edge.

    :raises PassFailedException: If an incoming value occupies a different register
        than the argument, which a verified ``q1_cf`` body cannot produce and which
        a rename alone cannot honour.
    """
    arg_representative: dict[BlockArgument, SSAValue] = {}
    for block in layout.ordered:
        if block.args and not layout.predecessors[block]:
            raise PassFailedException(
                f"Sequence '{seq.channel_id.data}': block argument has no incoming"
                f" edge to coalesce against. The sequence entry must not take"
                f" arguments"
            )
        for position, arg in enumerate(block.args):
            source = resolve_block_argument(
                _primary_incoming(layout, block, position), arg_representative
            )
            if source is arg:
                raise PassFailedException(
                    f"Sequence '{seq.channel_id.data}': block argument is only fed from "
                    f"itself. q1_cf edges must provide a non-self incoming value to coalesce"
                )
            if source.type != arg.type:
                raise PassFailedException(
                    f"Sequence '{seq.channel_id.data}': block argument in"
                    f" {arg.type} is fed from {source.type}. q1_cf edges must be"
                    f" register-coalesced before linearising"
                )
            arg_representative[arg] = source

    for block in layout.ordered:
        for arg in list(block.args):
            arg.replace_all_uses_with(resolve_block_argument(arg, arg_representative))
        for arg in list(block.args):
            block.erase_arg(arg)


def _emit_fall_through(
    layout: _Layout, target: Block, next_block: Block | None, into: list[Operation]
) -> None:
    """Emit a fall-through edge: jump unless the target is the next block."""
    if target is not next_block:
        into.append(JmpImmOp(layout.label_of[target]))


def _lower_terminator(
    layout: _Layout,
    block: Block,
    next_block: Block | None,
    into: list[Operation],
    halt_redirect_label: str,
) -> None:
    """Lower ``block``'s terminator into flat jumps appended to ``into``.

    Every branch target is emitted as a label reference. Unreferenced labels are
    pruned later by :func:`_elide_redundant_control`, so no separate bookkeeping of
    which labels are live is needed here.

    :param layout: Read-only control-flow facts naming each block's label.
    :param block: The block whose terminator is lowered.
    :param next_block: The block laid out immediately after, or ``None`` at the
        end. A successor equal to it falls through without a jump.
    :param into: Accumulator the lowered ops are appended to in place.
    :param halt_redirect_label: Label of the shared halt point used for
        redirected non-sink halts.
    """
    term = block.last_op
    if term is None:
        raise AssertionError("q1_cf block has no terminator to lower")
    match term:
        case JmpBranchOp():
            _emit_fall_through(layout, term.successor, next_block, into)
        case UnaryPredicateBranchOp():
            into.append(TestRsRsOp(term.rs, term.rs))
            jump = _UNARY_PREDICATE_JUMP[term.predicate.data]
            into.append(jump(layout.label_of[term.then_block]))
            _emit_fall_through(layout, term.else_block, next_block, into)
        case BinaryPredicateBranchOp():
            into.append(CmpRsRsOp(term.lhs, term.rhs))
            jump = _BINARY_PREDICATE_JUMP[term.predicate.data]
            into.append(jump(layout.label_of[term.then_block]))
            _emit_fall_through(layout, term.else_block, next_block, into)
        case LoopBranchOp():
            into.append(LoopRdImmOp(term.counter.type, layout.label_of[term.body_block]))
            _emit_fall_through(layout, term.exit_block, next_block, into)
        case _:
            # A non-sink halt of the same terminal state as the sink (mixed states
            # are rejected in _pick_sink), so redirecting to one shared halt point
            # is lossless.
            into.append(JmpImmOp(halt_redirect_label))


def _fresh_aux_label(layout: _Layout, stem: str) -> str:
    """Return a fresh label name derived from ``stem``.

    The label is deconflicted against generated block labels and all labels already defined
    or referenced in the sequence body.
    """
    reserved: set[str] = set(layout.label_of.values())
    for block in layout.ordered:
        for op in block.ops:
            if isinstance(op, LabelOp):
                reserved.add(op.reference.data)
            referenced = _referenced_label(op)
            if referenced is not None:
                reserved.add(referenced)

    return assign_unique_name(reserved, f"{stem}_", preferred_name=stem)


def _assemble_as_single_block(layout: _Layout) -> list[Operation]:
    """Concatenate the ordered blocks into one label-referenced stream of q1 flat ISA ops.

    Non-sink blocks lower their terminator to jumps. The sink contributes its body
    followed by the halt that ends the program. Each block head is labelled, and
    redundant jumps and dead labels collapse in :func:`_elide_redundant_control`.
    """
    non_sink = [block for block in layout.ordered if block is not layout.sink]
    halt = layout.sink.last_op
    if halt is None:
        raise AssertionError("sink block has no halt after _pick_sink selected it")

    sink_ops = detach_non_terminator_ops(layout.sink)
    has_non_sink_halt = any(isinstance(block.last_op, _HALT_OPS) for block in non_sink)
    halt_redirect_label = layout.label_of[layout.sink]
    if has_non_sink_halt and sink_ops:
        halt_redirect_label = _fresh_aux_label(
            layout, f"{layout.label_of[layout.sink]}_halt"
        )

    lowered_ops_by_block: dict[Block, list[Operation]] = {}
    for i, block in enumerate(non_sink):
        next_block = non_sink[i + 1] if i + 1 < len(non_sink) else None
        ops = detach_non_terminator_ops(block)
        _lower_terminator(layout, block, next_block, ops, halt_redirect_label)
        lowered_ops_by_block[block] = ops

    halt.detach()
    if halt_redirect_label == layout.label_of[layout.sink]:
        sink_ops.append(halt)
    else:
        sink_ops.append(JmpImmOp(halt_redirect_label))

    q1_flat_isa_ops: list[Operation] = []
    for block in non_sink:
        q1_flat_isa_ops.append(LabelOp(layout.label_of[block]))
        q1_flat_isa_ops.extend(lowered_ops_by_block[block])
    q1_flat_isa_ops.append(LabelOp(layout.label_of[layout.sink]))
    q1_flat_isa_ops.extend(sink_ops)
    if halt_redirect_label != layout.label_of[layout.sink]:
        q1_flat_isa_ops.append(LabelOp(halt_redirect_label))
        q1_flat_isa_ops.append(halt)
    return _elide_redundant_control(q1_flat_isa_ops)


def _linearise_sequence(seq: SequenceOp) -> None:
    """Rewrite one :class:`SequenceOp` body in place to a single flat ``q1`` block.

    A body already reduced to one block by folding and pruning is left untouched. Otherwise
    the block arguments are erased by coalescing rename, the blocks are assembled into a
    stream of q1 flat ISA ops, and that stream replaces the body's region.
    """
    layout = _prepare_layout(seq)
    if len(layout.ordered) == 1:
        return
    _coalescing_rename_block_arguments(seq, layout)
    install_single_block(seq.body, _assemble_as_single_block(layout))


class LineariseQ1CfToQ1Pass(ModulePass):
    """Lower every ``q1_cf`` CFG in the module to a single flat ``q1`` block."""

    name = "linearise-q1-cf-to-q1"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for seq in [child for child in op.walk() if isinstance(child, SequenceOp)]:
            _linearise_sequence(seq)
