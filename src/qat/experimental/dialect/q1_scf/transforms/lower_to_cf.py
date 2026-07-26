# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Lower structured ``q1_scf`` regions to a ``q1_cf`` block CFG.

``q1_scf`` expresses control flow with structured containers: an ``if`` conditional
and ``while``/``for`` loops whose bodies are regions terminated by ``condition`` and
``yield``. ``q1_cf`` is the layer below, an unstructured CFG of basic blocks wired by
branch terminators. This pass expands each container into blocks and selects the
matching ``q1_cf`` terminator, carrying the source predicate across unchanged.

Each construct is lowered by splitting its parent block into the ops preceding it and
a continuation holding the ops that follow. The container regions inline between the
two, and the branch terminators thread live values as successor block arguments:

1. ``q1_scf.if`` becomes a conditional branch into ``then``/``else`` blocks, both
   rejoining the continuation. Region ``yield`` operands become the continuation's
   block arguments, which stand in for the conditional's results.
2. ``q1_scf.while`` becomes a header block that tests the condition and branches to
   the body or the exit, with the body's ``yield`` forming the back-edge to the
   header.
3. ``q1_scf.for`` becomes a self-looping body whose terminator is ``q1_cf.loop_branch``,
   the native decrement-and-branch. The induction counter, its back-edge argument, and
   the loop-branch counter operand are one value, decremented in place each iteration.

The lowering is register-neutral. It introduces no new register values: every value a
branch forwards already exists as an SSA value at the ``q1_scf`` level. Conditions carry
a predicate over ``q1.reg`` operands rather than a boolean, so predicate selection is a
direct match with no value to materialise. Whether a pre-allocation body still needs
relocation is left to register allocation.

A ``q1_cf`` conditional branch falls through to its ``else`` successor, which its
verifier requires to be the branch block's layout neighbour. The lowering honours this
by placing the ``else`` block, or a forwarding stub for edges that leave through the
continuation, immediately after the branch.

Reference: https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html
"""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import BlockInsertPoint, InsertPoint
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.q1 import INT_REGISTER_VALUE_MASK, MoveImmRdOp
from qat.experimental.dialect.q1_cf.ir.attrs import BinaryPredicateAttr, UnaryPredicateAttr
from qat.experimental.dialect.q1_cf.ir.ops import (
    BinaryPredicateBranchOp,
    JmpBranchOp,
    LoopBranchOp,
    UnaryPredicateBranchOp,
)
from qat.experimental.dialect.q1_scf.ir.ops import (
    ConditionOp,
    ForOp,
    IfOp,
    WhileOp,
    YieldOp,
)


def _conditional_branch(
    predicate: UnaryPredicateAttr | BinaryPredicateAttr,
    predicate_args: Sequence[SSAValue],
    then_block: Block,
    then_arguments: Sequence[SSAValue],
    else_block: Block,
    else_arguments: Sequence[SSAValue],
) -> UnaryPredicateBranchOp | BinaryPredicateBranchOp:
    """Build the ``q1_cf`` conditional branch selected by the predicate kind.

    A comparison predicate reads two operands and lowers to
    :class:`BinaryPredicateBranchOp`; a unary predicate reads one and lowers to
    :class:`UnaryPredicateBranchOp`. The predicate attribute passes through unchanged, so the
    signed or unsigned character of the source condition is preserved.

    :param predicate: The source predicate, a unary or binary predicate attribute.
    :param predicate_args: The register operands the predicate tests.
    :param then_block: Successor taken when the predicate holds.
    :param then_arguments: Block arguments forwarded on the ``then`` edge.
    :param else_block: Fall-through successor, taken when the predicate fails.
    :param else_arguments: Block arguments forwarded on the ``else`` edge.
    :returns: The matching ``q1_cf`` conditional branch terminator.
    """
    if isinstance(predicate, BinaryPredicateAttr):
        lhs, rhs = predicate_args
        return BinaryPredicateBranchOp(
            predicate, lhs, rhs, then_arguments, else_arguments, then_block, else_block
        )
    (rs,) = predicate_args
    return UnaryPredicateBranchOp(
        predicate, rs, then_arguments, else_arguments, then_block, else_block
    )


def _split_continuation(
    op: IfOp | WhileOp | ForOp, rewriter: PatternRewriter
) -> tuple[Block, Block]:
    """Split the parent block, returning the head and the continuation.

    The head retains the operations preceding ``op`` and receives the branch that
    enters the lowered CFG. The continuation holds the operations that follow. When
    ``op`` has results, the continuation is a fresh block whose arguments stand in for
    those results and which branches to the trailing operations, so the values reach
    their uses by the branches that target it.

    :param op: The structured operation being lowered.
    :param rewriter: The active pattern rewriter.
    :returns: The head block and the continuation block.
    """
    head = op.parent_block()
    if head is None:
        raise PassFailedException(f"{op.name}: operation is detached from its parent block")
    trailing = head.split_before(op)
    if not op.results:
        return head, trailing
    region = head.parent_region()
    if region is None:
        raise PassFailedException(f"{op.name}: parent block is detached from its region")
    continuation = Block(arg_types=op.result_types)
    region.insert_block_before(continuation, trailing)
    rewriter.insert_op(JmpBranchOp([], trailing), InsertPoint.at_end(continuation))
    return head, continuation


class IfLowering(RewritePattern):
    """Lower ``q1_scf.if`` to a conditional branch over ``then``/``else`` blocks."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IfOp, rewriter: PatternRewriter) -> None:
        head, continuation = _split_continuation(op, rewriter)

        then_region = op.then_region
        then_entry = then_region.first_block
        then_last = then_region.last_block
        if then_entry is None or then_last is None:
            raise PassFailedException(
                f"{op.name}: then region must contain at least one block"
            )
        then_yield = then_last.last_op
        if then_yield is None:
            raise PassFailedException(f"{op.name}: then region must terminate with yield")
        if not isinstance(then_yield, YieldOp):
            raise PassFailedException(
                f"{op.name}: then region terminator must be q1_scf.yield"
            )
        rewriter.insert_op(
            JmpBranchOp(then_yield.operands, continuation),
            InsertPoint.at_end(then_last),
        )
        rewriter.erase_op(then_yield)

        if op.else_region.blocks:
            else_region = op.else_region
            else_entry = else_region.first_block
            else_last = else_region.last_block
            if else_entry is None or else_last is None:
                raise PassFailedException(
                    f"{op.name}: else region must contain at least one block"
                )
            else_yield = else_last.last_op
            if else_yield is None:
                raise PassFailedException(
                    f"{op.name}: else region must terminate with yield"
                )
            if not isinstance(else_yield, YieldOp):
                raise PassFailedException(
                    f"{op.name}: else region terminator must be q1_scf.yield"
                )
            rewriter.insert_op(
                JmpBranchOp(else_yield.operands, continuation),
                InsertPoint.at_end(else_last),
            )
            rewriter.erase_op(else_yield)
            rewriter.inline_region(else_region, BlockInsertPoint.before(continuation))
        else:
            else_entry = Block()
            rewriter.insert_op(
                JmpBranchOp([], continuation), InsertPoint.at_end(else_entry)
            )
            region = head.parent_region()
            if region is None:
                raise PassFailedException(
                    f"{op.name}: parent block is detached from its region"
                )
            region.insert_block_before(else_entry, continuation)

        rewriter.inline_region(then_region, BlockInsertPoint.before(continuation))

        rewriter.insert_op(
            _conditional_branch(
                op.predicate, op.predicate_args, then_entry, (), else_entry, ()
            ),
            InsertPoint.at_end(head),
        )
        rewriter.replace_op(op, [], continuation.args)


class WhileLowering(RewritePattern):
    """Lower ``q1_scf.while`` to a header, body, and exit CFG with the back-edge."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: WhileOp, rewriter: PatternRewriter) -> None:
        head, continuation = _split_continuation(op, rewriter)

        before_region = op.before_region
        header = before_region.first_block
        if header is None:
            raise PassFailedException(
                f"{op.name}: before region must contain a header block"
            )
        condition = header.last_op
        if condition is None:
            raise PassFailedException(
                f"{op.name}: before region must terminate with condition"
            )
        if not isinstance(condition, ConditionOp):
            raise PassFailedException(
                f"{op.name}: before region terminator must be q1_scf.condition"
            )

        after_region = op.after_region
        body_entry = after_region.first_block
        after_last = after_region.last_block
        if body_entry is None or after_last is None:
            raise PassFailedException(
                f"{op.name}: after region must contain at least one block"
            )
        body_yield = after_last.last_op
        if body_yield is None:
            raise PassFailedException(f"{op.name}: after region must terminate with yield")
        if not isinstance(body_yield, YieldOp):
            raise PassFailedException(
                f"{op.name}: after region terminator must be q1_scf.yield"
            )

        rewriter.inline_region(before_region, BlockInsertPoint.before(continuation))

        exit_stub = Block()
        rewriter.insert_op(
            JmpBranchOp(condition.forward_args, continuation),
            InsertPoint.at_end(exit_stub),
        )
        region = head.parent_region()
        if region is None:
            raise PassFailedException(
                f"{op.name}: parent block is detached from its region"
            )
        region.insert_block_before(exit_stub, continuation)

        rewriter.insert_op(
            JmpBranchOp(body_yield.operands, header),
            InsertPoint.at_end(after_last),
        )
        rewriter.erase_op(body_yield)
        rewriter.inline_region(after_region, BlockInsertPoint.before(continuation))

        rewriter.insert_op(
            _conditional_branch(
                condition.predicate,
                condition.predicate_args,
                body_entry,
                condition.forward_args,
                exit_stub,
                (),
            ),
            InsertPoint.at_end(header),
        )
        rewriter.erase_op(condition)

        rewriter.insert_op(JmpBranchOp(op.init_args, header), InsertPoint.at_end(head))
        rewriter.replace_op(op, [], continuation.args)


class ForLowering(RewritePattern):
    """Lower ``q1_scf.for`` to a self-looping body terminated by ``loop_branch``.

    The counted loop maps onto the native decrement-and-branch. The body runs at
    least once, so a well-formed ``q1_scf.for`` has a positive iteration count; a
    zero-trip loop is not expressible on the hardware and is expected to be removed
    upstream.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ForOp, rewriter: PatternRewriter) -> None:
        head, continuation = _split_continuation(op, rewriter)

        iter_count = op.iter_count
        if (
            isinstance(iter_count, OpResult)
            and isinstance(iter_count.op, MoveImmRdOp)
            and (iter_count.op.imm.data & INT_REGISTER_VALUE_MASK) == 0
        ):
            raise PassFailedException(
                f"{op.name}: statically zero iter_count cannot lower to q1_cf.loop_branch"
            )

        body = op.body.first_block
        if body is None:
            raise PassFailedException(f"{op.name}: body region must contain a block")
        body_yield = body.last_op
        if body_yield is None:
            raise PassFailedException(f"{op.name}: body region must terminate with yield")
        if not isinstance(body_yield, YieldOp):
            raise PassFailedException(
                f"{op.name}: body region terminator must be q1_scf.yield"
            )
        induction = body.args[0]
        carried = body_yield.operands

        rewriter.insert_op(
            LoopBranchOp(
                induction, [induction, *carried], list(carried), body, continuation
            ),
            InsertPoint.at_end(body),
        )
        rewriter.erase_op(body_yield)
        rewriter.inline_region(op.body, BlockInsertPoint.before(continuation))

        rewriter.insert_op(
            JmpBranchOp([op.iter_count, *op.iter_args], body),
            InsertPoint.at_end(head),
        )
        rewriter.replace_op(op, [], continuation.args)


class LowerQ1ScfToQ1CfPass(ModulePass):
    """Lower every ``q1_scf`` container in the module to a ``q1_cf`` block CFG."""

    name = "lower-q1-scf-to-q1-cf"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Run lowering in a fixed inside-out order across structured forms.

        Rewrites run as ``if -> while -> for`` in one greedy walk so nested containers
        dissolve from the smallest structured region toward enclosing loop forms.
        """
        PatternRewriteWalker(
            GreedyRewritePatternApplier([IfLowering(), WhileLowering(), ForLowering()])
        ).rewrite_module(op)
