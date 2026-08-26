# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Lower the builtin ``scf`` dialect into ``q1_scf``.

This pass converts standard MLIR structured control flow (:mod:`xdsl.dialects.scf`)
into the Q1-sequencer-specific structured control flow dialect ``q1_scf``.

Motivation
----------
The frontend importer produces ``scf.for`` loops to represent shot iteration and
similar counted patterns.  Before the ``q1_scf`` pipeline can lower those loops to
hardware, the generic ``scf`` containers must first be mapped onto their Q1
counterparts.

For loops
---------
``scf.for %i = %lb to %ub step %step iter_args(%a, ...) { ... scf.yield ... }``
maps onto ``q1_scf.for %n : q1.reg iter_args(%a, ...) { ... q1_scf.yield ... }``
where ``%n`` is loaded with the iteration count ``ceil((ub - lb) / step)`` by a
``q1.ir.move`` instruction inserted immediately before the loop.

The ``scf`` induction variable counts *up* from ``lb`` toward ``ub``; the ``q1_scf``
induction counter counts *down* from ``n`` toward zero.  When the loop body uses the
induction variable, arithmetic instructions are inserted at the top of the body to
reconstruct the ascending value ``lb + (count - counter) * step``.

Bounds that are not ``arith.constant`` integer literals, non-positive step values,
zero-trip counts, and loop-carried values whose type is not ``q1.reg`` are also
rejected.

Stubs
-----
``scf.if`` and ``scf.while`` are not yet supported.  The corresponding rewrite
patterns raise :class:`~xdsl.utils.exceptions.PassFailedException` immediately.

Reference: https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html
"""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.dialects.scf import (
    ForOp as ScfForOp,
    IfOp as ScfIfOp,
    WhileOp as ScfWhileOp,
    YieldOp as ScfYieldOp,
)
from xdsl.ir import OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.q1 import (
    AddRsImmRdOp,
    MoveImmRdOp,
    Muls16RsImmRdOp,
    NotRsRdOp,
)
from qat.experimental.dialect.q1.ir.imm_desc import SI16Imm, SU32Imm
from qat.experimental.dialect.q1.ir.reg_desc import IntRegisterType
from qat.experimental.dialect.q1_scf.ir.ops import ForOp, YieldOp


def _get_static_integer(value: SSAValue) -> int | None:
    """Return the integer value of a static ``arith.constant`` result, or ``None``.

    :returns: The integer constant, or ``None`` if not a static integer constant.
    """
    if not isinstance(value, OpResult):
        return None
    if not isinstance(value.op, ArithConstantOp):
        return None
    attr = value.op.value
    if not isinstance(attr, IntegerAttr):
        return None
    return int(attr.value.data)


def _remap_induction_uses(
    induction_var: SSAValue,
    countdown: SSAValue,
    lb: int,
    count: int,
    step: int,
    body_block,
    rewriter: PatternRewriter,
) -> None:
    """Replace uses of the ``scf.for`` induction variable with the ascending index.

    Inserts arithmetic at the top of *body_block* to compute
    ``lb + (count - countdown) * step`` from the ``q1_scf.for`` countdown arg.
    """
    # ~countdown + (count + 1)  ==  count - countdown  (two's complement)
    not_op = NotRsRdOp(countdown, IntRegisterType.unallocated())
    rewriter.insert_op(not_op, InsertPoint.at_start(body_block))

    if step == 1:
        ascending = AddRsImmRdOp(
            not_op.rd, SU32Imm(lb + count + 1), IntRegisterType.unallocated()
        )
        rewriter.insert_op(ascending, InsertPoint.after(not_op))
        induction_var.replace_all_uses_with(ascending.rd)
    else:
        count_minus_c = AddRsImmRdOp(
            not_op.rd, SU32Imm(count + 1), IntRegisterType.unallocated()
        )
        rewriter.insert_op(count_minus_c, InsertPoint.after(not_op))
        scaled = Muls16RsImmRdOp(
            count_minus_c.rd, SI16Imm(step), IntRegisterType.unallocated()
        )
        rewriter.insert_op(scaled, InsertPoint.after(count_minus_c))
        if lb == 0:
            induction_var.replace_all_uses_with(scaled.rd)
        else:
            shifted = AddRsImmRdOp(scaled.rd, SU32Imm(lb), IntRegisterType.unallocated())
            rewriter.insert_op(shifted, InsertPoint.after(scaled))
            induction_var.replace_all_uses_with(shifted.rd)


class ForLowering(RewritePattern):
    """Lower ``scf.for`` to ``q1_scf.for``.
    The iteration count ``ceil((ub - lb) / step)`` is materialised as a ``q1.ir.move``
    inserted immediately before the loop.  The body induction argument is retyped
    from ``index`` to ``!q1.reg``, and ``scf.yield`` is replaced by ``q1_scf.yield``.
    When the body uses the induction variable, arithmetic is inserted at the top of
    the body block to reconstruct the ascending index from the countdown counter.

    Example (``lb=0``, ``ub=10``, ``step=2``, ``count=5``; body uses ``%i``):

    .. code-block:: mlir

        // before
        %lb   = arith.constant 0 : index
        %ub   = arith.constant 10 : index
        %step = arith.constant 2 : index
        scf.for %i = %lb to %ub step %step {
          ... uses of %i ...
          scf.yield
        }

        // after
        %lb   = arith.constant 0 : index
        %ub   = arith.constant 10 : index
        %step = arith.constant 2 : index
        %n    = q1.ir.move () <{imm = #q1.su32_imm<5>}> : () -> !q1.reg
        q1_scf.for %n : !q1.reg {
        ^bb0(%counter: !q1.reg):
          %not_c  = q1.rr.not (%counter) : (!q1.reg) -> !q1.reg
          %diff   = q1.rir.add (%not_c) <{imm = #q1.su32_imm<6>}> : (!q1.reg) -> !q1.reg
          %i      = q1.rir.muls16 (%diff) <{imm = #q1.si16_imm<2>}> : (!q1.reg) -> !q1.reg
          ... uses of %i (now !q1.reg, ascending 0, 2, 4, 6, 8) ...
          q1_scf.yield
        }
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ScfForOp, rewriter: PatternRewriter) -> None:
        lb_int = _get_static_integer(op.lb)
        ub_int = _get_static_integer(op.ub)
        step_int = _get_static_integer(op.step)

        if lb_int is None:
            raise PassFailedException(
                f"{op.name}: lower bound is not a static integer constant"
            )
        if ub_int is None:
            raise PassFailedException(
                f"{op.name}: upper bound is not a static integer constant"
            )
        if step_int is None:
            raise PassFailedException(f"{op.name}: step is not a static integer constant")
        if step_int <= 0:
            raise PassFailedException(f"{op.name}: step must be positive, got {step_int}")

        # ceiling division: -((-diff) // step) avoids float and matches scf.for semantics
        count = -(-(ub_int - lb_int) // step_int)
        if count <= 0:
            raise PassFailedException(
                f"{op.name}: iteration count {count} is not positive;"
                " zero-trip loops cannot be represented by q1_scf.for"
            )

        for i, iter_arg in enumerate(op.iter_args):
            if not isinstance(iter_arg.type, IntRegisterType):
                raise PassFailedException(
                    f"{op.name}: iter_arg {i} has type {iter_arg.type!r};"
                    " only q1.reg loop-carried values are supported"
                )

        body_block = list(op.body.blocks)[0]
        induction_var = body_block.args[0]

        new_induction = body_block.insert_arg(IntRegisterType.unallocated(), 0)
        if any(True for _ in induction_var.uses):
            _remap_induction_uses(
                induction_var, new_induction, lb_int, count, step_int, body_block, rewriter
            )
        body_block.erase_arg(induction_var)

        body_yield = body_block.last_op
        if not isinstance(body_yield, ScfYieldOp):
            raise PassFailedException(
                f"{op.name}: body region terminator must be scf.yield, got"
                f" {body_yield.name if body_yield is not None else 'nothing'}"
            )
        yield_values = list(body_yield.operands)
        rewriter.erase_op(body_yield)
        rewriter.insert_op(YieldOp(*yield_values), InsertPoint.at_end(body_block))

        move_op = MoveImmRdOp(SU32Imm(count), IntRegisterType.unallocated())
        rewriter.insert_op(move_op, InsertPoint.before(op))

        # Detach body_block from op.body before passing it to ForOp; xDSL's
        # Region.__init__ raises if a block is already owned by another region.
        op.body.detach_block(body_block)
        new_for = ForOp(move_op.rd, list(op.iter_args), body_block)
        rewriter.replace_op(op, new_for)


class WhileLowering(RewritePattern):
    """Stub for ``scf.while`` → ``q1_scf.while`` lowering.

    Not yet implemented.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ScfWhileOp, rewriter: PatternRewriter) -> None:
        raise PassFailedException(
            f"{op.name}: lowering scf.while to q1_scf is not yet supported"
        )


class IfLowering(RewritePattern):
    """Stub for ``scf.if`` → ``q1_scf.if`` lowering.

    Not yet implemented.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ScfIfOp, rewriter: PatternRewriter) -> None:
        raise PassFailedException(
            f"{op.name}: lowering scf.if to q1_scf is not yet supported"
        )


class LowerScfToQ1ScfPass(ModulePass):
    """Lower every ``scf`` operation in the module to its ``q1_scf`` equivalent.

    Currently only ``scf.for`` is fully implemented.  ``scf.if`` and ``scf.while``
    raise :class:`~xdsl.utils.exceptions.PassFailedException` if encountered.
    """

    name = "lower-scf-to-q1-scf"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [ForLowering(), WhileLowering(), IfLowering()],
                dce_enabled=False,
            )
        ).rewrite_module(op)
