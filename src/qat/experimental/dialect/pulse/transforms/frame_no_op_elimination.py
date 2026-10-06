# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements canonicalization patterns that remove operations that act as a no-op.

This includes:
- Phase shifts that are modulo 2pi equal to zero.
- Waits that are equal to zero.
"""

from math import isclose, tau

from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern

from qat.experimental.dialect.pulse.ir import ConstantOp, PhaseShiftOp, WaitOp

_PHASE_COMPARISON_TOLERANCE = 1e-12
_WAIT_COMPARISON_TOLERANCE = 1  # Tolerance for comparing wait durations to zero is 1 ps.


class FoldZeroPhaseShiftOp(RewritePattern):
    """Finds :class:`PhaseShiftOp` with a constant operand that is modulo 2pi equal to zero
    and removes the operation."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PhaseShiftOp, rewriter: PatternRewriter):
        """Matches and applies the rewrite if an appropriate condition is found."""

        operand = op.phase.owner
        if not isinstance(operand, ConstantOp):
            return

        operand_value = operand.fold()
        if not operand_value:
            return

        literal_value = operand_value[0].literal_value
        modulo_value = literal_value % tau
        if not (
            isclose(modulo_value, 0.0, abs_tol=_PHASE_COMPARISON_TOLERANCE)
            or isclose(modulo_value, tau, abs_tol=_PHASE_COMPARISON_TOLERANCE)
        ):
            return

        rewriter.replace_op(op, [], (op.frame,))


class FoldZeroWaitOp(RewritePattern):
    """Finds :class:`WaitOp` with a constant operand that is equal to zero and removes the
    operation.

    This only applies to WaitOp instructions (not other duration-based operations). Wait
    operations with zero duration are no-ops and can be safely eliminated.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: WaitOp, rewriter: PatternRewriter):
        """Matches and applies the rewrite if an appropriate condition is found."""

        operand = op.duration.owner
        if not isinstance(operand, ConstantOp):
            return

        operand_value = operand.fold()
        if not operand_value:
            return

        literal_value = operand_value[0].literal_value
        if not isclose(literal_value, 0.0, abs_tol=_WAIT_COMPARISON_TOLERANCE):
            return

        rewriter.replace_op(op, [], (op.frame,))
