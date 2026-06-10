# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Contains traits that are used to describe properties of operations in the pulse dialect,
and to apply canonicalization patterns to operations in the pulse dialect."""

from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait, OpTrait


class AdvancesTimeTrait(OpTrait):
    """A trait that signifies an operation advances time on the frame(s) it acts on.

    The time does not need to be known at compile time, and in that sense, can be runtime
    dynamic.
    """

    ...


class PulseTypesCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    """Applied to arithmetic binary operations on types in the pulse dialect that resemble
    floating point or complex numbers."""

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from qat.experimental.dialect.pulse.transforms.constants import (
            FoldConstantConstantOp,
            FoldMaxTimeOp,
        )

        return (FoldConstantConstantOp(), FoldMaxTimeOp())


class FrameCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    """Applies canonicalization to operations that act on frames.

    Including:

    * Phase shifts that are modulo 2pi equal to zero.
    * Waits that are equal to zero.
    """

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from qat.experimental.dialect.pulse.transforms.frame_no_op_elimination import (
            FoldZeroPhaseShiftOp,
            FoldZeroWaitOp,
        )

        return (FoldZeroPhaseShiftOp(), FoldZeroWaitOp())
