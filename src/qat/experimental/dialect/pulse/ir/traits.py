# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Contains traits that are used to describe properties of operations in the pulse dialect,
and to apply canonicalization patterns to operations in the pulse dialect."""

from xdsl.ir import Operation
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    OpTrait,
    SymbolTable,
    SymbolUserOpInterface,
)
from xdsl.utils.exceptions import VerifyException


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


class CallKernelOpUserOpInterface(SymbolUserOpInterface):
    """Symbol-user trait for call operations that target kernels.

    Inheriting :class:`SymbolUserOpInterface` registers the operation as a symbol user.
    This trait also verifies that the resolved callee is a :class:`KernelOp` and that call
    operand/result signatures match the referenced kernel function type.
    """

    def verify(self, op: Operation):
        """Verify symbol resolution and signature compatibility for a kernel call."""
        found_callee = SymbolTable.lookup_symbol(op, op.callee)
        if not found_callee:
            raise VerifyException(
                f"CallKernelOp must reference a KernelOp, but no symbol was found for "
                f"{op.callee}."
            )

        from .ops import KernelOp

        if not isinstance(found_callee, KernelOp):
            raise VerifyException(
                f"CallKernelOp must reference a KernelOp, but found {found_callee}"
            )

        if len(found_callee.function_type.inputs) != len(op.arguments):
            raise VerifyException(
                f"CallKernelOp must have the same number of arguments as the KernelOp it "
                f"references, but found {len(op.arguments)} arguments and "
                f"{len(found_callee.function_type.inputs)} inputs."
            )

        if len(found_callee.function_type.outputs) != len(op.results):
            raise VerifyException(
                f"CallKernelOp must have the same number of results as the KernelOp it "
                f"references, but found {len(op.results)} results and "
                f"{len(found_callee.function_type.outputs)} outputs."
            )

        for idx, (found_operand, operand) in enumerate(
            zip(found_callee.function_type.inputs, op.arguments.types, strict=False)
        ):
            if found_operand != operand:
                raise VerifyException(
                    f"CallKernelOp must have the same argument types as the KernelOp it "
                    f"references, but found argument {idx} with type {operand} and "
                    f"KernelOp input type {found_operand}."
                )

        for idx, (found_output, result) in enumerate(
            zip(found_callee.function_type.outputs, op.results.types, strict=False)
        ):
            if found_output != result:
                raise VerifyException(
                    f"CallKernelOp must have the same result types as the KernelOp it "
                    f"references, but found result {idx} with type {result} and KernelOp "
                    f"output type {found_output}."
                )
