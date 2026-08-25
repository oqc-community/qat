# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Pass and pipeline definitions for the Pulse-to-Q1 conversion."""

import math
from dataclasses import dataclass, field

import numpy as np
from xdsl.context import Context
from xdsl.dialects import scf
from xdsl.dialects.arith import AddiOp, ConstantOp as ArithConstantOp, MuliOp
from xdsl.dialects.builtin import IndexType, IntAttr, ModuleOp
from xdsl.ir import SSAValue
from xdsl.irdl import IRDLOperation
from xdsl.passes import ModulePass, PassPipeline
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import PassFailedException

from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.experimental.conversion.pulse_to_q1.pre_q1_ir import PreQ1AcquireOp
from qat.experimental.conversion.pulse_to_q1.rewrite_patterns import (
    create_legalisation_patterns,
    create_pulse_to_q1_lowering_patterns,
)
from qat.experimental.conversion.pulse_to_q1.sequence_outlining import Q1OutliningPass
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    ConstantOp,
    CreateFrameOp,
    PhaseSetOp,
    PhaseShiftOp,
    WaitOp,
)
from qat.experimental.dialect.pulse.utils import (
    extract_frequency_hz,
    extract_phase_radians,
    extract_time_seconds,
)
from qat.experimental.passes.pass_ordering import OrderedPassPipeline

_TIME_ROUNDING_TOLERANCE_NS = 1e-3


@dataclass(frozen=True)
class Q1PulseValidationPass(ModulePass):
    """Validate QBlox-specific pre-conditions on constant Pulse operands.

    Enforces hardware constraints that cannot be expressed as Pulse dialect
    invariants, ahead of the legalisation and lowering stages. Only constant
    operands are checked. Dynamic operands are deferred without error.

    The following constraints are enforced:

    * ``pulse.wait`` constant duration: finite, non-negative, and an integer
      number of nanoseconds.
    * ``pulse.create_frame`` constant frequency: finite.
    * ``pulse.phase_set`` and ``pulse.phase_shift`` constant phase: finite.
    """

    name = "q1-pulse-validation"
    target_data: QbloxTargetData = field(default=TARGET_DATA)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for pulse_op in op.walk():
            if isinstance(pulse_op, WaitOp):
                self._validate_wait(pulse_op)
            elif isinstance(pulse_op, CreateFrameOp):
                self._validate_create_frame(pulse_op)
            elif isinstance(pulse_op, PhaseSetOp | PhaseShiftOp):
                self._validate_phase(pulse_op)

    def _validate_wait(self, op: WaitOp) -> None:
        if not isinstance(op.duration.owner, ConstantOp):
            return
        seconds = extract_time_seconds(op)
        if not math.isfinite(seconds):
            raise PassFailedException(f"{op.name} time must be finite. Got {seconds}.")
        if seconds < 0:
            raise PassFailedException(
                f"{op.name} time must be non-negative. Got {seconds}."
            )

        ns_float = seconds * 1e9
        if 0 < ns_float < 1:
            raise PassFailedException(
                f"pulse.wait duration smaller than one nanosecond is illegal. Got {ns_float} ns."
            )

        ns_int = round(ns_float)
        if not math.isclose(
            ns_float, ns_int, abs_tol=_TIME_ROUNDING_TOLERANCE_NS, rel_tol=0
        ):
            raise PassFailedException(
                "pulse.wait duration must map to integer nanoseconds within tolerance. "
                f"Got {ns_float} ns."
            )

    def _validate_create_frame(self, op: CreateFrameOp) -> None:
        if not isinstance(op.frequency.owner, ConstantOp):
            return
        frequency_hz = extract_frequency_hz(op)
        if not math.isfinite(frequency_hz):
            raise PassFailedException(
                f"{op.name} frequency must be finite. Got {frequency_hz}."
            )

    def _validate_phase(self, op: PhaseSetOp | PhaseShiftOp) -> None:
        if not isinstance(op.phase.owner, ConstantOp):
            return
        radians = extract_phase_radians(op)
        if not math.isfinite(radians):
            raise PassFailedException(f"{op.name} phase must be finite. Got {radians}.")


@dataclass(frozen=True)
class Q1PulseLegalisationPass(ModulePass):
    """Apply Pulse phase legalisation before Pulse-to-Q1 lowering.

    This stage applies the legalisation pattern set to Pulse-level operands after
    validation. The pattern set is expected to grow over time as more Pulse operations
    acquire legalisation support.
    """

    name = "q1-pulse-legalisation"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(create_legalisation_patterns()),
            apply_recursively=False,
        ).rewrite_module(op)


@dataclass
class AcquireAnalysisStack:
    """Mutable stack state tracked while walking the enclosing loop nest.

    Both lists behave as stacks that are pushed on entry to an ``scf.for`` and popped on
    exit, so at any acquire they describe exactly the loops surrounding it.

    :ivar for_op_number_repeats: A stack tracking the number of repetitions of each
        enclosing ``scf.for`` loop.
    :ivar for_op_indexes: Induction-variable SSA values of the currently enclosing
        ``scf.for`` loops, aligned with ``for_op_number_repeats``.
    """

    for_op_number_repeats: list[int]
    for_op_indexes: list[SSAValue[IndexType]]


@dataclass(frozen=True)
class Q1PreAcquireTransformationPass(ModulePass):
    """Lower ``pulse.acquire`` to :class:`PreQ1AcquireOp` with QBlox acquisition context.

    QBlox acquires need a result store index (bin) and a repetition count, neither of which
    is expressed by ``pulse.acquire`` itself. This pass walks the module, tracking the
    enclosing ``scf.for`` nest, and for each ``pulse.acquire`` computes:

    * a ``store_idx`` from the loop induction variables, so each iteration writes to a
      distinct bin, and
    * ``number_runs`` from the product of the enclosing loop trip counts.

    It then replaces the ``pulse.acquire`` with an equivalent :class:`PreQ1AcquireOp`
    carrying that context, ready for the context-free lowering in
    :class:`RewritePreQ1AcquireOp`.

    Assumes there are no ``scf.while`` loops and that all ``scf.for`` bounds are constant.
    """

    name = "acquire-pre-q1-transformation"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Run the transformation over ``op`` in place.

        :param ctx: The xDSL context (unused; present for the pass interface).
        :param op: The module to transform.
        """
        _ = self._walk_op(
            op, AcquireAnalysisStack(for_op_number_repeats=[], for_op_indexes=[])
        )

    @staticmethod
    def _on_enter(op: scf.ForOp, for_data: AcquireAnalysisStack) -> AcquireAnalysisStack:
        """Push loop state when entering an ``scf.for``.

        :param op: The ``scf.for`` operation being entered.
        :param for_data: The analysis state to update.
        :returns: The updated analysis state.
        :raises PassFailedException: If the loop has non-constant bounds or a non-``index``
            induction variable.
        """
        if any(
            not isinstance(v_op.owner, ArithConstantOp) for v_op in [op.ub, op.lb, op.step]
        ):
            raise PassFailedException("Dynamic For loop bounds not currently supported.")

        # Constant bounds are guaranteed above, so the trip count is known statically.
        ub_int = op.ub.owner.value.value.data
        lb_int = op.lb.owner.value.value.data
        step_int = op.step.owner.value.value.data
        repetitions = int((ub_int - lb_int) / step_int)
        if repetitions < 0:
            raise PassFailedException("For loop has negative repeat count.")
        for_data.for_op_number_repeats.append(repetitions)

        index_ssa = op.body.block.args[0]
        if index_ssa.type != IndexType():
            raise PassFailedException("For loop index must be of IndexType.")
        for_data.for_op_indexes.append(index_ssa)
        return for_data

    @staticmethod
    def _on_exit(for_data: AcquireAnalysisStack) -> AcquireAnalysisStack:
        """Pop loop state when leaving an ``scf.for``.

        :param for_data: The analysis state to update.
        :returns: The updated analysis state.
        """
        del for_data.for_op_number_repeats[-1]
        del for_data.for_op_indexes[-1]
        return for_data

    @staticmethod
    def _generate_acquire_ops(
        acquire_op: AcquireOp, for_data: AcquireAnalysisStack
    ) -> list[IRDLOperation]:
        """Build the replacement ops for a single ``pulse.acquire``.

        Derives the acquisition's store index (bin) from the enclosing loop induction
        variables and its repetition count from the loop trip counts, emitting any helper
        ops needed to materialise the store index followed by the :class:`PreQ1AcquireOp`
        itself.

        :param acquire_op: The ``pulse.acquire`` op being lowered.
        :param for_data: The loop-nest analysis state describing the enclosing loops.
        :returns: The ordered replacement ops, ending with the ``PreQ1AcquireOp``.
        """
        new_ops: list[IRDLOperation] = []

        # Derive the per-iteration store index from the enclosing loops.
        if len(for_data.for_op_indexes) == 1:
            # Single loop: the induction variable is the bin index.
            store_idx = for_data.for_op_indexes[-1]
        elif len(for_data.for_op_indexes) > 1:
            # Flatten the enclosing induction variables into a single row-major bin
            # index, with the outermost loop most significant.
            #
            # For loops with trip counts r_0 (outermost) .. r_{m-1} (innermost) and
            # induction variables i_0 .. i_{m-1}, the flattened store index is:
            #     idx = i_0 * (r_1 * r_2 * ... * r_{m-1})
            #         + i_1 * (r_2 * ... * r_{m-1})
            #         + ...
            #         + i_{m-2} * r_{m-1}
            #         + i_{m-1}
            # evaluated via:
            #     acc = i_0
            #     for j in 1 .. m-1:  acc = acc * r_j + i_j

            index_acc = for_data.for_op_indexes[0]
            for index_j, repeats_j in zip(
                for_data.for_op_indexes[1:],
                for_data.for_op_number_repeats[1:],
                strict=True,
            ):
                repeats_op = ArithConstantOp.from_int_and_width(repeats_j, IndexType())
                scaled_op = MuliOp(index_acc, repeats_op.result)
                sum_op = AddiOp(scaled_op.result, index_j)
                new_ops.extend([repeats_op, scaled_op, sum_op])
                index_acc = sum_op.result
            store_idx = index_acc
        else:
            # No enclosing loop: a single acquisition into bin 0.
            const_index = ArithConstantOp.from_int_and_width(0, IndexType())
            new_ops.append(const_index)
            store_idx = const_index.result

        new_ops.append(
            PreQ1AcquireOp(
                frame=acquire_op.frame,
                duration=acquire_op.duration,
                store_idx=store_idx,
                number_runs=IntAttr(int(np.prod(for_data.for_op_number_repeats))),
                weights=acquire_op.weights,
                label=acquire_op.label,
            )
        )
        return new_ops

    def _walk_op(
        self, op: IRDLOperation, for_data: AcquireAnalysisStack
    ) -> AcquireAnalysisStack:
        """Recursively walk ``op`` and rewrite enclosed ``pulse.acquire`` ops.

        :param op: The operation to walk into.
        :param for_data: The loop-nest analysis state, maintained across the walk.
        :returns: The analysis state after visiting ``op`` and its children.
        """
        if isinstance(op, scf.ForOp):
            for_data = self._on_enter(op, for_data)

        for region in op.regions:
            for block in region.blocks:
                for child_op in block.ops:
                    if isinstance(child_op, AcquireOp):
                        new_ops = self._generate_acquire_ops(child_op, for_data)
                        Rewriter.replace_op(
                            child_op,
                            new_ops,
                        )
                    else:
                        self._walk_op(child_op, for_data)

        if isinstance(op, scf.ForOp):
            for_data = self._on_exit(for_data)
        return for_data


@dataclass(frozen=True)
class PulseToQ1LoweringPass(ModulePass):
    """Apply the Pulse-to-Q1 rewrite stage inside outlined sequences.

    ``Q1OutliningPass`` first isolates one logical sequence envelope for each
    frame partition. This pass then traverses those envelopes and applies the
    per-operation rewrite set that converts Pulse-level instructions into the
    flat Q1 instruction dialect.
    """

    name = "pulse-to-q1-lowering"
    target_data: QbloxTargetData = field(default=TARGET_DATA)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                create_pulse_to_q1_lowering_patterns(self.target_data)
            ),
            apply_recursively=False,
        ).rewrite_module(op)


def create_default_pulse_to_q1_pipeline(
    target_data: QbloxTargetData = TARGET_DATA,
) -> PassPipeline:
    """Create the default pass pipeline for Pulse-to-Q1 conversion.

    The pipeline has five stages. ``Q1OutliningPass`` partitions the Pulse program
    into per-frame ``q1_sequence.sequence`` envelopes. ``Q1PulseValidationPass``
    enforces QBlox-specific constant operand constraints. ``Q1PulseLegalisationPass``
    canonicalises phase operands in the Pulse dialect. ``Q1PreAcquireTransformationPass``
    transforms acquires into a pre-lowering form that legalises the instruction ready
    for lowering. ``PulseToQ1LoweringPass`` completes the conversion within those
    sequences.

    :param target_data: QBlox target description passed to outlining, validation,
        and lowering stages.
    :returns: Pass pipeline for the default Pulse-to-Q1 conversion flow.
    """

    return OrderedPassPipeline(
        (
            Q1OutliningPass(target_data=target_data),
            Q1PulseValidationPass(target_data=target_data),
            Q1PulseLegalisationPass(),
            Q1PreAcquireTransformationPass(),
            PulseToQ1LoweringPass(target_data=target_data),
        )
    )
