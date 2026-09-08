# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Pass and pipeline definitions for the Pulse-to-Q1 conversion."""

from dataclasses import dataclass, field
from math import isclose, isfinite

from numpy import prod
from xdsl.context import Context
from xdsl.dialects.arith import AddiOp, ConstantOp as ArithConstantOp, MuliOp
from xdsl.dialects.builtin import IndexType, IntAttr, ModuleOp
from xdsl.dialects.scf import ForOp
from xdsl.ir import SSAValue
from xdsl.irdl import IRDLOperation
from xdsl.passes import ModulePass, PassPipeline
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.rewriter import Rewriter
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.utils.exceptions import PassFailedException

# TODO: Migrate this lowering boundary to QbloxTargetDescription.
from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.experimental.backend.qblox.pre_emission_verification import (
    QbloxPreEmissionVerificationPass,
)
from qat.experimental.conversion.pulse_to_q1.hardware_binding import (
    QbloxHardwareBindingPass,
)
from qat.experimental.conversion.pulse_to_q1.pre_q1_ir import PreQ1AcquireOp
from qat.experimental.conversion.pulse_to_q1.rewrite_patterns import (
    create_legalisation_patterns,
    create_pulse_to_q1_lowering_patterns,
)
from qat.experimental.conversion.pulse_to_q1.sequence_outlining import Q1OutliningPass
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    PhaseSetOp,
    PhaseShiftOp,
    StartContinuousWaveformOp,
    WaitOp,
)
from qat.experimental.dialect.pulse.utils import (
    extract_frequency_hz,
    extract_phase_radians,
    extract_time_seconds,
)
from qat.experimental.dialect.q1.transforms.reg_alloc import (
    LinearScanRegisterAllocationPass,
)
from qat.experimental.dialect.q1_cf.transforms.linearise_q1_cf import LineariseQ1CfToQ1Pass
from qat.experimental.dialect.q1_scf.transforms.lower_scf import LowerScfToQ1ScfPass
from qat.experimental.dialect.q1_scf.transforms.lower_to_cf import LowerQ1ScfToQ1CfPass
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.passes.pass_ordering import OrderedPass, OrderedPassPipeline
from qat.experimental.system_data.canonical.schema import CanonicalSystemData

_TIME_ROUNDING_TOLERANCE_NS = 1e-3


@dataclass(frozen=True)
class Q1PulseValidationPass(OrderedPass, ModulePass):
    """Validate QBlox-specific pre-conditions on constant Pulse operands.

    Enforces hardware constraints that cannot be expressed as Pulse dialect invariants,
    ahead of legalisation and lowering. Dynamic waits, frame frequencies, continuous
    amplitudes, and phase operands are rejected because no Q1 lowering exists for them:
    Q1 phase instructions only accept an immediate, so a dynamic phase would require a
    real radians-to-NCO-steps runtime conversion that does not exist.

    The following constraints are enforced:

    * ``pulse.wait`` constant duration: finite, non-negative, and an integer
      number of nanoseconds.
    * ``pulse.create_frame`` constant frequency: finite.
    * ``pulse.phase_set`` and ``pulse.phase_shift`` constant phase: finite.
    """

    name = "q1-pulse-validation"
    target_data: QbloxTargetData = field(default=TARGET_DATA)

    def required_predecessors(self) -> frozenset[type[ModulePass]]:
        return frozenset({Q1OutliningPass})

    def runs_before(self) -> frozenset[type[ModulePass]]:
        return frozenset({Q1PulseLegalisationPass})

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for pulse_op in op.walk():
            if isinstance(pulse_op, WaitOp):
                self._validate_wait(pulse_op)
            elif isinstance(pulse_op, CreateFrameOp):
                self._validate_create_frame(pulse_op)
            elif isinstance(pulse_op, StartContinuousWaveformOp):
                self._validate_amplitude(pulse_op)
            elif isinstance(pulse_op, PhaseSetOp | PhaseShiftOp):
                self._validate_phase(pulse_op)

    def _validate_wait(self, op: WaitOp) -> None:
        if not isinstance(op.duration.owner, ConstantOp):
            raise PassFailedException("Dynamic pulse.wait duration is not supported.")
        seconds = extract_time_seconds(op)
        if not isfinite(seconds):
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
        if not isclose(ns_float, ns_int, abs_tol=_TIME_ROUNDING_TOLERANCE_NS, rel_tol=0):
            raise PassFailedException(
                "pulse.wait duration must map to integer nanoseconds within tolerance. "
                f"Got {ns_float} ns."
            )

    def _validate_create_frame(self, op: CreateFrameOp) -> None:
        if not isinstance(op.frequency.owner, ConstantOp):
            raise PassFailedException(
                "Dynamic pulse.create_frame frequency is not supported."
            )
        frequency_hz = extract_frequency_hz(op)
        if not isfinite(frequency_hz):
            raise PassFailedException(
                f"{op.name} frequency must be finite. Got {frequency_hz}."
            )

    def _validate_phase(self, op: PhaseSetOp | PhaseShiftOp) -> None:
        if not isinstance(op.phase.owner, ConstantOp):
            raise PassFailedException(f"Dynamic {op.name} phase is not supported.")
        radians = extract_phase_radians(op)
        if not isfinite(radians):
            raise PassFailedException(f"{op.name} phase must be finite. Got {radians}.")

    @staticmethod
    def _validate_amplitude(op: StartContinuousWaveformOp) -> None:
        if not isinstance(op.amplitude.owner, ConstantOp):
            raise PassFailedException(
                "Dynamic pulse.start_continuous_waveform amplitude is not supported."
            )
        amplitude = op.amplitude.owner.value
        if not isinstance(amplitude, AmplitudeAttr):
            raise PassFailedException(
                "pulse.start_continuous_waveform expects a pulse.amplitude constant."
            )
        value = amplitude.literal_value
        if not isfinite(value.real) or not isfinite(value.imag):
            raise PassFailedException(
                f"pulse.start_continuous_waveform amplitude must be finite. Got {value}."
            )


@dataclass(frozen=True)
class Q1PulseLegalisationPass(OrderedPass, ModulePass):
    """Apply Pulse phase legalisation before Pulse-to-Q1 lowering.

    This stage applies the legalisation pattern set to Pulse-level operands after
    validation. The pattern set is expected to grow over time as more Pulse operations
    acquire legalisation support.
    """

    name = "q1-pulse-legalisation"

    def required_predecessors(self) -> frozenset[type[ModulePass]]:
        return frozenset({Q1PulseValidationPass})

    def runs_before(self) -> frozenset[type[ModulePass]]:
        return frozenset({Q1PreAcquireTransformationPass})

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
class Q1PreAcquireTransformationPass(OrderedPass, ModulePass):
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

    def required_predecessors(self) -> frozenset[type[ModulePass]]:
        return frozenset({Q1PulseLegalisationPass})

    def runs_before(self) -> frozenset[type[ModulePass]]:
        return frozenset({QbloxHardwareBindingPass})

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Run the transformation over ``op`` in place.

        :param ctx: The xDSL context (unused; present for the pass interface).
        :param op: The module to transform.
        """
        _ = self._walk_op(
            op, AcquireAnalysisStack(for_op_number_repeats=[], for_op_indexes=[])
        )

    @staticmethod
    def _on_enter(op: ForOp, for_data: AcquireAnalysisStack) -> AcquireAnalysisStack:
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
                number_runs=IntAttr(int(prod(for_data.for_op_number_repeats))),
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
        if isinstance(op, ForOp):
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

        if isinstance(op, ForOp):
            for_data = self._on_exit(for_data)
        return for_data


@dataclass(frozen=True)
class PulseToQ1LoweringPass(OrderedPass, ModulePass):
    """Apply the Pulse-to-Q1 rewrite stage inside outlined sequences.

    ``Q1OutliningPass`` first isolates one logical sequence envelope for each
    frame partition. This pass then traverses those envelopes and applies the
    per-operation rewrite set that converts Pulse-level instructions into the
    flat Q1 instruction dialect.
    """

    name = "pulse-to-q1-lowering"
    target_data: QbloxTargetData = field(default=TARGET_DATA)

    def required_predecessors(self) -> frozenset[type[ModulePass]]:
        return frozenset({Q1PreAcquireTransformationPass, QbloxHardwareBindingPass})

    def runs_before(self) -> frozenset[type[ModulePass]]:
        return frozenset({LowerScfToQ1ScfPass})

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                create_pulse_to_q1_lowering_patterns(self.target_data)
            ),
            apply_recursively=False,
        ).rewrite_module(op)


@dataclass(frozen=True)
class BoundDeadFrameEliminationPass(OrderedPass, ModulePass):
    """Erase frame metadata after binding and lowering have consumed it.

    This deliberately runs after :class:`PulseToQ1LoweringPass` rather than marking
    ``pulse.create_frame`` globally pure. Frames identify physical channels during outlining
    and binding, including channels with no timed instructions, so generic Pulse
    canonicalization must preserve them until that information has been captured.
    """

    name = "bound-dead-frame-elimination"

    # TODO(COMPILER-1281): Replace this cleanup with general dead-frame elimination.
    def required_predecessors(self) -> frozenset[type[ModulePass]]:
        return frozenset({PulseToQ1LoweringPass})

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for sequence in (nested for nested in op.walk() if isinstance(nested, SequenceOp)):
            frames = [
                nested for nested in sequence.walk() if isinstance(nested, CreateFrameOp)
            ]
            for frame in frames:
                if frame.result.uses:
                    continue
                Rewriter.erase_op(frame)


def create_qblox_configured_q1_pipeline(
    canonical_data: CanonicalSystemData,
    target_data: QbloxTargetData = TARGET_DATA,
) -> PassPipeline:
    """Create the Pulse-to-emission-ready configured Qblox Q1 conversion pipeline.

    This is a Q1 conversion/configuration pipeline only: it assumes the input module has
    already been through Pulse-level preprocessing (see
    :meth:`~qat.experimental.dialect.pulse.transforms.pipeline.PulsePipelineManager.build_default_pipeline`).
    Callers that need both stages (for example
    :func:`qat.experimental.backend.qblox.codegen.compile_qblox_program`) are responsible
    for invoking Pulse preprocessing themselves before applying this pipeline.

    Binding runs after Pulse validation, legalisation, and acquisition preparation while
    ``pulse.create_frame`` still carries generator-selection metadata.
    :class:`~xdsl.transforms.reconcile_unrealized_casts.ReconcileUnrealizedCastsPass` runs
    after all lowering (including the arith integer constant to ``q1.ir.move`` rewrite
    applied by :class:`PulseToQ1LoweringPass`) has substituted the values that made the
    index-to-Q1-register bridging casts identity casts, and before register allocation,
    which requires the module to be free of ``builtin.unrealized_conversion_cast``.

    :param canonical_data: Canonical hardware data used for physical binding.
    :param target_data: Qblox target description used by conversion stages.
    :returns: Ordered pipeline ending in strict Qblox pre-emission verification.
    """

    # TODO(COMPILER-1440): Resolve target capabilities from a typed DLTI description attached
    # to the IR instead of passing QbloxTargetData separately through conversion stages.
    return OrderedPassPipeline(
        (
            Q1OutliningPass(target_data=target_data),
            Q1PulseValidationPass(target_data=target_data),
            Q1PulseLegalisationPass(),
            Q1PreAcquireTransformationPass(),
            QbloxHardwareBindingPass(canonical_data),
            PulseToQ1LoweringPass(target_data=target_data),
            BoundDeadFrameEliminationPass(),
            DeadCodeElimination(),
            LowerScfToQ1ScfPass(),
            LowerQ1ScfToQ1CfPass(),
            LineariseQ1CfToQ1Pass(),
            ReconcileUnrealizedCastsPass(),
            LinearScanRegisterAllocationPass(),
            QbloxPreEmissionVerificationPass(),
        )
    )
