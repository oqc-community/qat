# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Rewrite patterns for the Pulse-to-Q1 phase legalisation and lowering stages."""

import hashlib
from collections import defaultdict
from collections.abc import Callable
from math import ceil

import numpy as np
from xdsl.context import Context
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Operation
from xdsl.irdl import IRDLOperation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException, VerifyException

from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.experimental.conversion.pulse_to_q1.phase import PhaseLegalisation, PhaseLowering
from qat.experimental.conversion.pulse_to_q1.pre_q1_ir import PreQ1AcquireOp
from qat.experimental.dialect.pulse.ir import (
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    SampledWaveformAttr,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SynchronizeOp,
    WaitOp,
    WeightsAttr,
)
from qat.experimental.dialect.pulse.ir.ops import extract_constant_scalar
from qat.experimental.dialect.pulse.units import TIME_UNIT_EXPONENTS, TimeUnits
from qat.experimental.dialect.pulse.utils import require_constant_operand
from qat.experimental.dialect.q1 import (
    AcquireWeightedImmRsRsRsImmOp,
    IntRegisterType,
    MoveImmRdOp,
    PlayImmImmImmOp,
    SetAwgOffsImmImmOp,
    SI16Imm,
    UI10Imm,
    WaitImmOp,
)
from qat.experimental.dialect.q1.ir.attrs import DebugInfoAttr, ProvenanceInfoAttr
from qat.experimental.dialect.q1.ir.imm_desc import DurationImm, SU32Imm, UI5Imm
from qat.experimental.dialect.q1.ir.ops import AcquireImmRsImmOp
from qat.experimental.dialect.q1_sequence import SequenceOp, find_enclosing_sequence
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    make_acquisition,
    make_waveform,
    make_weight,
)


class LowerArithIntegerConstantToMoveOp(ModulePass, RewritePattern):
    """Lower ``arith.constant`` with an integer or index type to ``q1.ir.move``.

    An ``arith.constant`` whose result type is
    :class:`~xdsl.dialects.builtin.IntegerType` or
    :class:`~xdsl.dialects.builtin.IndexType` is replaced by a ``q1.ir.move``
    instruction that loads the same immediate value into a virtual register.
    Constants whose result type is not an integer or index (e.g. floating-point)
    are left unchanged.

    The immediate must fit in the 32-bit signed-or-unsigned range accepted by
    ``q1.ir.move`` (``SU32Imm``: ``[-2**31, 2**32 - 1]``).  Values outside that
    range raise :class:`~xdsl.utils.exceptions.PassFailedException`.
    """

    name = "lower-arith-integer-constant-to-move"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ArithConstantOp, rewriter: PatternRewriter) -> None:
        if not isinstance(op.result.type, IntegerType | IndexType):
            return
        if not isinstance(op.value, IntegerAttr):
            return
        value = op.value.value.data
        try:
            imm = SU32Imm(value)
        except VerifyException as exc:
            raise PassFailedException(
                f"arith.constant value {value} is out of range for q1.ir.move "
                f"(SU32Imm supports [{SU32Imm._MIN}, {SU32Imm._MAX}])"
            ) from exc
        rewriter.replace_op(op, MoveImmRdOp(imm, IntRegisterType.unallocated()))

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(self, apply_recursively=False).rewrite_module(op)


def _register_waveform(
    sequence_op: SequenceOp,
    samples: list[float],
    index: int,
    name: str,
) -> None:
    """Append a Q1 waveform-table entry to a sequence.

    Duplicate waveforms are assumed to have been folded and de-duplicated upstream, so this
    simply appends a new entry without scanning the existing table for matching samples.

    :param sequence_op: The sequence whose waveform table is extended in place.
    :param samples: The float samples for the new waveform entry.
    :param index: The waveform-table index to assign to the new entry.
    :param name: The name assigned to the new waveform entry.
    """
    q1_waveform = make_waveform(name, index, samples)
    sequence_op.waveforms = ArrayAttr(list(sequence_op.waveforms.data) + [q1_waveform])


def _get_enclosing_port(op: Operation) -> str | None:
    """Return the channel port token of the ``q1_sequence.sequence`` that contains *op*.

    After outlining, pulse ops live inside a ``SequenceOp`` body region. Walking up
    through the block → region → parent-op chain retrieves the sequence's ``sym_name``,
    which is the normalised channel token (e.g. ``"q0_drive"``).

    :param op: The operation whose enclosing sequence is to be found.
    :returns: The ``sym_name`` data string, or ``None`` if *op* is not inside a
        ``SequenceOp``.
    """
    region = op.parent_region()
    if region is None:
        return None
    parent = region.parent
    if not isinstance(parent, SequenceOp):
        return None
    return parent.sym_name.data


def _make_debug_info(op: IRDLOperation) -> DebugInfoAttr | None:
    """Build a :class:`~qat.experimental.dialect.q1.ir.attrs.DebugInfoAttr` for *op*.

    Combines the pulse dialect op name (``op.name``) with the physical channel token
    returned by :func:`_get_enclosing_port`.  Returns ``None`` when *op* is not inside
    a ``SequenceOp`` (e.g. during isolated unit tests).

    :param op: The pulse op being lowered.
    :returns: A ``DebugInfoAttr`` recording the source op and port, or ``None``.
    """
    port = _get_enclosing_port(op)
    if port is None:
        return None
    return ProvenanceInfoAttr(source_op=op.name, port=port)


class RewriteCreateFrameOp(RewritePattern):
    """Skeleton for frequency initialisation from ``pulse.create_frame``.

    The intended lowering emits ``q1.set_freq`` + ``q1.upd_param`` using the
    frame's NCO intermediate frequency. ``CreateFrameOp.frequency`` carries
    the total carrier frequency. Extracting the IF requires the LO frequency,
    which is hardware-model information unavailable at this pipeline stage.
    A prior legalisation pass must decompose ``carrier = LO + IF`` and rewrite
    the operand to the IF before this pattern can fire safely.

    TODO(COMPILER-1386): Implement once the carrier-to-IF legalisation pass is in place.
    """

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CreateFrameOp, _rewriter: PatternRewriter) -> None:
        # TODO(COMPILER-1386): Emit set_freq + upd_param once IF is available.
        return


class RewriteSynchronizeOp(RewritePattern):
    """Skeleton for COMPILER-1344 synchronize macro-expansion."""

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SynchronizeOp, _rewriter: PatternRewriter) -> None:
        # TODO(COMPILER-1343): Replace pulse.sync with Q1 macro-expansion.
        return


class RewriteWaitOp(RewritePattern):
    """Lower ``pulse.wait`` to one or more ``q1.wait`` instructions.

    The requested duration is converted from seconds to nanoseconds and aligned up to
    the sequencer grid time. Durations that exceed the maximum wait immediate are split
    into a chain of ``q1.wait`` instructions whose durations sum to the requested value.
    The frame carried by ``pulse.wait`` is forwarded to downstream operations.

    Register-driven durations that do not fold to a compile-time constant are left
    untouched; those are handled elsewhere and are out of scope for this lowering.
    """

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: WaitOp, rewriter: PatternRewriter) -> None:
        duration_s = extract_constant_scalar(op.duration)
        if duration_s is None:
            return

        grid_time = self.target_data.CONTROL_SEQUENCER_DATA.grid_time
        max_wait_time = self.target_data.Q1ASM_DATA.max_wait_time

        total_ns = int(
            ceil(duration_s * self.target_data.CONTROL_SEQUENCER_DATA.sample_rate)
        )
        remainder = total_ns % grid_time
        if remainder:
            total_ns += grid_time - remainder

        # Note: dosen't have a breakpoint to turn unrolling to
        # hardware-based loops for now.
        debug_info = _make_debug_info(op)
        wait_ops: list[Operation] = []
        while total_ns > max_wait_time:
            wait_ops.append(
                WaitImmOp(DurationImm(max_wait_time)).with_debug_info(debug_info)
            )
            total_ns -= max_wait_time
        if total_ns > 0:
            wait_ops.append(WaitImmOp(DurationImm(total_ns)).with_debug_info(debug_info))

        rewriter.replace_matched_op(wait_ops, new_results=[op.frame])


class RewritePhaseSetOp(RewritePattern):
    """Match ``pulse.phase_set`` and delegate stage policy to ``rewrite_callable``.

    The same matcher is used in legalisation and lowering. The callable decides whether the
    rewrite remains in Pulse or emits Q1 instructions.
    """

    def __init__(
        self,
        target_data: QbloxTargetData,
        rewrite_callable: Callable,
    ) -> None:
        self.target_data = target_data
        self.rewrite_callable = rewrite_callable

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PhaseSetOp, rewriter: PatternRewriter) -> None:
        self.rewrite_callable(op, rewriter, self.target_data, _make_debug_info(op))


class RewritePhaseShiftOp(RewritePattern):
    """Match ``pulse.phase_shift`` and delegate stage policy to ``rewrite_callable``.

    Mirrors ``RewritePhaseSetOp`` in structure.
    """

    def __init__(
        self,
        target_data: QbloxTargetData,
        rewrite_callable: Callable,
    ) -> None:
        self.target_data = target_data
        self.rewrite_callable = rewrite_callable

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PhaseShiftOp, rewriter: PatternRewriter) -> None:
        self.rewrite_callable(op, rewriter, self.target_data, _make_debug_info(op))


class RewritePulseOp(RewritePattern):
    """Lowers a ``pulse.pulse`` op to a Q1 ``play`` instruction.

    The pattern resolves the pulse's sampled waveform, registers its real (I) and
    imaginary (Q) sample vectors in the enclosing sequence's waveform table, and
    replaces the ``pulse.pulse`` op with a ``PlayImmImmImmOp`` that references the
    two table indices and the waveform duration. The now-unused waveform
    ``ConstantOp[SampledWaveformAttr]`` is erased once no other op consumes it.

    Distinct sampled waveforms are assumed to have been folded and de-duplicated
    upstream, so identical waveforms share a single ``ConstantOp``. Each such op is
    therefore registered in the waveform table only once per sequence; subsequent
    pulses referencing the same op reuse the cached table indices instead of
    re-comparing sample arrays.
    """

    def __init__(self, target_data: QbloxTargetData) -> None:
        """Initialise the pattern.

        :param target_data: The QBlox target description used during lowering.
        """
        self.target_data = target_data
        self.sequence_to_waveform_to_index_map: dict[
            SequenceOp, dict[ConstantOp, tuple[int, int]]
        ] = defaultdict(dict)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PulseOp, rewriter: PatternRewriter) -> None:
        """Rewrite a single ``pulse.pulse`` op into a Q1 ``play`` instruction.

        :param op: The ``pulse.pulse`` op to lower.
        :param rewriter: The pattern rewriter used to mutate the IR.
        :raises PassFailedException: If the pulse's waveform is not a ``ConstantOp``
            holding a ``SampledWaveformAttr``, or if the pulse duration is outside the Q1
            hardware limits.
        """
        pulse_op = op
        waveform_op = pulse_op.waveform.owner
        if not isinstance(waveform_op, ConstantOp) or not isinstance(
            waveform_op.value, SampledWaveformAttr
        ):
            raise PassFailedException(
                "Can only handle pulses that point to a SampledWaveformAttr, "
                f"got {waveform_op}"
            )

        sequence_op = find_enclosing_sequence(pulse_op)
        waveform_to_index_map = self.sequence_to_waveform_to_index_map[sequence_op]

        if waveform_op not in waveform_to_index_map:
            samples = waveform_op.value.literal_value
            name = f"waveform_{len(waveform_to_index_map)}"

            i_index = len(sequence_op.waveforms.data)
            _register_waveform(sequence_op, samples.real.tolist(), i_index, name + "_I")
            q_index = len(sequence_op.waveforms.data)
            _register_waveform(sequence_op, samples.imag.tolist(), q_index, name + "_Q")

            waveform_to_index_map[waveform_op] = (i_index, q_index)
        else:
            i_index, q_index = waveform_to_index_map[waveform_op]

        # Convert duration to ns for Q1; each sample is assumed to be 1 ns, so we don't need
        # to worry about rounding error.
        duration_ns = round(waveform_op.value.width.value_in_unit(TimeUnits.NANOSECOND))
        min_duration = self.target_data.CONTROL_SEQUENCER_DATA.grid_time

        # TODO (COMPILER-1389): Remove this validation in favour of a dedicated pulse-level
        # pass
        if duration_ns < min_duration:
            raise PassFailedException(
                f"Pulse duration {duration_ns} ns is below minimum {min_duration} ns."
            )

        # Fails if pulse is too long
        max_duration = self.target_data.Q1ASM_DATA.max_wait_time
        if duration_ns > max_duration:
            raise PassFailedException(
                f"Pulse duration {duration_ns} ns is above maximum {max_duration} ns."
            )

        q1_pulse = PlayImmImmImmOp(
            UI10Imm(i_index),
            UI10Imm(q_index),
            DurationImm(duration_ns),
        ).with_debug_info(_make_debug_info(op))

        rewriter.replace_op(pulse_op, [q1_pulse], new_results=[pulse_op.frame])
        if not waveform_op.result.uses:
            rewriter.erase_op(waveform_op)


class RewriteStartContinuousWaveformOp(RewritePattern):
    """Lowers a ``pulse.start_continuous_waveform`` op to a Q1 AWG offset.

    A continuous waveform is emitted on Q1 hardware by latching a constant AWG
    offset on both output paths. This pattern replaces the pulse op with a
    ``SetAwgOffsImmImmOp`` carrying the I and Q offsets scaled to the DAC range.
    Assumes Square waves have been legalised to `StartContinuousWaveformOp`,
    `Delay`, and `StopContinuousWaveformOp`.
    """

    def __init__(self, target_data: QbloxTargetData) -> None:
        """Initialise the pattern.

        :param target_data: The QBlox target description used during lowering.
        """
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: StartContinuousWaveformOp, rewriter: PatternRewriter
    ) -> None:
        """Rewrite the op into a ``SetAwgOffsImmImmOp``.

        :param op: The ``pulse.start_continuous_waveform`` op to lower.
        :param rewriter: The pattern rewriter used to mutate the IR.
        :raises PassFailedException: If the op's amplitude is not a ``ConstantOp``
            holding an ``AmplitudeAttr``.
        """
        amplitude_op = op.amplitude.owner
        if not isinstance(amplitude_op, ConstantOp) or not isinstance(
            amplitude_op.value, AmplitudeAttr
        ):
            raise PassFailedException(
                "Can only handle continuous waveforms with a constant AmplitudeAttr, "
                f"got {amplitude_op}"
            )

        amplitude = amplitude_op.value.literal_value
        max_offset = self.target_data.Q1ASM_DATA.max_offset
        q1_start = SetAwgOffsImmImmOp(
            SI16Imm(int(amplitude.real * max_offset)),
            SI16Imm(int(amplitude.imag * max_offset)),
        ).with_debug_info(_make_debug_info(op))
        rewriter.replace_op(op, [q1_start], new_results=[op.frame])
        if not amplitude_op.result.uses:
            rewriter.erase_op(amplitude_op)


class RewriteStopContinuousWaveformOp(RewritePattern):
    """Lowers a ``pulse.stop_continuous_waveform`` op to a Q1 AWG offset reset.

    Stopping a continuous waveform corresponds to clearing the latched AWG offset
    on both output paths, emitted as a ``SetAwgOffsImmImmOp`` with zero offsets.
    Assumes Square waves have been legalised to `StartContinuousWaveformOp`,
    `Delay`, and `StopContinuousWaveformOp`.
    """

    def __init__(self, target_data: QbloxTargetData) -> None:
        """Initialise the pattern.

        :param target_data: The QBlox target description used during lowering.
        """
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: StopContinuousWaveformOp, rewriter: PatternRewriter
    ) -> None:
        """Rewrite the op into a zero-offset ``SetAwgOffsImmImmOp``.

        :param op: The ``pulse.stop_continuous_waveform`` op to lower.
        :param rewriter: The pattern rewriter used to mutate the IR.
        """
        q1_stop = SetAwgOffsImmImmOp(
            SI16Imm(0),
            SI16Imm(0),
        ).with_debug_info(_make_debug_info(op))
        rewriter.replace_op(op, [q1_stop], new_results=[op.frame])


class RewritePreQ1AcquireOp(RewritePattern):
    """Lower :class:`PreQ1AcquireOp` to a Q1 acquire instruction.

    * :class:`AcquireImmRsImmOp` when no integration weights are present.
    * :class:`AcquireWeightedImmRsRsRsImmOp` when a :class:`WeightsAttr` is present.

    The acquisition, and any integration weights, are registered on the enclosing
    :class:`SequenceOp` and assigned hardware indices in encounter order. Q1 requires the
    bin and weight indices to be supplied in registers, so index values are materialised as
    ``builtin.unrealized_conversion_cast`` results typed as unallocated integer registers;
    a later register-allocation pass assigns concrete Q1 GPRs.
    """

    def __init__(self, target_data: QbloxTargetData) -> None:
        self.target_data = target_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PreQ1AcquireOp, rewriter: PatternRewriter) -> None:
        """Lower a single ``pre_q1_pulse.acquire`` op to Q1 acquire instructions.

        :param op: The ``pre_q1_pulse.acquire`` op to lower.
        :param rewriter: The pattern rewriter used to mutate the IR.
        :raises NotImplementedError: If the acquisition result is consumed downstream;
            lowering of ``pulse.acquire`` result consumers is not yet supported.
        """
        # TODO: Support lowering of pulse.acquire acquisition_result consumers
        # (e.g. pulse.integrate). Once supported, replace the guard below with proper
        # lowering logic. Post COMPILER-1369 work.
        if op.acquisition_result.uses:
            raise NotImplementedError(
                "pre_q1_pulse.acquire acquisition_result consumers are not yet supported"
            )
        debug_info = _make_debug_info(op)

        # Ops that replace ``op``, emitted in order: index casts first, then the acquire.
        new_ops: list[Operation] = []
        sequencer = find_enclosing_sequence(op)

        current_no_acquires = len(sequencer.acquisitions)

        acq_name = (
            op.label.data
            if op.label is not None
            else op.frame.type.port.data.replace("/", "_") + f"_{current_no_acquires}"
        )

        num_bins = op.number_runs.data
        acq_idx = self._register_acquisition(sequencer, acq_name, num_bins)

        # The store index is an ``index``-typed SSA value computed upstream. Q1 acquires
        # take the bin in a register, so cast it to an unallocated integer register and let
        # register allocation assign a concrete GPR later.
        cast_start_idx_op = UnrealizedConversionCastOp.get(
            [op.store_idx],
            [IntRegisterType.unallocated()],
        )
        new_ops.append(cast_start_idx_op)
        start_idx = cast_start_idx_op.results[0]

        duration_ns = self._get_ns_duration(op)
        # TODO(COMPILER-1349): Write ``duration_ns`` as the integration length into the
        # ``SequencerDataAttr`` on the enclosing ``SequenceOp`` once that attribute is
        # defined (pending the physical Qblox system data PR).
        if isinstance(weights_attr := op.weights, WeightsAttr):
            weights_data = weights_attr.weights.data
            weight_index_i = self._register_weight(sequencer, weights_data.real)
            weight_index_q = self._register_weight(sequencer, weights_data.imag)

            # QBloxs forces use of resister which seems wasteful.
            # Change static indexs to register values
            casted_weight_ops = []
            for weight_index in [weight_index_i, weight_index_q]:
                const_weight_index = ArithConstantOp.from_int_and_width(
                    weight_index, IndexType()
                )
                cast_const_weight_index_op = UnrealizedConversionCastOp.get(
                    [const_weight_index.result],
                    [IntRegisterType.unallocated()],
                )
                new_ops.extend([const_weight_index, cast_const_weight_index_op])
                casted_weight_ops.append(cast_const_weight_index_op)

            new_acquire_op = AcquireWeightedImmRsRsRsImmOp(
                UI5Imm(acq_idx),
                start_idx,
                casted_weight_ops[0].results[0],
                casted_weight_ops[1].results[0],
                DurationImm(duration_ns),
            ).with_debug_info(debug_info)
        else:
            new_acquire_op = AcquireImmRsImmOp(
                UI5Imm(acq_idx), start_idx, DurationImm(duration_ns)
            ).with_debug_info(debug_info)
        new_ops.append(new_acquire_op)
        rewriter.replace_op(op, new_ops, new_results=[op.frame, None])
        return

    def _get_ns_duration(self, op: PreQ1AcquireOp) -> int:
        """Get the duration of the acquisition in nanoseconds.

        TODO: Change this to ps as apart of COMPILER-1388.

        :param op: The ``pre_q1_pulse.acquire`` operation to extract from.
        :returns: Duration in nanoseconds as an integer.
        :raises PassFailedException: If the duration operand is not a constant or its
            attribute is not a ``TimeAttr``.
        """
        const = require_constant_operand(op.name, "duration", op.duration)
        time_attr = const.fold()[0]
        unit = time_attr.unit.data
        value = time_attr.value.data
        ns = value * 10 ** (
            TIME_UNIT_EXPONENTS[unit] - TIME_UNIT_EXPONENTS[TimeUnits.NANOSECOND]
        )
        return round(ns)

    def _register_weight(self, sequence_op: SequenceOp, weights_data: np.ndarray) -> int:
        """Register a new weight in the target data and return its index.

        :param sequence_op: The enclosing ``SequenceOp`` to register the weight in.
        :param weights_data: Weight coefficients to register.
        :returns: Index of the registered weight.
        """
        weight_list = weights_data.tolist()
        name = hashlib.md5(str(weight_list).encode(), usedforsecurity=False).hexdigest()[:8]
        for i, entry in enumerate(sequence_op.weights):
            if entry.weight_name.data == name:
                return i
        weight_index = len(sequence_op.weights)
        weight_attr = make_weight(name, weight_index, weight_list)
        sequence_op.properties["weights"] = ArrayAttr([*sequence_op.weights, weight_attr])
        return weight_index

    def _register_acquisition(
        self, sequence_op: SequenceOp, name: str, num_bins: int
    ) -> int:
        """Register a new acquisition in the target data and return its index.

        :param sequence_op: The enclosing ``SequenceOp`` to register the acquisition in.
        :param name: Unique name for the acquisition entry.
        :param num_bins: Number of bins to allocate for the acquisition.
        :returns: Index of the registered acquisition.
        """
        if name in (entry.acquisition_name.data for entry in sequence_op.acquisitions):
            raise ValueError(
                f"Acquisition name '{name}' already exists in sequence '{sequence_op.channel_id.data}'"
            )
        acq_index = len(sequence_op.acquisitions)
        acq_attr = make_acquisition(name, acq_index, num_bins)
        sequence_op.properties["acquisitions"] = ArrayAttr(
            [*sequence_op.acquisitions, acq_attr]
        )
        return acq_index


def create_legalisation_patterns() -> tuple[RewritePattern, ...]:
    """Create the rewrite set used by the legalisation stage.

    Canonicalises ``pulse.phase_set`` and ``pulse.phase_shift`` operands inside
    the Pulse dialect. New legalisation behaviour can be added here without
    touching the lowering factory.

    :returns: Ordered pattern tuple for the legalisation pass.
    """
    canonicalise = PhaseLegalisation()
    return (
        RewritePhaseSetOp(
            TARGET_DATA,
            rewrite_callable=lambda op, rewriter, _target_data, _debug_info=None: (
                canonicalise(op, rewriter)
            ),
        ),
        RewritePhaseShiftOp(
            TARGET_DATA,
            rewrite_callable=lambda op, rewriter, _target_data, _debug_info=None: (
                canonicalise(op, rewriter)
            ),
        ),
    )


def create_pulse_to_q1_lowering_patterns(
    target_data: QbloxTargetData | None = None,
) -> tuple[RewritePattern, ...]:
    """Create the rewrite set used by the lowering stage.

    Phase entries are configured with ``PhaseLowering``. ``RewritePreQ1AcquireOp`` fully
    lowers ``pre_q1_pulse.acquire`` to Q1 acquire instructions. All other entries are
    scaffold patterns that preserve IR shape pending their dedicated lowering
    implementations.

    :param target_data: QBlox target description. When omitted, the repository
        default is used.
    :returns: Ordered pattern tuple for the lowering pass.
    """

    resolved_target_data = target_data or TARGET_DATA
    return (
        LowerArithIntegerConstantToMoveOp(),
        RewritePhaseSetOp(resolved_target_data, rewrite_callable=PhaseLowering()),
        RewritePhaseShiftOp(resolved_target_data, rewrite_callable=PhaseLowering()),
        RewriteCreateFrameOp(resolved_target_data),
        RewriteSynchronizeOp(resolved_target_data),
        RewriteWaitOp(resolved_target_data),
        RewritePulseOp(resolved_target_data),
        RewriteStartContinuousWaveformOp(resolved_target_data),
        RewriteStopContinuousWaveformOp(resolved_target_data),
        RewritePreQ1AcquireOp(resolved_target_data),
    )
