# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import io
import math

import pytest
from xdsl.context import Context
from xdsl.dialects import func, scf
from xdsl.dialects.arith import AddiOp, ConstantOp as ArithConstantOp, MuliOp
from xdsl.dialects.builtin import (
    IndexType,
    IntAttr,
    ModuleOp,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Block, BlockArgument, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def
from xdsl.printer import Printer
from xdsl.utils.exceptions import PassFailedException

from qat.backend.qblox.target_data import TARGET_DATA
from qat.experimental.conversion.pulse_to_q1.passes import (
    PulseToQ1LoweringPass,
    Q1PreAcquireTransformationPass,
    Q1PulseLegalisationPass,
    Q1PulseValidationPass,
    create_default_pulse_to_q1_pipeline,
)
from qat.experimental.conversion.pulse_to_q1.pre_q1_ir import PreQ1AcquireOp
from qat.experimental.conversion.pulse_to_q1.sequence_outlining import Q1OutliningPass
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AcquisitionType,
    AdvancesTimeTrait,
    ConstantOp,
    CreateFrameOp,
    FrameType,
    FrequencyAttr,
    FrequencyType,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    PhaseType,
    TimeAttr,
    WaitOp,
    WeightsAttr,
)
from qat.experimental.dialect.pulse.units import TimeUnits
from qat.experimental.dialect.q1 import (
    AcquireImmRsImmOp,
    AcquireWeightedImmRsRsRsImmOp,
    DurationImm,
    IntRegisterType,
    SetMrkImmOp,
    StopOp,
    UI5Imm,
)
from qat.experimental.dialect.q1_sequence import SequenceOp


def _module_with_main(ops) -> ModuleOp:
    return ModuleOp([func.FuncOp("main", ((), ()), Region(Block(ops)))])


@irdl_op_definition
class _DynamicPhaseSourceOp(IRDLOperation):
    name = "test.dynamic_phase_source"
    result = result_def(PhaseType)

    def __init__(self):
        super().__init__(result_types=[PhaseType()])


@irdl_op_definition
class _DynamicFrequencySourceOp(IRDLOperation):
    name = "test.dynamic_frequency_source"
    result = result_def(FrequencyType)

    def __init__(self):
        super().__init__(result_types=[FrequencyType()])


def _sequence_module(*ops, channel_id="q0_drive") -> ModuleOp:
    return ModuleOp([SequenceOp(channel_id, [*ops, StopOp()])])


def _frame(channel_id: str = "q0/drive") -> tuple[ConstantOp, CreateFrameOp]:
    freq = ConstantOp(FrequencyAttr(4.8e9))
    return freq, CreateFrameOp(freq, StringAttr(channel_id))


def _sequence_body_ops(module: ModuleOp) -> list:
    [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    return list(seq.body.block.ops)


def test_default_pulse_to_q1_pipeline_runs_outlining_pass():
    """Verify that the default pipeline outlines one sequence per frame."""
    freq = ConstantOp(FrequencyAttr(4.8e9))
    frame = CreateFrameOp(freq, StringAttr("q0.drive"))
    module = _module_with_main([freq, frame, func.ReturnOp()])

    pipeline = create_default_pulse_to_q1_pipeline()
    pipeline.apply(Context(), module)

    [seq] = list(module.body.block.ops)
    assert isinstance(seq, SequenceOp)
    assert seq.channel_id.data == "q0.drive"
    assert isinstance(seq.body.block.first_op, SetMrkImmOp)
    assert seq.body.block.first_op.mrk.data == 3
    assert isinstance(seq.body.block.last_op, StopOp)


def test_default_pulse_to_q1_pipeline_includes_all_passes():
    """Verify that the default pipeline contains all four stages."""
    pipeline = create_default_pulse_to_q1_pipeline()

    assert len(pipeline.passes) == 5
    assert isinstance(pipeline.passes[0], Q1OutliningPass)
    assert isinstance(pipeline.passes[1], Q1PulseValidationPass)
    assert isinstance(pipeline.passes[2], Q1PulseLegalisationPass)
    assert isinstance(pipeline.passes[3], Q1PreAcquireTransformationPass)
    assert isinstance(pipeline.passes[4], PulseToQ1LoweringPass)


class TestQ1PulseValidationPass:
    def _run(self, module: ModuleOp) -> None:
        Q1PulseValidationPass().apply(Context(), module)

    def test_accepts_integer_nanosecond_wait(self):
        freq, frame = _frame()
        time = ConstantOp(TimeAttr(5e-9))
        wait = WaitOp(frame, time)
        self._run(_sequence_module(freq, frame, time, wait))

    def test_accepts_zero_wait_duration(self):
        freq, frame = _frame()
        zero_time = ConstantOp(TimeAttr(0.0))
        wait = WaitOp(frame, zero_time)
        self._run(_sequence_module(freq, frame, zero_time, wait))

    def test_rejects_non_integer_nanosecond_wait(self):
        freq, frame = _frame()
        bad_time = ConstantOp(TimeAttr(4.5e-9))
        wait = WaitOp(frame, bad_time)
        with pytest.raises(
            PassFailedException, match="must map to integer nanoseconds within tolerance"
        ):
            self._run(_sequence_module(freq, frame, bad_time, wait))

    @pytest.mark.parametrize("duration", [math.inf, -math.inf, math.nan])
    def test_rejects_non_finite_wait_duration(self, duration: float):
        freq, frame = _frame()
        time = ConstantOp(TimeAttr(duration))
        wait = WaitOp(frame, time)
        with pytest.raises(PassFailedException, match="time must be finite"):
            self._run(_sequence_module(freq, frame, time, wait))

    def test_rejects_negative_wait_duration(self):
        freq, frame = _frame()
        time = ConstantOp(TimeAttr(-16e-9))
        wait = WaitOp(frame, time)
        with pytest.raises(PassFailedException, match="time must be non-negative"):
            self._run(_sequence_module(freq, frame, time, wait))

    def test_accepts_minimum_nanosecond_duration(self):
        freq, frame = _frame()
        time = ConstantOp(TimeAttr(1e-9))
        wait = WaitOp(frame, time)
        self._run(_sequence_module(freq, frame, time, wait))

    def test_rejects_sub_nanosecond_non_zero_duration(self):
        freq, frame = _frame()
        time = ConstantOp(TimeAttr(0.5e-9))
        wait = WaitOp(frame, time)
        with pytest.raises(PassFailedException, match="smaller than one nanosecond"):
            self._run(_sequence_module(freq, frame, time, wait))

    def test_accepts_dynamic_wait_duration(self):
        from qat.experimental.dialect.pulse.ir import TimeType

        @irdl_op_definition
        class _DynamicTimeSourceOp(IRDLOperation):
            name = "test.dynamic_time_source_val"
            result = result_def(TimeType)

            def __init__(self):
                super().__init__(result_types=[TimeType()])

        freq, frame = _frame()
        dynamic_time = _DynamicTimeSourceOp()
        wait = WaitOp(frame, dynamic_time)
        self._run(_sequence_module(freq, frame, dynamic_time, wait))

    @pytest.mark.parametrize("frequency", [math.inf, -math.inf, math.nan])
    def test_rejects_non_finite_frame_frequency(self, frequency: float):
        freq_const = ConstantOp(FrequencyAttr(frequency))
        frame = CreateFrameOp(freq_const, StringAttr("q0/drive"))
        with pytest.raises(PassFailedException, match="frequency must be finite"):
            self._run(_sequence_module(freq_const, frame))

    def test_accepts_dynamic_frame_frequency(self):
        @irdl_op_definition
        class _DynamicFreqSourceOp(IRDLOperation):
            name = "test.dynamic_freq_source_val"
            result = result_def(FrequencyType)

            def __init__(self):
                super().__init__(result_types=[FrequencyType()])

        dynamic_freq = _DynamicFreqSourceOp()
        frame = CreateFrameOp(dynamic_freq, StringAttr("q0/drive"))
        self._run(_sequence_module(dynamic_freq, frame))

    @pytest.mark.parametrize("phase_value", [math.inf, -math.inf, math.nan])
    def test_rejects_non_finite_phase_set(self, phase_value: float):
        freq, frame = _frame()
        phase = ConstantOp(PhaseAttr(phase_value))
        phase_set = PhaseSetOp(frame, phase)
        with pytest.raises(PassFailedException, match="phase must be finite"):
            self._run(_sequence_module(freq, frame, phase, phase_set))

    def test_accepts_zero_phase_set(self):
        freq, frame = _frame()
        phase = ConstantOp(PhaseAttr(0.0))
        phase_set = PhaseSetOp(frame, phase)
        self._run(_sequence_module(freq, frame, phase, phase_set))

    @pytest.mark.parametrize(
        "phase_value",
        [math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi, -math.pi / 6],
    )
    def test_accepts_valid_phase_set_constants(self, phase_value: float):
        freq, frame = _frame()
        phase = ConstantOp(PhaseAttr(phase_value))
        phase_set = PhaseSetOp(frame, phase)
        self._run(_sequence_module(freq, frame, phase, phase_set))

    @pytest.mark.parametrize("phase_value", [math.inf, -math.inf, math.nan])
    def test_rejects_non_finite_phase_shift(self, phase_value: float):
        freq, frame = _frame()
        phase = ConstantOp(PhaseAttr(phase_value))
        phase_shift = PhaseShiftOp(frame, phase)
        with pytest.raises(PassFailedException, match="phase must be finite"):
            self._run(_sequence_module(freq, frame, phase, phase_shift))

    def test_accepts_zero_phase_shift(self):
        freq, frame = _frame()
        phase = ConstantOp(PhaseAttr(0.0))
        phase_shift = PhaseShiftOp(frame, phase)
        self._run(_sequence_module(freq, frame, phase, phase_shift))

    @pytest.mark.parametrize(
        "phase_value",
        [math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi, -math.pi / 6],
    )
    def test_accepts_valid_phase_shift_constants(self, phase_value: float):
        freq, frame = _frame()
        phase = ConstantOp(PhaseAttr(phase_value))
        phase_shift = PhaseShiftOp(frame, phase)
        self._run(_sequence_module(freq, frame, phase, phase_shift))


class TestQ1PulseLegalisationPass:
    def _run(self, module: ModuleOp) -> None:
        Q1PulseLegalisationPass().apply(Context(), module)

    def test_accepts_constant_phase(self):
        freq, frame = _frame()
        phase = ConstantOp(PhaseAttr(0.0))
        phase_set = PhaseSetOp(frame, phase)
        self._run(_sequence_module(freq, frame, phase, phase_set))

    def test_rejects_non_phase_constant_attribute(self):
        malformed_phase = ConstantOp(TimeAttr(0), result_type=PhaseType())
        freq, frame = _frame()
        phase_set = PhaseSetOp(frame, malformed_phase)
        with pytest.raises(
            PassFailedException, match="expects pulse.constant phase operand"
        ):
            self._run(_sequence_module(freq, frame, malformed_phase, phase_set))

    def test_passes_through_dynamic_phase_set(self):
        freq, frame = _frame()
        dynamic_phase = _DynamicPhaseSourceOp()
        phase_set = PhaseSetOp(frame, dynamic_phase)
        module = _sequence_module(freq, frame, dynamic_phase, phase_set)
        self._run(module)
        body_ops = _sequence_body_ops(module)
        assert any(isinstance(op, UnrealizedConversionCastOp) for op in body_ops)
        assert any(isinstance(op, PhaseSetOp) for op in body_ops)

    def test_passes_through_dynamic_phase_shift(self):
        freq, frame = _frame()
        dynamic_phase = _DynamicPhaseSourceOp()
        phase_shift = PhaseShiftOp(frame, dynamic_phase)
        module = _sequence_module(freq, frame, dynamic_phase, phase_shift)
        self._run(module)
        body_ops = _sequence_body_ops(module)
        assert any(isinstance(op, UnrealizedConversionCastOp) for op in body_ops)
        assert any(isinstance(op, PhaseShiftOp) for op in body_ops)

    def test_passes_through_dynamic_frame_frequency(self):
        dynamic_freq = _DynamicFrequencySourceOp()
        frame = CreateFrameOp(dynamic_freq, StringAttr("q0/drive"))
        self._run(_sequence_module(dynamic_freq, frame))


@irdl_op_definition
class _DynamicIndexSourceOp(IRDLOperation):
    name = "test.dynamic_index_source"
    result = result_def(IndexType)

    def __init__(self):
        super().__init__(result_types=[IndexType()])


def _create_acquire_module(
    duration_ns,
    channel_id,
    repeats,
    weights: WeightsAttr | None = None,
    label: str | None = None,
    no_acquires: int = 1,
) -> ModuleOp:
    """Build a flat Pulse module with one ``pulse.acquire`` wrapped in ``repeats`` loops.

    :param duration_ns: The acquisition duration in nanoseconds.
    :param channel_id: The frame channel identifier.
    :param repeats: Trip counts for the enclosing ``scf.for`` loops, innermost first.
    :param weights: Optional integration weights for the acquisition.
    :param label: Optional label for the acquisition.
    :param no_acquires: Number of acquires.
    :returns: A module carrying the (possibly loop-nested) acquisition.
    """
    freq, frame = _frame(channel_id)
    duration = ConstantOp(TimeAttr(duration_ns, TimeUnits.NANOSECOND))

    acquires = [
        AcquireOp(frame, duration, weights=weights, label=label) for _ in range(no_acquires)
    ]
    ops = [freq, frame, duration, *acquires]

    for repeat in repeats:
        ops.append(scf.YieldOp())

        start_op = ArithConstantOp.from_int_and_width(0, IndexType())
        stop_op = ArithConstantOp.from_int_and_width(repeat, IndexType())
        step_op = ArithConstantOp.from_int_and_width(1, IndexType())
        loop = scf.ForOp(
            start_op,
            stop_op,
            step_op,
            [],
            Block(ops=ops, arg_types=[IndexType()]),
        )
        ops = [start_op, stop_op, step_op, loop]

    return ModuleOp(ops)


class TestQ1PreAcquireTransformationPass:
    @staticmethod
    def _pre_acquire_op(module: ModuleOp) -> PreQ1AcquireOp:
        [pre_acquire] = [op for op in module.walk() if isinstance(op, PreQ1AcquireOp)]
        return pre_acquire

    def test_replaces_acquire_with_pre_q1_acquire(self):
        module = _create_acquire_module(1000, "q0/readout", [])
        Q1PreAcquireTransformationPass().apply(Context(), module)
        ops = list(module.walk())
        assert not any(isinstance(op, AcquireOp) for op in ops)
        assert len([op for op in ops if isinstance(op, PreQ1AcquireOp)]) == 1

    def test_no_loop_sets_single_run_and_zero_store_index(self):
        module = _create_acquire_module(1000, "q0/readout", [])
        Q1PreAcquireTransformationPass().apply(Context(), module)
        pre_acquire = self._pre_acquire_op(module)
        assert pre_acquire.number_runs.data == 1
        assert isinstance(pre_acquire.store_idx.owner, ArithConstantOp)
        assert pre_acquire.store_idx.owner.value.value.data == 0

    def test_single_loop_uses_induction_variable_as_store_index(self):
        module = _create_acquire_module(1000, "q0/readout", [7])
        Q1PreAcquireTransformationPass().apply(Context(), module)
        pre_acquire = self._pre_acquire_op(module)
        assert pre_acquire.number_runs.data == 7
        # The single loop induction variable is used directly as the store index.
        assert isinstance(pre_acquire.store_idx, BlockArgument)
        assert pre_acquire.store_idx.type == IndexType()

    def test_nested_loops_multiply_run_count_and_store_index(self):
        module = _create_acquire_module(1000, "q0/readout", [5, 4])
        Q1PreAcquireTransformationPass().apply(Context(), module)
        pre_acquire = self._pre_acquire_op(module)
        assert pre_acquire.number_runs.data == 20
        # Nested induction variables are flattened by a Horner multiply-add step, so the
        # store index is produced by an AddiOp fed by the scaling MuliOp.
        add_op = pre_acquire.store_idx.owner
        assert isinstance(add_op, AddiOp)
        mul_op = add_op.lhs.owner
        assert isinstance(mul_op, MuliOp)
        # The outer index is scaled by the inner loop's trip count (5), not its own (4).
        assert isinstance(mul_op.rhs.owner, ArithConstantOp)
        assert mul_op.rhs.owner.value.value.data == 5

    def test_many_nested_loops(self):
        module = _create_acquire_module(1000, "q0/readout", [5, 4, 21, 8, 2])
        Q1PreAcquireTransformationPass().apply(Context(), module)
        pre_acquire = self._pre_acquire_op(module)
        assert pre_acquire.number_runs.data == 5 * 4 * 21 * 8 * 2

        muli_ops = [op for op in module.walk() if isinstance(op, MuliOp)]
        addi_ops = [op for op in module.walk() if isinstance(op, AddiOp)]
        # Each of the four inner loops contributes one Horner multiply-add step.
        assert len(muli_ops) == 4
        assert len(addi_ops) == 4

        # Each step is `acc = acc * r_j + i_j`: the AddiOp consumes its MuliOp, and each
        # subsequent MuliOp consumes the previous AddiOp, chaining outermost to innermost.
        for mul_op, add_op in zip(muli_ops, addi_ops, strict=False):
            assert add_op.lhs == mul_op.result
        for prev_add, next_mul in zip(addi_ops, muli_ops[1:], strict=False):
            assert next_mul.lhs == prev_add.result

        # The store index is the final AddiOp in the chain.
        assert pre_acquire.store_idx == addi_ops[-1].result

        # Each multiply scales the running index by the next (inner) loop's trip count.
        inner_trip_counts = [8, 21, 4, 5]
        for mul_op, trip in zip(muli_ops, inner_trip_counts, strict=False):
            assert isinstance(mul_op.rhs.owner, ArithConstantOp)
            assert mul_op.rhs.owner.value.value.data == trip

    def test_preserves_weights_and_label(self):
        weights = WeightsAttr([0.1 + 0j, 0.2 + 0.1j])
        module = _create_acquire_module(
            1000, "q0/readout", [], weights=weights, label="my_acquire"
        )
        Q1PreAcquireTransformationPass().apply(Context(), module)
        pre_acquire = self._pre_acquire_op(module)
        assert pre_acquire.weights == weights
        assert pre_acquire.label.data == "my_acquire"

    def test_preserves_acquire_duration_operand(self):
        module = _create_acquire_module(500, "q0/readout", [])
        Q1PreAcquireTransformationPass().apply(Context(), module)
        pre_acquire = self._pre_acquire_op(module)
        duration_attr = pre_acquire.duration.owner.fold()[0]
        assert duration_attr.value.data == 500
        assert duration_attr.unit.data == TimeUnits.NANOSECOND

    def test_rejects_dynamic_loop_bounds(self):
        freq, frame = _frame("q0/readout")
        duration = ConstantOp(TimeAttr(1000, TimeUnits.NANOSECOND))
        acquire = AcquireOp(frame, duration)
        yield_op = scf.YieldOp()
        body = Block(
            ops=[freq, frame, duration, acquire, yield_op],
            arg_types=[IndexType()],
        )
        start_op = ArithConstantOp.from_int_and_width(0, IndexType())
        dynamic_stop = _DynamicIndexSourceOp()
        step_op = ArithConstantOp.from_int_and_width(1, IndexType())
        loop = scf.ForOp(start_op, dynamic_stop, step_op, [], body)
        module = ModuleOp([start_op, dynamic_stop, step_op, loop])

        with pytest.raises(
            PassFailedException, match="Dynamic For loop bounds not currently supported"
        ):
            Q1PreAcquireTransformationPass().apply(Context(), module)

    def test_multi_acquires_on_same_channel(self):
        module = _create_acquire_module(1000, "q0/readout", [4, 8], no_acquires=3)
        Q1PreAcquireTransformationPass().apply(Context(), module)
        ops = list(module.walk())
        assert not any(isinstance(op, AcquireOp) for op in ops)
        assert len([op for op in ops if isinstance(op, PreQ1AcquireOp)]) == 3


class TestPreQ1AcquireOp:
    @staticmethod
    def _operands(
        channel_id: str = "q0/readout",
    ) -> tuple[CreateFrameOp, ConstantOp, ArithConstantOp]:
        _, frame = _frame(channel_id)
        duration = ConstantOp(TimeAttr(1000, TimeUnits.NANOSECOND))
        store_idx = ArithConstantOp.from_int_and_width(0, IndexType())
        return frame, duration, store_idx

    def test_operands_and_results(self):
        frame, duration, store_idx = self._operands()
        op = PreQ1AcquireOp(
            frame=frame, duration=duration, store_idx=store_idx, number_runs=1
        )
        assert op.name == "pre_q1_pulse.acquire"
        assert list(op.operands) == [
            frame.results[0],
            duration.results[0],
            store_idx.result,
        ]
        assert isinstance(op.frame.type, FrameType)
        assert isinstance(op.store_idx.type, IndexType)
        assert isinstance(op.frame_result.type, FrameType)
        assert isinstance(op.acquisition_result.type, AcquisitionType)
        assert op.has_trait(AdvancesTimeTrait)

    def test_number_runs_from_int(self):
        frame, duration, store_idx = self._operands()
        op = PreQ1AcquireOp(
            frame=frame, duration=duration, store_idx=store_idx, number_runs=8
        )
        assert isinstance(op.number_runs, IntAttr)
        assert op.number_runs.data == 8

    def test_number_runs_from_int_attr(self):
        frame, duration, store_idx = self._operands()
        op = PreQ1AcquireOp(
            frame=frame, duration=duration, store_idx=store_idx, number_runs=IntAttr(3)
        )
        assert op.number_runs.data == 3

    def test_optional_attributes_default_to_none(self):
        frame, duration, store_idx = self._operands()
        op = PreQ1AcquireOp(
            frame=frame, duration=duration, store_idx=store_idx, number_runs=1
        )
        assert op.weights is None
        assert op.label is None

    def test_optional_attributes_are_stored(self):
        frame, duration, store_idx = self._operands()
        weights = WeightsAttr([0.25 + 0j, -0.5 + 0j])
        op = PreQ1AcquireOp(
            frame=frame,
            duration=duration,
            store_idx=store_idx,
            number_runs=1,
            weights=weights,
            label="readout_0",
        )
        assert op.weights == weights
        assert op.label.data == "readout_0"

    def test_label_accepts_string_attr(self):
        frame, duration, store_idx = self._operands()
        op = PreQ1AcquireOp(
            frame=frame,
            duration=duration,
            store_idx=store_idx,
            number_runs=1,
            label=StringAttr("labelled"),
        )
        assert op.label.data == "labelled"

    def test_is_printable(self):
        frame, duration, store_idx = self._operands()
        op = PreQ1AcquireOp(
            frame=frame, duration=duration, store_idx=store_idx, number_runs=1
        )
        module = ModuleOp([frame, duration, store_idx, op])
        stream = io.StringIO()
        Printer(stream=stream).print_op(module)
        assert "pre_q1_pulse.acquire" in stream.getvalue()


class TestPulseToQ1AcquireLowering:
    """Full-pipeline lowering of ``pulse.acquire`` to Q1 acquire instructions.

    The pipeline (``Q1PreAcquireTransformationPass`` -> ``Q1OutliningPass`` ->
    ``PulseToQ1LoweringPass``) currently only supports acquisitions outside of an
    ``scf.for`` loop. A loop-enclosed acquisition is rejected during outlining.
    """

    @staticmethod
    def _lower(module: ModuleOp) -> SequenceOp:
        Q1PreAcquireTransformationPass().apply(Context(), module)
        Q1OutliningPass(target_data=TARGET_DATA).apply(Context(), module)
        PulseToQ1LoweringPass().apply(Context(), module)
        [sequence] = list(module.body.block.ops)
        assert isinstance(sequence, SequenceOp)
        return sequence

    def test_lowers_to_unweighted_acquire(self):
        module = _create_acquire_module(1000, "q0/readout", [])
        sequence = self._lower(module)

        acquires = [op for op in sequence.walk() if isinstance(op, AcquireImmRsImmOp)]
        assert len(acquires) == 1
        acquire = acquires[0]
        assert not any(
            isinstance(op, AcquireWeightedImmRsRsRsImmOp) for op in sequence.walk()
        )
        assert isinstance(acquire.acq_idx, UI5Imm)
        assert acquire.acq_idx.data == 0
        assert isinstance(acquire.duration, DurationImm)
        assert acquire.duration.data == 1000
        # The bin index is supplied via a register materialised by a conversion cast.
        assert isinstance(acquire.bin_idx.type, IntRegisterType)
        assert isinstance(acquire.bin_idx.owner, UnrealizedConversionCastOp)

    def test_unweighted_acquire_registers_acquisition(self):
        module = _create_acquire_module(1000, "q0/readout", [])
        sequence = self._lower(module)

        acquisitions = list(sequence.acquisitions)
        assert len(acquisitions) == 1
        assert acquisitions[0].acquisition_name.data == "q0_readout_0"
        assert acquisitions[0].index.data == 0
        assert acquisitions[0].num_bins.data == 1
        assert len(list(sequence.weights)) == 0

    def test_lowers_to_weighted_acquire(self):
        weights = WeightsAttr([0.1 + 0j, 0.2 + 0.1j, 0.3 + 0j])
        module = _create_acquire_module(1000, "q0/readout", [], weights=weights)
        sequence = self._lower(module)

        acquires = [
            op for op in sequence.walk() if isinstance(op, AcquireWeightedImmRsRsRsImmOp)
        ]
        assert len(acquires) == 1
        acquire = acquires[0]
        assert not any(isinstance(op, AcquireImmRsImmOp) for op in sequence.walk())
        assert acquire.acq_idx.data == 0
        assert acquire.duration.data == 1000
        # Real and imaginary integration weights are registered as two entries.
        assert len(list(sequence.weights)) == 2
        assert isinstance(acquire.bin_idx.type, IntRegisterType)
        assert isinstance(acquire.weight_idx0.type, IntRegisterType)
        assert isinstance(acquire.weight_idx1.type, IntRegisterType)

    def test_label_used_as_acquisition_name(self):
        module = _create_acquire_module(1000, "q0/readout", [], label="custom_acq")
        sequence = self._lower(module)

        acquisitions = list(sequence.acquisitions)
        assert len(acquisitions) == 1
        assert acquisitions[0].acquisition_name.data == "custom_acq"

    def test_raises_when_acquire_inside_loop(self):
        module = _create_acquire_module(1000, "q0/readout", [1000])
        Q1PreAcquireTransformationPass().apply(Context(), module)
        with pytest.raises(PassFailedException, match="region-free entry blocks"):
            Q1OutliningPass(target_data=TARGET_DATA).apply(Context(), module)
