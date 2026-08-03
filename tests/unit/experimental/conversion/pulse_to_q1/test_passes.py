# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import math

import pytest
from xdsl.context import Context
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, StringAttr, UnrealizedConversionCastOp
from xdsl.ir import Block, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.conversion.pulse_to_q1.passes import (
    PulseToQ1LoweringPass,
    Q1PulseLegalisationPass,
    Q1PulseValidationPass,
    create_default_pulse_to_q1_pipeline,
)
from qat.experimental.conversion.pulse_to_q1.sequence_outlining import Q1OutliningPass
from qat.experimental.dialect.pulse.ir import (
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    FrequencyType,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    PhaseType,
    TimeAttr,
    WaitOp,
)
from qat.experimental.dialect.q1 import SetMrkImmOp, StopOp
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


def _sequence_module(*ops) -> ModuleOp:
    return ModuleOp([SequenceOp("q0_drive", [*ops, StopOp()])])


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

    assert len(pipeline.passes) == 4
    assert isinstance(pipeline.passes[0], Q1OutliningPass)
    assert isinstance(pipeline.passes[1], Q1PulseValidationPass)
    assert isinstance(pipeline.passes[2], Q1PulseLegalisationPass)
    assert isinstance(pipeline.passes[3], PulseToQ1LoweringPass)


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
