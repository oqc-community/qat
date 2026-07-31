# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import math

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr, UnrealizedConversionCastOp
from xdsl.ir import Operation
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.conversion.pulse_to_q1.passes import (
    PulseToQ1LoweringPass,
    Q1PulseLegalisationPass,
)
from qat.experimental.conversion.pulse_to_q1.rewrite_patterns import (
    RewriteAcquireOp,
    RewriteCreateFrameOp,
    RewritePhaseSetOp,
    RewritePhaseShiftOp,
    RewritePulseOp,
    RewriteStartContinuousWaveformOp,
    RewriteStopContinuousWaveformOp,
    RewriteSynchronizeOp,
    RewriteWaitOp,
    create_legalisation_patterns,
    create_pulse_to_q1_lowering_patterns,
)
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    PhaseType,
    PulseOp,
    SquareWaveformOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SynchronizeOp,
    TimeAttr,
    WaitOp,
)
from qat.experimental.dialect.q1 import (
    AddRsImmRdOp,
    CmpRsImmOp,
    JaeImmOp,
    JbImmOp,
    JgeImmOp,
    JlImmOp,
    LabelOp,
    SetPhDeltaImmOp,
    SetPhDeltaRsOp,
    SetPhImmOp,
    SetPhRsOp,
    StopOp,
    SubRsImmRdOp,
)
from qat.experimental.dialect.q1.ir.ops import UpdParamImmOp
from qat.experimental.dialect.q1_sequence import SequenceOp


@irdl_op_definition
class _DynamicPhaseSourceOp(IRDLOperation):
    name = "test.dynamic_phase_source"
    result = result_def(PhaseType)

    def __init__(self):
        super().__init__(result_types=[PhaseType()])


def _sequence_module(*ops) -> ModuleOp:
    """Build a module containing one q1_sequence.sequence with the given body ops.

    This is the correct IR context for ``PulseToQ1LoweringPass``: after
    ``Q1OutliningPass``, the module contains ``q1_sequence.sequence`` envelopes
    and the lowering pass traverses those envelopes to rewrite Pulse operations.
    """
    return ModuleOp([SequenceOp("q0_drive", [*ops, StopOp()])])


def _frame(channel_id: str = "q0/drive") -> tuple[ConstantOp, CreateFrameOp]:
    freq = ConstantOp(FrequencyAttr(4.8e9))
    return freq, CreateFrameOp(freq, StringAttr(channel_id))


def _run_q1_pipeline(module: ModuleOp) -> None:
    Q1PulseLegalisationPass().apply(Context(), module)
    PulseToQ1LoweringPass().apply(Context(), module)


def _sequence_body_ops(module: ModuleOp) -> list[Operation]:
    [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    return list(seq.body.block.ops)


def test_lowering_pattern_factory_returns_nine_patterns():
    """Verify that the pattern factory returns the full rewrite set in order."""
    patterns = create_pulse_to_q1_lowering_patterns()
    assert len(patterns) == 9
    assert isinstance(patterns[0], RewritePhaseSetOp)
    assert isinstance(patterns[1], RewritePhaseShiftOp)
    assert isinstance(patterns[2], RewriteCreateFrameOp)
    assert isinstance(patterns[3], RewriteSynchronizeOp)
    assert isinstance(patterns[4], RewriteWaitOp)
    assert isinstance(patterns[5], RewritePulseOp)
    assert isinstance(patterns[6], RewriteStartContinuousWaveformOp)
    assert isinstance(patterns[7], RewriteStopContinuousWaveformOp)
    assert isinstance(patterns[8], RewriteAcquireOp)


def test_legalisation_pattern_factory_returns_two_patterns():
    """Verify that the legalisation factory returns the two phase rewrite patterns."""
    patterns = create_legalisation_patterns()
    assert len(patterns) == 2
    assert isinstance(patterns[0], RewritePhaseSetOp)
    assert isinstance(patterns[1], RewritePhaseShiftOp)


def test_rewrite_synchronize_op_is_noop_skeleton():
    """Verify that RewriteSynchronizeOp leaves pulse.sync unchanged.

    Replace this body with the actual Q1 macro-expansion assertion once
    COMPILER-1343 is implemented. The module uses the post-outline IR shape:
    a ``q1_sequence.sequence`` envelope containing the Pulse op.
    """
    freq_0, frame_0 = _frame("q0/drive")
    freq_1, frame_1 = _frame("q1/drive")
    sync = SynchronizeOp(frame_0, frame_1)
    module = _sequence_module(freq_0, frame_0, freq_1, frame_1, sync)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    assert any(isinstance(op, SynchronizeOp) for op in body_ops)


def test_rewrite_wait_op_is_noop_skeleton():
    """Verify that RewriteWaitOp leaves pulse.wait unchanged.

    Replace this body with the actual Q1 macro-expansion assertion once COMPILER-1343 is
    implemented.
    """
    freq, frame = _frame()
    duration = ConstantOp(TimeAttr(16e-9))
    wait = WaitOp(frame, duration)
    module = _sequence_module(freq, frame, duration, wait)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    assert any(isinstance(op, WaitOp) for op in body_ops)


def test_rewrite_phase_set_op_lowers_to_set_ph_and_upd_param():
    """Verify pulse.phase_set is lowered to q1.set_ph + q1.upd_param with PhaseSetOp
    removed."""
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(0.5))
    phase_set = PhaseSetOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_set)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    assert not any(isinstance(op, PhaseSetOp) for op in body_ops)
    assert any(isinstance(op, SetPhImmOp) for op in body_ops)
    assert any(isinstance(op, UpdParamImmOp) for op in body_ops)


def test_rewrite_phase_set_op_converts_radians_to_nco_phase_steps():
    """Verify radian phase is converted to NCO phase steps using nco_phase_steps_per_deg."""
    from qat.backend.qblox.target_data import CONTROL_SEQUENCER_DATA

    phase_rad = math.pi / 2
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(phase_rad))
    phase_set = PhaseSetOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_set)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    [set_ph] = [op for op in body_ops if isinstance(op, SetPhImmOp)]
    expected_steps = round(
        math.degrees(phase_rad) % 360 * CONTROL_SEQUENCER_DATA.nco_phase_steps_per_deg
    )
    assert set_ph.imm.data == expected_steps


def test_rewrite_phase_shift_op_lowers_to_set_ph_delta_and_upd_param():
    """Verify pulse.phase_shift is lowered to q1.set_ph_delta + q1.upd_param with
    PhaseShiftOp removed."""
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(0.25))
    phase_shift = PhaseShiftOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_shift)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    assert not any(isinstance(op, PhaseShiftOp) for op in body_ops)
    assert any(isinstance(op, SetPhDeltaImmOp) for op in body_ops)
    assert any(isinstance(op, UpdParamImmOp) for op in body_ops)


def test_rewrite_phase_shift_op_converts_radians_to_nco_phase_steps():
    """Verify radian phase is converted to NCO phase steps using nco_phase_steps_per_deg."""
    from qat.backend.qblox.target_data import CONTROL_SEQUENCER_DATA

    phase_rad = math.pi
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(phase_rad))
    phase_shift = PhaseShiftOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_shift)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    [set_ph_delta] = [op for op in body_ops if isinstance(op, SetPhDeltaImmOp)]
    expected_steps = round(
        math.degrees(phase_rad) % 360 * CONTROL_SEQUENCER_DATA.nco_phase_steps_per_deg
    )
    assert set_ph_delta.imm.data == expected_steps


def test_rewrite_phase_shift_op_wraps_negative_radians_to_valid_nco_range():
    """Verify negative radian phase wraps to valid NCO phase range via degree modulo 360."""
    from qat.backend.qblox.target_data import CONTROL_SEQUENCER_DATA

    phase_rad = -math.pi / 2
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(phase_rad))
    phase_shift = PhaseShiftOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_shift)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    [set_ph_delta] = [op for op in body_ops if isinstance(op, SetPhDeltaImmOp)]
    expected_steps = round(
        math.degrees(phase_rad) % 360 * CONTROL_SEQUENCER_DATA.nco_phase_steps_per_deg
    )
    assert set_ph_delta.imm.data == expected_steps


def test_rewrite_phase_shift_op_maps_full_rotation_to_zero():
    """Verify 2π radian phase (full rotation) converts to zero NCO phase steps."""
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(2 * math.pi))
    phase_shift = PhaseShiftOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_shift)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    [set_ph_delta] = [op for op in body_ops if isinstance(op, SetPhDeltaImmOp)]
    assert set_ph_delta.imm.data == 0


def test_phase_lowering_requires_canonical_phase_without_legalisation():
    """Lowering-only execution rejects non-canonical phase operands."""
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(3 * math.pi))
    phase_set = PhaseSetOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_set)

    with pytest.raises(PassFailedException, match="phase operand is not canonical"):
        PulseToQ1LoweringPass().apply(Context(), module)


def test_rewrite_phase_shift_op_near_full_rotation_stays_in_nco_range():
    """Verify phases near 2π map to an in-range immediate via modulo normalisation."""
    from qat.backend.qblox.target_data import CONTROL_SEQUENCER_DATA

    phase_rad = math.nextafter(2 * math.pi, 0.0)
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(phase_rad))
    phase_shift = PhaseShiftOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_shift)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    [set_ph_delta] = [op for op in body_ops if isinstance(op, SetPhDeltaImmOp)]
    assert 0 <= set_ph_delta.imm.data < CONTROL_SEQUENCER_DATA.nco_max_phase_steps


@pytest.mark.parametrize(
    "phase_rad",
    [
        math.nextafter(2 * math.pi, math.inf),
        -(10 * math.pi + math.pi / 3),
        1234567.89,
    ],
)
def test_rewrite_phase_set_op_wraps_wide_radian_range_to_valid_nco_steps(phase_rad: float):
    """Wide-range phase_set constants are normalised into the valid NCO step interval."""
    from qat.backend.qblox.target_data import CONTROL_SEQUENCER_DATA

    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(phase_rad))
    phase_set = PhaseSetOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_set)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    [set_ph] = [op for op in body_ops if isinstance(op, SetPhImmOp)]
    expected_steps = (
        round(
            math.degrees(phase_rad) % 360 * CONTROL_SEQUENCER_DATA.nco_phase_steps_per_deg
        )
        % CONTROL_SEQUENCER_DATA.nco_max_phase_steps
    )
    assert set_ph.imm.data == expected_steps


@pytest.mark.parametrize(
    "phase_rad",
    [
        math.nextafter(2 * math.pi, math.inf),
        -(10 * math.pi + math.pi / 3),
        1234567.89,
    ],
)
def test_rewrite_phase_shift_op_wraps_wide_radian_range_to_valid_nco_steps(
    phase_rad: float,
):
    """Wide-range phase_shift constants are normalised into the valid NCO step interval."""
    from qat.backend.qblox.target_data import CONTROL_SEQUENCER_DATA

    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(phase_rad))
    phase_shift = PhaseShiftOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_shift)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    [set_ph_delta] = [op for op in body_ops if isinstance(op, SetPhDeltaImmOp)]
    expected_steps = (
        round(
            math.degrees(phase_rad) % 360 * CONTROL_SEQUENCER_DATA.nco_phase_steps_per_deg
        )
        % CONTROL_SEQUENCER_DATA.nco_max_phase_steps
    )
    assert set_ph_delta.imm.data == expected_steps


def test_rewrite_phase_set_op_lowers_dynamic_radian_phase():
    """Dynamic pulse.phase_set in radians lowers through register conversion and modulo
    loops."""
    freq, frame = _frame()
    dynamic_phase = _DynamicPhaseSourceOp()
    phase_set = PhaseSetOp(frame, dynamic_phase)
    module = _sequence_module(freq, frame, dynamic_phase, phase_set)
    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    assert not any(isinstance(op, PhaseSetOp) for op in body_ops)
    assert any(isinstance(op, UnrealizedConversionCastOp) for op in body_ops)
    assert any(isinstance(op, SetPhRsOp) for op in body_ops)
    assert any(isinstance(op, UpdParamImmOp) for op in body_ops)
    assert any(isinstance(op, CmpRsImmOp) for op in body_ops)
    assert any(isinstance(op, JgeImmOp) for op in body_ops)
    assert any(isinstance(op, JlImmOp) for op in body_ops)
    assert any(isinstance(op, JbImmOp) for op in body_ops)
    assert any(isinstance(op, JaeImmOp) for op in body_ops)
    assert any(isinstance(op, AddRsImmRdOp) for op in body_ops)
    assert any(isinstance(op, SubRsImmRdOp) for op in body_ops)
    assert len([op for op in body_ops if isinstance(op, LabelOp)]) >= 3


def test_rewrite_phase_shift_op_lowers_dynamic_radian_phase():
    """Dynamic pulse.phase_shift in radians lowers through register conversion and modulo
    loops."""
    freq, frame = _frame()
    dynamic_phase = _DynamicPhaseSourceOp()
    phase_shift = PhaseShiftOp(frame, dynamic_phase)
    module = _sequence_module(freq, frame, dynamic_phase, phase_shift)
    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    assert not any(isinstance(op, PhaseShiftOp) for op in body_ops)
    assert any(isinstance(op, UnrealizedConversionCastOp) for op in body_ops)
    assert any(isinstance(op, SetPhDeltaRsOp) for op in body_ops)
    assert any(isinstance(op, UpdParamImmOp) for op in body_ops)


def test_rewrite_pulse_op_is_noop_skeleton():
    """Verify that RewritePulseOp leaves pulse.pulse unchanged.

    Replace this body with the actual Q1 macro-expansion assertion once COMPILER-1345 is
    implemented.
    """
    freq, frame = _frame()
    width = ConstantOp(TimeAttr(40e-9))
    amp = ConstantOp(AmplitudeAttr(0.5))
    waveform = SquareWaveformOp(width, amp)
    pulse = PulseOp(frame, waveform)
    module = _sequence_module(freq, frame, width, amp, waveform, pulse)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    assert any(isinstance(op, PulseOp) for op in body_ops)


def test_rewrite_start_continuous_waveform_op_is_noop_skeleton():
    """Verify that RewriteStartContinuousWaveformOp leaves pulse.start_continuous_waveform
    unchanged.

    Replace this body with the actual Q1 macro-expansion assertion once COMPILER-1345 is
    implemented.
    """
    freq, frame = _frame()
    amp = ConstantOp(AmplitudeAttr(0.5))
    start = StartContinuousWaveformOp(frame, amp)
    module = _sequence_module(freq, frame, amp, start)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    assert any(isinstance(op, StartContinuousWaveformOp) for op in body_ops)


def test_rewrite_stop_continuous_waveform_op_is_noop_skeleton():
    """Verify that RewriteStopContinuousWaveformOp leaves pulse.stop_continuous_waveform
    unchanged.

    Replace this body with the actual Q1 macro-expansion assertion once COMPILER-1345 is
    implemented.
    """
    freq, frame = _frame()
    amp = ConstantOp(AmplitudeAttr(0.5))
    start = StartContinuousWaveformOp(frame, amp)
    stop = StopContinuousWaveformOp(start)
    module = _sequence_module(freq, frame, amp, start, stop)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    assert any(isinstance(op, StopContinuousWaveformOp) for op in body_ops)


def test_rewrite_acquire_op_is_noop_skeleton():
    """Verify that RewriteAcquireOp leaves pulse.acquire unchanged.

    Replace this body with the actual Q1 macro-expansion assertion once COMPILER-1346 is
    implemented.
    """
    freq, frame = _frame("q0/measure")
    duration = ConstantOp(TimeAttr(1e-6))
    acquire = AcquireOp(frame, duration)
    module = _sequence_module(freq, frame, duration, acquire)

    _run_q1_pipeline(module)

    body_ops = _sequence_body_ops(module)
    assert any(isinstance(op, AcquireOp) for op in body_ops)
