# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr

from qat.experimental.conversion.pulse_to_q1.passes import PulseToQ1LoweringPass
from qat.experimental.conversion.pulse_to_q1.rewrite_patterns import (
    RewriteAcquireOp,
    RewritePhaseSetOp,
    RewritePhaseShiftOp,
    RewritePulseOp,
    RewriteStartContinuousWaveformOp,
    RewriteStopContinuousWaveformOp,
    RewriteSynchronizeOp,
    RewriteWaitOp,
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
    PulseOp,
    SquareWaveformOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SynchronizeOp,
    TimeAttr,
    WaitOp,
)
from qat.experimental.dialect.q1 import StopOp
from qat.experimental.dialect.q1_sequence import SequenceOp


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


def test_lowering_pattern_factory_returns_eight_skeleton_patterns():
    """Verify that the pattern factory returns the full rewrite set in order."""
    patterns = create_pulse_to_q1_lowering_patterns()
    assert len(patterns) == 8
    assert isinstance(patterns[0], RewriteSynchronizeOp)
    assert isinstance(patterns[1], RewriteWaitOp)
    assert isinstance(patterns[2], RewritePhaseSetOp)
    assert isinstance(patterns[3], RewritePhaseShiftOp)
    assert isinstance(patterns[4], RewritePulseOp)
    assert isinstance(patterns[5], RewriteStartContinuousWaveformOp)
    assert isinstance(patterns[6], RewriteStopContinuousWaveformOp)
    assert isinstance(patterns[7], RewriteAcquireOp)


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

    PulseToQ1LoweringPass().apply(Context(), module)

    [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    assert any(isinstance(op, SynchronizeOp) for op in seq.body.block.ops)


def test_rewrite_wait_op_is_noop_skeleton():
    """Verify that RewriteWaitOp leaves pulse.wait unchanged.

    Replace this body with the actual Q1 macro-expansion assertion once COMPILER-1343 is
    implemented.
    """
    freq, frame = _frame()
    duration = ConstantOp(TimeAttr(16e-9))
    wait = WaitOp(frame, duration)
    module = _sequence_module(freq, frame, duration, wait)

    PulseToQ1LoweringPass().apply(Context(), module)

    [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    assert any(isinstance(op, WaitOp) for op in seq.body.block.ops)


def test_rewrite_phase_set_op_is_noop_skeleton():
    """Verify that RewritePhaseSetOp leaves pulse.phase_set unchanged.

    Replace this body with the actual Q1 macro-expansion assertion once COMPILER-1344 is
    implemented.
    """
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(0.5))
    phase_set = PhaseSetOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_set)

    PulseToQ1LoweringPass().apply(Context(), module)

    [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    assert any(isinstance(op, PhaseSetOp) for op in seq.body.block.ops)


def test_rewrite_phase_shift_op_is_noop_skeleton():
    """Verify that RewritePhaseShiftOp leaves pulse.phase_shift unchanged.

    Replace this body with the actual Q1 macro-expansion assertion once COMPILER-1344 is
    implemented.
    """
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(0.25))
    phase_shift = PhaseShiftOp(frame, phase)
    module = _sequence_module(freq, frame, phase, phase_shift)

    PulseToQ1LoweringPass().apply(Context(), module)

    [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    assert any(isinstance(op, PhaseShiftOp) for op in seq.body.block.ops)


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

    PulseToQ1LoweringPass().apply(Context(), module)

    [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    assert any(isinstance(op, PulseOp) for op in seq.body.block.ops)


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

    PulseToQ1LoweringPass().apply(Context(), module)

    [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    assert any(isinstance(op, StartContinuousWaveformOp) for op in seq.body.block.ops)


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

    PulseToQ1LoweringPass().apply(Context(), module)

    [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    assert any(isinstance(op, StopContinuousWaveformOp) for op in seq.body.block.ops)


def test_rewrite_acquire_op_is_noop_skeleton():
    """Verify that RewriteAcquireOp leaves pulse.acquire unchanged.

    Replace this body with the actual Q1 macro-expansion assertion once COMPILER-1346 is
    implemented.
    """
    freq, frame = _frame("q0/measure")
    duration = ConstantOp(TimeAttr(1e-6))
    acquire = AcquireOp(frame, duration)
    module = _sequence_module(freq, frame, duration, acquire)

    PulseToQ1LoweringPass().apply(Context(), module)

    [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    assert any(isinstance(op, AcquireOp) for op in seq.body.block.ops)
