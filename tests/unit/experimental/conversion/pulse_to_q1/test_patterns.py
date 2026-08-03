# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import math

import numpy as np
import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr, UnrealizedConversionCastOp
from xdsl.ir import Operation
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def
from xdsl.utils.exceptions import PassFailedException, VerifyException

from qat.backend.qblox.target_data import TARGET_DATA
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
    IntegrateOp,
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
    WeightsAttr,
)
from qat.experimental.dialect.pulse.units import TimeUnits
from qat.experimental.dialect.q1 import (
    AcquireImmImmImmOp,
    AcquireWeightedImmImmImmImmImmOp,
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


def _sequence_module(*ops, channel_id: str = "q0_drive") -> ModuleOp:
    """Build a module containing one q1_sequence.sequence with the given body ops.

    This is the correct IR context for ``PulseToQ1LoweringPass``: after
    ``Q1OutliningPass``, the module contains ``q1_sequence.sequence`` envelopes
    and the lowering pass traverses those envelopes to rewrite Pulse operations.
    """
    return ModuleOp([SequenceOp(channel_id, [*ops, StopOp()])])


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


class TestRewriteAcquireOp:
    """Tests for ``RewriteAcquireOp``: lowering ``pulse.acquire`` to the appropriate Q1
    acquire-family op (``AcquireImmImmImmOp`` for square-weight acquisitions,
    ``AcquireWeightedImmImmImmImmImmOp`` for custom-weight acquisitions).

    Each test verifies the emitted op type, immediate field values (acq_idx, bin_idx,
    weight indices, duration), and the resulting acquisitions/weights table entries on
    the enclosing ``SequenceOp``.
    """

    @staticmethod
    def _run(
        *acquire_params: tuple[WeightsAttr | None, int, str],
    ) -> tuple[SequenceOp, list]:
        """Build a module from ``acquire_params``, apply ``PulseToQ1LoweringPass``, and
        return the lowered ``SequenceOp`` together with all Q1 acquire ops found in its
        body.

        :param acquire_params: One entry per ``AcquireOp`` to emit, each a triple of
            ``(weights, duration_ns, channel_id)`` where ``duration_ns`` is an integer
            number of nanoseconds, reflecting the IR state after the duration
            unit-normalisation pass.
        :returns: ``(sequencer, q1_acquire_ops)`` where ``q1_acquire_ops`` preserves
            the original program order.
        """
        ops = []
        channel_id = None
        for weights, duration_ns, channel_id in acquire_params:
            freq, frame = _frame(channel_id)
            duration = ConstantOp(TimeAttr(duration_ns, TimeUnits.NANOSECOND))
            acquire = AcquireOp(frame, duration, weights=weights)
            ops.extend([freq, frame, duration, acquire])
        module = _sequence_module(*ops, channel_id="seq_0")
        PulseToQ1LoweringPass().apply(Context(), module)
        [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        q1_acq_ops = [
            op
            for op in seq.body.block.ops
            if isinstance(op, AcquireImmImmImmOp | AcquireWeightedImmImmImmImmImmOp)
        ]
        return seq, q1_acq_ops

    def test_single_unweighted(self):
        """Single unweighted acquire lowers to AcquireImmImmImmOp with correct acq_idx,
        bin_idx and duration, and registers one acquisition table entry."""
        seq, acq_ops = self._run((None, 1000, "q0/measure"))

        assert len([op for op in seq.body.block.ops if isinstance(op, AcquireOp)]) == 0
        [op] = acq_ops
        assert isinstance(op, AcquireImmImmImmOp)
        assert op.acq_idx.data == 0
        assert op.bin_idx.data == 0
        assert op.duration.data == 1000  # 1e-6 s → 1000 ns

        [entry] = seq.acquisitions
        assert entry.acquisition_name.data == "q0_measure"
        assert entry.index.data == 0
        assert entry.num_bins.data == 1

    def test_single_weighted(self):
        """Single weighted acquire lowers to AcquireWeightedImmImmImmImmImmOp with correct
        acq_idx, bin_idx, weight_idx0/1 and duration, and registers one acquisition and two
        weight table entries."""
        seq, acq_ops = self._run(
            (WeightsAttr(np.array([0.5 + 0.5j, 0.3 + 0.1j])), 1000, "q0/measure")
        )

        assert len([op for op in seq.body.block.ops if isinstance(op, AcquireOp)]) == 0
        [op] = acq_ops
        assert isinstance(op, AcquireWeightedImmImmImmImmImmOp)
        assert op.acq_idx.data == 0
        assert op.bin_idx.data == 0
        assert op.weight_idx0.data == 0
        assert op.weight_idx1.data == 1
        assert op.duration.data == 1000  # 1e-6 s → 1000 ns

        [acq_entry] = seq.acquisitions
        assert acq_entry.acquisition_name.data == "q0_measure"
        assert acq_entry.index.data == 0
        assert acq_entry.num_bins.data == 1

        assert [w.weight_name.data for w in seq.weights] == [
            "53874763",
            "2778593f",
        ]
        assert [w.index.data for w in seq.weights] == [0, 1]

    def test_two_unweighted_different_frames(self):
        """Two unweighted acquires on different frames lower to AcquireImmImmImmOp with
        distinct acq_idx and duration values and two acquisition table entries."""
        seq, acq_ops = self._run(
            (None, 1000, "q0/measure"),
            (None, 2000, "q1/measure"),
        )

        assert len([op for op in seq.body.block.ops if isinstance(op, AcquireOp)]) == 0
        assert len(acq_ops) == 2
        assert all(isinstance(op, AcquireImmImmImmOp) for op in acq_ops)
        assert acq_ops[0].acq_idx.data == 0
        assert acq_ops[0].bin_idx.data == 0
        assert acq_ops[0].duration.data == 1000
        assert acq_ops[1].acq_idx.data == 1
        assert acq_ops[1].bin_idx.data == 0
        assert acq_ops[1].duration.data == 2000

        assert [a.acquisition_name.data for a in seq.acquisitions] == [
            "q0_measure",
            "q1_measure",
        ]
        assert [a.index.data for a in seq.acquisitions] == [0, 1]

    def test_two_weighted_different_frames(self):
        """Two weighted acquires on different frames lower to
        AcquireWeightedImmImmImmImmImmOp with consecutive acq_idx and weight_idx values,
        correct durations, and four weight table entries."""
        seq, acq_ops = self._run(
            (WeightsAttr(np.array([0.5 + 0.5j, 0.3 + 0.1j])), 1000, "q0/measure"),
            (WeightsAttr(np.array([1.0 + 0.0j, 0.0 + 1.0j])), 800, "q1/measure"),
        )

        assert len([op for op in seq.body.block.ops if isinstance(op, AcquireOp)]) == 0
        assert len(acq_ops) == 2
        assert all(isinstance(op, AcquireWeightedImmImmImmImmImmOp) for op in acq_ops)
        assert acq_ops[0].acq_idx.data == 0
        assert acq_ops[0].bin_idx.data == 0
        assert acq_ops[0].weight_idx0.data == 0
        assert acq_ops[0].weight_idx1.data == 1
        assert acq_ops[0].duration.data == 1000
        assert acq_ops[1].acq_idx.data == 1
        assert acq_ops[1].bin_idx.data == 0
        assert acq_ops[1].weight_idx0.data == 2
        assert acq_ops[1].weight_idx1.data == 3
        assert acq_ops[1].duration.data == 800

        assert [a.acquisition_name.data for a in seq.acquisitions] == [
            "q0_measure",
            "q1_measure",
        ]
        assert [w.weight_name.data for w in seq.weights] == [
            "53874763",
            "2778593f",
            "b40c35e7",
            "29743b86",
        ]
        assert [w.index.data for w in seq.weights] == [0, 1, 2, 3]

    @pytest.mark.parametrize(
        "time_attr, expected_ns",
        [
            pytest.param(TimeAttr(1000, TimeUnits.NANOSECOND), 1000, id="nanoseconds"),
            pytest.param(TimeAttr(1.1, TimeUnits.MICROSECOND), 1100, id="microseconds"),
            pytest.param(TimeAttr(0.0008, TimeUnits.MILLISECOND), 800, id="milliseconds"),
            pytest.param(TimeAttr(6e-7, TimeUnits.SECOND), 600, id="seconds"),
        ],
    )
    def test_duration_unit_conversion(self, time_attr, expected_ns):
        """Duration TimeAttrs in any time unit are correctly converted to integer
        nanoseconds in the emitted Q1 acquire op."""
        freq, frame = _frame("q0/measure")
        duration = ConstantOp(time_attr)
        acquire = AcquireOp(frame, duration, weights=None)
        module = _sequence_module(freq, frame, duration, acquire, channel_id="q0/measure")

        PulseToQ1LoweringPass().apply(Context(), module)

        [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        [op] = [op for op in seq.body.block.ops if isinstance(op, AcquireImmImmImmOp)]
        assert op.duration.data == expected_ns

    def test_acquisition_result_consumer_raises(self):
        """If anything consumes the acquisition_result SSA value of a pulse.acquire op,
        lowering raises NotImplementedError because that path is not yet implemented."""
        freq, frame = _frame("q0/measure")
        duration = ConstantOp(TimeAttr(1000, TimeUnits.NANOSECOND))
        acquire = AcquireOp(frame, duration, weights=None)
        # Attach a use to the acquisition_result via pulse.integrate — any use
        # is sufficient to trigger the guard.
        integrate = IntegrateOp(acquire.acquisition_result)
        module = _sequence_module(
            freq, frame, duration, acquire, integrate, channel_id="q0/measure"
        )

        with pytest.raises(NotImplementedError, match="acquisition_result consumers"):
            PulseToQ1LoweringPass().apply(Context(), module)

    def test_duplicate_acquisition_name_raises(self):
        """Two acquires sharing the same channel_id produce the same acquisition name; the
        second registration raises ValueError."""
        with pytest.raises(ValueError, match="already exists"):
            self._run(
                (None, 1000, "q0/measure"),
                (None, 1000, "q0/measure"),
            )

    def test_acq_table_overflow_raises(self):
        """Registering a 33rd distinct acquisition on the same sequencer exceeds the
        hardware table limit of 32 entries (acq_idx 0–31) and raises VerifyException."""
        pattern = RewriteAcquireOp(TARGET_DATA)
        seq = SequenceOp("test", [StopOp()])
        for i in range(32):
            pattern._register_acquisition(seq, f"acq_{i}", num_bins=1)
        with pytest.raises(VerifyException):
            pattern._register_acquisition(seq, "acq_32", num_bins=1)

    def test_weight_table_overflow_raises(self):
        """Registering a 33rd weight on the same sequencer exceeds the hardware table limit
        of 32 entries (indices 0–31) and raises VerifyException."""
        pattern = RewriteAcquireOp(TARGET_DATA)
        seq = SequenceOp("test", [StopOp()])
        for i in range(32):
            pattern._register_weight(seq, np.full(4, i / 33))
        with pytest.raises(VerifyException):
            pattern._register_weight(seq, np.full(4, 32 / 33))

    def test_duplicate_weight_payload_returns_existing_index(self):
        """Registering the same weight payload twice returns the existing index without
        adding a duplicate entry to the weight table."""
        pattern = RewriteAcquireOp(TARGET_DATA)
        seq = SequenceOp("test", [StopOp()])
        coeffs = np.ones(4)
        first_index = pattern._register_weight(seq, coeffs)
        second_index = pattern._register_weight(seq, coeffs)
        assert second_index == first_index
        assert len(seq.weights) == 1

    def test_dynamic_bin_idx_raises(self, mocker):
        """If _get_bin_info returns an SSAValue as bin_idx, NotImplementedError is raised
        because register-based bin indexing is not yet wired up in the lowering pass."""
        freq, frame = _frame("q0/measure")
        duration = ConstantOp(TimeAttr(1000, TimeUnits.NANOSECOND))
        acquire = AcquireOp(frame, duration, weights=None)
        module = _sequence_module(freq, frame, duration, acquire, channel_id="q0/measure")

        # Use the duration op result as a stand-in for a future bin-counter register.
        mocker.patch.object(
            RewriteAcquireOp,
            "_get_bin_info",
            return_value=(8192, duration.results[0]),
        )

        with pytest.raises(NotImplementedError, match="Dynamic bin indices"):
            PulseToQ1LoweringPass().apply(Context(), module)

    def test_label_used_as_acquisition_name(self):
        """When a label is provided on pulse.acquire, it should be used as the acquisition
        name in the table rather than the frame channel_id."""
        freq, frame = _frame("q0/measure")
        duration = ConstantOp(TimeAttr(1000, TimeUnits.NANOSECOND))
        acquire = AcquireOp(frame, duration, weights=None, label="my_readout")
        module = _sequence_module(freq, frame, duration, acquire, channel_id="q0/measure")

        PulseToQ1LoweringPass().apply(Context(), module)

        [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        [entry] = seq.acquisitions
        assert entry.acquisition_name.data == "my_readout"

    def test_acquire_outside_sequence_op_raises(self):
        """pulse.acquire with no SequenceOp ancestor raises ValueError.

        The PatternRewriter is not touched before the error fires, so None is passed as a
        stand-in.
        """
        freq, frame = _frame("q0/measure")
        duration = ConstantOp(TimeAttr(1000, TimeUnits.NANOSECOND))
        acquire = AcquireOp(frame, duration, weights=None)
        # acquire is a standalone op with no SequenceOp ancestor

        pattern = RewriteAcquireOp(TARGET_DATA)
        with pytest.raises(ValueError, match="No SequenceOp found in the parent chain"):
            pattern.match_and_rewrite(acquire, None)
