# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from dataclasses import replace
from math import pi
from types import SimpleNamespace

import numpy as np
import pytest

from qat.experimental.dialect.q1.ir.ops import (
    AcquireImmImmImmOp,
    AcquireImmRsImmOp,
    AcquireTtlImmImmImmImmOp,
    AcquireTtlImmRsImmImmOp,
    AcquireWeightedImmImmImmImmImmOp,
    AcquireWeightedImmRsRsRsImmOp,
    LatchRstImmOp,
    MoveImmRdOp,
    NopOp,
    PlayImmImmImmOp,
    ResetPhOp,
    SetAwgGainImmImmOp,
    SetAwgOffsImmImmOp,
    SetCondImmImmImmImmOp,
    SetCondRsRsRsImmOp,
    SetFreqImmOp,
    SetLatchEnImmImmOp,
    SetLatchEnRsImmOp,
    SetMrkImmOp,
    SetMrkRsOp,
    SetPhDeltaImmOp,
    SetPhImmOp,
    StopImmOp,
    StopOp,
    StopRsOp,
    UpdParamImmOp,
    WaitImmOp,
    WaitSyncImmOp,
)
from qat.experimental.dialect.q1.ir.schedule import Q1ScheduleVisitor
from qat.experimental.system_data.hardware.qblox import DEFAULT_QBLOX_TARGET
from qat.experimental.system_data.hardware.qblox.target import Q1SequencerType
from qat.experimental.tools.schedule import ScheduleTracker


def test_q1_schedule_visitor_dispatches_state_and_timing_operations():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(
        schedule_tracker,
        waveforms={0: [1.0, 1.0], 1: [0.0, 0.0]},
    )
    dispatch = Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry
    sequence = schedule_tracker.select("sequence")

    dispatch[NopOp](schedule_visitor, SimpleNamespace())
    dispatch[WaitImmOp](schedule_visitor, SimpleNamespace(duration=SimpleNamespace(data=8)))
    dispatch[WaitSyncImmOp](
        schedule_visitor, SimpleNamespace(duration=SimpleNamespace(data=4))
    )
    dispatch[PlayImmImmImmOp](
        schedule_visitor,
        SimpleNamespace(
            wave0=SimpleNamespace(data=0),
            wave1=SimpleNamespace(data=1),
            duration=SimpleNamespace(data=2),
        ),
    )
    dispatch[AcquireImmImmImmOp](
        schedule_visitor, SimpleNamespace(duration=SimpleNamespace(data=3))
    )
    dispatch[SetFreqImmOp](
        schedule_visitor, SimpleNamespace(nco_freq=SimpleNamespace(data=10))
    )
    dispatch[SetPhImmOp](schedule_visitor, SimpleNamespace(nco_po=SimpleNamespace(data=20)))
    dispatch[SetPhDeltaImmOp](
        schedule_visitor, SimpleNamespace(nco_delta_po=SimpleNamespace(data=5))
    )
    dispatch[ResetPhOp](schedule_visitor, SimpleNamespace())
    dispatch[SetAwgGainImmImmOp](
        schedule_visitor,
        SimpleNamespace(gain0=SimpleNamespace(data=2), gain1=SimpleNamespace(data=3)),
    )
    dispatch[UpdParamImmOp](
        schedule_visitor, SimpleNamespace(duration=SimpleNamespace(data=4))
    )

    assert sequence.frequency == 2.5
    assert sequence.phase == pytest.approx(2 * pi * 2.5 * 4e-9)
    assert schedule_tracker.timelines["sequence"][-1].end == 25
    assert schedule_tracker.timelines["sequence"][3].signal is not None


@pytest.mark.parametrize(
    "operation_type",
    [
        AcquireImmImmImmOp,
        AcquireImmRsImmOp,
        AcquireTtlImmImmImmImmOp,
        AcquireTtlImmRsImmImmOp,
        AcquireWeightedImmImmImmImmImmOp,
        AcquireWeightedImmRsRsRsImmOp,
    ],
)
def test_q1_schedule_visitor_records_all_statically_timed_acquisitions(operation_type):
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker)

    Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry[operation_type](
        schedule_visitor,
        SimpleNamespace(duration=SimpleNamespace(data=8)),
    )

    assert schedule_tracker.records[0].label == "acquire"
    assert schedule_tracker.records[0].duration == 8


def test_q1_schedule_visitor_accepts_lowered_register_setup():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker)

    Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry[MoveImmRdOp](
        schedule_visitor, SimpleNamespace()
    )

    assert schedule_tracker.records == ()


def test_q1_schedule_visitor_records_timed_latch_reset():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker)

    Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry[LatchRstImmOp](
        schedule_visitor,
        SimpleNamespace(duration=SimpleNamespace(data=8)),
    )

    assert schedule_tracker.records[0].label == "latch_rst"
    assert schedule_tracker.records[0].duration == 8


def test_q1_schedule_visitor_uses_target_clock_period_for_nop():
    schedule_tracker = ScheduleTracker()
    sequencer_spec = replace(
        DEFAULT_QBLOX_TARGET.sequencer_spec(Q1SequencerType.control),
        clock_period_ns=8,
    )
    target = SimpleNamespace(
        q1asm=DEFAULT_QBLOX_TARGET.q1asm,
        sequencer_spec=lambda _: sequencer_spec,
    )
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker, target_description=target)

    NopOp().accept(schedule_visitor)

    assert schedule_tracker.records[0].duration == 8


def test_q1_schedule_visitor_selects_explicit_sequencer_type():
    requested_types = []
    target = SimpleNamespace(
        q1asm=DEFAULT_QBLOX_TARGET.q1asm,
        sequencer_spec=lambda type_: (
            requested_types.append(type_) or DEFAULT_QBLOX_TARGET.sequencer_spec(type_)
        ),
    )

    Q1ScheduleVisitor(
        ScheduleTracker(),
        target_description=target,
        sequencer_type=Q1SequencerType.readout,
    )

    assert requested_types == [Q1SequencerType.readout]


@pytest.mark.parametrize("operation_type", [StopOp, StopImmOp, StopRsOp])
def test_q1_schedule_visitor_accepts_zero_duration_stop_variants(operation_type):
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker)

    Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry[operation_type](
        schedule_visitor, SimpleNamespace()
    )

    assert schedule_tracker.records == ()


def test_q1_schedule_visitor_records_update_parameter_duration():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker)
    dispatch = Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry

    dispatch[UpdParamImmOp](
        schedule_visitor, SimpleNamespace(duration=SimpleNamespace(data=4))
    )

    events = schedule_tracker.timelines["sequence"]
    assert events[-1].label == "upd_param"
    assert events[-1].end == 4
    np.testing.assert_array_equal(events[-1].signal, np.zeros(4))


def test_q1_schedule_visitor_converts_and_accumulates_nco_phase_steps():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker)
    dispatch = Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry

    dispatch[SetPhImmOp](
        schedule_visitor, SimpleNamespace(nco_po=SimpleNamespace(data=250_000_000))
    )
    dispatch[UpdParamImmOp](
        schedule_visitor, SimpleNamespace(duration=SimpleNamespace(data=4))
    )
    assert schedule_tracker.phase == 0.5 * pi

    dispatch[SetPhDeltaImmOp](
        schedule_visitor,
        SimpleNamespace(nco_delta_po=SimpleNamespace(data=750_000_000)),
    )
    dispatch[UpdParamImmOp](
        schedule_visitor, SimpleNamespace(duration=SimpleNamespace(data=4))
    )
    assert schedule_tracker.phase == pytest.approx(0.0)


def test_q1_schedule_visitor_latches_only_latest_phase_delta():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker)
    dispatch = Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry

    for phase_steps in (100_000_000, 200_000_000):
        dispatch[SetPhDeltaImmOp](
            schedule_visitor,
            SimpleNamespace(nco_delta_po=SimpleNamespace(data=phase_steps)),
        )
    dispatch[UpdParamImmOp](
        schedule_visitor, SimpleNamespace(duration=SimpleNamespace(data=4))
    )

    assert schedule_tracker.phase == pytest.approx(0.4 * pi)


def test_q1_phase_kicks_do_not_change_active_absolute_offset():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker)
    dispatch = Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry

    dispatch[SetPhDeltaImmOp](
        schedule_visitor,
        SimpleNamespace(nco_delta_po=SimpleNamespace(data=250_000_000)),
    )
    dispatch[UpdParamImmOp](
        schedule_visitor, SimpleNamespace(duration=SimpleNamespace(data=4))
    )
    dispatch[SetPhImmOp](schedule_visitor, SimpleNamespace(nco_po=SimpleNamespace(data=0)))
    dispatch[UpdParamImmOp](
        schedule_visitor, SimpleNamespace(duration=SimpleNamespace(data=4))
    )

    assert schedule_tracker.phase == pytest.approx(0.5 * pi)


@pytest.mark.parametrize("frequency_steps", [-2_147_483_648, 2_147_483_647])
def test_q1_schedule_visitor_rejects_frequency_outside_target_limits(frequency_steps):
    schedule_visitor = Q1ScheduleVisitor(ScheduleTracker())

    with pytest.raises(ValueError, match="NCO frequency.*outside"):
        Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry[SetFreqImmOp](
            schedule_visitor,
            SimpleNamespace(nco_freq=SimpleNamespace(data=frequency_steps)),
        )


@pytest.mark.parametrize(
    "operation_type",
    [
        SetCondImmImmImmImmOp,
        SetCondRsRsRsImmOp,
        SetMrkImmOp,
        SetMrkRsOp,
    ],
)
def test_q1_schedule_visitor_accepts_zero_duration_configuration_operations(
    operation_type,
):
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker)

    Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry[operation_type](
        schedule_visitor, SimpleNamespace()
    )

    assert schedule_tracker.records == ()


@pytest.mark.parametrize("operation_type", [SetLatchEnImmImmOp, SetLatchEnRsImmOp])
def test_q1_schedule_visitor_records_timed_latch_configuration(operation_type):
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker)

    Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry[operation_type](
        schedule_visitor,
        SimpleNamespace(duration=SimpleNamespace(data=4)),
    )

    assert schedule_tracker.records[0].label == "set_latch_en"
    assert schedule_tracker.records[0].duration == 4


def test_q1_schedule_visitor_stops_waveform_playback_at_instruction_end():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(
        schedule_tracker,
        waveforms={
            0: [1.0, 0.5, 0.0, -0.5, -1.0, 0.0],
            1: [0.0, 0.5, 1.0, 0.5, 0.0, -0.5],
        },
    )
    dispatch = Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry

    dispatch[SetAwgGainImmImmOp](
        schedule_visitor,
        SimpleNamespace(
            gain0=SimpleNamespace(data=16_384),
            gain1=SimpleNamespace(data=16_384),
        ),
    )
    dispatch[PlayImmImmImmOp](
        schedule_visitor,
        SimpleNamespace(
            wave0=SimpleNamespace(data=0),
            wave1=SimpleNamespace(data=1),
            duration=SimpleNamespace(data=4),
        ),
    )
    dispatch[WaitImmOp](
        schedule_visitor,
        SimpleNamespace(duration=SimpleNamespace(data=4)),
    )

    play, wait = schedule_tracker.timelines["sequence"]
    scale = 0.5 / np.sqrt(2)
    np.testing.assert_allclose(
        play.signal,
        scale * np.array([1.0, 0.5 + 0.5j, 1.0j, -0.5 + 0.5j]),
    )
    np.testing.assert_allclose(
        wait.signal,
        np.zeros(4),
    )


def test_q1_schedule_visitor_applies_awg_offsets_after_waveform_ends():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(
        schedule_tracker,
        waveforms={0: [0.0, 0.0], 1: [0.0, 0.0]},
    )
    dispatch = Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry

    dispatch[SetAwgOffsImmImmOp](
        schedule_visitor,
        SimpleNamespace(
            offs0=SimpleNamespace(data=16_384),
            offs1=SimpleNamespace(data=-16_384),
        ),
    )
    dispatch[PlayImmImmImmOp](
        schedule_visitor,
        SimpleNamespace(
            wave0=SimpleNamespace(data=0),
            wave1=SimpleNamespace(data=1),
            duration=SimpleNamespace(data=2),
        ),
    )
    dispatch[WaitImmOp](
        schedule_visitor,
        SimpleNamespace(duration=SimpleNamespace(data=2)),
    )

    expected = np.full(2, (0.5 - 0.5j) / np.sqrt(2))
    play, wait = schedule_tracker.timelines["sequence"]
    np.testing.assert_allclose(play.signal, expected)
    np.testing.assert_allclose(wait.signal, expected)


def test_q1_schedule_visitor_applies_complex_nco_modulation_equation():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(
        schedule_tracker,
        waveforms={
            0: [1.0, 0.0, 0.0, 0.0],
            1: [1.0, 0.0, 0.0, 0.0],
        },
    )
    dispatch = Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry

    dispatch[SetPhImmOp](
        schedule_visitor,
        SimpleNamespace(nco_po=SimpleNamespace(data=250_000_000)),
    )
    dispatch[PlayImmImmImmOp](
        schedule_visitor,
        SimpleNamespace(
            wave0=SimpleNamespace(data=0),
            wave1=SimpleNamespace(data=1),
            duration=SimpleNamespace(data=4),
        ),
    )

    signal = schedule_tracker.timelines["sequence"][0].signal
    np.testing.assert_allclose(signal[0], (-1.0 + 1.0j) / np.sqrt(2), atol=1e-12)
    np.testing.assert_array_equal(signal[1:], np.zeros(3))


def test_q1_schedule_visitor_rejects_missing_waveform_reference():
    schedule_visitor = Q1ScheduleVisitor(
        ScheduleTracker(),
        waveforms={0: [1.0]},
    )
    dispatch = Q1ScheduleVisitor.__dict__["visit"].dispatcher.registry

    with pytest.raises(ValueError, match="missing waveform index 1"):
        dispatch[PlayImmImmImmOp](
            schedule_visitor,
            SimpleNamespace(
                wave0=SimpleNamespace(data=0),
                wave1=SimpleNamespace(data=1),
                duration=SimpleNamespace(data=4),
            ),
        )


@pytest.mark.parametrize("sample", [1.01, np.nan, np.inf, -np.inf])
def test_q1_schedule_visitor_rejects_invalid_waveform_samples(sample):
    with pytest.raises(ValueError, match=r"outside \[-1.0, 1.0\]"):
        Q1ScheduleVisitor(
            ScheduleTracker(),
            waveforms={0: [sample]},
        )


def test_q1_schedule_visitor_materialises_waveform_iterables():
    visitor = Q1ScheduleVisitor(
        ScheduleTracker(),
        waveforms={0: (sample for sample in [0.25, 0.5])},
    )

    np.testing.assert_array_equal(visitor._waveforms[0], [0.25, 0.5])
