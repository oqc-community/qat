# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.experimental.tools.schedule import ResourceKind, ScheduleTracker


def test_schedule_tracker_applies_state_to_signal_and_advances_time():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("q0", ResourceKind.FRAME, "s")
    schedule_tracker.set_amplitude(2.0)
    schedule_tracker.set_phase(np.pi / 2)

    schedule_event = schedule_tracker.advance(2e-9, [1.0, 0.5], label="pulse")

    assert schedule_event.start == 0.0
    assert schedule_event.end == 2e-9
    np.testing.assert_allclose(schedule_event.signal, [2j, 1j])


def test_schedule_tracker_records_semantic_intervals():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("q0", ResourceKind.FRAME, "s")

    wait = schedule_tracker.wait(1.0)
    pulse = schedule_tracker.pulse(1.0, [1.0])
    acquire = schedule_tracker.acquire(1.0, label="readout")
    marker = schedule_tracker.mark("state change")

    assert [event.label for event in (wait, pulse, acquire, marker)] == [
        "wait",
        "pulse",
        "readout",
        "state change",
    ]
    assert marker.duration == 0


def test_schedule_tracker_materialises_signal_iterables():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("q0", ResourceKind.FRAME, "s")

    event = schedule_tracker.advance(2e-9, (sample for sample in [1.0, 0.5]))

    np.testing.assert_array_equal(event.signal, [1.0, 0.5])


def test_schedule_tracker_continuously_modulates_signal_and_phase():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("q0", ResourceKind.FRAME, "s")
    schedule_tracker.set_frequency(250e6)

    first_event = schedule_tracker.advance(1e-9, [1.0, 1.0])
    second_event = schedule_tracker.advance(1e-9, [1.0, 1.0])

    np.testing.assert_allclose(
        first_event.signal, [1.0, np.exp(1j * np.pi / 4)], atol=1e-12
    )
    np.testing.assert_allclose(
        second_event.signal,
        [1.0j, np.exp(1j * 3 * np.pi / 4)],
        atol=1e-12,
    )


def test_schedule_tracker_synchronises_resources():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("q0", ResourceKind.FRAME, "s").advance(2.0)
    schedule_tracker.resource("q1", ResourceKind.FRAME, "s").advance(5.0)

    schedule_tracker.synchronise(["q0", "q1"])

    assert schedule_tracker.timelines["q0"][-1].label == "synchronise"
    assert schedule_tracker.timelines["q0"][-1].end == 5.0


def test_schedule_tracker_synchronises_resources_with_different_time_units():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("pulse", ResourceKind.FRAME, "s").advance(1e-9)
    schedule_tracker.resource("q1", ResourceKind.SEQUENCE, "ns").advance(2.0)

    schedule_tracker.synchronise(["pulse", "q1"])

    assert schedule_tracker.timelines["pulse"][-1].label == "synchronise"
    assert schedule_tracker.timelines["pulse"][-1].end == pytest.approx(2e-9)
    assert len(schedule_tracker.timelines["q1"]) == 1


def test_schedule_tracker_preserves_global_event_insertion_order():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("q0", ResourceKind.FRAME, "s").advance(1.0, label="q0-1")
    schedule_tracker.resource("q1", ResourceKind.FRAME, "s").advance(1.0, label="q1-1")
    schedule_tracker.select("q0").advance(1.0, label="q0-2")

    assert [event.label for event in schedule_tracker.records] == [
        "q0-1",
        "q1-1",
        "q0-2",
    ]


def test_schedule_tracker_rejects_dynamic_duration():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("sequence", ResourceKind.SEQUENCE, "ns")

    with pytest.raises(ValueError, match="duration must be"):
        schedule_tracker.advance("duration")  # type: ignore[arg-type]


def test_schedule_tracker_rejects_invalid_resource_selection_and_metadata():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("q0", ResourceKind.FRAME, "s")

    with pytest.raises(KeyError, match="Unknown schedule resource"):
        schedule_tracker.select("missing")
    with pytest.raises(ValueError, match="conflicting metadata"):
        schedule_tracker.resource("q0", ResourceKind.SEQUENCE, "ns")


@pytest.mark.parametrize("phase_scale", [0.0, -1.0, float("inf")])
def test_schedule_tracker_rejects_invalid_phase_scale(phase_scale):
    schedule_tracker = ScheduleTracker()

    with pytest.raises(ValueError, match="phase scale"):
        schedule_tracker.resource(
            "q0",
            ResourceKind.FRAME,
            "s",
            phase_scale=phase_scale,
        )


def test_schedule_tracker_accepts_repeated_matching_resource_metadata():
    schedule_tracker = ScheduleTracker()

    selected = schedule_tracker.resource(
        "q0",
        ResourceKind.FRAME,
        "s",
        "Amplitude",
        "turns",
        1 / (2 * np.pi),
    ).resource(
        "q0",
        ResourceKind.FRAME,
        "s",
        "Amplitude",
        "turns",
        1 / (2 * np.pi),
    )

    assert selected.selected_resource == "q0"


def test_schedule_tracker_exposes_resource_metadata():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource(
        "q0",
        ResourceKind.FRAME,
        "s",
        signal_unit="DAC/ADC range",
        signal_limits=(-1.0, 1.0),
    )

    resource = schedule_tracker.resource_info("q0")

    assert resource.signal_unit == "DAC/ADC range"
    assert resource.signal_limits == (-1.0, 1.0)


@pytest.mark.parametrize(
    "signal_limits",
    [
        (1.0, -1.0),
        (0.0, 0.0),
        (float("-inf"), 1.0),
        (-1.0, float("inf")),
    ],
)
def test_schedule_tracker_rejects_invalid_signal_limits(signal_limits):
    schedule_tracker = ScheduleTracker()

    with pytest.raises(ValueError, match="signal"):
        schedule_tracker.resource(
            "q0",
            ResourceKind.FRAME,
            "s",
            signal_limits=signal_limits,
        )


def test_schedule_tracker_rejects_negative_duration():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("q0", ResourceKind.FRAME, "s")

    with pytest.raises(ValueError, match="duration must be non-negative"):
        schedule_tracker.advance(-1.0)


def test_schedule_tracker_exposes_state_and_records():
    schedule_tracker = ScheduleTracker()
    with pytest.raises(RuntimeError, match="No schedule resource"):
        _ = schedule_tracker.selected_resource

    schedule_tracker.resource("q0", ResourceKind.FRAME, "s")
    schedule_tracker.set_frequency(5.0)
    schedule_tracker.set_phase(0.25)
    schedule_tracker.shift_phase(0.5)
    schedule_tracker.set_amplitude(2.0 + 1.0j)
    schedule_event = schedule_tracker.advance(1.0)

    assert schedule_tracker.selected_resource == "q0"
    assert schedule_tracker.frequency == 5.0
    assert schedule_tracker.phase == 0.75
    assert schedule_tracker.amplitude == 2.0 + 1.0j
    assert schedule_event.duration == 1.0
    assert schedule_tracker.records == (schedule_event,)
    schedule_tracker.synchronise()


def test_schedule_tracker_synchronise_accepts_no_resources():
    schedule_tracker = ScheduleTracker()

    schedule_tracker.synchronise()

    assert schedule_tracker.resources == ()
    assert schedule_tracker.records == ()


@pytest.mark.parametrize(
    ("unit", "seconds"),
    [
        ("s", 1.0),
        ("ms", 1e-3),
        ("us", 1e-6),
        ("ns", 1e-9),
        ("ps", 1e-12),
    ],
)
def test_schedule_tracker_supports_native_time_units(unit, seconds):
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("q0", ResourceKind.FRAME, unit)
    schedule_tracker.set_frequency(0.25 / seconds)

    event = schedule_tracker.advance(1.0, signal=[1.0])

    assert event.signal == pytest.approx([1.0])
    assert schedule_tracker.phase == pytest.approx(np.pi / 2)


def test_schedule_tracker_rejects_unsupported_time_unit_without_recording_event():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("q0", ResourceKind.FRAME, "minute")

    with pytest.raises(ValueError, match="Unsupported schedule time unit"):
        schedule_tracker.advance(1.0)

    assert schedule_tracker.records == ()
    assert schedule_tracker.timelines["q0"] == ()


@pytest.mark.parametrize(
    ("method", "value", "name"),
    [
        ("set_frequency", float("inf"), "frequency"),
        ("set_phase", float("nan"), "phase"),
        ("set_amplitude", complex(float("inf")), "amplitude"),
    ],
)
def test_schedule_tracker_rejects_non_finite_state(method, value, name):
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("q0", ResourceKind.FRAME, "s")

    with pytest.raises(ValueError, match=name):
        getattr(schedule_tracker, method)(value)
