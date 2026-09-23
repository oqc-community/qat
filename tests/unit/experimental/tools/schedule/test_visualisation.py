# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import numpy as np
import pytest
from matplotlib import pyplot as plt

from qat.experimental.tools.schedule import (
    ResourceKind,
    ScheduleTracker,
    visualise_schedule,
)


def test_visualise_schedule_is_dialect_independent():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("sequence", ResourceKind.SEQUENCE, "ns").advance(
        8, label="wait"
    )

    figure, axes = visualise_schedule(schedule_tracker)

    assert len(axes) == 2
    assert axes[0].get_xlabel() == "Time (ns)"
    assert axes[0].get_title() == "sequence"
    assert axes[1].get_ylabel() == "Phase (rad)"
    assert figure.axes == list(axes)
    assert figure.number not in plt.get_fignums()


def test_visualise_schedule_uses_resource_metadata_for_empty_resources():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource(
        "sequence",
        ResourceKind.SEQUENCE,
        "ns",
        signal_unit="DAC/ADC range",
        phase_label="NCO phase",
        phase_unit="steps",
        signal_limits=(-1.0, 1.0),
    )

    _, axes = visualise_schedule(schedule_tracker)

    assert [axis.get_xlabel() for axis in axes] == ["Time (ns)", "Time (ns)"]
    assert axes[0].get_ylabel() == "DAC/ADC range"
    assert axes[0].get_ylim() == (-1.0, 1.0)
    assert axes[1].get_ylabel() == "NCO phase (steps)"


def test_visualise_schedule_accepts_tracker_without_resources():
    figure, axes = visualise_schedule(ScheduleTracker())

    assert len(axes) == 2
    assert [axis.get_xlabel() for axis in axes] == ["Time", "Time"]
    assert figure.number not in plt.get_fignums()


def test_visualise_schedule_marks_instantaneous_events():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("sequence", ResourceKind.SEQUENCE, "ns")
    schedule_tracker.advance(0, label="barrier")

    _, axes = visualise_schedule(schedule_tracker)

    assert axes[0].texts[0].get_text() == "barrier"
    assert len(axes[0].lines) == 1
    assert len(axes[1].lines) == 1


def test_visualise_schedule_labels_timing_only_events_with_state():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("sequence", ResourceKind.SEQUENCE, "ns")
    schedule_tracker.set_frequency(31.25e6)
    schedule_tracker.advance(8, label="play")

    _, axes = visualise_schedule(schedule_tracker)

    assert any(
        text.get_text().startswith("play\nf=") and "phase=0" in text.get_text()
        for text in axes[0].texts
    )


@pytest.mark.parametrize(
    ("signal_interpolation", "drawstyle"),
    [("linear", "default"), ("zero_order_hold", "steps-post")],
)
def test_visualise_schedule_uses_native_sample_grid(signal_interpolation, drawstyle):
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource(
        "sequence",
        ResourceKind.SEQUENCE,
        "ns",
        "DAC/ADC range",
        phase_label="NCO phase",
        phase_unit="steps",
        phase_scale=1_000_000_000 / (2 * np.pi),
        signal_limits=(-1.0, 1.0),
    )
    schedule_tracker.advance(4, signal=[1.0, 1.0, 1.0, 1.0], label="play")
    schedule_tracker.advance(4, signal=[1.0, 1.0, 1.0, 1.0], label="wait")

    _, axes = visualise_schedule(
        schedule_tracker,
        signal_interpolation=signal_interpolation,
    )

    signal_lines = [line for line in axes[0].lines if line.get_color() in {"C0", "C1"}]
    assert signal_lines[0].get_xdata().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [line.get_drawstyle() for line in signal_lines] == [drawstyle] * 4
    assert [line.get_color() for line in signal_lines] == ["C0", "C1", "C0", "C1"]
    assert axes[0].get_ylabel() == "DAC/ADC range"
    assert axes[0].get_ylim() == (-1.0, 1.0)
    assert axes[1].get_ylabel() == "NCO phase (steps)"


def test_visualise_schedule_rejects_unknown_signal_interpolation():
    with pytest.raises(ValueError, match="Unsupported signal interpolation"):
        visualise_schedule(
            ScheduleTracker(),
            signal_interpolation="spline",  # type: ignore[arg-type]
        )


def test_visualise_schedule_plots_wrapped_nco_phase():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("sequence", ResourceKind.SEQUENCE, "ns")
    schedule_tracker.set_frequency(250e6)
    schedule_tracker.advance(4, signal=[1.0, 1.0, 1.0, 1.0], label="play")

    _, axes = visualise_schedule(schedule_tracker)

    phase_line = next(line for line in axes[1].lines if line.get_color() == "C2")
    phase_times = phase_line.get_xdata()
    phases = phase_line.get_ydata()
    assert phase_times[0] == 0.0
    assert phase_times[-1] == 4.0
    assert len(phase_times) >= 17
    assert np.isnan(phases).any()
    assert np.nanmin(phases) >= -np.pi
    assert np.nanmax(phases) <= np.pi
    assert axes[1].get_ylim() == pytest.approx((-np.pi, np.pi))


def test_visualise_schedule_scales_wrapped_phase_to_native_steps():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource(
        "sequence",
        ResourceKind.SEQUENCE,
        "ns",
        phase_label="NCO phase",
        phase_unit="steps",
        phase_scale=1_000_000_000 / (2 * np.pi),
    )
    schedule_tracker.set_phase(np.pi / 2)
    schedule_tracker.advance(1, signal=[1.0], label="play")

    _, axes = visualise_schedule(schedule_tracker)

    phase_line = next(line for line in axes[1].lines if line.get_color() == "C2")
    assert phase_line.get_ydata().tolist() == pytest.approx(
        [250_000_000, 250_000_000, 250_000_000]
    )
    assert axes[1].get_ylim() == pytest.approx((-500_000_000, 500_000_000))


def test_visualise_schedule_samples_phase_independently_of_sparse_signal():
    schedule_tracker = ScheduleTracker()
    schedule_tracker.resource("frame", ResourceKind.FRAME, "s")
    schedule_tracker.set_frequency(5e9)
    schedule_tracker.advance(2e-9, signal=[0.0, 0.0], label="wait")

    _, axes = visualise_schedule(schedule_tracker)

    phase_line = next(line for line in axes[1].lines if line.get_color() == "C2")
    assert np.count_nonzero(np.isfinite(phase_line.get_ydata())) >= 161
