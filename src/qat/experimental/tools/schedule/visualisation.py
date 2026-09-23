# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Visualise generic schedule tracker output."""

from __future__ import annotations

from math import ceil
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from qat.experimental.tools.schedule.tracker import ScheduleEvent, ScheduleTracker

_SECONDS_PER_UNIT = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
}
SignalInterpolation = Literal["linear", "zero_order_hold"]


def visualise_schedule(
    tracker: ScheduleTracker,
    signal_interpolation: SignalInterpolation = "linear",
) -> tuple[Figure, np.ndarray]:
    """Render tracker events without displaying the resulting figure.

    :param tracker: Generic schedule tracker containing recorded events.
    :param signal_interpolation: Whether to join samples linearly or hold each sample until
        the next one.
    :returns: The matplotlib figure and consecutive amplitude and phase axes for each
        tracked resource.
    """
    try:
        drawstyle = {
            "linear": "default",
            "zero_order_hold": "steps-post",
        }[signal_interpolation]
    except KeyError as error:
        raise ValueError(
            f"Unsupported signal interpolation {signal_interpolation!r}"
        ) from error

    count = max(len(tracker.resources), 1)
    figure, axes = plt.subplots(2 * count, 1, squeeze=False, figsize=(10, 5 * count))
    flat_axes: np.ndarray = axes[:, 0]

    if not tracker.resources:
        for axis in flat_axes:
            axis.set_xlabel("Time")
        plt.close(figure)
        return figure, flat_axes

    for resource_index, resource_name in enumerate(tracker.resources):
        amplitude_axis = flat_axes[2 * resource_index]
        phase_axis = flat_axes[2 * resource_index + 1]
        events = tracker.timelines[resource_name]
        resource = tracker.resource_info(resource_name)
        unit = resource.unit
        signal_unit = resource.signal_unit
        phase_label = resource.phase_label
        phase_unit = resource.phase_unit
        phase_scale = resource.phase_scale
        amplitude_axis.set_title(resource_name)
        amplitude_axis.set_ylabel(signal_unit)
        phase_axis.set_ylabel(f"{phase_label} ({phase_unit})")
        time_label = f"Time ({unit})" if unit else "Time"
        amplitude_axis.set_xlabel(time_label)
        phase_axis.set_xlabel(time_label)
        phase_limit = np.pi * phase_scale
        phase_axis.set_ylim(-phase_limit, phase_limit)
        if phase_unit == "rad":
            phase_axis.set_yticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
        signal_labels: set[str] = set()
        for event in events:
            if event.start == event.end:
                amplitude_axis.axvline(event.start, color="black", linewidth=0.8, alpha=0.8)
                phase_axis.axvline(event.start, color="black", linewidth=0.8, alpha=0.8)
                amplitude_axis.text(
                    event.start,
                    0.95,
                    event.label,
                    ha="left",
                    va="top",
                    fontsize=8,
                    rotation=90,
                    transform=amplitude_axis.get_xaxis_transform(),
                )
                continue
            if event.signal is None or event.signal.size == 0:
                amplitude_axis.axvspan(
                    event.start,
                    event.end,
                    alpha=0.3,
                    label=event.label,
                    edgecolor="black",
                    linewidth=0.5,
                )
                midpoint = (event.start + event.end) / 2
                amplitude_axis.text(
                    midpoint,
                    0.5,
                    f"{event.label}\nf={event.frequency:g} Hz\nphase={event.phase:.3g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    rotation=90,
                    transform=amplitude_axis.get_xaxis_transform(),
                )
            else:
                times = np.linspace(
                    event.start,
                    event.end,
                    event.signal.size,
                    endpoint=False,
                )
                plot_times = np.append(times, event.end)
                amplitude_axis.axvline(
                    event.start,
                    color="black",
                    linewidth=0.5,
                    alpha=0.2,
                )
                amplitude_axis.text(
                    event.start,
                    0.98,
                    event.label,
                    ha="left",
                    va="top",
                    fontsize=7,
                    rotation=90,
                    transform=amplitude_axis.get_xaxis_transform(),
                )
                for component, values in (
                    ("I", event.signal.real),
                    ("Q", event.signal.imag),
                ):
                    label = component if component not in signal_labels else None
                    colour = "C0" if component == "I" else "C1"
                    plot_values = np.append(values, values[-1])
                    amplitude_axis.plot(
                        plot_times,
                        plot_values,
                        color=colour,
                        label=label,
                        drawstyle=drawstyle,
                    )
                    signal_labels.add(component)

            phase_times, phases = _phase_trace(event)
            phase_axis.axvline(
                event.start,
                color="black",
                linewidth=0.5,
                alpha=0.2,
            )
            phase_axis.plot(phase_times, phases, color="C2")
        handles, labels = amplitude_axis.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles, strict=True))
            amplitude_axis.legend(unique.values(), unique.keys())
        amplitude_axis.autoscale()
        if resource.signal_limits is not None:
            amplitude_axis.set_ylim(resource.signal_limits)

    figure.tight_layout()
    plt.close(figure)
    return figure, flat_axes


def _phase_trace(event: ScheduleEvent) -> tuple[np.ndarray, np.ndarray]:
    signal_sample_count = event.signal.size if event.signal is not None else 0
    seconds = event.duration * _SECONDS_PER_UNIT[event.unit]
    cycles = abs(event.frequency) * seconds
    sample_count = min(max(signal_sample_count, ceil(cycles * 16), 2), 10_000)
    times = np.linspace(event.start, event.end, sample_count + 1)
    elapsed_seconds = (times - event.start) * _SECONDS_PER_UNIT[event.unit]
    phase = event.phase + 2 * np.pi * event.frequency * elapsed_seconds
    wrapped_phase = (phase + np.pi) % (2 * np.pi) - np.pi
    displayed_phase = wrapped_phase * event.phase_scale
    wrap_indices = np.flatnonzero(np.abs(np.diff(wrapped_phase)) > np.pi) + 1
    if wrap_indices.size:
        times = np.insert(times, wrap_indices, times[wrap_indices])
        displayed_phase = np.insert(displayed_phase, wrap_indices, np.nan)
    return times, displayed_phase


plot_schedule = visualise_schedule

__all__ = ["plot_schedule", "visualise_schedule"]
