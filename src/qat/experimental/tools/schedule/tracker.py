# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""IR-independent ledger for statically scheduled resources."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from math import isclose, isfinite
from numbers import Real

import numpy as np
from numpy.typing import NDArray


class ResourceKind(str, Enum):
    """The two resource kinds that can be plotted."""

    FRAME = "frame"
    SEQUENCE = "sequence"


@dataclass(frozen=True)
class ScheduleResource:
    """Static metadata for a tracked schedule resource."""

    name: str
    kind: ResourceKind
    unit: str
    signal_unit: str
    phase_label: str
    phase_unit: str
    phase_scale: float
    signal_limits: tuple[float, float] | None


@dataclass(frozen=True)
class ScheduleEvent:
    """A single interval recorded for a schedule resource."""

    resource: str
    kind: ResourceKind
    unit: str
    signal_unit: str
    phase_label: str
    phase_unit: str
    phase_scale: float
    start: float
    end: float
    frequency: float
    phase: float
    amplitude: complex
    signal: NDArray[np.complexfloating] | None = None
    label: str | None = None

    @property
    def duration(self) -> float:
        """Return the interval duration in the resource's native unit."""
        return self.end - self.start


@dataclass
class _Resource:
    name: str
    kind: ResourceKind
    unit: str
    signal_unit: str
    phase_label: str
    phase_unit: str
    phase_scale: float
    signal_limits: tuple[float, float] | None
    time: float = 0.0
    frequency: float = 0.0
    phase: float = 0.0
    amplitude: complex = 1.0 + 0.0j
    events: list[ScheduleEvent] = field(default_factory=list)


class ScheduleTracker:
    """Track static frame or sequence timelines with a small ledger-like API.

    The tracker deliberately knows nothing about an IR. Dialect visitors select a resource,
    update its state, and append intervals through this API.
    """

    def __init__(self) -> None:
        self._resources: dict[str, _Resource] = {}
        self._selected: str | None = None
        self._records: list[ScheduleEvent] = []

    def resource(
        self,
        name: str,
        kind: ResourceKind,
        unit: str,
        signal_unit: str = "Amplitude",
        phase_unit: str = "rad",
        phase_scale: float = 1.0,
        phase_label: str = "Phase",
        signal_limits: tuple[Real, Real] | None = None,
    ) -> ScheduleTracker:
        """Select or create a resource.

        :param name: Stable resource name used by the visualisation.
        :param kind: Whether the resource is a frame or a sequence.
        :param unit: Native unit label for time values.
        :param signal_unit: Label describing the recorded signal scale.
        :param phase_unit: Native unit label for displayed phase values.
        :param phase_scale: Displayed phase units per radian.
        :param phase_label: Label describing the recorded phase.
        :param signal_limits: Optional lower and upper bounds for the signal axis.
        :returns: This tracker, for fluent visitor code.
        """
        phase_scale = self._finite(phase_scale, "phase scale")
        if phase_scale <= 0:
            raise ValueError(f"phase scale must be positive, got {phase_scale}")
        limits = None
        if signal_limits is not None:
            limits = (
                self._finite(signal_limits[0], "signal lower limit"),
                self._finite(signal_limits[1], "signal upper limit"),
            )
            if limits[0] >= limits[1]:
                raise ValueError("signal lower limit must be below signal upper limit")
        existing = self._resources.get(name)
        if existing is None:
            self._resources[name] = _Resource(
                name,
                kind,
                unit,
                signal_unit,
                phase_label,
                phase_unit,
                phase_scale,
                limits,
            )
        elif (
            existing.kind != kind
            or existing.unit != unit
            or existing.signal_unit != signal_unit
            or existing.phase_label != phase_label
            or existing.phase_unit != phase_unit
            or existing.phase_scale != phase_scale
            or existing.signal_limits != limits
        ):
            raise ValueError(f"Resource {name!r} was registered with conflicting metadata")
        self._selected = name
        return self

    def select(self, name: str) -> ScheduleTracker:
        """Select an existing resource."""
        if name not in self._resources:
            raise KeyError(f"Unknown schedule resource {name!r}")
        self._selected = name
        return self

    @property
    def selected_resource(self) -> str:
        """Return the selected resource name."""
        if self._selected is None:
            raise RuntimeError("No schedule resource is selected")
        return self._selected

    def _current(self) -> _Resource:
        return self._resources[self.selected_resource]

    def set_frequency(self, frequency: Real) -> None:
        """Set the selected resource frequency."""
        self._current().frequency = self._finite(frequency, "frequency")

    def set_phase(self, phase: Real) -> None:
        """Set the selected resource phase."""
        phase_value = self._finite(phase, "phase") % (2 * np.pi)
        self._current().phase = 0.0 if np.isclose(phase_value, 2 * np.pi) else phase_value

    def shift_phase(self, phase: Real) -> None:
        """Shift the selected resource phase."""
        self.set_phase(self.phase + self._finite(phase, "phase shift"))

    def set_amplitude(self, amplitude: complex | Real) -> None:
        """Set the selected resource amplitude."""
        value = complex(amplitude)
        if not isfinite(value.real) or not isfinite(value.imag):
            raise ValueError(f"amplitude must be finite, got {amplitude!r}")
        self._current().amplitude = value

    @property
    def phase(self) -> float:
        """Return the selected resource phase."""
        return self._current().phase

    @property
    def frequency(self) -> float:
        """Return the selected resource frequency."""
        return self._current().frequency

    @property
    def amplitude(self) -> complex:
        """Return the selected resource amplitude."""
        return self._current().amplitude

    def advance(
        self,
        duration: Real,
        signal: Iterable[complex] | NDArray[np.complexfloating] | None = None,
        label: str | None = None,
    ) -> ScheduleEvent:
        """Append an interval to the selected resource and advance its time."""
        duration_value = self._finite(duration, "duration")
        if duration_value < 0:
            raise ValueError(f"duration must be non-negative, got {duration_value}")

        resource = self._current()
        seconds_per_unit = self._seconds_per_unit(resource.unit)
        samples = (
            None
            if signal is None
            else np.asarray(
                signal if isinstance(signal, np.ndarray) else tuple(signal),
                dtype=complex,
            )
        )
        if samples is not None:
            sample_times = np.linspace(0.0, duration_value, samples.size, endpoint=False)
            sample_times *= seconds_per_unit
            samples = (
                samples
                * resource.amplitude
                * np.exp(
                    1j * (resource.phase + 2 * np.pi * resource.frequency * sample_times)
                )
            )
        event = ScheduleEvent(
            resource=resource.name,
            kind=resource.kind,
            unit=resource.unit,
            signal_unit=resource.signal_unit,
            phase_label=resource.phase_label,
            phase_unit=resource.phase_unit,
            phase_scale=resource.phase_scale,
            start=resource.time,
            end=resource.time + duration_value,
            frequency=resource.frequency,
            phase=resource.phase,
            amplitude=resource.amplitude,
            signal=samples,
            label=label,
        )
        resource.events.append(event)
        self._records.append(event)
        resource.time = event.end
        phase = (
            resource.phase
            + 2 * np.pi * resource.frequency * duration_value * seconds_per_unit
        ) % (2 * np.pi)
        resource.phase = 0.0 if np.isclose(phase, 2 * np.pi) else phase
        return event

    def wait(
        self,
        duration: Real,
        signal: Iterable[complex] | NDArray[np.complexfloating] | None = None,
    ) -> ScheduleEvent:
        """Record a wait interval on the selected resource."""
        return self.advance(duration, signal, label="wait")

    def pulse(
        self,
        duration: Real,
        signal: Iterable[complex] | NDArray[np.complexfloating],
    ) -> ScheduleEvent:
        """Record a pulse interval on the selected resource."""
        return self.advance(duration, signal, label="pulse")

    def acquire(
        self,
        duration: Real,
        signal: Iterable[complex] | NDArray[np.complexfloating] | None = None,
        label: str = "acquire",
    ) -> ScheduleEvent:
        """Record an acquisition interval on the selected resource."""
        return self.advance(duration, signal, label=label)

    def mark(self, label: str) -> ScheduleEvent:
        """Record a zero-duration state transition on the selected resource."""
        return self.advance(0, label=label)

    def synchronise(self, resources: Iterable[str] | None = None) -> None:
        """Advance selected resources to the latest current time."""
        names = list(self._resources if resources is None else resources)
        if not names:
            return
        end_seconds = max(
            self._resources[name].time * self._seconds_per_unit(self._resources[name].unit)
            for name in names
        )
        for name in names:
            resource = self._resources[name]
            seconds_per_unit = self._seconds_per_unit(resource.unit)
            end = end_seconds / seconds_per_unit
            if resource.time < end and not isclose(
                resource.time, end, rel_tol=1e-12, abs_tol=0.0
            ):
                self.select(name).advance(end - resource.time, label="synchronise")

    @property
    def records(self) -> tuple[ScheduleEvent, ...]:
        """Return all recorded events in insertion order."""
        return tuple(self._records)

    @property
    def timelines(self) -> dict[str, tuple[ScheduleEvent, ...]]:
        """Return recorded events grouped by resource."""
        return {name: tuple(resource.events) for name, resource in self._resources.items()}

    @property
    def resources(self) -> tuple[str, ...]:
        """Return resource names in registration order."""
        return tuple(self._resources)

    def resource_info(self, name: str) -> ScheduleResource:
        """Return static metadata for a resource."""
        resource = self._resources[name]
        return ScheduleResource(
            resource.name,
            resource.kind,
            resource.unit,
            resource.signal_unit,
            resource.phase_label,
            resource.phase_unit,
            resource.phase_scale,
            resource.signal_limits,
        )

    @staticmethod
    def _finite(value: Real, name: str) -> float:
        if not isinstance(value, Real) or not isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real number, got {value!r}")
        return float(value)

    @staticmethod
    def _seconds_per_unit(unit: str) -> float:
        try:
            return {
                "s": 1.0,
                "ms": 1e-3,
                "us": 1e-6,
                "ns": 1e-9,
                "ps": 1e-12,
            }[unit]
        except KeyError as error:
            raise ValueError(f"Unsupported schedule time unit {unit!r}") from error
