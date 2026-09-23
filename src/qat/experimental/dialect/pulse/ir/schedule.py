# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Pulse operation visitor for the generic schedule ledger."""

from __future__ import annotations

from functools import singledispatchmethod

import numpy as np
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.traits import ConstantLike

from qat.experimental.dialect.pulse.ir.attributes import SampledWaveformAttr
from qat.experimental.dialect.pulse.ir.interfaces import IsAnalyticalWaveformInterface
from qat.experimental.dialect.pulse.ir.ops import (
    AcquireOp,
    AddOp,
    ConstantOp,
    CreateFrameOp,
    MixOp,
    PhaseOp,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SubOp,
    SynchronizeOp,
    WaitOp,
)
from qat.experimental.dialect.pulse.ir.types import WaveformType
from qat.experimental.tools.schedule import ResourceKind, ScheduleTracker
from qat.experimental.waveforms.evaluate import evaluate_waveform


class PulseScheduleVisitor:
    """Translate statically scheduled Pulse operations into ledger updates."""

    def __init__(
        self,
        tracker: ScheduleTracker,
        sample_time: float = 1e-9,
        signal_unit: str = "DAC/ADC range",
        signal_limits: tuple[float, float] | None = (-1.0, 1.0),
    ) -> None:
        self.tracker = tracker
        self.sample_time = sample_time
        self.signal_unit = signal_unit
        self.signal_limits = signal_limits
        self._waveforms: dict[SSAValue, tuple[float, np.ndarray]] = {}
        self._frames: dict[SSAValue, str] = {}
        self._port_frame_counts: dict[str, int] = {}
        self._continuous_amplitudes: dict[str, complex | None] = {}

    @singledispatchmethod
    def visit(self, operation: Operation) -> None:
        raise NotImplementedError(
            f"Pulse schedule visitor does not support {type(operation).__name__}"
        )

    @visit.register
    def _(self, operation: ConstantOp) -> None:
        value = ConstantLike.get_constant_value(operation.result)
        if isinstance(value, SampledWaveformAttr):
            self._waveforms[operation.result] = (
                float(value.width.literal_value),
                value.literal_value,
            )

    @visit.register
    def _(self, operation: IsAnalyticalWaveformInterface) -> None:
        width = self._scalar(operation.width)
        amplitude = self._scalar(operation.amplitude)
        shape = operation.build_shape()
        if width is None or amplitude is None or shape is None:
            raise ValueError(f"Pulse waveform {operation.name} requires static operands")
        width_ps = round(width * 1e12)
        sample_time_ps = round(self.sample_time * 1e12)
        samples = evaluate_waveform(
            width=width_ps,
            sample_time=sample_time_ps,
            shape=shape,
            amplitude=amplitude,
        )
        self._waveforms[operation.result] = (width, samples)

    @visit.register(AddOp)
    @visit.register(SubOp)
    @visit.register(MixOp)
    def _(self, operation: AddOp | SubOp | MixOp) -> None:
        try:
            lhs_width, lhs_samples = self._waveforms[operation.lhs]
            rhs_width, rhs_samples = self._waveforms[operation.rhs]
        except KeyError as error:
            raise ValueError(
                f"Pulse waveform expression {operation.name} requires static operands"
            ) from error
        if (
            not np.isclose(lhs_width, rhs_width, rtol=1e-12, atol=0.0)
            or lhs_samples.shape != rhs_samples.shape
        ):
            raise ValueError(
                f"Pulse waveform expression {operation.name} requires matching operands"
            )
        self._waveforms[operation.result] = (
            lhs_width,
            np.asarray(operation.py_operation(lhs_samples, rhs_samples), dtype=complex),
        )

    @visit.register
    def _(self, operation: CreateFrameOp) -> None:
        frequency = self._scalar(operation.frequency)
        if frequency is None:
            raise ValueError("Pulse frame frequency must be statically known")
        port = operation.port.data
        frame_count = self._port_frame_counts.get(port, 0) + 1
        self._port_frame_counts[port] = frame_count
        resource = port if frame_count == 1 else f"{port} [{frame_count}]"
        self.tracker.resource(
            resource,
            ResourceKind.FRAME,
            "s",
            signal_unit=self.signal_unit,
            signal_limits=self.signal_limits,
        ).set_frequency(frequency)
        self._frames[operation.result] = resource
        self._continuous_amplitudes[resource] = None

    @visit.register
    def _(self, operation: PhaseOp) -> None:
        phase = self._scalar(operation.phase)
        if phase is None:
            raise ValueError(f"{operation.name} requires a static phase")
        resource = self._select_frame(operation.frame)
        if operation.name == PhaseShiftOp.name:
            self.tracker.shift_phase(phase)
        elif operation.name == PhaseSetOp.name:
            self.tracker.set_phase(phase)
        else:
            raise NotImplementedError(f"Unsupported Pulse phase operation {operation.name}")
        self._frames[operation.result] = resource

    @visit.register
    def _(self, operation: WaitOp) -> None:
        duration = self._scalar(operation.duration)
        if duration is None:
            raise ValueError("Pulse wait duration must be statically known")
        resource = self._select_frame(operation.frame)
        self.tracker.wait(
            duration,
            self._continuous_signal(resource, duration),
        )
        self._frames[operation.result] = resource

    @visit.register
    def _(self, operation: SynchronizeOp) -> None:
        resources = [self._resource_for(frame) for frame in operation.frames]
        self._frames.update(zip(operation.results, resources, strict=True))

    @visit.register
    def _(self, operation: PulseOp) -> None:
        resource = self._select_frame(operation.frame)
        try:
            duration, signal = self._waveforms[operation.waveform]
        except KeyError as error:
            raise ValueError(
                "Pulse waveform must be visited before pulse operation"
            ) from error
        self.tracker.pulse(duration, signal)
        self._frames[operation.result] = resource

    @visit.register
    def _(self, operation: StartContinuousWaveformOp) -> None:
        amplitude = self._scalar(operation.amplitude)
        if amplitude is None:
            raise ValueError("Pulse continuous waveform amplitude must be statically known")
        resource = self._select_frame(operation.frame)
        self._continuous_amplitudes[resource] = complex(amplitude)
        self.tracker.mark("start continuous waveform")
        self._frames[operation.result] = resource

    @visit.register
    def _(self, operation: StopContinuousWaveformOp) -> None:
        resource = self._select_frame(operation.frame)
        self._continuous_amplitudes[resource] = None
        self.tracker.mark("stop continuous waveform")
        self._frames[operation.result] = resource

    @visit.register
    def _(self, operation: AcquireOp) -> None:
        duration = self._scalar(operation.duration)
        if duration is None:
            raise ValueError("Pulse acquisition duration must be statically known")
        resource = self._select_frame(operation.frame)
        self.tracker.acquire(
            duration,
            self._continuous_signal(resource, duration),
            operation.label.data if operation.label is not None else "acquire",
        )
        self._frames[operation.frame_result] = resource

    def _resource_for(self, frame: SSAValue) -> str:
        try:
            return self._frames[frame]
        except KeyError as error:
            raise ValueError("Pulse frame must be visited before it is used") from error

    def _select_frame(self, frame: SSAValue) -> str:
        resource = self._resource_for(frame)
        self.tracker.select(resource)
        return resource

    def _continuous_signal(self, resource: str, duration: float) -> np.ndarray | None:
        amplitude = self._continuous_amplitudes[resource]
        if duration == 0:
            return None
        if amplitude is None:
            return np.zeros(2, dtype=complex)
        sample_count = min(max(round(duration / self.sample_time), 2), 10_000)
        return np.full(sample_count, amplitude, dtype=complex)

    @staticmethod
    def _scalar(value: SSAValue) -> float | complex | None:
        constant = ConstantLike.get_constant_value(value)
        literal = getattr(constant, "literal_value", None)
        if literal is None:
            return None
        return literal


def build_pulse_schedule(
    module: ModuleOp,
) -> ScheduleTracker:
    """Walk a Pulse module and record its entry-block schedule.

    :param module: Pulse module containing the executable entry block.
    :returns: The populated schedule tracker.
    """
    from qat.experimental.dialect.pulse.ir.interfaces import PulseOperationInterface
    from qat.experimental.dialect.pulse.utils import pulse_entry_block

    tracker = ScheduleTracker()
    visitor = PulseScheduleVisitor(tracker)
    for operation in pulse_entry_block(module).ops:
        if operation.regions:
            raise ValueError(
                f"Pulse schedule plotting requires flat control flow, found "
                f"{operation.name}"
            )
        if isinstance(operation, PulseOperationInterface):
            operation.accept(visitor)
        elif isinstance(operation, AddOp | SubOp | MixOp) and isinstance(
            operation.result.type, WaveformType
        ):
            visitor.visit(operation)
    return tracker
