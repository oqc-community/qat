# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2026 Oxford Quantum Circuits Ltd
"""Contains elementary IR units for waveforms, including waveform types and pulses."""

from __future__ import annotations

import numpy as np
from pydantic import Field, PositiveFloat, model_validator

from qat.ir.instructions import QuantumInstruction
from qat.utils.pydantic import (
    AllowExtraFieldsModel,
    ComplexNDArray,
    FloatNDArray,
    find_all_subclasses,
)
from qat.utils.waveform import (
    BlackmanFunction,
    ComplexFunction,
    Cos,
    DragGaussianFunction,
    ExtraSoftSquareFunction,
    GaussianFunction,
    GaussianSquareFunction,
    GaussianZeroEdgeFunction,
    RoundedSquareFunction,
    SechFunction,
    SetupHoldFunction,
    Sin,
    SofterGaussianFunction,
    SofterSquareFunction,
    SoftSquareFunction,
    SquareFunction,
)


class AbstractWaveform(AllowExtraFieldsModel):
    @classmethod
    def name(cls) -> str:
        return cls.__name__.replace("Waveform", "")


class Waveform(AbstractWaveform):
    """Stores the attributes which define a waveform."""

    width: float = Field(ge=0, default=0)
    amp: float | complex = 0.0
    phase: float | complex = 0.0

    # All of the following are shape specific, we should probably deal with this better.
    drag: float = 0.0
    rise: float = 0.0
    amp_setup: float = 0.0
    scale_factor: float | complex = 1.0
    zero_at_edges: bool = False
    beta: float = 0.0
    frequency: float = 0.0
    internal_phase: float = 0.0
    std_dev: float = 0.0
    square_width: float = 0.0

    shape_function_type: type[ComplexFunction] | None = None

    @property
    def duration(self):
        return self.width

    def __repr__(self):
        return f"{self.__class__.__name__}(width={self.width}, amp={self.amp}, phase={self.phase})"

    def sample(self, t: np.ndarray, phase_offset: float = 0.0):
        if self.shape_function_type is None:
            raise AttributeError(
                f"Waveform of type `{self.__class__.__name__}` cannot be evaluated, please provide a valid shape function type."
            )

        # Generate a shape function based on the given type. The shape function
        # will only use whatever member attributes from this class it needs.
        shape_function = self.shape_function_type(**self.model_dump())

        amplitude = shape_function(t)
        phase_offset = self.phase + phase_offset
        samples = self.scale_factor * self.amp * np.exp(1.0j * phase_offset) * amplitude

        if self.drag:
            amplitude_differential = shape_function.derivative(t, amplitude)
            if len(amplitude_differential) < len(samples):
                amplitude_differential = np.pad(
                    amplitude_differential,
                    (0, len(samples) - len(amplitude_differential)),
                    "edge",
                )
            samples += (
                self.drag
                * 1.0j
                * self.amp
                * self.scale_factor
                * np.exp(1.0j * phase_offset)
                * amplitude_differential
            )

        return SampledWaveform(samples=samples)

    def __hash__(self):
        return hash(
            (
                self.__class__.__name__,
                self.width,
                self.amp,
                self.phase,
                self.drag,
                self.rise,
                self.amp_setup,
                self.scale_factor,
                self.zero_at_edges,
                self.beta,
                self.frequency,
                self.internal_phase,
                self.std_dev,
                self.square_width,
            )
        )


class SampledWaveform(AbstractWaveform):
    """Provide a list of amplitudes to define a sampled waveform."""

    samples: ComplexNDArray | FloatNDArray
    sample_time: float | None = None  # Time between samples, in seconds

    @property
    def duration(self):
        if self.sample_time is None:
            # TODO: COMPILER-723 -- Do we want to raise an error here, or return NAN or None or 0?
            raise ValueError(
                "Cannot determine duration of SampledWaveform without sample_time being set."
            )
        return self.sample_time * len(self.samples)

    def __repr__(self):
        return "sampled waveform"

    def __eq__(self, other: SampledWaveform):
        return np.array_equal(self.samples, other.samples)


class SquareWaveform(Waveform):
    shape_function_type: type[SquareFunction] = SquareFunction


class SoftSquareWaveform(Waveform):
    shape_function_type: type[SoftSquareFunction] = SoftSquareFunction


class SofterSquareWaveform(Waveform):
    shape_function_type: type[SofterSquareFunction] = SofterSquareFunction


class ExtraSoftSquareWaveform(Waveform):
    shape_function_type: type[ExtraSoftSquareFunction] = ExtraSoftSquareFunction


class GaussianWaveform(Waveform):
    shape_function_type: type[GaussianFunction] = GaussianFunction


class SofterGaussianWaveform(Waveform):
    shape_function_type: type[SofterGaussianFunction] = SofterGaussianFunction


class BlackmanWaveform(Waveform):
    shape_function_type: type[BlackmanFunction] = BlackmanFunction


class SetupHoldWaveform(Waveform):
    shape_function_type: type[SetupHoldFunction] = SetupHoldFunction


class RoundedSquareWaveform(Waveform):
    shape_function_type: type[RoundedSquareFunction] = RoundedSquareFunction


class GaussianSquareWaveform(Waveform):
    shape_function_type: type[GaussianSquareFunction] = GaussianSquareFunction


class DragGaussianWaveform(Waveform):
    shape_function_type: type[DragGaussianFunction] = DragGaussianFunction


class GaussianZeroEdgeWaveform(Waveform):
    shape_function_type: type[GaussianZeroEdgeFunction] = GaussianZeroEdgeFunction


class CosWaveform(Waveform):
    shape_function_type: type[Cos] = Cos


class SinWaveform(Waveform):
    shape_function_type: type[Sin] = Sin


class SechWaveform(Waveform):
    shape_function_type: type[SechFunction] = SechFunction


waveform_classes = tuple(find_all_subclasses(Waveform) + [SampledWaveform])


class Pulse(QuantumInstruction):
    """Instructs a pulse channel to send a waveform.

    The intention of the waveform (e.g. a drive or measure pulse) can be specified using the
    type.
    """

    ignore_channel_scale: bool = False
    waveform: Waveform | SampledWaveform

    @model_validator(mode="before")
    def validate_duration(cls, data):
        # TODO: Review with COMPILER-723
        if isinstance(data, dict) and isinstance(data.get("waveform"), Waveform):
            data["duration"] = data["waveform"].duration
        return data

    def __repr__(self):
        return f"{self.__class__.__name__} on targets {set(self.targets)} with {self.waveform}."

    def update_duration(self, duration: float, sample_time: float | None = None):
        if isinstance(self.waveform, Waveform):
            self.duration = duration
            self.waveform.width = duration
        elif isinstance(self.waveform, SampledWaveform):
            if sample_time is None:
                sample_time = self.waveform.sample_time
            else:
                self.waveform.sample_time = sample_time

            current_duration = self.waveform.duration
            if current_duration > duration and not np.isclose(current_duration, duration):
                raise NotImplementedError(
                    "Cannot update the duration of a SampledWaveform to a smaller value. "
                    f"{current_duration} > {duration}\n"
                    "This would require removing samples, which is not supported."
                )
            # If the new duration is larger, we can pad the samples with zeros.
            padding = int(np.round((duration - current_duration) / sample_time, 0))
            self.waveform.samples = np.pad(
                self.waveform.samples,
                (0, padding),
                mode="constant",
                constant_values=0,
            )
            self.duration = self.waveform.duration
        else:
            raise ValueError(
                f"{type(self.waveform)} does not support updating duration. "
                "Can only apply with a Waveform or SampledWaveform."
            )

    @property
    def target(self):
        return next(iter(self.targets))

    @property
    def pulse_channel(self):
        return self.target


def sample_waveform(waveform: Waveform, sample_time: PositiveFloat) -> SampledWaveform:
    """Utility function to sample a waveform at a given time per sample (sample rate).

    :param waveform: The analytical waveform to sample.
    :param sample_time: The time between samples, in seconds.
    :return: A SampledWaveform containing the sampled values.
    """

    edge = (waveform.duration - sample_time) / 2.0
    num_samples = int(np.ceil(waveform.duration / sample_time - 1e-10))
    t = np.linspace(start=-edge, stop=edge, num=num_samples)
    samples = waveform.sample(t)
    return SampledWaveform(samples=samples.samples, sample_time=sample_time)
