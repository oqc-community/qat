# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2025 Oxford Quantum Circuits Ltd
from __future__ import annotations

from typing import Optional, Type, Union

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import Field, model_validator

from qat.ir.instructions import QuantumInstruction
from qat.utils.pydantic import AllowExtraFieldsModel, find_all_subclasses
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
    @property
    def name(self):
        return self.__class__.__name__.replace("Waveform", "")


class Waveform(AbstractWaveform):
    """
    Stores the attributes which define a waveform.
    """

    width: float = Field(ge=0, default=0)
    amp: float | complex = 0.0
    phase: float | complex = 0.0

    # All of the following are shape specific, we should probably deal with this better.
    drag: float = 0.0
    rise: float = 0.0
    amp_setup: float = 0.0
    scale_factor: float = 1.0
    zero_at_edges: bool = False
    beta: float = 0.0
    frequency: float = 0.0
    internal_phase: float = 0.0
    std_dev: float = 0.0
    square_width: float = 0.0

    shape_function_type: Optional[Type[ComplexFunction]] = None

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
    """
    Provide a list of amplitudes to define a sampled waveform.
    """

    # TODO: Investigate linting issue with typehint Shape["* x"]
    samples: NDArray[Shape["* x"], int | float | complex]  # noqa: F722

    @property
    def duration(self):
        # TODO: Calculating this duration requires knowledge of the pulse channel, which
        # is not stored here. I think what we should actually be doing is calculating
        # the duration of a Pulse instruction when we calculate the position map.
        # TODO: Review for COMPILER-642 changes
        return 0.0

    def __repr__(self):
        return "sampled waveform"

    def __eq__(self, other: SampledWaveform):
        return np.array_equal(self.samples, other.samples)


class SquareWaveform(Waveform):
    shape_function_type: Type[SquareFunction] = SquareFunction


class SoftSquareWaveform(Waveform):
    shape_function_type: Type[SoftSquareFunction] = SoftSquareFunction


class SofterSquareWaveform(Waveform):
    shape_function_type: Type[SofterSquareFunction] = SofterSquareFunction


class ExtraSoftSquareWaveform(Waveform):
    shape_function_type: Type[ExtraSoftSquareFunction] = ExtraSoftSquareFunction


class GaussianWaveform(Waveform):
    shape_function_type: Type[GaussianFunction] = GaussianFunction


class SofterGaussianWaveform(Waveform):
    shape_function_type: Type[SofterGaussianFunction] = SofterGaussianFunction


class BlackmanWaveform(Waveform):
    shape_function_type: Type[BlackmanFunction] = BlackmanFunction


class SetupHoldWaveform(Waveform):
    shape_function_type: Type[SetupHoldFunction] = SetupHoldFunction


class RoundedSquareWaveform(Waveform):
    shape_function_type: Type[RoundedSquareFunction] = RoundedSquareFunction


class GaussianSquareWaveform(Waveform):
    shape_function_type: Type[GaussianSquareFunction] = GaussianSquareFunction


class DragGaussianWaveform(Waveform):
    shape_function_type: Type[DragGaussianFunction] = DragGaussianFunction


class GaussianZeroEdgeWaveform(Waveform):
    shape_function_type: Type[GaussianZeroEdgeFunction] = GaussianZeroEdgeFunction


class CosWaveform(Waveform):
    shape_function_type: Type[Cos] = Cos


class SinWaveform(Waveform):
    shape_function_type: Type[Sin] = Sin


class SechWaveform(Waveform):
    shape_function_type: Type[SechFunction] = SechFunction


waveform_classes = tuple(find_all_subclasses(Waveform) + [SampledWaveform])


class Pulse(QuantumInstruction):
    """
    Instructs a pulse channel to send a waveform. The intention of the waveform
    (e.g. a drive or measure pulse) can be specified using the type.
    """

    ignore_channel_scale: bool = False
    waveform: Union[waveform_classes]

    @model_validator(mode="before")
    def validate_duration(cls, data):
        # The duration of a pulse is equal to the width of the underlying waveform of the pulse.
        # Since `SampledWaveform`s do not have a width, only a set of samples, we cannot derive a
        # duration from such custom waveform and the duration must be supplied to the pulse.
        # TODO: Review for COMPILER-642 changes
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
            # TODO: Review for COMPILER-642 changes
            if sample_time is None:
                raise ValueError(
                    "Updating the duration of a pulse with 'SampledWaveform' waveform "
                    "requires 'sample_time' but None was provided."
                )
            current_duration = sample_time * self.waveform.samples.size
            if current_duration > duration:
                raise NotImplementedError(
                    "Cannot update the duration of a SampledWaveform to a smaller value. "
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
