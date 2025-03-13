# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2025 Oxford Quantum Circuits Ltd
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Type, Union

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import Field, model_validator

from qat.ir.instructions import QuantumInstruction
from qat.utils.pydantic import AllowExtraFieldsModel, FrozenSet, find_all_subclasses
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


class SampledWaveform(AbstractWaveform):
    """
    Provide a list of amplitudes to define a sampled waveform.
    """

    samples: NDArray[Shape["* x"], int | float | complex]

    @property
    def duration(self):
        # TODO: Calculating this duration requires knowledge of the pulse channel, which
        # is not stored here. I think what we should actually be doing is calculating
        # the duration of a Pulse instruction when we calculate the position map.
        return 0.0

    def __repr__(self):
        return f"sampled waveform"

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


class PulseType(Enum):
    # TODO: Remove `PulseType` class as we do not really need this, the pulse type is implicitly encoded in the pulse channel class. #327
    """
    States the intention of a pulse, e.g., to drive a Qubit, or take a measurement.
    """

    DRIVE = "drive"
    MEASURE = "measure"
    SECOND_STATE = "second_state"
    CROSS_RESONANCE = "cross_resonance"
    CROSS_RESONANCE_CANCEL = "cross_resonance_cancel"
    OTHER = "other"


waveform_classes = tuple(find_all_subclasses(Waveform) + [SampledWaveform])


class Pulse(QuantumInstruction):
    """
    Instructs a pulse channel to send a waveform. The intention of the waveform
    (e.g. a drive or measure pulse) can be specified using the type.
    """

    inst: Literal["Pulse"] = "Pulse"
    targets: FrozenSet[str] = Field(max_length=1)
    ignore_channel_scale: bool = False
    type: PulseType = PulseType.OTHER
    waveform: Union[waveform_classes]

    @model_validator(mode="before")
    def validate_duration(cls, data):
        # The duration of a pulse is equal to the width of the underlying waveform of the pulse.
        # Since `SampledWaveform`s do not have a width, only a set of samples, we cannot derive a
        # duration from such custom waveform and the duration must be supplied to the pulse.
        if isinstance(data["waveform"], Waveform):
            data["duration"] = data["waveform"].duration
        return data

    def __repr__(self):
        return f"{self.__class__.__name__} with type '{self.type.value}' on targets {set(self.targets)} with {self.waveform}."

    @property
    def target(self):
        return next(iter(self.targets))

    @property
    def pulse_channel(self):
        return self.target
