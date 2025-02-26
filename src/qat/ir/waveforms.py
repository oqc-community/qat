# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2024 Oxford Quantum Circuits Ltd
from enum import Enum
from typing import List, Literal, Union

import numpy as np
from pydantic import Field, model_validator

from qat.ir.instructions import QuantumInstruction
from qat.purr.compiler.devices import PulseShapeType
from qat.utils.pydantic import NoExtraFieldsModel, ValidatedSet


class AbstractWaveform(NoExtraFieldsModel): ...


class Waveform(AbstractWaveform):
    """
    Stores the attributes which define a waveform.
    """

    shape: PulseShapeType
    width: float = Field(ge=0, default=0)
    amp: float = 0.0
    phase: float = 0.0

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

    @property
    def duration(self):
        return self.width

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, width={self.width}, amp={self.amp}, phase={self.phase})"


class CustomWaveform(AbstractWaveform):
    """
    Provide a list of amplitudes to define a custom waveform.
    """

    samples: List[Union[complex, float]]

    @property
    def duration(self):
        # TODO: Calculating this duration requires knowledge of the pulse channel, which
        # is not stored here. I think what we should actually be doing is calculating
        # the duration of a Pulse instruction when we calculate the position map.
        return 0.0

    def __repr__(self):
        return f"custom waveform"


class PulseType(Enum):
    """
    States the intention of a pulse, e.g., to drive a Qubit, or take a measurement.
    """

    DRIVE = "drive"
    MEASURE = "measure"
    SECOND_STATE = "second_state"
    CROSS_RESONANCE = "cross_resonance"
    CROSS_RESONANCE_CANCEL = "cross_resonance_cancel"
    OTHER = "other"


class Pulse(QuantumInstruction):
    """
    Instructs a pulse channel to send a waveform. The intention of the waveform
    (e.g. a drive or measure pulse) can be specified using the type.
    """

    inst: Literal["Pulse"] = "Pulse"
    targets: ValidatedSet[str] = Field(max_length=1)
    ignore_channel_scale: bool = False
    type: PulseType = PulseType.OTHER
    waveform: Union[Waveform, CustomWaveform]

    @model_validator(mode="before")
    def validate_duration(cls, data):
        # The duration of a pulse is equal to the width of the underlying waveform of the pulse.
        # Since `CustomWaveform`s do not have a width, only a set of samples, we cannot derive a
        # duration from such custom waveform and the duration must be supplied to the pulse.
        if isinstance(data["waveform"], Waveform):
            data["duration"] = data["waveform"].duration
        elif isinstance(data["waveform"], CustomWaveform) and np.isclose(
            data["duration"], 0
        ):
            raise ValueError("Duration of a `CustomWaveform` cannot be zero.")
        return data

    def __repr__(self):
        return f"{self.type} pulse on targets {self.targets} with waveform {self.waveform}"

    @property
    def channel(self):
        return next(iter(self.targets))
