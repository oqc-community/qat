# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from enum import Enum
from typing import List, Literal, Union

from pydantic import BaseModel

from qat.ir.instructions import QuantumInstruction, Variable
from qat.purr.compiler.devices import PulseShapeType


class AbstractWaveform(BaseModel): ...


class Waveform(AbstractWaveform):
    """
    Stores the attributes which define a waveform.
    """

    shape: PulseShapeType
    width: Union[Variable, float] = 0.0
    amp: Union[Variable, float] = 0.0
    phase: Union[Variable, float] = 0.0

    # All of the following are shape specific, we should probably deal with this better.
    drag: Union[Variable, float] = 0.0
    rise: Union[Variable, float] = 0.0
    amp_setup: Union[Variable, float] = 0.0
    scale_factor: Union[Variable, float] = 1.0
    zero_at_edges: bool = False
    beta: Union[Variable, float] = 0.0
    frequency: Union[Variable, float] = 0.0
    internal_phase: Union[Variable, float] = 0.0
    std_dev: Union[Variable, float] = 0.0
    square_width: Union[Variable, float] = 0.0

    @property
    def duration(self):
        return self.width

    def __repr__(self):
        return f"waveform {self.shape.value},{self.width},{self.amp},{self.phase}"


class CustomWaveform(AbstractWaveform):
    samples: List[Union[complex, float]]

    @property
    def duration(self):
        # TODO: this depends on the pulse channel too, so we need to decide
        # how to deal with this.
        return 0.0

    def __repr__(self):
        return f"custom waveform"


class PulseType(Enum):
    """
    States the intention of a pulse, e.g., to drive a Qubit, or take a measurement.
    """

    DRIVE = "DRIVE"
    MEASURE = "MEASURE"
    SECOND_STATE = "SECOND_STATE"
    CROSS_RESONANCE = "CROSS_RESONANCE"
    CROSS_RESONANCE_CANCEL = "CROSS_RESONANCE_CANCEL"
    OTHER = "OTHER"


class Pulse(QuantumInstruction):
    """
    Instructs a pulse channel to send a waveform. The intention of the waveform
    (e.g. a drive or measure pulse) can be specified using the type.
    """

    inst: Literal["Pulse"] = "Pulse"
    targets: str
    ignore_channel_scale: bool = False
    type: PulseType = PulseType.OTHER
    waveform: Union[Waveform, CustomWaveform]

    def __repr__(self):
        return f"pulse {self.channel},{self.waveform}"

    @property
    def duration(self):
        return self.waveform.duration

    @property
    def channel(self):
        return self.targets
