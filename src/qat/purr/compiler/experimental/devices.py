from typing import List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from qat.purr.compiler.devices import ChannelType


class QuantumComponent(BaseModel, validate_assignment=True):
    """
    Base class for any logical object which can act as a target of a quantum action
    - a Qubit or various channels for a simple example.
    """

    id: str = ""

    def full_id(self):
        return self.id

    def __repr__(self):
        return self.id


class PhysicalBaseband(QuantumComponent):
    frequency: float = Field(ge=0.0)
    if_frequency: Optional[float] = Field(ge=0.0, default=250e6)


class PhysicalChannel(QuantumComponent, validate_assignment=True):
    sample_time: float
    baseband: PhysicalBaseband
    block_size: int | None = Field(ge=1, default=1)
    phase_offset: float = 0.0
    imbalance: float = 1.0

    acquire_allowed: bool = False
    pulse_channel_min_frequency: float = Field(ge=0.0, default=0.0)
    pulse_channel_max_frequency: float = np.inf

    @field_validator("block_size")
    def set_block_size(cls, block_size):
        return block_size or 1


class PulseChannel(QuantumComponent, validate_assignment=True):
    """Models a pulse channel on a particular device."""

    physical_channel: PhysicalChannel

    frequency: float = Field(ge=0.0, default=0.0)
    bias: complex = 0.0 + 0.0j
    scale: complex = 1.0 + 0.0j

    fixed_if: bool = False

    @model_validator(mode="after")
    def check_frequency(self):
        min_frequency = self.physical_channel.pulse_channel_min_frequency
        max_frequency = self.physical_channel.pulse_channel_max_frequency
        full_id = self.id_

        if self.frequency < min_frequency or self.frequency > max_frequency:
            raise ValueError(
                f"Pulse channel frequency '{self.frequency}' must be between the bounds "
                f"({min_frequency}, {max_frequency}) on physical "
                f"channel with id {full_id}."
            )
        return self

    @property
    def sample_time(self):
        return self.physical_channel.sample_time

    @property
    def block_size(self):
        return self.physical_channel.block_size

    @property
    def block_time(self):
        return self.physical_channel.block_time

    @property
    def phase_offset(self):
        return self.physical_channel.phase_offset

    @property
    def imbalance(self):
        return self.physical_channel.imbalance

    @property
    def acquire_allowed(self):
        return self.physical_channel.acquire_allowed

    @property
    def baseband_frequency(self):
        return self.physical_channel.baseband_frequency

    @property
    def baseband_if_frequency(self):
        return self.physical_channel.baseband_if_frequency

    @property
    def physical_channel_id(self):
        return self.physical_channel.full_id()

    @property
    def min_frequency(self):
        return self.physical_channel.pulse_channel_min_frequency

    @property
    def max_frequency(self):
        return self.physical_channel.pulse_channel_max_frequency

    def partial_id(self):
        return self.id

    def full_id(self):
        return self.physical_channel_id + "." + self.partial_id()

    def __eq__(self, other):
        if not isinstance(other, PulseChannel):
            return False

        return self.full_id() == other.full_id()

    def __hash__(self):
        return hash(self.full_id())


class PulseChannelView(PulseChannel):
    """
    Each quantum device will have a unique view of a PulseChannel, which this class
    helps encapsulate. For example, a PulseChannel may be the driving channel for one
    qubit but to another qubit the same PulseChannel might be driving its second state.
    We need to be able to share pulse channel instances with different usages in this
    particular case.

    Functionally this acts as an opaque wrapper to PulseChannel, forwarding all calls to
    the wrapped object.
    """

    pulse_channel: PulseChannel
    channel_type: ChannelType
    auxiliary_devices: List["QuantumDevice"] = []


class QuantumDevice(QuantumComponent, validate_assignment=True):
    measure_device: Union["QuantumDevice", None] = None
    default_pulse_channel_type: Literal[ChannelType.measure] = ChannelType.measure
    # pulse_channels: Dict[str, Union[PulseChannel, PulseChannelView]]
    physical_channel: PhysicalChannel


class QubitCoupling(BaseModel, validate_assignment=True):
    direction: tuple = Field(min_length=2, max_length=2)
    quality: float = Field(ge=0.0, le=1.0)

    def __repr__(self):
        return f"{str(self.direction)} @{self.quality}"
