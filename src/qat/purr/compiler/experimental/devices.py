from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qat.purr.compiler.devices import ChannelType, PulseShapeType


class QuantumComponent(BaseModel):
    """
    Base class for any logical object which can act as a target of a quantum action
    - a Qubit or various channels for a simple example.
    """

    model_config = ConfigDict(validate_assignment=True)

    id: str | None = ""

    @field_validator("id")
    def set_id(cls, id):
        return id or ""

    def full_id(self):
        return self.id

    def __repr__(self):
        return self.id


class PhysicalBaseband(QuantumComponent):
    frequency: float = Field(ge=0.0)
    if_frequency: Optional[float] = Field(ge=0.0, default=250e6)


class PhysicalChannel(QuantumComponent):
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


class PulseChannel(QuantumComponent):
    """Models a pulse channel on a particular device."""

    channel_type: ChannelType | None = None
    auxiliary_devices: List[QuantumDevice] = None

    physical_channel: PhysicalChannel

    frequency: float = Field(ge=0.0, default=0.0)
    bias: complex = 0.0 + 0.0j
    scale: complex = 1.0 + 0.0j

    fixed_if: bool = False

    @model_validator(mode="after")
    def check_frequency(self):
        min_frequency = self.physical_channel.pulse_channel_min_frequency
        max_frequency = self.physical_channel.pulse_channel_max_frequency

        if self.frequency < min_frequency or self.frequency > max_frequency:
            raise ValueError(
                f"Pulse channel frequency '{self.frequency}' must be between the bounds "
                f"({min_frequency}, {max_frequency}) on physical "
                f"channel with id {self.id}."
            )
        return self

    @field_validator("auxiliary_devices")
    def check_auxiliary_devices(cls, auxiliary_devices):
        return auxiliary_devices or []

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


class QuantumDevice(QuantumComponent):
    pulse_channels: Dict[str, PulseChannel]
    physical_channel: PhysicalChannel
    measure_device: Union[QuantumDevice, None] = None
    default_pulse_channel_type: Literal[ChannelType.measure] = ChannelType.measure

    def add_pulse_channel(
        self,
        pulse_channel: PulseChannel,
        channel_type: ChannelType | None = None,
        auxiliary_devices: List[QuantumDevice] = [],
    ):
        if pulse_channel.physical_channel != self.physical_channel:
            raise ValueError(
                "Pulse channel has a physical channel and this must be equal to"
                f"the device physical channel. Device {self} has physical channel {self.physical_channel} "
                f"while pulse channel has physical channel {pulse_channel.physical_channel}."
            )

        id_ = self._create_pulse_channel_id(channel_type, auxiliary_devices)
        if id_ in self.pulse_channels:
            raise KeyError(f"Pulse channel with id '{id_}' already exists.")

        self.pulse_channels[id_] = PulseChannel(
            pulse_channel, channel_type, auxiliary_devices
        )

    def create_pulse_channel(
        self,
        channel_type: ChannelType,
        frequency=0.0,
        bias=0.0 + 0.0j,
        scale=1.0 + 0.0j,
        amp=0.0,
        active: bool = True,
        fixed_if: bool = False,
        auxiliary_devices: List["QuantumDevice"] = None,
        id_: str = None,
    ):
        auxiliary_devices = auxiliary_devices or []
        id_ = id_ or self._create_pulse_channel_id(channel_type, [self] + auxiliary_devices)

        if channel_type == ChannelType.freq_shift:
            pulse_channel = self.physical_channel.create_freq_shift_pulse_channel(
                id_, frequency, bias, scale, amp, active, fixed_if
            )
        else:
            pulse_channel = self.physical_channel.create_pulse_channel(
                id_, frequency, bias, scale, fixed_if
            )
        self.add_pulse_channel(pulse_channel, channel_type, auxiliary_devices)

        return pulse_channel

    def _create_pulse_channel_id(
        self, channel_type: ChannelType, auxiliary_devices: List[QuantumDevice]
    ):
        if (
            channel_type in self.multi_device_pulse_channel_types
            and len(auxiliary_devices) == 0
        ):
            raise ValueError(
                f"Channel type {channel_type.name} requires at least one "
                "auxillary_device"
            )
        return ".".join(
            [str(x.full_id()) for x in (auxiliary_devices)] + [channel_type.name]
        )

    def get_auxiliary_devices(self, pulse_channel: PulseChannel):
        return pulse_channel.auxiliary_devices

    def get_pulse_channel(
        self,
        channel_type: ChannelType = None,
        auxiliary_devices: List[QuantumDevice] = None,
    ) -> PulseChannel:
        channel_type = channel_type or self.default_pulse_channel_type

        auxiliary_devices = auxiliary_devices or []
        if not isinstance(auxiliary_devices, List):
            auxiliary_devices = [auxiliary_devices]

        id_ = self._create_pulse_channel_id(channel_type, auxiliary_devices)
        if id_ not in self.pulse_channels:
            raise KeyError(
                f"Pulse channel with stored as '{id_}' not found on device '{self.id}'."
            )

        return self.pulse_channels[id_]

    def get_default_pulse_channel(self):
        return self.get_pulse_channel(self.default_pulse_channel_type)


class Resonator(QuantumDevice):
    """Models a resonator on a chip. Can be connected to multiple qubits."""

    def get_measure_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.measure)

    def get_acquire_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.acquire)


class Qubit(QuantumDevice):
    """
    Class modelling our superconducting qubit and holds all information relating to them.
    """

    index: int
    resonator: Resonator
    coupled_qubits: List[Qubit] | None = []
    drive_amp: float = 1.0
    default_pulse_channel_type: Literal[ChannelType.drive] = ChannelType.drive

    mean_z_map_args: List[float] = Field(min_length=2, max_length=2, default=[1.0, 0.0])
    discriminator: List[float] = Field(min_length=1, max_length=1, default=[0.0])

    pulse_hw_zx_pi_4: Dict[str, Dict[str, Any]] = dict()

    pulse_hw_x_pi_2: Dict[str, Any] = {
        "shape": PulseShapeType.GAUSSIAN,
        "width": 100e-9,
        "rise": 1.0 / 3.0,
        "amp": 0.25 / (100e-9 * 1.0 / 3.0 * np.pi**0.5),
        "drag": 0.0,
        "phase": 0.0,
    }

    pulse_measure: Dict[str, Any] = {
        "shape": PulseShapeType.SQUARE,
        "width": 1.0e-6,
        "amp": drive_amp,
        "amp_setup": 0.0,
        "rise": 0.0,
    }

    measure_acquire: Dict[str, Any] = {
        "delay": 180e-9,
        "sync": True,
        "width": 1e-6,
        "weights": None,
        "use_weights": False,
    }

    @field_validator("id", mode="after")
    def set_id(cls, id):
        return id or f"Q{cls.index}"

    @field_validator("coupled_qubits")
    def set_coupled_qubits(cls, coupled_qubits):
        return coupled_qubits or []

    def add_coupled_qubit(self, qubit: Qubit):
        if qubit is None:
            return

        if qubit not in self.coupled_qubits:
            self.coupled_qubits.append(qubit)

        if qubit.full_id() not in self.pulse_hw_zx_pi_4:
            self.pulse_hw_zx_pi_4[qubit.full_id()] = {
                "shape": PulseShapeType.SOFT_SQUARE,
                "width": 125e-9,
                "rise": 10e-9,
                "amp": 1e6,
                "drag": 0.0,
                "phase": 0.0,
            }

    def get_acquire_channel(self) -> PulseChannel:
        return self.measure_device.get_acquire_channel()

    def get_cross_resonance_channel(self, linked_qubits: List[Qubit]) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.cross_resonance, linked_qubits)

    def get_cross_resonance_cancellation_channel(
        self, linked_qubits: List[Qubit]
    ) -> PulseChannel:
        return self.get_pulse_channel(
            ChannelType.cross_resonance_cancellation, linked_qubits
        )

    def get_drive_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.drive)

    def get_freq_shift_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.freq_shift)

    def get_measure_channel(self) -> PulseChannel:
        return self.measure_device.get_measure_channel()

    def get_second_state_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.second_state)

    def get_all_channels(self):
        """
        Returns all channels associated with this qubit, including resonator channel and
        other auxiliary devices that act as if they are on this object.
        """
        return [
            *self.pulse_channels.values(),
            self.get_measure_channel(),
            self.get_acquire_channel(),
        ]


class QubitCoupling(BaseModel, validate_assignment=True):
    direction: tuple = Field(min_length=2, max_length=2)
    quality: float = Field(ge=0.0, le=1.0)

    def __repr__(self):
        return f"{str(self.direction)} @{self.quality}"
