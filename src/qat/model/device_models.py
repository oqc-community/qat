from __future__ import annotations

import uuid
from typing import Dict, List, Literal, Optional

import numpy as np
from pydantic import Field, field_serializer, model_validator

from qat.purr.compiler.devices import ChannelType
from qat.utils.pydantic import WarnOnExtraFieldsModel


class DeviceTagMixin(WarnOnExtraFieldsModel):
    """
    Base class for any logical object which can act as a target of a quantum action
    - a Qubit or various channels for a simple example.

    Attributes:
        id: The string representation of the quantum component.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), allow_mutation=False)
    tag_type: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.tag_type = self.__class__.__name__


class PhysicalBasebandId(DeviceTagMixin):
    pass


class PhysicalChannelId(DeviceTagMixin):
    pass


class PulseChannelId(DeviceTagMixin):
    pass


class QuantumDeviceId(DeviceTagMixin):
    pass


class ResonatorId(QuantumDeviceId):
    pass


class QubitId(QuantumDeviceId):
    pass


class QuantumComponent(WarnOnExtraFieldsModel):
    id: str = ""
    tag: DeviceTagMixin = DeviceTagMixin()

    @model_validator(mode="after")
    def set_id(self):
        if not self.id:
            self.id = self.tag.uuid
        return self

    @property
    def full_id(self):
        return self.tag.uuid

    def __hash__(self):
        return hash(self.tag.uuid)

    def __repr__(self):
        return f"{type(self).__name__}({self.id})"


class PhysicalBaseband(QuantumComponent):
    """
    Models the Local Oscillator (LO) used with a mixer to change the
    frequency of a carrier signal.

    Attributes:
        frequency: The frequency of the LO.
        if_frequency: The intermediate frequency (IF) resulting from
                      mixing the baseband with the carrier signal.
    """

    frequency: float = Field(ge=0.0)
    if_frequency: Optional[float] = Field(ge=0.0, default=250e6)

    tag: PhysicalBasebandId = PhysicalBasebandId()


class PhysicalChannel(QuantumComponent):
    """
    Models a physical channel that can carry one or multiple pulses.

    Attributes:
        baseband: The physical baseband.
        sample_time: The rate at which the pulse is sampled.

        block_size: The number of samples within a block ???
        phase_iq_offset: Deviation of the phase difference of the I
                         and Q components from 90° due to imperfections
                         in the mixing of the LO and unmodulated signal.
        bias: The bias in voltages V_I / V_Q for the I and Q components.
        acquire_allowed: If the physical channel allows acquire pulses.
        min_frequency: Min frequency allowed in this physical channel.
        max_frequency: Max frequency allowed in this physical channel.
    """

    baseband: PhysicalBaseband
    sample_time: float = Field(ge=0.0)
    block_size: Optional[int] = Field(ge=1, default=1)
    phase_iq_offset: float = 0.0
    bias: float = 1.0

    acquire_allowed: bool = False
    min_frequency: float = Field(ge=0.0, default=0.0)
    max_frequency: float = np.inf

    tag: PhysicalChannelId = PhysicalChannelId()

    @field_serializer("baseband")
    def serialise_baseband(self, baseband: PhysicalBaseband):
        return baseband.tag


class PulseChannel(QuantumComponent):
    """
    Models a pulse channel on a particular device.

    Attributes:
        physical_channel: Physical channel that carries the pulse.
        frequency: Frequency of the pulse.
        bias: Mean value of the signal, quantifies offset relative to zero mean.
        scale: Scale factor for mapping the voltage of the pulse to frequencies.
        fixed_if: Flag which determines if the intermediate frequency is fixed.
        channel_type: Type of the pulse.
        auxiliary_devices: Any extra devices this PulseChannel could be affecting except
                           the current one. For example in cross resonance pulses.
    """

    physical_channel: PhysicalChannel
    frequency: float = Field(ge=0.0, default=0.0)
    bias: complex = 0.0 + 0.0j
    scale: complex = 1.0 + 0.0j
    fixed_if: bool = False

    channel_type: Optional[ChannelType] = Field(allow_mutation=False, default=None)
    auxiliary_devices: List[QuantumDevice] = Field(allow_mutation=False, default=None)

    tag: PulseChannelId = PulseChannelId()

    @field_serializer("physical_channel")
    def serialise_physical_channel(self, physical_channel: PhysicalChannel):
        return physical_channel.tag

    @field_serializer("auxiliary_devices")
    def serialise_auxiliary_devices(self, auxiliary_devices: List[QuantumDevice]):
        if auxiliary_devices:
            return [auxiliary_device.tag for auxiliary_device in auxiliary_devices]


class QuantumDevice(QuantumComponent):
    """
    A physical device whose main form of operation involves pulse channels.

    Attributes:
        pulse_channels: Pulse channels with their ids as keys.
        physical_channel: Physical channel associated with the pulse channels.
                          Note that this physical channel must be equal to the
                          physical channel associated with the pulse channels.
        measure_device: A quantum device used to measure the state of this device.
        default_pulse_channel_type: Default type of pulse for the quantum device.
    """

    pulse_channels: Dict[str, PulseChannel]
    physical_channel: PhysicalChannel
    measure_device: Optional[QuantumDevice] = None
    default_pulse_channel_type: ChannelType = ChannelType.measure

    tag: QuantumDeviceId = QuantumDeviceId()

    @field_serializer("pulse_channels")
    def serialise_pulse_channels(self, pulse_channels: Dict[str, PulseChannel]):
        return [pulse_channel.tag for pulse_channel in pulse_channels.values()]

    @field_serializer("physical_channel")
    def serialise_physical_channel(self, physical_channel: PhysicalChannel):
        return physical_channel.tag

    @field_serializer("measure_device")
    def serialise_measure_device(self, measure_device: QuantumDevice):
        if measure_device:
            return measure_device.tag


class Resonator(QuantumDevice):
    """Models a resonator on a chip. Can be connected to multiple qubits."""

    measure_device: None = None

    tag: ResonatorId = ResonatorId()


class Qubit(QuantumDevice):
    """
    Models a superconducting qubit on a chip, and holds all information relating to it.

    Attributes:
        index: The index of the qubit on the chip.
        resonator: The resonator coupled to the qubit.
        coupled_qubits: The qubits connected to this qubit in the QPU architecture.
        drive_amp: Amplitude for the pulse controlling the |0> -> |1> transition of the qubit.
        default_pulse_channel_type: Default type of pulse for the qubit.
    """

    index: int = Field(ge=0)
    drive_amp: float = 1.0
    default_pulse_channel_type: Literal[ChannelType.drive] = ChannelType.drive

    measure_device: Resonator = None

    tag: QubitId = QubitId()
