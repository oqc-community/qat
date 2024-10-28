from __future__ import annotations

import uuid
from typing import Dict, List, Literal, Optional

import numpy as np
from pydantic import Field, field_serializer, model_validator

from qat.purr.compiler.devices import ChannelType
from qat.utils.pydantic import WarnOnExtraFieldsModel


class DeviceIdMixin(WarnOnExtraFieldsModel):
    """
    Base class for any logical object which can act as a target of a quantum action
    - a Qubit or various channels for a simple example.

    Attributes:
        id: The string representation of the quantum component.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), allow_mutation=False)

    def __init__(self, **data):
        super().__init__(**data)
        self.id_type = self.__class__.__name__


class PhysicalBasebandId(DeviceIdMixin):
    pass


class PhysicalChannelId(DeviceIdMixin):
    pass


class PulseChannelId(DeviceIdMixin):
    pass


class QuantumDeviceId(DeviceIdMixin):
    pass


class ResonatorId(QuantumDeviceId):
    pass


class QubitId(QuantumDeviceId):
    pass


class QuantumComponent(WarnOnExtraFieldsModel):
    id: DeviceIdMixin = DeviceIdMixin()
    custom_id: str = ""

    @property
    def full_id(self):
        return self.id.uuid

    def __hash__(self):
        return hash(self.id.uuid)

    def __repr__(self):
        return (
            self.custom_id if self.custom_id else f"{type(self).__name__}({self.id.uuid})"
        )

    def __eq__(self, other: QuantumComponent):
        return self.full_id == other.full_id

    def __ne__(self, other: QuantumComponent):
        return self.full_id != other.full_id


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

    id: PhysicalBasebandId = PhysicalBasebandId()


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

    id: PhysicalChannelId = PhysicalChannelId()

    @field_serializer("baseband")
    def serialise_baseband(self, baseband: PhysicalBaseband):
        return baseband.id


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
        auxiliary_qubits: Any extra devices this PulseChannel could be affecting except
                           the current one. For example in cross resonance pulses.
    """

    physical_channel: PhysicalChannel
    frequency: float = Field(ge=0.0, default=0.0)
    bias: complex = 0.0 + 0.0j
    scale: complex = 1.0 + 0.0j
    fixed_if: bool = False

    channel_type: Optional[ChannelType] = Field(allow_mutation=False, default=None)
    auxiliary_qubits: List[QubitData] = Field(allow_mutation=False, default=None)

    id: PulseChannelId = PulseChannelId()

    @field_serializer("physical_channel")
    def serialise_physical_channel(self, physical_channel: PhysicalChannel):
        return physical_channel.tag

    @field_serializer("auxiliary_qubits")
    def serialise_auxiliary_qubits(self, auxiliary_qubits: List[QubitData]):
        if auxiliary_qubits:
            return [qubit.id for qubit in auxiliary_qubits]


class QuantumDeviceData(QuantumComponent):
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
    physical_channel: PhysicalChannelId
    measure_device: Optional[ResonatorData] = None
    default_pulse_channel_type: ChannelType = ChannelType.measure

    id: QuantumDeviceId = QuantumDeviceId()

    @field_serializer("pulse_channels")
    def serialise_pulse_channels(self, pulse_channels: Dict[str, PulseChannel]):
        return [pulse_channel.id for pulse_channel in pulse_channels.values()]

    @field_serializer("physical_channel")
    def serialise_physical_channel(self, physical_channel: PhysicalChannel):
        return physical_channel.id

    @field_serializer("measure_device")
    def serialise_measure_device(self, measure_device: ResonatorData):
        if measure_device:
            return measure_device.id

    @model_validator(mode="after")
    def check_pulse_channels(self):
        invalid_pulse_channels = [
            pulse_channel
            for pulse_channel in self.pulse_channels.values()
            if pulse_channel.physical_channel != self.physical_channel
        ]

        if any(invalid_pulse_channels):
            error_pulse_channels_str = ",".join(
                [str(pulse_channel) for pulse_channel in invalid_pulse_channels]
            )
            error_physical_channels_str = ",".join(
                [
                    str(pulse_channel.physical_channel)
                    for pulse_channel in invalid_pulse_channels
                ]
            )
            raise ValueError(
                "Pulse channel has a physical channel and this must be equal to"
                f"the device physical channel. Device {self} has physical channel {self.physical_channel} "
                f"while pulse channels {error_pulse_channels_str} have associated physical channels {error_physical_channels_str}."
            )

        return self


class ResonatorData(QuantumDeviceData):
    """Models a resonator on a chip. Can be connected to multiple qubits."""

    measure_device: None = None

    id: ResonatorId = ResonatorId()

    def get_measure_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.measure)

    def get_acquire_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.acquire)


class QubitData(QuantumDeviceData):
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

    measure_device: ResonatorData = None

    tag: QubitId = QubitId()
