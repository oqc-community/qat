from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from qat.purr.compiler.devices import ChannelType, PulseShapeType
from qat.purr.utils.logger import get_default_logger
from qat.utils.pydantic import WarnOnExtraFieldsModel

log = get_default_logger()


class QuantumComponent(WarnOnExtraFieldsModel):
    """
    Base class for any logical object which can act as a target of a quantum action
    - a Qubit or various channels for a simple example.

    Attributes:
        id: The string representation of the quantum component.
    """

    model_config = ConfigDict(validate_assignment=True)

    id: Optional[str] = ""
    _uuid: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))

    @field_validator("id")
    def set_id(cls, id):
        return id or ""

    def __hash__(self):
        return hash(self._uuid)

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


class PhysicalChannel(QuantumComponent):
    """
    Models a physical channel that can carry one or multiple pulses.

    Attributes:
        sample_time: The rate at which the pulse is sampled.
        baseband: The physical baseband.
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

    @field_validator("block_size")
    def set_block_size(cls, block_size):
        return block_size or 1

    @property
    def block_time(self):
        return self.block_size * self.sample_time

    @property
    def baseband_frequency(self):
        return self.baseband.frequency

    @property
    def baseband_if_frequency(self):
        return self.baseband.if_frequency

    def create_pulse_channel(
        self,
        id_: str,
        frequency: float = 0.0,
        bias: complex = 0.0 + 0.0j,
        scale: complex = 1.0 + 0.0j,
        fixed_if: bool = False,
    ):
        return PulseChannel(
            id=id_,
            physical_channel=self,
            frequency=frequency,
            bias=bias,
            scale=scale,
            fixed_if=fixed_if,
        )

    def create_freq_shift_pulse_channel(
        self,
        id_: str,
        frequency: float = 0.0,
        bias: complex = 0.0 + 0.0j,
        scale: complex = 1.0 + 0.0j,
        amp: float = 0.0,
        fixed_if: bool = False,
        active: bool = True,
    ):
        pulse_channel = FreqShiftPulseChannel(
            id_, self, frequency, bias, scale, amp, active, fixed_if
        )

        return pulse_channel


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.auxiliary_devices:
            self.id += "-[" + ",".join([ad.id for ad in self.auxiliary_devices]) + "]"

    @model_validator(mode="after")
    def check_id(self):
        if not self.id and self.channel_type is not None:
            self.id = self.channel_type.name + "@" + self.physical_channel.id
        return self

    @model_validator(mode="after")
    def check_channel_type(self):
        if (
            self.channel_type in MULTI_DEVICE_CHANNEL_TYPES
            and len(self.auxiliary_devices) == 0
        ):
            raise ValueError(
                f"Channel type {self.channel_type.name} requires at least one auxillary_device."
            )
        return self

    @model_validator(mode="after")
    def check_frequency(self):
        min_frequency = self.physical_channel.min_frequency
        max_frequency = self.physical_channel.max_frequency

        if self.frequency < min_frequency or self.frequency > max_frequency:
            raise ValueError(
                f"Pulse channel frequency '{self.frequency}' must be between the bounds "
                f"({min_frequency}, {max_frequency}) on physical channel with id {self.id}."
            )
        return self

    @field_validator("auxiliary_devices")
    def check_auxiliary_devices(cls, auxiliary_devices):
        auxiliary_devices = auxiliary_devices or []

        if not isinstance(auxiliary_devices, List):
            auxiliary_devices = [auxiliary_devices]

        return auxiliary_devices

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
        return self.physical_channel.id

    @property
    def min_frequency(self):
        return self.physical_channel.pulse_channel_min_frequency

    @property
    def max_frequency(self):
        return self.physical_channel.pulse_channel_max_frequency

    def __eq__(self, other):
        if not isinstance(other, PulseChannel):
            return False

        return self.id == other.id and self.physical_channel_id == other.physical_channel_id


class FreqShiftPulseChannel(PulseChannel):
    amp: float = 0.0
    active: bool = False


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

    def _create_pulse_channel_id(
        self, channel_type: ChannelType, auxiliary_devices: List[QuantumDevice]
    ):
        if channel_type in MULTI_DEVICE_CHANNEL_TYPES and len(auxiliary_devices) == 0:
            raise ValueError(
                f"Channel type {channel_type.name} requires at least one "
                "auxillary_device"
            )
        return ".".join([str(x.id) for x in (auxiliary_devices)] + [channel_type.name])

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

    measure_device: None = None

    def get_measure_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.measure)

    def get_acquire_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.acquire)


class Qubit(QuantumDevice):
    """
    Models a superconducting qubit on a chip, and holds all information relating to it.

    Attributes:
        index: The index of the qubit on the chip.
        resonator: The resonator coupled to the qubit.
        coupled_qubits: The qubits connected to this qubit in the QPU architecture.
        drive_amp: Amplitude for the pulse controlling the |0> -> |1> transition of the qubit.
        default_pulse_channel_type: Default type of pulse for the qubit.

        mean_z_map_args: The state of the qubit is determined through mapping the acquired readout
                         voltage (I + jQ) onto the line in the complex plane that maximally separates
                         them by applying a rotation vector z-map [A, B]. This variable determines the
                         mean value of the parameters A and B.
        discriminator: The threshold value along the separation that distinguishes the two states.
        pulse_hw_zx_pi_4: A 2-qubit $Z \times X$ (maximally entangling) gate.
        pulse_hw_x_pi_2: A single-qubit X gate.
    """

    index: int = Field(ge=0)
    coupled_qubits: Optional[List[Qubit]] = []
    drive_amp: float = 1.0
    default_pulse_channel_type: Literal[ChannelType.drive] = ChannelType.drive

    measure_device: Resonator = None

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
        else:
            log.warning(f"Qubits {self} and {qubit} are already coupled.")

        if qubit.id not in self.pulse_hw_zx_pi_4:
            self.pulse_hw_zx_pi_4[qubit.id] = {
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

    def get_all_channels(self) -> List[PulseChannel]:
        """
        Returns all channels associated with this qubit, including resonator channel and
        other auxiliary devices that act as if they are on this object.
        """
        return [
            *self.pulse_channels.values(),
            self.get_measure_channel(),
            self.get_acquire_channel(),
        ]


class QubitCoupling(WarnOnExtraFieldsModel):
    """The coupling between two qubits.

    Attributes:
        direction: The indices of the coupled qubits.
        quality: The quality-level of the coupling.
    """

    direction: tuple = Field(min_length=2, max_length=2)
    quality: float = Field(ge=0.0, le=1.0)

    def __repr__(self):
        return f"{str(self.direction)} @{self.quality}"


MULTI_DEVICE_CHANNEL_TYPES = (
    ChannelType.cross_resonance,
    ChannelType.cross_resonance_cancellation,
)
