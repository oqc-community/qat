from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
from pydantic import Field
from pydantic_core import core_schema

from qat.model.hardware_base import QubitId
from qat.utils.pydantic import WarnOnExtraFieldsModel


class CalibratablePositiveFloat(float):
    @classmethod
    def validate(cls, v):
        if not np.isnan(v) and v < 0.0:
            raise ValueError("value must be >= 0")
        return cls(v)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    def __eq__(self, other: CalibratablePositiveFloat):
        if np.isnan(self) and np.isnan(other):
            return True
        else:
            return self.__eq__(other)


class Component(WarnOnExtraFieldsModel):
    """
    Base class for any logical object which can act as a target of a quantum action
    - a Qubit or various channels for a simple example.

    :param uuid: The unique string representation of the component.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), frozen=True)

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other: Component):
        if self.uuid != other.uuid:
            return False

        if not type(self) is type(other):
            return False

        if self.model_fields != other.model_fields:
            return False

        for field_name in self.model_fields:
            field_s = getattr(self, field_name)
            field_o = getattr(other, field_name)
            if isinstance(field_s, float) and isinstance(field_o, float):
                if np.isnan(field_s) and np.isnan(field_o):
                    continue

            if field_s != field_o:
                return False

        return True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.uuid})"

    def __str__(self):
        return self.__repr__()

    @property
    def is_calibrated(self):
        for field_name in self.model_fields:
            field_value = getattr(self, field_name)
            if (
                isinstance(field_value, (Component, PulseChannelSet))
                and not field_value.is_calibrated
            ):
                return False
            elif isinstance(field_value, float) and np.isnan(field_value):
                return False
        return True


class PhysicalBaseband(Component):
    """
    Models the Local Oscillator (LO) used with a mixer to change the
    frequency of a carrier signal.

    :param frequency: The frequency of the LO.
    :param if_frequency: The intermediate frequency (IF) resulting from
                      mixing the baseband with the carrier signal.
    """

    frequency: CalibratablePositiveFloat = Field(default=np.nan)
    if_frequency: CalibratablePositiveFloat = Field(default=np.nan)


class PhysicalChannel(Component):
    """
    Models a physical channel that can carry one or multiple pulses.

    Attributes:
        baseband: The physical baseband.

        sample_time: The rate at which the pulse is sampled.
        block_size: The number of samples within a single block.
        phase_iq_offset: Deviation of the phase difference of the I
                         and Q components from 90° due to imperfections
                         in the mixing of the LO and unmodulated signal.
        bias: The bias in voltages V_I / V_Q for the I and Q components.
    """

    baseband: PhysicalBaseband = Field(frozen=True)

    sample_time: CalibratablePositiveFloat = Field(default=np.nan)
    block_size: Optional[int] = Field(ge=1, default=1)
    phase_iq_offset: complex = 0.0 + 0.0j
    bias: complex = 0.0 + 0.0j


class PulseChannel(Component):
    """
    Models a pulse channel on a particular device.

    :param physical_channel: Physical channel that carries the pulse.
    :param frequency: Frequency of the pulse.
    :param bias: Mean value of the signal, quantifies offset relative to zero mean.
    :param scale: Scale factor for mapping the voltage of the pulse to frequencies.
    :param fixed_if: Flag which determines if the intermediate frequency is fixed.
    """

    frequency: CalibratablePositiveFloat = Field(default=np.nan)
    bias: complex = 0.0 + 0.0j
    scale: complex = 1.0 + 0.0j
    fixed_if: bool = False


class MeasurePulseChannel(PulseChannel): ...


class DrivePulseChannel(PulseChannel): ...


class AcquirePulseChannel(PulseChannel): ...


class MeasureAcquirePulseChannel(PulseChannel): ...


class SecondStatePulseChannel(PulseChannel): ...


class FreqShiftPulseChannel(PulseChannel): ...


class CrossResonancePulseChannel(PulseChannel):
    auxiliary_qubit: QubitId


class CrossResonanceCancellationPulseChannel(PulseChannel):
    auxiliary_qubit: QubitId


class PulseChannelSet(WarnOnExtraFieldsModel):
    """
    Encapsulates a set of pulse channels.
    """

    @property
    def is_calibrated(self):
        for field_name in self.model_fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, Component) and not field_value.is_calibrated:
                return False
            elif isinstance(field_value, tuple):
                for pulse_channel in field_value:
                    if (
                        isinstance(pulse_channel, Component)
                        and not pulse_channel.is_calibrated
                    ):
                        return False
        return True

    def __eq__(self, other: PulseChannelSet):
        if type(self) != type(other):
            return False

        if self.model_fields != other.model_fields:
            return False

        for field_name in self.model_fields:
            if getattr(self, field_name) != getattr(other, field_name):
                return False

        return True

    def __ne__(self, other: PulseChannelSet):
        return not self.__eq__(other)


class ResonatorPulseChannels(PulseChannelSet):
    measure: MeasurePulseChannel = Field(default=MeasurePulseChannel())
    acquire: AcquirePulseChannel = Field(default=AcquirePulseChannel())


class Resonator(Component):
    """
    Models a resonator on a chip. Can be connected to multiple qubits.

    :param physical_channel: The physical channel that carries the pulses to the physical resonator.
    :param pulse_channels: The pulse channels for controlling the resonator.
    """

    physical_channel: PhysicalChannel
    pulse_channels: ResonatorPulseChannels = Field(
        frozen=True, default=ResonatorPulseChannels()
    )


class QubitPulseChannels(PulseChannelSet):
    drive: DrivePulseChannel = Field(frozen=True, default=DrivePulseChannel())
    second_state: SecondStatePulseChannel = Field(
        frozen=True, default=SecondStatePulseChannel()
    )
    freq_shift: FreqShiftPulseChannel = Field(frozen=True, default=FreqShiftPulseChannel())

    cross_resonance_channels: tuple[CrossResonancePulseChannel, ...] = Field(
        frozen=True, max_length=3, default=tuple()
    )
    cross_resonance_cancellation_channels: tuple[
        CrossResonanceCancellationPulseChannel, ...
    ] = Field(frozen=True, max_length=3, default=tuple())


class Qubit(Component):
    """
    Models a superconducting qubit on a chip, and holds all information relating to it.

    :param physical_channel: The physical channel that carries the pulses to the physical qubit.
    :param pulse_channels: The pulse channels for controlling the qubit.
    :param resonator: The measure device of the qubit.
    """

    physical_channel: PhysicalChannel
    pulse_channels: QubitPulseChannels
    resonator: Resonator
