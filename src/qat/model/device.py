from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from qat.utils.pydantic import WarnOnExtraFieldsModel

QubitId = int


class Component(WarnOnExtraFieldsModel):
    """
    Base class for any logical object which can act as a target of a quantum action
    - a Qubit or various channels for a simple example.

    Attributes:
        uuid: The unique string representation of the component.
    """

    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), frozen=True)

    def __init__(self, **data):
        super().__init__(**data)

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other: Component):
        if type(self) != type(other):
            return False

        if self.model_fields != other.model_fields:
            return False

        if self.uuid != other.uuid:
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

    def __ne__(self, other: Component):
        return not self.__eq__(other)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.uuid})"

    def __str__(self):
        return self.__repr__()

    @property
    def calibrated(self):
        for field_name in self.model_fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, Component) and not field_value.calibrated:
                return False
            elif isinstance(field_value, float) and np.isnan(field_value):
                return False
        return True


class PhysicalBaseband(Component):
    """
    Models the Local Oscillator (LO) used with a mixer to change the
    frequency of a carrier signal.
    Attributes:
        frequency: The frequency of the LO.
        if_frequency: The intermediate frequency (IF) resulting from
                      mixing the baseband with the carrier signal.
    """

    frequency: float = Field(default=np.nan)
    if_frequency: Optional[float] = Field(default=np.nan)


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
        acquire_allowed: If the physical channel allows acquire pulses.
    """

    baseband: PhysicalBaseband = Field(frozen=True)

    sample_time: float = Field(default=np.nan)
    block_size: Optional[int] = Field(ge=1, default=1)
    phase_iq_offset: complex = 0.0 + 0.0j
    bias: complex = 0.0 + 0.0j


class PulseChannel(Component):
    """
    Models a pulse channel on a particular device.
    Attributes:
        physical_channel: Physical channel that carries the pulse.
        frequency: Frequency of the pulse.
        bias: Mean value of the signal, quantifies offset relative to zero mean.
        scale: Scale factor for mapping the voltage of the pulse to frequencies.
        fixed_if: Flag which determines if the intermediate frequency is fixed.
    """

    frequency: float = Field(default=np.nan)
    bias: complex = 0.0 + 0.0j
    scale: complex = 1.0 + 0.0j
    fixed_if: bool = False


class MeasurePulseChannel(PulseChannel):
    pass


class DrivePulseChannel(PulseChannel):
    pass


class AcquirePulseChannel(PulseChannel):
    pass


class MeasureAcquirePulseChannel(PulseChannel):
    pass


class SecondStatePulseChannel(PulseChannel):
    pass


class FreqShiftPulseChannel(PulseChannel):
    pass


class CrossResonancePulseChannel(PulseChannel):
    target_qubit: QubitId


class CrossResonanceCancellationPulseChannel(PulseChannel):
    target_qubit: QubitId


class PulseChannels(WarnOnExtraFieldsModel):
    @property
    def calibrated(self):
        for field_name in self.model_fields:
            field_value = getattr(self, field_name)
            if not field_value.calibrated:
                return False
        return True

    def __eq__(self, other: PulseChannels):
        if type(self) != type(other):
            return False

        if self.model_fields != other.model_fields:
            return False

        for field_name in self.model_fields:
            if getattr(self, field_name) != getattr(other, field_name):
                return False

        return True

    def __ne__(self, other: PulseChannels):
        return not self.__eq__(other)


class ResonatorPulseChannels(PulseChannels):
    measure: MeasurePulseChannel
    acquire: AcquirePulseChannel


class QbloxResonatorPulseChannels(PulseChannels):
    measure: MeasureAcquirePulseChannel
    acquire: MeasureAcquirePulseChannel

    @model_validator(mode="after")
    def validate_pulse_channels(self):
        if self.measure != self.acquire:
            raise ValueError(
                f"Measure and acquire pulse channels must be equal in {self.__class__.__name__}, got: {self.measure} and {self.acquire}."
            )
        return self


class Resonator(Component):
    physical_channel: PhysicalChannel
    pulse_channels: ResonatorPulseChannels


class QubitPulseChannels(PulseChannels):
    drive: DrivePulseChannel = Field(frozen=True, default=DrivePulseChannel())
    second_state: SecondStatePulseChannel = Field(
        frozen=True, default=SecondStatePulseChannel()
    )
    freq_shift: FreqShiftPulseChannel = Field(frozen=True, default=FreqShiftPulseChannel())

    cross_resonance_channels: tuple[CrossResonancePulseChannel, ...] = Field(
        frozen=True, max_length=3
    )
    cross_resonance_cancellation_channels: tuple[
        CrossResonanceCancellationPulseChannel, ...
    ] = Field(frozen=True, max_length=3)


class Qubit(Component):
    physical_channel: PhysicalChannel
    pulse_channels: QubitPulseChannels
    resonator: Resonator
