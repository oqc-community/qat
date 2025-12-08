# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from __future__ import annotations

import re
from typing import Any

import numpy as np
from pydantic import Field, field_validator, model_validator

from qat.ir.waveforms import GaussianWaveform, SoftSquareWaveform
from qat.utils.pydantic import (
    CalibratablePositiveFloat,
    ComplexNDArray,
    FloatNDArray,
    FrozenDict,
    NoExtraFieldsModel,
    QubitId,
    WaveformType,
)
from qat.utils.uuid import uuid4


class Component(NoExtraFieldsModel):
    """
    Base class for any logical object which can act as a target of a quantum action
    - a Qubit or various channels for a simple example.

    :param uuid: The unique string representation of the component.
    """

    uuid: str = Field(default_factory=lambda: str(uuid4()), frozen=True)

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other: Component):
        if self.uuid != other.uuid:
            return False

        if type(self) is not type(other):
            return False

        if self.__class__.model_fields != other.__class__.model_fields:
            return False

        for field_name in self.__class__.model_fields:
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
        for field_name in self.__class__.model_fields:
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


class IQBias(NoExtraFieldsModel):
    """
    Models the bias in voltages V_I / V_Q for the I and Q components in a physical channel.

    :param bias: The bias as a complex number, where the real part is the I component
                and the imaginary part is the Q component.
    """

    bias: complex | float = 0 + 0j


class PhysicalChannel(Component):
    """
    Models a physical channel that can carry one or multiple pulses.

    Attributes:
        baseband: The physical baseband.

        block_size: The number of samples within a single block.
        iq_voltage_bias: The bias in voltages V_I / V_Q for the I and Q components.

        acquire_allowed: Whether the physical channel allows acquire operations.
        swap_readout_iq: Whether to swap the I and Q components for readout operations.
        name_index: integer used in human readable name, `channel_<index>`.
    """

    baseband: PhysicalBaseband = Field(frozen=True)

    block_size: int | None = Field(ge=1, default=1)
    iq_voltage_bias: IQBias = Field(default=IQBias())
    name_index: int = Field(ge=0, frozen=True)

    @property
    def I_bias(self) -> float:
        return self.iq_voltage_bias.bias.real

    @property
    def Q_bias(self) -> float:
        return self.iq_voltage_bias.bias.imag


class QubitPhysicalChannel(PhysicalChannel): ...


class ResonatorPhysicalChannel(PhysicalChannel):
    swap_readout_iq: bool = False


pulse_channel_check = re.compile(r"PulseChannel$")


class PulseChannel(Component):
    """
    Models a pulse channel on a particular device.

    :param physical_channel: Physical channel that carries the pulse.
    :param frequency: Frequency of the pulse.
    :param imbalance: Ratio between I and Q AC voltage that is sent out of the FPGA.
    :param phase_iq_offset: Phase offset between the I and Q plane for AC voltage that is send out the FPGA.
    :param scale: Scale factor for mapping the voltage of the pulse to frequencies.
    """

    frequency: CalibratablePositiveFloat = Field(default=np.nan)
    imbalance: float = 1.0
    phase_iq_offset: float = 0.0
    scale: complex | float = 1.0 + 0.0j

    def model_post_init(self, context: Any, /) -> None:
        if isinstance(self.scale, np.complexfloating):
            self.scale = complex(self.scale)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not pulse_channel_check.search(cls.__name__):
            raise TypeError(
                f"The class name '{cls.__name__}' must contain `PulseChannel` at the end."
            )

    @property
    def pulse_type(self):
        return self.__class__.__name__.replace("PulseChannel", "").lower()


class QubitPulseChannel(PulseChannel): ...


class ResonatorPulseChannel(PulseChannel): ...


class CalibratablePulse(NoExtraFieldsModel):
    waveform_type: WaveformType = GaussianWaveform
    width: CalibratablePositiveFloat = Field(default=100e-09, ge=0)
    amp: float = 0.25 / (100e-9 * 1.0 / 3.0 * np.pi**0.5)
    phase: float = 0.0
    drag: float = 0.0
    rise: float = 1.0 / 3.0
    amp_setup: float = 0.0
    std_dev: float = 0.0


class CalibratableAcquire(NoExtraFieldsModel):
    delay: CalibratablePositiveFloat = Field(default=180e-08, ge=0)
    width: CalibratablePositiveFloat = Field(default=1e-06, ge=0)
    sync: bool = True
    weights: ComplexNDArray | FloatNDArray = Field(
        default_factory=lambda: ComplexNDArray([])
    )
    use_weights: bool = False


class DrivePulseChannel(QubitPulseChannel):
    """
    The pulse channel that drives the qubit from |0> -> |1>.

    :param pulse: Calibratable parameters for the X(pi/2) drive pulse.
    """

    pulse: CalibratablePulse = Field(
        default=CalibratablePulse(
            waveform_type=GaussianWaveform, width=100e-9, rise=1.0 / 3.0
        ),
        frozen=True,
    )

    pulse_x_pi: CalibratablePulse | None = Field(default=None)


class MeasurePulseChannel(ResonatorPulseChannel):
    """
    The pulse channel that measures the quantum state of the resonator.

    :param pulse: Calibratable parameters for the measure pulse.
    """

    pulse: CalibratablePulse = Field(default=CalibratablePulse(width=1e-06), frozen=True)


class AcquirePulseChannel(ResonatorPulseChannel):
    acquire: CalibratableAcquire = Field(default=CalibratableAcquire(), frozen=True)


class MeasureAcquirePulseChannel(MeasurePulseChannel, AcquirePulseChannel): ...


class SecondStatePulseChannel(QubitPulseChannel):
    active: bool = False
    delay: float = 0.0

    pulse: CalibratablePulse = Field(
        default=CalibratablePulse(
            waveform_type=GaussianWaveform, width=100e-9, rise=1.0 / 3.0
        ),
        frozen=True,
    )


class FreqShiftPulseChannel(QubitPulseChannel):
    active: bool = False
    amp: float = 1.0
    phase: float = 0.0


class CrossResonancePulseChannel(QubitPulseChannel):
    auxiliary_qubit: QubitId
    zx_pi_4_pulse: CalibratablePulse = Field(
        default=CalibratablePulse(
            waveform_type=SoftSquareWaveform, width=125e-9, rise=10e-9, amp=1e6
        ),
        frozen=True,
    )

    def __repr__(self):
        return f"{self.__class__.__name__}(@Q{self.auxiliary_qubit})"


class CrossResonanceCancellationPulseChannel(QubitPulseChannel):
    auxiliary_qubit: QubitId

    def __repr__(self):
        return f"{self.__class__.__name__}(@Q{self.auxiliary_qubit})"


class PulseChannelSet(NoExtraFieldsModel):
    """
    Encapsulates a set of pulse channels.
    """

    @property
    def is_calibrated(self):
        for field_name in self.__class__.model_fields:
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
        if type(self) is not type(other):
            return False

        if self.__class__.model_fields != other.__class__.model_fields:
            return False

        for field_name in self.__class__.model_fields:
            if getattr(self, field_name) != getattr(other, field_name):
                return False

        return True

    def __ne__(self, other: PulseChannelSet):
        return not self.__eq__(other)

    def pulse_channel_with_id(self, id_: str):
        for field_name in self.__class__.model_fields:
            if (
                isinstance(comp := getattr(self, field_name), Component)
                and comp.uuid == id_
            ):
                return comp

        return None


class ResonatorPulseChannels(PulseChannelSet):
    measure: MeasurePulseChannel = Field(default=MeasurePulseChannel())
    acquire: AcquirePulseChannel = Field(default=AcquirePulseChannel())

    @property
    def all_pulse_channels(self):
        return [self.measure, self.acquire]


class Resonator(Component):
    """
    Models a resonator on a chip. Can be connected to multiple qubits.

    :param physical_channel: The physical channel that carries the pulses to the physical resonator.
    :param pulse_channels: The pulse channels for controlling the resonator.
    :param measure_pulse: Calibrated parameters for the measure pulse on the resonator.
    :param acquire: Calibrated parameters for the acquire instruction.
    """

    physical_channel: ResonatorPhysicalChannel
    pulse_channels: ResonatorPulseChannels = Field(
        frozen=True, default=ResonatorPulseChannels()
    )

    @property
    def all_pulse_channels(self):
        return self.pulse_channels.all_pulse_channels

    @property
    def measure_pulse_channel(self):
        return self.pulse_channels.measure

    @property
    def acquire_pulse_channel(self):
        return self.pulse_channels.acquire


class QubitPulseChannels(PulseChannelSet):
    drive: DrivePulseChannel = Field(frozen=True, default=DrivePulseChannel())
    second_state: SecondStatePulseChannel = Field(
        frozen=True, default=SecondStatePulseChannel()
    )
    freq_shift: FreqShiftPulseChannel = Field(frozen=True, default=FreqShiftPulseChannel())

    cross_resonance_channels: FrozenDict[QubitId, CrossResonancePulseChannel] = Field(
        frozen=True, max_length=3, default=FrozenDict({})
    )
    cross_resonance_cancellation_channels: FrozenDict[
        QubitId, CrossResonanceCancellationPulseChannel
    ] = Field(frozen=True, max_length=3, default=FrozenDict({}))

    @field_validator("cross_resonance_channels", "cross_resonance_cancellation_channels")
    def validate_channels_qubit_mapping(cls, channels):
        for aux_qubit_id, pulse_channel in channels.items():
            assert aux_qubit_id == pulse_channel.auxiliary_qubit, (
                f"Mismatch in mapping for qubit id in {channels}."
            )
        return channels

    @model_validator(mode="after")
    def validate_cross_resonance_pulse_channels(self):
        assert (
            self.cross_resonance_channels.keys()
            == self.cross_resonance_cancellation_channels.keys()
        ), (
            "Mismatch between auxiliary qubit ids for cross resonance and cross resonance cancellation channels."
        )
        return self

    @property
    def all_pulse_channels(self):
        return [
            self.drive,
            self.second_state,
            self.freq_shift,
            *self.cross_resonance_channels.values(),
            *self.cross_resonance_cancellation_channels.values(),
        ]

    def pulse_channel_with_id(self, id_: str):
        for pulse_ch in self.all_pulse_channels:
            if pulse_ch.uuid == id_:
                return pulse_ch

        return None


class Qubit(Component):
    """
    Models a superconducting qubit on a chip, and holds all information relating to it.

    :param physical_channel: The physical channel that carries the pulses to the physical qubit.
    :param pulse_channels: The pulse channels for controlling the qubit.
    :param resonator: The measure device of the qubit.
    :param pulse: Calibrated parameters for the X(pi/2) pulse.
    """

    physical_channel: QubitPhysicalChannel
    pulse_channels: QubitPulseChannels = Field(frozen=True, default=QubitPulseChannels())
    resonator: Resonator

    mean_z_map_args: list[complex | float] = Field(max_length=2, default=[1.0, 0.0])
    discriminator: complex | float = 0.0

    direct_x_pi: bool = False

    def model_post_init(self, context: Any, /) -> None:
        if isinstance(self.discriminator, np.complexfloating):
            self.discriminator = complex(self.discriminator)

        for i in range(len(self.mean_z_map_args)):
            if isinstance(self.mean_z_map_args[i], np.complexfloating):
                self.mean_z_map_args[i] = complex(self.mean_z_map_args[i])

    @property
    def all_pulse_channels(self):
        return self.pulse_channels.all_pulse_channels

    @property
    def all_qubit_and_resonator_pulse_channels(self):
        return self.all_pulse_channels + self.resonator.all_pulse_channels

    @property
    def drive_pulse_channel(self):
        return self.pulse_channels.drive

    @property
    def second_state_pulse_channel(self):
        return self.pulse_channels.second_state

    @property
    def freq_shift_pulse_channel(self):
        return self.pulse_channels.freq_shift

    @property
    def cross_resonance_pulse_channels(self):
        return self.pulse_channels.cross_resonance_channels

    @property
    def cross_resonance_cancellation_pulse_channels(self):
        return self.pulse_channels.cross_resonance_cancellation_channels

    @property
    def measure_pulse_channel(self):
        return self.resonator.pulse_channels.measure

    @property
    def acquire_pulse_channel(self):
        return self.resonator.pulse_channels.acquire
