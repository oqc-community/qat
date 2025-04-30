# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import random

import pytest
from pydantic import ValidationError

from qat.model.builder import PhysicalHardwareModelBuilder
from qat.model.device import (
    CalibratableAcquire,
    CalibratablePulse,
    CrossResonanceCancellationPulseChannel,
    CrossResonancePulseChannel,
    DrivePulseChannel,
    FreqShiftPulseChannel,
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    QubitPulseChannels,
    Resonator,
    ResonatorPulseChannels,
    SecondStatePulseChannel,
)
from qat.utils.hardware_model import generate_hw_model


class TestCalibratable:
    @pytest.mark.parametrize("invalid_width", ["invalid_width", -0.001, -5])
    def test_invalid_width(self, invalid_width):
        with pytest.raises(ValidationError):
            CalibratablePulse(width=invalid_width)

        with pytest.raises(ValidationError):
            CalibratableAcquire(width=invalid_width)

    @pytest.mark.parametrize("invalid_delay", ["invalid_delay", -0.001, -5])
    def test_invalid_width(self, invalid_delay):
        with pytest.raises(ValidationError):
            CalibratableAcquire(width=invalid_delay)


@pytest.mark.parametrize("seed", [21, 22, 23, 24])
class TestDevicesValidation:
    def test_physical_baseband(self, seed):
        bb = PhysicalBaseband()
        assert not bb.is_calibrated

        bb.frequency = random.Random(seed).uniform(1e05, 1e07)
        bb.if_frequency = random.Random(seed + 1).uniform(1e05, 1e07)
        assert bb.is_calibrated

        with pytest.raises(ValidationError):
            bb.frequency = random.Random(seed).uniform(-1e05, -1e07)

        with pytest.raises(ValidationError):
            bb.if_frequency = random.Random(seed).uniform(-1e05, -1e07)

    def test_physical_channel(self, seed):
        bb = PhysicalBaseband(
            frequency=random.Random(seed).uniform(1e05, 1e07),
            if_frequency=random.Random(seed).uniform(1e05, 1e07),
        )

        physical_channel = PhysicalChannel(baseband=bb)
        assert not physical_channel.is_calibrated

        physical_channel.sample_time = random.Random(seed).uniform(1e-08, 1e-10)
        assert physical_channel.is_calibrated

        with pytest.raises(ValidationError):
            physical_channel.sample_time = random.Random(seed).uniform(-1e-08, -1e-10)

        with pytest.raises(ValidationError):
            physical_channel.baseband = 1.0

    def test_pulse_channel(self, seed):
        pulse_channel = PulseChannel()
        assert not pulse_channel.is_calibrated

        pulse_channel.frequency = random.Random(seed).uniform(1e08, 1e10)
        assert pulse_channel.is_calibrated

        with pytest.raises(ValidationError):
            pulse_channel.frequency = random.Random(seed).uniform(-1e08, -1e10)

    def test_custom_pulse_channel(self, seed):
        class CustomPulseChannel(PulseChannel): ...

        custom_pulse_ch = CustomPulseChannel()
        assert custom_pulse_ch.pulse_type == "custom"

        with pytest.raises(TypeError):
            # `PulseChannel` should be in the class name
            class InvalidChannel(PulseChannel): ...

        with pytest.raises(TypeError):
            # `PulseChannel` should be at the end in the class name
            class InvalidPulseChannelSuffix(PulseChannel): ...

    def test_resonator(self, seed):
        bb = PhysicalBaseband(
            frequency=random.Random(seed).uniform(1e05, 1e07),
            if_frequency=random.Random(seed + 1).uniform(1e05, 1e07),
        )
        physical_channel = PhysicalChannel(
            baseband=bb, sample_time=random.Random(seed).uniform(1e-08, 1e-10)
        )

        resonator = Resonator(physical_channel=physical_channel)
        assert not resonator.is_calibrated

        for pulse_channel_name in ResonatorPulseChannels.model_fields:
            pulse_channel = getattr(resonator.pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed).uniform(1e08, 1e10)
        assert resonator.is_calibrated

        for pulse_channel_name in ResonatorPulseChannels.model_fields:
            pulse_channel = getattr(resonator.pulse_channels, pulse_channel_name)
            with pytest.raises(ValidationError):
                pulse_channel.frequency = random.Random(seed).uniform(-1e08, -1e10)

    def test_qubit(self, seed):
        bb = PhysicalBaseband(
            frequency=random.Random(seed).uniform(1e05, 1e07),
            if_frequency=random.Random(seed + 1).uniform(1e05, 1e07),
        )
        physical_channel = PhysicalChannel(
            baseband=bb, sample_time=random.Random(seed).uniform(1e-08, 1e-10)
        )

        resonator = Resonator(physical_channel=physical_channel)
        for i, pulse_channel_name in enumerate(ResonatorPulseChannels.model_fields):
            pulse_channel = getattr(resonator.pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)

        qubit_pulse_channels = QubitPulseChannels()
        for i, pulse_channel_name in enumerate(["drive", "second_state", "freq_shift"]):
            pulse_channel = getattr(qubit_pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)
        assert qubit_pulse_channels.is_calibrated

        qubit = Qubit(
            physical_channel=physical_channel,
            pulse_channels=qubit_pulse_channels,
            resonator=resonator,
        )
        assert qubit.is_calibrated

        for i, pulse_channel_name in enumerate(["drive", "second_state", "freq_shift"]):
            pulse_channel = getattr(qubit.pulse_channels, pulse_channel_name)
            with pytest.raises(ValidationError):
                pulse_channel.frequency = random.Random(seed + i).uniform(-1e08, -1e10)

    def test_qubit_pulse_channel_access(self, seed):
        hw = generate_hw_model(8, seed=seed)

        for qubit in hw.qubits.values():
            assert isinstance(qubit.drive_pulse_channel, DrivePulseChannel)
            assert isinstance(qubit.freq_shift_pulse_channel, FreqShiftPulseChannel)
            assert isinstance(qubit.second_state_pulse_channel, SecondStatePulseChannel)

            for cr_pulse_channel in qubit.cross_resonance_pulse_channels.values():
                assert isinstance(cr_pulse_channel, CrossResonancePulseChannel)

            for (
                crc_pulse_channel
            ) in qubit.cross_resonance_cancellation_pulse_channels.values():
                assert isinstance(crc_pulse_channel, CrossResonanceCancellationPulseChannel)

            # No setter for the pulse channel attribute.
            with pytest.raises(AttributeError):
                qubit.drive_pulse_channel = DrivePulseChannel()

    def test_qubit_pair(self, seed):
        physical_topology = {0: {1}, 1: {0}}
        builder = PhysicalHardwareModelBuilder(physical_connectivity=physical_topology)
        hw = builder.model
        for qubit in hw.qubits.values():
            assert not qubit.is_calibrated

            qubit.physical_channel.baseband.frequency = random.Random(seed).uniform(
                1e05, 1e07
            )
            qubit.physical_channel.baseband.if_frequency = random.Random(seed + 1).uniform(
                1e05, 1e07
            )

            qubit.resonator.physical_channel.baseband.frequency = random.Random(
                seed
            ).uniform(1e05, 1e07)
            qubit.resonator.physical_channel.baseband.if_frequency = random.Random(
                seed + 1
            ).uniform(1e05, 1e07)

            qubit.physical_channel.sample_time = random.Random(seed).uniform(1e-08, 1e-10)

            qubit.resonator.physical_channel.sample_time = random.Random(seed).uniform(
                1e-08, 1e-10
            )

            for i, pulse_channel_name in enumerate(["drive", "second_state", "freq_shift"]):
                pulse_channel = getattr(qubit.pulse_channels, pulse_channel_name)
                pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)

                with pytest.raises(ValidationError):
                    pulse_channel.frequency = random.Random(seed + i).uniform(-1e08, -1e10)

            for i, pulse_channel_name in enumerate(
                qubit.resonator.pulse_channels.__class__.model_fields
            ):
                pulse_channel = getattr(qubit.resonator.pulse_channels, pulse_channel_name)
                pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)

                with pytest.raises(ValidationError):
                    pulse_channel.frequency = random.Random(seed + i).uniform(-1e08, -1e10)

            for i, cr_channel in enumerate(qubit.cross_resonance_pulse_channels.values()):
                cr_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)
                with pytest.raises(ValidationError):
                    cr_channel.frequency = random.Random(seed + i).uniform(-1e08, -1e10)

            for i, crc_channel in enumerate(
                qubit.cross_resonance_cancellation_pulse_channels.values()
            ):
                crc_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)
                with pytest.raises(ValidationError):
                    crc_channel.frequency = random.Random(seed + i).uniform(-1e08, -1e10)

            assert qubit.is_calibrated


@pytest.mark.parametrize("seed", [21, 22, 23, 24])
class TestFrozenQubit:
    def test_cross_resonance_channels(self, seed):
        bb = PhysicalBaseband(
            frequency=random.Random(seed).uniform(1e05, 1e07),
            if_frequency=random.Random(seed + 1).uniform(1e05, 1e07),
        )

        physical_channel = PhysicalChannel(
            baseband=bb, sample_time=random.Random(seed).uniform(1e-08, 1e-10)
        )

        resonator = Resonator(physical_channel=physical_channel)
        for i, pulse_channel_name in enumerate(ResonatorPulseChannels.model_fields):
            pulse_channel = getattr(resonator.pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)

        qubit_pulse_channels = QubitPulseChannels()
        for i, pulse_channel_name in enumerate(["drive", "second_state", "freq_shift"]):
            pulse_channel = getattr(qubit_pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)

        qubit = Qubit(
            physical_channel=physical_channel,
            pulse_channels=qubit_pulse_channels,
            resonator=resonator,
        )

        # Cannot add items to a frozen pulse channel dict.
        with pytest.raises(TypeError):
            qubit.cross_resonance_pulse_channels[1] = CrossResonancePulseChannel(
                auxiliary_qubit=2
            )

        with pytest.raises(TypeError):
            qubit.cross_resonance_cancellation_pulse_channels[1] = (
                CrossResonanceCancellationPulseChannel(auxiliary_qubit=2)
            )
