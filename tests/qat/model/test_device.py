import random

import pytest
from pydantic import ValidationError

from qat.model.builder import PhysicalHardwareModelBuilder
from qat.model.device import (
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    QubitPulseChannels,
    Resonator,
)


@pytest.mark.parametrize("seed", [21, 22, 23, 24])
class TestDevicesValidation:
    def test_physical_baseband(self, seed):
        bb = PhysicalBaseband()
        assert not bb.is_calibrated

        bb.frequency = random.Random(seed).uniform(1e05, 1e07)
        bb.if_frequency = random.Random(seed).uniform(1e05, 1e07)
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

    def test_resonator(self, seed):
        bb = PhysicalBaseband(
            frequency=random.Random(seed).uniform(1e05, 1e07),
            if_frequency=random.Random(seed).uniform(1e05, 1e07),
        )
        physical_channel = PhysicalChannel(
            baseband=bb, sample_time=random.Random(seed).uniform(1e-08, 1e-10)
        )

        resonator = Resonator(physical_channel=physical_channel)
        assert not resonator.is_calibrated

        for pulse_channel_name in resonator.pulse_channels.model_fields:
            pulse_channel = getattr(resonator.pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed).uniform(1e08, 1e10)
        assert resonator.is_calibrated

        for pulse_channel_name in resonator.pulse_channels.model_fields:
            pulse_channel = getattr(resonator.pulse_channels, pulse_channel_name)
            with pytest.raises(ValidationError):
                pulse_channel.frequency = random.Random(seed).uniform(-1e08, -1e10)

    def test_qubit(self, seed):
        bb = PhysicalBaseband(
            frequency=random.Random(seed).uniform(1e05, 1e07),
            if_frequency=random.Random(seed).uniform(1e05, 1e07),
        )
        physical_channel = PhysicalChannel(
            baseband=bb, sample_time=random.Random(seed).uniform(1e-08, 1e-10)
        )

        resonator = Resonator(physical_channel=physical_channel)
        for pulse_channel_name in resonator.pulse_channels.model_fields:
            pulse_channel = getattr(resonator.pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed).uniform(1e08, 1e10)

        qubit_pulse_channels = QubitPulseChannels()
        for pulse_channel_name in ["drive", "second_state", "freq_shift"]:
            pulse_channel = getattr(qubit_pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed).uniform(1e08, 1e10)
        assert qubit_pulse_channels.is_calibrated

        qubit = Qubit(
            physical_channel=physical_channel,
            pulse_channels=qubit_pulse_channels,
            resonator=resonator,
        )
        assert qubit.is_calibrated

        for pulse_channel_name in ["drive", "second_state", "freq_shift"]:
            pulse_channel = getattr(qubit.pulse_channels, pulse_channel_name)
            with pytest.raises(ValidationError):
                pulse_channel.frequency = random.Random(seed).uniform(-1e08, -1e10)

    def test_qubit_pair(self, seed):
        physical_topology = {0: {1}, 1: {0}}
        builder = PhysicalHardwareModelBuilder(physical_connectivity=physical_topology)
        hw = builder.model
        for qubit in hw.qubits.values():
            assert not qubit.is_calibrated

            qubit.physical_channel.baseband.frequency = random.Random(seed).uniform(
                1e05, 1e07
            )
            qubit.physical_channel.baseband.if_frequency = random.Random(seed).uniform(
                1e05, 1e07
            )

            qubit.resonator.physical_channel.baseband.frequency = random.Random(
                seed
            ).uniform(1e05, 1e07)
            qubit.resonator.physical_channel.baseband.if_frequency = random.Random(
                seed
            ).uniform(1e05, 1e07)

            qubit.physical_channel.sample_time = random.Random(seed).uniform(1e-08, 1e-10)

            qubit.resonator.physical_channel.sample_time = random.Random(seed).uniform(
                1e-08, 1e-10
            )

            for pulse_channel_name in ["drive", "second_state", "freq_shift"]:
                pulse_channel = getattr(qubit.pulse_channels, pulse_channel_name)
                pulse_channel.frequency = random.Random(seed).uniform(1e08, 1e10)

                with pytest.raises(ValidationError):
                    pulse_channel.frequency = random.Random(seed).uniform(-1e08, -1e10)

            for pulse_channel_name in qubit.resonator.pulse_channels.model_fields:
                pulse_channel = getattr(qubit.resonator.pulse_channels, pulse_channel_name)
                pulse_channel.frequency = random.Random(seed).uniform(1e08, 1e10)

                with pytest.raises(ValidationError):
                    pulse_channel.frequency = random.Random(seed).uniform(-1e08, -1e10)

            for cr_channel in qubit.pulse_channels.cross_resonance_channels.values():
                cr_channel.frequency = random.Random(seed).uniform(1e08, 1e10)
                with pytest.raises(ValidationError):
                    cr_channel.frequency = random.Random(seed).uniform(-1e08, -1e10)

            for (
                crc_channel
            ) in qubit.pulse_channels.cross_resonance_cancellation_channels.values():
                crc_channel.frequency = random.Random(seed).uniform(1e08, 1e10)
                with pytest.raises(ValidationError):
                    crc_channel.frequency = random.Random(seed).uniform(-1e08, -1e10)

            assert qubit.is_calibrated
