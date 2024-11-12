import numpy as np
import pytest
from pydantic import ValidationError

from qat.model.device import (
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    Resonator,
)
from qat.purr.compiler.devices import ChannelType


class TestDevicesValidation:

    def test_physical_baseband_validation(self):
        bb = PhysicalBaseband(frequency=np.random.uniform(1e05, 1e07))

        with pytest.raises(ValidationError):
            bb.frequency = -1.0

    def test_physical_channel_validation(self):
        bb = PhysicalBaseband(frequency=np.random.uniform(1e05, 1e07))

        with pytest.raises(ValidationError):
            physical_channel = PhysicalChannel(
                baseband=bb, sample_time=np.random.uniform(-1e05, -1e07)
            )

        with pytest.raises(ValidationError):
            physical_channel = PhysicalChannel(
                baseband=bb,
                sample_time=np.random.uniform(1e-05, 1e-07),
                acquire_allowed="blah",
            )

        physical_channel = PhysicalChannel(
            baseband=bb, sample_time=np.random.uniform(1e05, 1e07)
        )

        bb_other = PhysicalBaseband(frequency=np.random.uniform(1e05, 1e07))
        with pytest.raises(ValidationError):
            physical_channel.baseband = bb_other

    def test_pulse_channel_validation(self):
        bb = PhysicalBaseband(frequency=np.random.uniform(1e05, 1e07))
        physical_channel = PhysicalChannel(
            baseband=bb, sample_time=np.random.uniform(1e-05, 1e-07)
        )

        with pytest.raises(ValidationError):
            pulse_channel = PulseChannel(
                physical_channel=physical_channel, frequency=np.random.uniform(-1e08, -1e10)
            )

        pulse_channel = PulseChannel(
            physical_channel=physical_channel, frequency=np.random.uniform(1e08, 1e10)
        )

        physical_channel_other = PhysicalChannel(
            baseband=bb, sample_time=np.random.uniform(1e-05, 1e-07)
        )
        with pytest.raises(ValidationError):
            pulse_channel.physical_channel = physical_channel_other

        with pytest.raises(ValidationError):
            pulse_channel = PulseChannel(
                physical_channel=physical_channel,
                frequency=np.random.uniform(-1e08, -1e10),
                channel_type="measure",
            )

    def test_resonator_validation(self):
        bb = PhysicalBaseband(frequency=np.random.uniform(1e05, 1e07))
        physical_channel = PhysicalChannel(
            baseband=bb, sample_time=np.random.uniform(1e-05, 1e-07)
        )

        pulse_channel_measure = PulseChannel(
            frequency=np.random.uniform(1e08, 1e10),
            physical_channel=physical_channel,
            channel_type=ChannelType.measure,
        )
        pulse_channel_acquire = PulseChannel(
            frequency=np.random.uniform(1e08, 1e10),
            physical_channel=physical_channel,
            channel_type=ChannelType.acquire,
        )
        pulse_channels = {
            pulse_channel_measure.to_component_id(): pulse_channel_measure,
            pulse_channel_acquire.to_component_id(): pulse_channel_acquire,
        }
        resonator = Resonator(
            pulse_channels=pulse_channels, physical_channel=physical_channel
        )

        other_physical_channel = PhysicalChannel(
            baseband=bb, sample_time=np.random.uniform(1e-05, 1e-07)
        )
        with pytest.raises(ValidationError):
            resonator.physical_channel = other_physical_channel

        with pytest.raises(ValidationError):
            pulse_channel_measure = PulseChannel(
                frequency=np.random.uniform(1e08, 1e10),
                physical_channel=other_physical_channel,
                channel_type=ChannelType.measure,
            )
            pulse_channel_acquire = PulseChannel(
                frequency=np.random.uniform(1e08, 1e10),
                physical_channel=physical_channel,
                channel_type=ChannelType.acquire,
            )
            pulse_channels = {
                pulse_channel_measure.to_component_id(): pulse_channel_measure,
                pulse_channel_acquire.to_component_id(): pulse_channel_acquire,
            }
            resonator = Resonator(
                pulse_channels=pulse_channels, physical_channel=physical_channel
            )

    def test_qubit_validation(self):
        bb = PhysicalBaseband(frequency=np.random.uniform(1e05, 1e07))
        physical_channel = PhysicalChannel(
            baseband=bb, sample_time=np.random.uniform(1e-05, 1e-07)
        )

        pulse_channel_measure = PulseChannel(
            frequency=np.random.uniform(1e08, 1e10),
            physical_channel=physical_channel,
            channel_type=ChannelType.measure,
        )
        pulse_channel_acquire = PulseChannel(
            frequency=np.random.uniform(1e08, 1e10),
            physical_channel=physical_channel,
            channel_type=ChannelType.acquire,
        )
        pulse_channels_r = {
            pulse_channel_measure.to_component_id(): pulse_channel_measure,
            pulse_channel_acquire.to_component_id(): pulse_channel_acquire,
        }
        resonator = Resonator(
            pulse_channels=pulse_channels_r, physical_channel=physical_channel
        )

        pulse_channel_drive = PulseChannel(
            frequency=np.random.uniform(1e08, 1e10),
            physical_channel=physical_channel,
            channel_type=ChannelType.drive,
        )
        pulse_channels_q = {pulse_channel_drive.to_component_id(): pulse_channel_drive}
        qubit = Qubit(
            pulse_channels=pulse_channels_q,
            measure_device=resonator,
            physical_channel=physical_channel,
            index=0,
        )

        with pytest.raises(ValidationError):
            qubit = Qubit(
                pulse_channels=pulse_channels_q,
                measure_device=resonator,
                physical_channel=physical_channel,
                index=-1,
            )

        other_qubit = Qubit(
            pulse_channels=pulse_channels_q,
            measure_device=resonator,
            physical_channel=physical_channel,
            index=0,
        )
        with pytest.raises(ValidationError):
            Qubit(
                pulse_channels=pulse_channels_q,
                measure_device=other_qubit,
                physical_channel=physical_channel,
                index=1,
            )

        other_physical_channel = PhysicalChannel(
            baseband=bb, sample_time=np.random.uniform(1e-05, 1e-07)
        )
        with pytest.raises(ValidationError):
            qubit.physical_channel = other_physical_channel
