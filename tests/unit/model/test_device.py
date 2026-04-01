# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import random

import numpy as np
import pytest
from pydantic import ValidationError

from qat.ir.waveforms import GaussianWaveform
from qat.model.builder import PhysicalHardwareModelBuilder
from qat.model.device import (
    CalibratableAcquire,
    CalibratablePulse,
    CrossResonanceCancellationPulseChannel,
    CrossResonancePulseChannel,
    DrivePulseChannel,
    FreqShiftPulseChannel,
    IQBias,
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    QubitPhysicalChannel,
    QubitPulseChannels,
    Resonator,
    ResonatorPhysicalChannel,
    ResonatorPulseChannels,
    SecondStatePulseChannel,
)
from qat.model.post_processing import LinearMapToRealMethod, MaxLikelihoodMethod, MLStateMap
from qat.utils.hardware_model import generate_hw_model


class TestCalibratable:
    @pytest.mark.parametrize("invalid_width", ["invalid_width", -0.001, -5])
    def test_invalid_width(self, invalid_width):
        with pytest.raises(ValidationError, match=r"Input should be"):
            CalibratablePulse(width=invalid_width)

        with pytest.raises(ValidationError, match=r"Input should be"):
            CalibratableAcquire(width=invalid_width)

    @pytest.mark.parametrize("invalid_delay", ["invalid_delay", -0.001, -5])
    def test_invalid_delay(self, invalid_delay):
        with pytest.raises(ValidationError, match=r"Input should be"):
            CalibratableAcquire(width=invalid_delay)


class TestCalibratablePulse:
    def test_defaults_give_valid_pulse(self):
        """Regression test to ensure default values give a sensible waveform that is free of
        errors and warnings."""
        pulse = CalibratablePulse()
        assert pulse.waveform_type == GaussianWaveform
        assert pulse.amp > 0.0
        assert pulse.width > 0.0
        assert pulse.rise > 0.0

        params = pulse.model_dump()
        params.pop("waveform_type")
        wf = pulse.waveform_type(**params)

        # evaluate the samples to check no divide by zeros
        samples = wf.sample(np.linspace(-pulse.width / 2, pulse.width / 2, 40))
        assert np.all(np.abs(samples.samples) > 0.0)


@pytest.mark.parametrize("seed", [21, 22, 23, 24])
class TestDevicesValidation:
    def test_physical_baseband(self, seed):
        bb = PhysicalBaseband()
        assert not bb.is_calibrated

        bb.frequency = random.Random(seed).uniform(1e05, 1e07)
        bb.if_frequency = random.Random(seed + 1).uniform(1e05, 1e07)
        assert bb.is_calibrated

        with pytest.raises(ValidationError, match=r"must be >=0"):
            bb.frequency = random.Random(seed).uniform(-1e05, -1e07)

        with pytest.raises(ValidationError, match=r"must be >=0"):
            bb.if_frequency = random.Random(seed).uniform(-1e05, -1e07)

    def test_physical_channel(self, seed):
        bb = PhysicalBaseband(
            frequency=random.Random(seed).uniform(1e05, 1e07),
            if_frequency=random.Random(seed).uniform(1e05, 1e07),
        )

        physical_channel = PhysicalChannel(baseband=bb, name_index=0)

        physical_channel.iq_voltage_bias.bias = random.Random(seed + 2).uniform(
            1e05, 1e07
        ) + 1.0j * random.Random(seed + 3).uniform(1e05, 1e07)
        assert physical_channel.is_calibrated
        assert physical_channel.I_bias != physical_channel.Q_bias

        with pytest.raises(ValidationError, match=r"Field is frozen"):
            physical_channel.baseband = 1.0

    def test_pulse_channel(self, seed):
        pulse_channel = PulseChannel()
        assert not pulse_channel.is_calibrated

        pulse_channel.frequency = random.Random(seed).uniform(1e08, 1e10)
        assert pulse_channel.is_calibrated

        with pytest.raises(ValidationError, match=r"must be >=0"):
            pulse_channel.frequency = random.Random(seed).uniform(-1e08, -1e10)

    @pytest.mark.parametrize(
        "input_val, output_type",
        [
            (0.5, float),
            (0.5 + 0.5j, complex),
            (np.complex128(0.5 + 0.5j), complex),
            (np.float64(0.5), float),
        ],
    )
    def test_pulse_channel_scale_types(self, input_val, output_type, seed):
        pulse_channel = PulseChannel(scale=input_val)
        assert isinstance(pulse_channel.scale, output_type)

    def test_custom_pulse_channel(self, seed):
        class CustomPulseChannel(PulseChannel): ...

        custom_pulse_ch = CustomPulseChannel()
        assert custom_pulse_ch.pulse_type == "custom"

        with pytest.raises(TypeError, match=r"must contain `PulseChannel` at the end"):
            # `PulseChannel` should be in the class name
            class InvalidChannel(PulseChannel): ...

        with pytest.raises(TypeError, match=r"must contain `PulseChannel` at the end"):
            # `PulseChannel` should be at the end in the class name
            class InvalidPulseChannelSuffix(PulseChannel): ...

    def test_resonator(self, seed):
        bb = PhysicalBaseband(
            frequency=random.Random(seed).uniform(1e05, 1e07),
            if_frequency=random.Random(seed + 1).uniform(1e05, 1e07),
        )
        physical_channel = ResonatorPhysicalChannel(
            baseband=bb,
            name_index=1,
        )

        resonator = Resonator(physical_channel=physical_channel)
        assert not resonator.is_calibrated

        for pulse_channel_name in ResonatorPulseChannels.model_fields:
            pulse_channel = getattr(resonator.pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed).uniform(1e08, 1e10)
        assert resonator.is_calibrated

        for pulse_channel_name in ResonatorPulseChannels.model_fields:
            pulse_channel = getattr(resonator.pulse_channels, pulse_channel_name)
            with pytest.raises(ValidationError, match=r"must be >=0"):
                pulse_channel.frequency = random.Random(seed).uniform(-1e08, -1e10)

    def test_qubit(self, seed):
        bb = PhysicalBaseband(
            frequency=random.Random(seed).uniform(1e05, 1e07),
            if_frequency=random.Random(seed + 1).uniform(1e05, 1e07),
        )
        physical_channel_r = ResonatorPhysicalChannel(baseband=bb, name_index=1)

        resonator = Resonator(physical_channel=physical_channel_r)
        for i, pulse_channel_name in enumerate(ResonatorPulseChannels.model_fields):
            pulse_channel = getattr(resonator.pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)

        qubit_pulse_channels = QubitPulseChannels()
        for i, pulse_channel_name in enumerate(["drive", "second_state", "freq_shift"]):
            pulse_channel = getattr(qubit_pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)
        assert qubit_pulse_channels.is_calibrated

        physical_channel_q = QubitPhysicalChannel(baseband=bb, name_index=0)
        qubit = Qubit(
            physical_channel=physical_channel_q,
            pulse_channels=qubit_pulse_channels,
            resonator=resonator,
        )
        assert qubit.is_calibrated

        for i, pulse_channel_name in enumerate(["drive", "second_state", "freq_shift"]):
            pulse_channel = getattr(qubit.pulse_channels, pulse_channel_name)
            with pytest.raises(ValidationError, match=r"must be >=0"):
                pulse_channel.frequency = random.Random(seed + i).uniform(-1e08, -1e10)

    def test_qubit_pulse_channel_access(self, seed, py_version):
        hw = generate_hw_model(8, seed=seed)
        if py_version >= (3, 11):
            match_str = "object has no setter"
        else:
            match_str = "can't set attribute"

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
            with pytest.raises(AttributeError, match=match_str):
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

            for i, pulse_channel_name in enumerate(["drive", "second_state", "freq_shift"]):
                pulse_channel = getattr(qubit.pulse_channels, pulse_channel_name)
                pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)

                with pytest.raises(ValidationError, match=r"must be >=0"):
                    pulse_channel.frequency = random.Random(seed + i).uniform(-1e08, -1e10)

            for i, pulse_channel_name in enumerate(
                qubit.resonator.pulse_channels.__class__.model_fields
            ):
                pulse_channel = getattr(qubit.resonator.pulse_channels, pulse_channel_name)
                pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)

                with pytest.raises(ValidationError, match=r"must be >=0"):
                    pulse_channel.frequency = random.Random(seed + i).uniform(-1e08, -1e10)

            for i, cr_channel in enumerate(qubit.cross_resonance_pulse_channels.values()):
                cr_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)
                with pytest.raises(ValidationError, match=r"must be >=0"):
                    cr_channel.frequency = random.Random(seed + i).uniform(-1e08, -1e10)

            for i, crc_channel in enumerate(
                qubit.cross_resonance_cancellation_pulse_channels.values()
            ):
                crc_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)
                with pytest.raises(ValidationError, match=r"must be >=0"):
                    crc_channel.frequency = random.Random(seed + i).uniform(-1e08, -1e10)

            assert qubit.is_calibrated

    @pytest.mark.parametrize(
        "device_type, wrong_phys_ch_type",
        [(Resonator, QubitPhysicalChannel), (Qubit, ResonatorPhysicalChannel)],
    )
    def test_device_wrong_physical_channel(self, seed, device_type, wrong_phys_ch_type):
        bb = PhysicalBaseband(
            frequency=random.Random(seed).uniform(1e05, 1e07),
            if_frequency=random.Random(seed + 1).uniform(1e05, 1e07),
        )
        physical_channel = wrong_phys_ch_type(baseband=bb, name_index=0)

        with pytest.raises(ValidationError, match=r"Input should be"):
            device_type(physical_channel=physical_channel)


@pytest.mark.parametrize("seed", [21, 22, 23, 24])
class TestFrozenQubit:
    def test_cross_resonance_channels(self, seed):
        bb = PhysicalBaseband(
            frequency=random.Random(seed).uniform(1e05, 1e07),
            if_frequency=random.Random(seed + 1).uniform(1e05, 1e07),
        )

        physical_channel_r = ResonatorPhysicalChannel(
            baseband=bb,
            name_index=1,
        )

        resonator = Resonator(physical_channel=physical_channel_r)
        for i, pulse_channel_name in enumerate(ResonatorPulseChannels.model_fields):
            pulse_channel = getattr(resonator.pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)

        qubit_pulse_channels = QubitPulseChannels()
        for i, pulse_channel_name in enumerate(["drive", "second_state", "freq_shift"]):
            pulse_channel = getattr(qubit_pulse_channels, pulse_channel_name)
            pulse_channel.frequency = random.Random(seed + i).uniform(1e08, 1e10)

        physical_channel_q = QubitPhysicalChannel(baseband=bb, name_index=0)
        qubit = Qubit(
            physical_channel=physical_channel_q,
            pulse_channels=qubit_pulse_channels,
            resonator=resonator,
        )

        # Cannot add items to a frozen pulse channel dict.
        with pytest.raises(TypeError, match=r"does not support item assignment"):
            qubit.cross_resonance_pulse_channels[1] = CrossResonancePulseChannel(
                auxiliary_qubit=2
            )

        with pytest.raises(TypeError, match=r"does not support item assignment"):
            qubit.cross_resonance_cancellation_pulse_channels[1] = (
                CrossResonanceCancellationPulseChannel(auxiliary_qubit=2)
            )


class TestPulseChannel:
    @pytest.mark.parametrize(
        "value",
        [
            1.0,
            1 + 0j,
            0.5 + 0.5j,
            np.float64(1.0),
            np.complex128(1 + 0j),
            np.complex128(0.5 + 0.5j),
        ],
    )
    def test_pulse_channel_scale_validator(self, value):
        """Regression test to test robustness against non-standard float types."""
        PulseChannel(uuid="test", scale=value, frequency=5e9)


# Test Qubit (de)serialization with mapper discriminator
class TestQubitMapperDiscriminator:
    """Tests serialization, deserialization, and validation logic for the Qubit model's
    post-processing discriminator field.

    .. rubric:: Covers

    - LinearMapToReal and MaxLikelihood post-processing
    - Exclusivity validation between mean_z_map_args and post_process_method
    - Error handling for invalid/missing discriminators
    """

    def _dummy_qubit(self, mean_z_map_args=None, post_process_method=None):
        """Helper to construct a Qubit instance for testing.

        :param mean_z_map_args: List of arguments for linear mapping (or None).
        :param post_process_method: Post-processing method instance (or None).
        :return: Qubit instance with provided configuration.
        """
        baseband = PhysicalBaseband(frequency=1.0, if_frequency=2.0)
        iq_bias = IQBias(bias=0.0)
        pc = QubitPhysicalChannel(
            baseband=baseband, block_size=1, iq_voltage_bias=iq_bias, name_index=0
        )
        qpc = QubitPulseChannels()
        rpc = ResonatorPulseChannels()
        resonator_pc = ResonatorPhysicalChannel(
            baseband=baseband, block_size=1, iq_voltage_bias=iq_bias, name_index=0
        )
        resonator = Resonator(physical_channel=resonator_pc, pulse_channels=rpc)
        return Qubit(
            physical_channel=pc,
            pulse_channels=qpc,
            resonator=resonator,
            mean_z_map_args=mean_z_map_args,
            discriminator=0.0,
            post_process_method=post_process_method,
            direct_x_pi=False,
        )

    def test_qubit_mapper_linear_map_to_real_json(self):
        """Test JSON serialization/deserialization for Qubit with LinearMapToReal post-
        processing.

        :raises AssertionError: If method discriminator or argument preservation fails.
        """
        post_process_method = LinearMapToRealMethod(mean_z_map_args=[1.0, 0.0])
        q = self._dummy_qubit(mean_z_map_args=None, post_process_method=post_process_method)
        s = q.model_dump_json()
        assert '"method":"linear_map_complex_to_real"' in s
        q2 = Qubit.model_validate_json(s)
        assert isinstance(q2.post_process_method, LinearMapToRealMethod)
        assert q2.post_process_method.mean_z_map_args == [1.0, 0.0]

    def test_qubit_mapper_max_likelihood_json(self):
        """Test JSON serialization/deserialization for Qubit with MaxLikelihood post-
        processing.

        :raises AssertionError: If method discriminator, noise estimate, or state mapping
            fails.
        """
        states = [
            MLStateMap(state="0", val=0, location=0.0),
            MLStateMap(state="1", val=1, location=1.0),
        ]
        post_process_method = MaxLikelihoodMethod(noise_est=0.1, states=states)
        q = self._dummy_qubit(mean_z_map_args=None, post_process_method=post_process_method)
        s = q.model_dump_json()
        assert '"method":"max_likelihood"' in s
        q2 = Qubit.model_validate_json(s)
        assert isinstance(q2.post_process_method, MaxLikelihoodMethod)
        assert q2.post_process_method.noise_est == 0.1
        assert len(q2.post_process_method.states) == 2

    def test_qubit_mapper_none_json(self):
        """Test Qubit JSON serialization/deserialization when no post-processing method is
        provided.

        :raises AssertionError: If None is not preserved for post_process_method.
        """
        q = self._dummy_qubit(mean_z_map_args=[1.0, 0.0], post_process_method=None)
        s = q.model_dump_json()
        assert '"post_process_method":null' in s
        q2 = Qubit.model_validate_json(s)
        assert q2.post_process_method is None

    def test_qubit_mapper_invalid_discriminator(self):
        """Test that an invalid post_process_method discriminator fails Qubit validation.

        :raises ValidationError: If an unknown method discriminator is accepted.
        """
        invalid_post_process_method = {
            "method": "INVALID_METHOD",
            "mean_z_map_args": [1.0, 0.0],
        }
        d = {
            "physical_channel": self._dummy_qubit(
                mean_z_map_args=[1.0, 0.0], post_process_method=None
            ).physical_channel.model_dump(),
            "pulse_channels": self._dummy_qubit(
                mean_z_map_args=[1.0, 0.0], post_process_method=None
            ).pulse_channels.model_dump(),
            "resonator": self._dummy_qubit(
                mean_z_map_args=[1.0, 0.0], post_process_method=None
            ).resonator.model_dump(),
            "mean_z_map_args": None,
            "discriminator": 0.0,
            "post_process_method": invalid_post_process_method,
            "direct_x_pi": False,
        }
        with pytest.raises(ValidationError, match=r"INVALID_METHOD"):
            Qubit.model_validate(d)

    def test_qubit_mapper_both_provided(self):
        """
        Test exclusivity validation: error if both mean_z_map_args and post_process_method are provided.

        :raises ValueError: If both are provided.
        """
        post_process_method = LinearMapToRealMethod(mean_z_map_args=[1.0, 0.0])
        with pytest.raises(
            ValueError,
            match="Exactly one of 'mean_z_map_args' or 'post_process_method' must be provided",
        ):
            self._dummy_qubit(
                mean_z_map_args=[1.0, 0.0], post_process_method=post_process_method
            )

    def test_qubit_mapper_neither_provided(self):
        """
        Test exclusivity validation: error if neither mean_z_map_args nor post_process_method is provided.

        :raises ValueError: If neither is provided.
        """
        with pytest.raises(
            ValueError,
            match="Exactly one of 'mean_z_map_args' or 'post_process_method' must be provided",
        ):
            self._dummy_qubit(mean_z_map_args=None, post_process_method=None)
