# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from contextlib import nullcontext

import numpy as np
import pytest

from qat.core.config.configure import get_config
from qat.core.result_base import ResultManager
from qat.middleend.passes.purr.analysis import ActiveChannelResults
from qat.middleend.passes.purr.transform import EvaluatePulses
from qat.middleend.passes.purr.validation import (
    DynamicFrequencyValidation,
    FixedIntermediateFrequencyValidation,
    FrequencySetupValidation,
    InstructionValidation,
    PhysicalChannelAmplitudeValidation,
    ReadoutValidation,
)
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.backends.live import LiveHardwareModel
from qat.purr.compiler.instructions import (
    AcquireMode,
    CustomPulse,
    PostProcessType,
    ProcessAxis,
    Pulse,
    PulseShapeType,
)

from tests.unit.utils.pulses import pulse_attributes

qatconfig = get_config()


class TestInstructionValidation:
    hw = EchoModelLoader(qubit_count=4).load()
    target_data = TargetData.default()

    @pytest.mark.parametrize("pulse_duration_limits", [True, False, None])
    def test_valid_instructions(self, pulse_duration_limits):
        builder = self.hw.create_builder()
        for qubit in self.hw.qubits:
            builder.X(target=qubit)

        InstructionValidation(
            self.target_data, pulse_duration_limits=pulse_duration_limits
        ).run(builder)

    @pytest.mark.parametrize("pulse_duration_limits", [True, False, None])
    def test_pulse_too_long(self, pulse_duration_limits):
        builder = self.hw.create_builder()
        for qubit in self.hw.qubits:
            builder.pulse(
                quantum_target=qubit.get_drive_channel(),
                width=0.1,
                shape=PulseShapeType.GAUSSIAN,
            )

        with (
            pytest.raises(ValueError, match="Max Waveform width is")
            if (pulse_duration_limits or pulse_duration_limits is None)
            else nullcontext()
        ):
            InstructionValidation(
                self.target_data, pulse_duration_limits=pulse_duration_limits
            ).run(builder)

    def test_too_many_instructions(self):
        builder = self.hw.create_builder()
        drive_pulse_ch = self.hw.get_qubit(0).get_drive_channel()

        for _ in range(self.target_data.QUBIT_DATA.instruction_memory_size + 1):
            builder.pulse(
                quantum_target=drive_pulse_ch, width=1e-08, shape=PulseShapeType.GAUSSIAN
            )

        with pytest.raises(
            ValueError, match="too large to be run in a single block on current hardware."
        ):
            InstructionValidation(self.target_data).run(builder)


class TestPhysicalChannelAmplitudeValidation:
    model = EchoModelLoader(qubit_count=2).load()

    def test_valid_custom_pulse(self):
        # Create custom sine waveform pulse
        t = np.linspace(0.0, 10 * np.pi, num=200)
        samples = (np.sin(t) + 1j * np.cos(t)).tolist()
        qubit = self.model.get_qubit(0)
        builder = self.model.create_builder()
        builder.repeat(2000)
        builder.add(CustomPulse(qubit.get_drive_channel(), samples))
        ir = PhysicalChannelAmplitudeValidation().run(builder)

        assert builder == ir

    @pytest.mark.parametrize("real_coeff, imag_coeff", [(1, 2), (2, 1), (2, 2)])
    def test_invalid_custom_pulse(self, real_coeff, imag_coeff):
        # Create custom sine waveform pulse
        t = np.linspace(0.0, 10 * np.pi, num=200)
        samples = (real_coeff * np.sin(t) + imag_coeff * 1j * np.cos(t)).tolist()
        qubit = self.model.get_qubit(0)
        builder = self.model.create_builder()
        builder.repeat(2000)
        builder.add(CustomPulse(qubit.get_drive_channel(), samples))
        with pytest.raises(ValueError):
            PhysicalChannelAmplitudeValidation().run(builder)

    def test_exceeding_custom_pulse_sum(self):
        # Create custom sine waveform pulse
        t = np.linspace(0.0, 10 * np.pi, num=200)
        samples = (np.sin(t) + 1j * np.cos(t)).tolist()
        qubit = self.model.get_qubit(0)
        builder = self.model.create_builder()
        builder.repeat(2000)
        builder.add(CustomPulse(qubit.get_drive_channel(), samples))
        builder.add(
            CustomPulse(qubit.get_cross_resonance_channel(self.model.get_qubit(1)), samples)
        )
        with pytest.raises(ValueError):
            PhysicalChannelAmplitudeValidation().run(builder)

    def test_valid_square_pulse(self):
        qubit = self.model.get_qubit(0)
        builder = self.model.create_builder()
        builder.repeat(2000)
        builder.add(
            Pulse(qubit.get_measure_channel(), PulseShapeType.SQUARE, width=1e-06, amp=0.9)
        )
        ir = PhysicalChannelAmplitudeValidation().run(builder)

        assert builder == ir

    @pytest.mark.parametrize(
        "shape", [s for s in list(PulseShapeType) if s != PulseShapeType.SQUARE]
    )
    def test_invalid_pulse_shape(self, shape):
        qubit = self.model.get_qubit(0)
        builder = self.model.create_builder()
        builder.repeat(2000)
        builder.add(Pulse(qubit.get_measure_channel(), shape=shape, width=1e-06, amp=0.9))
        with pytest.raises(ValueError, match=f"un-lowered {shape}"):
            PhysicalChannelAmplitudeValidation().run(builder)

    @pytest.mark.parametrize("amp, scale", [(1.2, 1), (1, 5)])
    @pytest.mark.parametrize("phase", [0, np.pi / 2, np.pi, 3 * np.pi / 2])
    def test_invalid_square_pulse(self, amp, scale, phase):
        qubit = self.model.get_qubit(0)
        builder = self.model.create_builder()
        builder.repeat(2000)
        builder.add(
            Pulse(
                qubit.get_measure_channel(),
                PulseShapeType.SQUARE,
                width=1e-06,
                amp=amp,
                scale_factor=scale,
                phase=phase,
            )
        )
        with pytest.raises(ValueError):
            PhysicalChannelAmplitudeValidation().run(builder)

    def test_exceeding_square_pulse_sum(self):
        qubit = self.model.get_qubit(0)
        builder = self.model.create_builder()
        builder.repeat(2000)
        builder.add(
            [
                Pulse(
                    qubit.get_drive_channel(),
                    PulseShapeType.SQUARE,
                    width=1e-06,
                    amp=0.9,
                    ignore_channel_scale=True,
                ),
                Pulse(
                    qubit.get_cross_resonance_channel(self.model.get_qubit(1)),
                    PulseShapeType.SQUARE,
                    width=1e-06,
                    amp=0.9,
                    ignore_channel_scale=True,
                ),
            ]
        )
        with pytest.raises(ValueError):
            PhysicalChannelAmplitudeValidation().run(builder)

    def test_exceeding_mixed_pulse_sum(self):
        # Create custom sine waveform pulse
        t = np.linspace(0.0, 10 * np.pi, num=200)
        samples = (np.sin(t) + 1j * np.cos(t)).tolist()
        qubit = self.model.get_qubit(0)
        builder = self.model.create_builder()
        builder.repeat(2000)
        builder.add(
            [
                Pulse(
                    qubit.get_drive_channel(),
                    PulseShapeType.SQUARE,
                    width=1e-06,
                    amp=0.9,
                    ignore_channel_scale=True,
                ),
                CustomPulse(
                    qubit.get_cross_resonance_channel(self.model.get_qubit(1)),
                    samples=samples,
                ),
            ]
        )
        with pytest.raises(ValueError):
            PhysicalChannelAmplitudeValidation().run(builder)

    def test_same_channel_not_summed(self):
        # Create custom sine waveform pulse
        t = np.linspace(0.0, 10 * np.pi, num=200)
        samples = (np.sin(t) + 1j * np.cos(t)).tolist()
        qubit = self.model.get_qubit(0)
        builder = self.model.create_builder()
        builder.repeat(2000)
        builder.add(
            Pulse(
                qubit.get_drive_channel(),
                PulseShapeType.SQUARE,
                width=1e-06,
                amp=0.9,
                ignore_channel_scale=True,
            )
        )
        builder.add(
            CustomPulse(
                qubit.get_drive_channel(),
                samples=samples,
            )
        )
        ir = PhysicalChannelAmplitudeValidation().run(builder)

        assert ir == builder

    def test_reset_by_sync(self):
        # Create custom sine waveform pulse
        t = np.linspace(0.0, 10 * np.pi, num=200)
        samples = (np.sin(t) + 1j * np.cos(t)).tolist()
        qubit = self.model.get_qubit(0)
        builder = self.model.create_builder()
        builder.repeat(2000)
        builder.add(
            Pulse(
                qubit.get_drive_channel(),
                PulseShapeType.SQUARE,
                width=1e-06,
                amp=0.9,
                ignore_channel_scale=True,
            )
        )
        builder.synchronize(qubit)
        builder.add(
            CustomPulse(
                qubit.get_cross_resonance_channel(self.model.get_qubit(1)),
                samples=samples,
            )
        )
        ir = PhysicalChannelAmplitudeValidation().run(builder)

        assert ir == builder

    def test_not_all_reset_by_sync(self):
        # Create custom sine waveform pulse
        t = np.linspace(0.0, 10 * np.pi, num=200)
        samples = (np.sin(t) + 1j * np.cos(t)).tolist()
        qubit0 = self.model.get_qubit(0)
        qubit1 = self.model.get_qubit(1)
        builder = self.model.create_builder()
        builder.repeat(2000)
        builder.add(
            Pulse(
                qubit0.get_drive_channel(),
                PulseShapeType.SQUARE,
                width=1e-06,
                amp=0.9,
                ignore_channel_scale=True,
            )
        )
        builder.synchronize(
            [
                qubit0.get_cross_resonance_cancellation_channel(qubit1),
                qubit1.get_cross_resonance_channel(qubit0),
            ]
        )
        builder.add(
            CustomPulse(
                qubit0.get_cross_resonance_channel(qubit1),
                samples=samples,
            )
        )
        with pytest.raises(ValueError):
            PhysicalChannelAmplitudeValidation().run(builder)

    def test_reset_by_sync_fails(self):
        # Create custom sine waveform pulse
        model = EchoModelLoader(qubit_count=3).load()
        qubit1 = model.get_qubit(0)
        qubit2 = model.get_qubit(1)
        qubit3 = model.get_qubit(2)
        builder = model.create_builder()
        builder.add(
            Pulse(
                qubit2.get_drive_channel(),
                PulseShapeType.SQUARE,
                width=1e-06,
                amp=0.4,
                ignore_channel_scale=True,
            )
        )
        builder.add(
            Pulse(
                qubit2.get_cross_resonance_channel(qubit3),
                PulseShapeType.SQUARE,
                width=1e-06,
                amp=0.4,
                ignore_channel_scale=True,
            )
        )
        builder.synchronize(
            [qubit2.get_drive_channel(), qubit2.get_cross_resonance_channel(qubit3)]
        )
        builder.add(
            Pulse(
                qubit2.get_cross_resonance_cancellation_channel(qubit1),
                PulseShapeType.SQUARE,
                width=1e-06,
                amp=0.4,
                ignore_channel_scale=True,
            )
        )
        with pytest.raises(ValueError):
            PhysicalChannelAmplitudeValidation().run(builder)

    @pytest.mark.parametrize("attributes1", pulse_attributes)
    @pytest.mark.parametrize("attributes2", pulse_attributes)
    def test_integration_with_evaluate_pulses_passes(self, attributes1, attributes2):
        builder = self.model.create_builder()
        chan1 = self.model.qubits[0].get_drive_channel()
        chan2 = self.model.qubits[0].get_cross_resonance_cancellation_channel(
            self.model.qubits[1]
        )
        builder.pulse(
            chan1, width=400e-9, amp=0.45, ignore_channel_scale=True, **attributes1
        )
        builder.pulse(
            chan2, width=400e-9, amp=0.49, ignore_channel_scale=True, **attributes2
        )
        EvaluatePulses().run(builder)
        PhysicalChannelAmplitudeValidation().run(builder)

    @pytest.mark.parametrize("attributes1", pulse_attributes)
    @pytest.mark.parametrize("attributes2", pulse_attributes)
    def test_integration_with_evaluate_pulses_fails(self, attributes1, attributes2):
        builder = self.model.create_builder()
        chan1 = self.model.qubits[0].get_drive_channel()
        chan2 = self.model.qubits[0].get_cross_resonance_cancellation_channel(
            self.model.qubits[1]
        )
        builder.pulse(
            chan1, width=400e-9, amp=0.49, ignore_channel_scale=True, **attributes1
        )
        builder.pulse(
            chan2, width=400e-9, amp=0.65, ignore_channel_scale=True, **attributes2
        )
        EvaluatePulses().run(builder)
        with pytest.raises(ValueError):
            PhysicalChannelAmplitudeValidation().run(builder)


class TestFrequencySetupValidation:
    target_data = TargetData.default()

    @staticmethod
    def target_data_with_non_zero_if_min():
        target_data = TargetData.default()
        return target_data.model_copy(
            update={
                "QUBIT_DATA": target_data.QUBIT_DATA.model_copy(
                    update={"pulse_channel_if_freq_min": 1e6}
                ),
                "RESONATOR_DATA": target_data.RESONATOR_DATA.model_copy(
                    update={"pulse_channel_if_freq_min": 1e6}
                ),
            }
        )

    def test_create_baseband_frequency_map(self):
        model = EchoModelLoader().load()
        baseband_frequencies = FrequencySetupValidation._create_baseband_frequency_map(
            model
        )
        assert len(baseband_frequencies) == len(model.physical_channels)

    def test_create_resonator_map(self):
        model = EchoModelLoader().load()
        is_resonator = FrequencySetupValidation._create_resonator_map(model)
        assert len([val for val in is_resonator.values() if val]) == len(
            [val for val in is_resonator.values() if not val]
        )
        assert len(is_resonator) == len(model.physical_channels)

    def test_create_pulse_channel_if_map(self):
        model = EchoModelLoader().load()
        pulse_channel_ifs = FrequencySetupValidation._create_pulse_channel_if_map(model)
        assert len(pulse_channel_ifs) == len(model.pulse_channels)

    def test_validate_baseband_frequencies(self):
        is_resonator = {
            "test1": False,
            "test2": False,
            "test3": False,
            "test4": True,
            "test5": True,
            "test6": True,
        }
        qubit_lo_freq_limits = (
            self.target_data.QUBIT_DATA.pulse_channel_lo_freq_min,
            self.target_data.QUBIT_DATA.pulse_channel_lo_freq_max,
        )
        resonator_lo_freq_limits = (
            self.target_data.RESONATOR_DATA.pulse_channel_lo_freq_min,
            self.target_data.RESONATOR_DATA.pulse_channel_lo_freq_max,
        )
        freqs = {
            "test1": 0.9 * qubit_lo_freq_limits[0],
            "test2": 1.1 * qubit_lo_freq_limits[1],
            "test3": 0.99 * qubit_lo_freq_limits[1],
            "test4": 0.9 * resonator_lo_freq_limits[0],
            "test5": 1.1 * resonator_lo_freq_limits[1],
            "test6": 0.99 * resonator_lo_freq_limits[1],
        }
        baseband_validations = FrequencySetupValidation._validate_baseband_frequencies(
            freqs, is_resonator, qubit_lo_freq_limits, resonator_lo_freq_limits
        )
        assert len(baseband_validations) == 6
        for i in range(6):
            assert f"test{i + 1}" in baseband_validations
            assert baseband_validations[f"test{i + 1}"] == (True if i in (2, 5) else False)

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_validate_pulse_channel_ifs(self, sign):
        model = EchoModelLoader().load()
        target_data = self.target_data_with_non_zero_if_min()
        qubits = model.qubits[0:2]
        physical_channels = [
            qubits[0].physical_channel,
            qubits[1].physical_channel,
            qubits[0].measure_device.physical_channel,
            qubits[1].measure_device.physical_channel,
        ]
        pulse_channels = [
            qubits[0].get_drive_channel(),
            qubits[0].get_cross_resonance_channel(model.qubits[1]),
            qubits[1].get_drive_channel(),
            qubits[0].get_measure_channel(),
            qubits[0].get_acquire_channel(),
            qubits[1].get_acquire_channel(),
        ]

        qubit_if_freq_limits = (
            target_data.QUBIT_DATA.pulse_channel_if_freq_min,
            target_data.QUBIT_DATA.pulse_channel_if_freq_max,
        )
        resonator_if_freq_limits = (
            target_data.RESONATOR_DATA.pulse_channel_if_freq_min,
            target_data.RESONATOR_DATA.pulse_channel_if_freq_max,
        )

        pulse_channel_ifs = {
            pulse_channels[0].partial_id(): sign * 0.99 * qubit_if_freq_limits[0],
            pulse_channels[1].partial_id(): sign * 1.1 * qubit_if_freq_limits[1],
            pulse_channels[2].partial_id(): (
                sign * 0.5 * (qubit_if_freq_limits[0] + qubit_if_freq_limits[1])
            ),
            pulse_channels[3].partial_id(): sign * 0.99 * resonator_if_freq_limits[0],
            pulse_channels[4].partial_id(): sign * 1.1 * resonator_if_freq_limits[1],
            pulse_channels[5].partial_id(): (
                sign * 0.5 * (resonator_if_freq_limits[0] + resonator_if_freq_limits[1])
            ),
        }

        is_resonator = {
            physical_channels[0].id: False,
            physical_channels[1].id: False,
            physical_channels[2].id: True,
            physical_channels[3].id: True,
        }

        pulse_to_physical_channels = {
            pulse_channel.partial_id(): pulse_channel.physical_channel.id
            for pulse_channel in pulse_channels
        }

        validate_ifs = FrequencySetupValidation._validate_pulse_channel_ifs(
            pulse_channel_ifs,
            is_resonator,
            pulse_to_physical_channels,
            qubit_if_freq_limits,
            resonator_if_freq_limits,
        )

        for pulse_channel, validation in validate_ifs.items():
            if pulse_channel in (
                pulse_channels[2].partial_id(),
                pulse_channels[5].partial_id(),
            ):
                assert validation is True
            else:
                assert validation is False

    @pytest.mark.parametrize("violation_type", ["lower", "higher"])
    @pytest.mark.parametrize("channel_type", ["qubit", "resonator"])
    def test_raises_error_for_invalid_baseband_frequency(
        self, violation_type, channel_type
    ):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]

        # Sets up a channel to have a bad baseband freq but not a bad IF freq.
        if channel_type == "qubit":
            channel = qubit.get_drive_channel()
            if_freq = 0.5 * (
                self.target_data.QUBIT_DATA.pulse_channel_if_freq_min
                + self.target_data.QUBIT_DATA.pulse_channel_if_freq_max
            )
            if violation_type == "lower":
                channel.physical_channel.baseband.frequency = (
                    0.99 * self.target_data.QUBIT_DATA.pulse_channel_lo_freq_min
                )
            elif violation_type == "higher":
                channel.physical_channel.baseband.frequency = (
                    1.1 * self.target_data.QUBIT_DATA.pulse_channel_lo_freq_max
                )
            channel.frequency = channel.physical_channel.baseband.frequency + if_freq
        elif channel_type == "resonator":
            channel = qubit.get_measure_channel()
            if_freq = 0.5 * (
                self.target_data.RESONATOR_DATA.pulse_channel_if_freq_min
                + self.target_data.RESONATOR_DATA.pulse_channel_if_freq_max
            )
            if violation_type == "lower":
                channel.physical_channel.baseband.frequency = (
                    0.99 * self.target_data.RESONATOR_DATA.pulse_channel_lo_freq_min
                )
            elif violation_type == "higher":
                channel.physical_channel.baseband.frequency = (
                    1.1 * self.target_data.RESONATOR_DATA.pulse_channel_lo_freq_max
                )
            channel.frequency = channel.physical_channel.baseband.frequency + if_freq

        builder = model.create_builder()
        builder.pulse(channel, width=80e-9, shape=PulseShapeType.SQUARE)
        res_mgr = ResultManager()
        res_mgr.add(ActiveChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            FrequencySetupValidation(model, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("violation_type", ["lower", "higher"])
    @pytest.mark.parametrize("channel_type", ["qubit", "resonator"])
    def test_raises_error_for_invalid_if(self, violation_type, channel_type):
        model = EchoModelLoader().load()
        target_data = self.target_data_with_non_zero_if_min()
        qubit = model.qubits[0]

        # Sets up a channel to have a bad IF freq but not a bad baseband freq.
        if channel_type == "qubit":
            channel = qubit.get_drive_channel()
            baseband_freq = channel.physical_channel.baseband.frequency
            if violation_type == "lower":
                channel.frequency = (
                    0.99 * target_data.QUBIT_DATA.pulse_channel_if_freq_min + baseband_freq
                )
            elif violation_type == "higher":
                channel.frequency = (
                    1.1 * target_data.QUBIT_DATA.pulse_channel_if_freq_max + baseband_freq
                )
        elif channel_type == "resonator":
            channel = qubit.get_measure_channel()
            baseband_freq = channel.physical_channel.baseband.frequency
            if violation_type == "lower":
                channel.frequency = (
                    0.99 * target_data.RESONATOR_DATA.pulse_channel_if_freq_min
                    + baseband_freq
                )
            elif violation_type == "higher":
                channel.frequency = (
                    1.1 * target_data.RESONATOR_DATA.pulse_channel_if_freq_max
                    + baseband_freq
                )

        builder = model.create_builder()
        builder.pulse(channel, width=80e-9, shape=PulseShapeType.SQUARE)
        res_mgr = ResultManager()
        res_mgr.add(ActiveChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            FrequencySetupValidation(model, target_data).run(builder, res_mgr)

    def test_with_custom_pulse_channel(self):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]
        physical_channel = qubit.get_drive_channel().physical_channel
        channel = physical_channel.create_pulse_channel("custom")
        channel.frequency = (
            channel.physical_channel.baseband_frequency
            + 1.1 * self.target_data.QUBIT_DATA.pulse_channel_if_freq_max
        )

        builder = model.create_builder()
        builder.add(Pulse(channel, width=80e-9, shape=PulseShapeType.SQUARE))
        res_mgr = ResultManager()
        res_mgr.add(ActiveChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            FrequencySetupValidation(model, self.target_data).run(builder, res_mgr)


class TestDynamicFrequencyValidation:
    target_data = TargetData.default()

    @staticmethod
    def get_single_pulse_channel(model):
        return next(iter(model.pulse_channels.values()))

    @staticmethod
    def get_two_pulse_channels_from_single_physical_channel(model):
        physical_channel = next(iter(model.physical_channels.values()))
        pulse_channels = iter(
            model.get_pulse_channels_from_physical_channel(physical_channel)
        )
        return next(pulse_channels), next(pulse_channels)

    @staticmethod
    def get_two_pulse_channels_from_different_physical_channels(model):
        physical_channels = iter(model.physical_channels.values())
        pulse_channel_1 = next(
            iter(model.get_pulse_channels_from_physical_channel(next(physical_channels)))
        )
        pulse_channel_2 = next(
            iter(model.get_pulse_channels_from_physical_channel(next(physical_channels)))
        )
        return pulse_channel_1, pulse_channel_2

    @staticmethod
    def target_data_with_non_zero_if_min():
        target_data = TargetData.default()
        return target_data.model_copy(
            update={
                "QUBIT_DATA": target_data.QUBIT_DATA.model_copy(
                    update={"pulse_channel_if_freq_min": 1e6}
                ),
                "RESONATOR_DATA": target_data.RESONATOR_DATA.model_copy(
                    update={"pulse_channel_if_freq_min": 1e6}
                ),
            }
        )

    def test_create_resonator_map(self):
        model = EchoModelLoader().load()
        is_resonator = FrequencySetupValidation._create_resonator_map(model)
        assert len([val for val in is_resonator.values() if val]) == len(
            [val for val in is_resonator.values() if not val]
        )
        assert len(is_resonator) == len(model.physical_channels)

    @pytest.mark.parametrize("scale", [-0.95, -0.5, 0.0, 0.5, 0.95])
    def test_raises_no_error_when_freq_shift_in_range(self, scale):
        """Shifts the frequency in the range [min_if, max_if] to check it does not fail."""
        model = EchoModelLoader().load()
        channel = self.get_single_pulse_channel(model)
        qubit_data = self.target_data.QUBIT_DATA
        target_freq = (
            channel.physical_channel.baseband_frequency
            + scale * qubit_data.pulse_channel_lo_freq_max
            - (1 - scale) * qubit_data.pulse_channel_lo_freq_min
        )
        delta_freq = target_freq - channel.frequency
        builder = model.create_builder()
        builder.frequency_shift(channel, delta_freq)
        res_mgr = ResultManager()
        res_mgr.add(ActiveChannelResults(target_map={channel: "doesn't matter"}))
        DynamicFrequencyValidation(model, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_raises_error_when_lower_than_min(self, sign):
        model = EchoModelLoader().load()
        target_data = self.target_data_with_non_zero_if_min()
        channel = self.get_single_pulse_channel(model)
        qubit_data = target_data.QUBIT_DATA
        assert qubit_data.pulse_channel_if_freq_min > 0.0
        target_freq = (
            channel.physical_channel.baseband_frequency
            + sign * 0.5 * qubit_data.pulse_channel_lo_freq_min
        )
        delta_freq = target_freq - channel.frequency
        builder = model.create_builder()
        builder.frequency_shift(channel, delta_freq)
        res_mgr = ResultManager()
        res_mgr.add(ActiveChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            DynamicFrequencyValidation(model, target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_raises_error_when_higher_than_max(self, sign):
        model = EchoModelLoader().load()
        channel = self.get_single_pulse_channel(model)
        qubit_data = self.target_data.QUBIT_DATA
        target_freq = (
            channel.physical_channel.baseband_frequency
            + sign * 1.1 * qubit_data.pulse_channel_lo_freq_max
        )
        delta_freq = target_freq - channel.frequency
        builder = model.create_builder()
        builder.frequency_shift(channel, delta_freq)
        res_mgr = ResultManager()
        res_mgr.add(ActiveChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            DynamicFrequencyValidation(model, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_raises_error_when_moving_out_and_back_in_of_range(self, sign):
        model = EchoModelLoader().load()
        channel = self.get_single_pulse_channel(model)
        qubit_data = self.target_data.QUBIT_DATA
        target_freq = (
            channel.physical_channel.baseband_frequency
            + sign * 1.1 * qubit_data.pulse_channel_lo_freq_max
        )
        delta_freq = target_freq - channel.frequency
        builder = model.create_builder()
        builder.frequency_shift(channel, delta_freq)
        builder.frequency_shift(channel, -delta_freq)
        res_mgr = ResultManager()
        res_mgr.add(ActiveChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            DynamicFrequencyValidation(model, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_no_value_error_for_two_channels_same_physical_channel(self, sign):
        model = EchoModelLoader().load()
        channels = self.get_two_pulse_channels_from_single_physical_channel(model)
        builder = model.create_builder()

        qubit_data = self.target_data.QUBIT_DATA
        delta_freq = 0.05 * (
            qubit_data.pulse_channel_lo_freq_max - qubit_data.pulse_channel_lo_freq_min
        )

        # interweave with random instructions
        builder.phase_shift(channels[0], np.pi)
        builder.frequency_shift(channels[0], sign * delta_freq)
        builder.phase_shift(channels[1], -np.pi)
        builder.frequency_shift(channels[1], sign * 2 * delta_freq)
        builder.frequency_shift(channels[0], sign * 3 * delta_freq)
        builder.phase_shift(channels[1], -2.54)
        builder.frequency_shift(channels[1], sign * 4 * delta_freq)

        res_mgr = ResultManager()
        res_mgr.add(
            ActiveChannelResults(
                target_map={channels[0]: "doesn't matter", channels[1]: "matters even less"}
            )
        )
        DynamicFrequencyValidation(model, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_value_error_for_two_channels_same_physical_channel(self, sign):
        model = EchoModelLoader().load()
        channels = self.get_two_pulse_channels_from_single_physical_channel(model)
        builder = model.create_builder()

        qubit_data = self.target_data.QUBIT_DATA
        delta_freq = 0.05 * (
            qubit_data.pulse_channel_lo_freq_max - qubit_data.pulse_channel_lo_freq_min
        )

        # interweave with random instructions
        builder.phase_shift(channels[0], np.pi)
        builder.frequency_shift(channels[0], sign * 10 * delta_freq)
        builder.phase_shift(channels[1], -np.pi)
        builder.frequency_shift(channels[1], sign * 2 * delta_freq)
        builder.frequency_shift(channels[0], sign * 16 * delta_freq)
        builder.phase_shift(channels[1], -2.54)
        builder.frequency_shift(channels[1], sign * 4 * delta_freq)

        res_mgr = ResultManager()
        res_mgr.add(
            ActiveChannelResults(
                target_map={channels[0]: "doesn't matter", channels[1]: "matters even less"}
            )
        )
        with pytest.raises(ValueError):
            DynamicFrequencyValidation(model, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_value_error_for_two_channels_different_physical_channel(self, sign):
        model = EchoModelLoader().load()
        channels = self.get_two_pulse_channels_from_different_physical_channels(model)
        qubit_data = self.target_data.QUBIT_DATA
        delta_freq = 0.05 * (
            qubit_data.pulse_channel_lo_freq_max - qubit_data.pulse_channel_lo_freq_min
        )

        builder = model.create_builder()

        # interweave with random instructions
        builder.phase_shift(channels[0], np.pi)
        builder.frequency_shift(channels[0], sign * 10 * delta_freq)
        builder.phase_shift(channels[1], -np.pi)
        builder.frequency_shift(channels[1], sign * 2 * delta_freq)
        builder.frequency_shift(channels[0], sign * 16 * delta_freq)
        builder.phase_shift(channels[1], -2.54)
        builder.frequency_shift(channels[1], sign * 4 * delta_freq)

        res_mgr = ResultManager()
        res_mgr.add(
            ActiveChannelResults(
                target_map={channels[0]: "doesn't matter", channels[1]: "matters even less"}
            )
        )
        with pytest.raises(ValueError):
            DynamicFrequencyValidation(model, self.target_data).run(builder, res_mgr)

    def test_small_change_to_if_is_valid(self):
        """Lil' regression test to test a bug fix for a bug found in pipeline tests.

        Does a little change to the IF frequency - ensures the current IF value is added.
        """

        target_data = self.target_data_with_non_zero_if_min()
        model = EchoModelLoader().load()
        channel = model.qubits[0].get_drive_channel()
        channel.frequency = channel.physical_channel.baseband.frequency + 1e7
        builder = model.create_builder()
        builder.frequency_shift(channel, 1e3)
        builder.pulse(channel, width=80e-9, shape=PulseShapeType.SQUARE)
        res_mgr = ResultManager()
        res_mgr.add(ActiveChannelResults(target_map={channel: "doesn't matter"}))
        DynamicFrequencyValidation(model, target_data).run(builder, res_mgr)


class TestFixedIntermediateFrequencyValidation:
    @staticmethod
    def get_two_pulse_channels_from_different_physical_channels(model):
        physical_channels = iter(model.physical_channels.values())
        pulse_channel_1 = next(
            iter(model.get_pulse_channels_from_physical_channel(next(physical_channels)))
        )
        pulse_channel_2 = next(
            iter(model.get_pulse_channels_from_physical_channel(next(physical_channels)))
        )
        return pulse_channel_1, pulse_channel_2

    def test_create_fixed_if_map(self):
        model = EchoModelLoader().load()
        fixed_if_channels = [qubit.get_drive_channel() for qubit in model.qubits[0:2]]
        for pulse_channel in model.pulse_channels.values():
            pulse_channel.fixed_if = pulse_channel in fixed_if_channels
        fixed_ifs = FixedIntermediateFrequencyValidation._create_fixed_if_map(model)
        assert len(fixed_ifs) == len(model.physical_channels)
        for channel in fixed_if_channels:
            assert fixed_ifs[channel.physical_channel.id] is True

    def test_fixed_if_raises_error(self):
        model = EchoModelLoader().load()
        channels = self.get_two_pulse_channels_from_different_physical_channels(model)
        builder = model.create_builder()
        channels[0].fixed_if = True
        builder.frequency_shift(channels[0], 1e8)
        builder.frequency_shift(channels[1], 1e8)
        res_mgr = ResultManager()
        res_mgr.add(
            ActiveChannelResults(
                target_map={channels[0]: "doesn't matter", channels[1]: "matters even less"}
            )
        )
        with pytest.raises(ValueError):
            FixedIntermediateFrequencyValidation(model).run(builder, res_mgr)

    def test_fixed_if_does_not_affect_other_channel(self):
        model = EchoModelLoader().load()
        channels = self.get_two_pulse_channels_from_different_physical_channels(model)
        builder = model.create_builder()
        channels[0].fixed_if = True
        builder.pulse(
            channels[0],
            width=80e-9,
            shape=PulseShapeType.SQUARE,
        )
        builder.frequency_shift(channels[1], 1e8)
        res_mgr = ResultManager()
        res_mgr.add(
            ActiveChannelResults(
                target_map={channels[0]: "doesn't matter", channels[1]: "matters even less"}
            )
        )
        FixedIntermediateFrequencyValidation(model).run(builder, res_mgr)


class TestReadoutValidation:
    hw = EchoModelLoader(qubit_count=4).load()

    def _mock_1Q_live_hw_model(self, hw_model):
        """Mock a 1Q live hardware model for testing."""
        hw = LiveHardwareModel()
        qubit = self.hw.qubits[0]
        resonator = self.hw.resonators[0]
        hw.add_device(qubit)
        hw.add_physical_channel(qubit.physical_channel)
        hw.add_device(resonator)
        hw.add_physical_channel(resonator.physical_channel)
        return hw

    def test_valid_readout(self):
        builder = self.hw.create_builder()
        for qubit in self.hw.qubits:
            builder.measure_single_shot_z(target=qubit)

        ReadoutValidation(self.hw).run(builder)

    @pytest.mark.parametrize("no_mid_circuit_measurement", [True, False, None])
    def test_mid_circuit_measurement(self, no_mid_circuit_measurement):
        # ReadoutValidation pass only checks `LiveHardwareModel`s.
        hw = self._mock_1Q_live_hw_model(self.hw)

        builder = hw.create_builder()
        for qubit in hw.qubits:
            builder.measure_single_shot_z(target=qubit, axis=ProcessAxis.TIME)
        builder.X(target=qubit)

        with (
            pytest.raises(
                ValueError, match="Mid-circuit measurements currently unable to be used."
            )
            if (no_mid_circuit_measurement or no_mid_circuit_measurement is None)
            else nullcontext()
        ):
            ReadoutValidation(
                hw, no_mid_circuit_measurement=no_mid_circuit_measurement
            ).run(builder)

    @pytest.mark.parametrize(
        "acquire_mode, process_axis",
        [
            (AcquireMode.SCOPE, ProcessAxis.SEQUENCE),
            (AcquireMode.INTEGRATOR, ProcessAxis.TIME),
            (AcquireMode.RAW, None),
        ],
    )
    def test_acquire_with_invalid_pp(self, acquire_mode, process_axis):
        # ReadoutValidation pass only checks `LiveHardwareModel`s.
        hw = self._mock_1Q_live_hw_model(self.hw)
        qubit = hw.qubits[0]

        builder = hw.create_builder()
        builder.acquire(qubit.get_drive_channel(), mode=acquire_mode)
        builder.post_processing(
            builder.instructions[0], PostProcessType.MEAN, process_axis, qubit
        )

        with pytest.raises(ValueError, match="Invalid"):
            ReadoutValidation(hw).run(builder)
