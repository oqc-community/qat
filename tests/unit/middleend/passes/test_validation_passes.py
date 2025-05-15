# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from copy import deepcopy

import numpy as np
import pytest
from compiler_config.config import (
    CompilerConfig,
    ErrorMitigationConfig,
    QuantumResultsFormat,
)

from qat import qatconfig
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.middleend.passes.legacy.validation import PhysicalChannelAmplitudeValidation
from qat.middleend.passes.transform import EvaluatePulses
from qat.middleend.passes.validation import (
    FrequencyValidation,
    PydHardwareConfigValidity,
    PydNoMidCircuitMeasurementValidation,
)
from qat.model.error_mitigation import ErrorMitigation, ReadoutMitigation
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.compiler.instructions import CustomPulse, Pulse, PulseShapeType
from qat.utils.hardware_model import generate_hw_model, generate_random_linear

from tests.unit.utils.pulses import pulse_attributes


class TestNoMidCircuitMeasurementValidation:
    hw = generate_hw_model(32)

    def test_no_mid_circuit_meas_found(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.measure_single_shot_z(target=qubit)

        p = PydNoMidCircuitMeasurementValidation(self.hw)
        builder_before = deepcopy(builder)
        p.run(builder)

        assert builder_before.number_of_instructions == builder.number_of_instructions
        for instr_before, instr_after in zip(builder_before, builder):
            assert instr_before == instr_after

    def test_throw_error_mid_circuit_meas(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.measure_single_shot_z(target=qubit)

        builder.X(target=self.hw.qubit_with_index(0))

        p = PydNoMidCircuitMeasurementValidation(self.hw)
        with pytest.raises(ValueError):
            p.run(builder)


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


class TestFrequencyValidation:
    res_mgr = ResultManager()
    model = EchoModelLoader().load()
    target_data = TargetData.default()

    def get_single_pulse_channel(self):
        return next(iter(self.model.pulse_channels.values()))

    def get_two_pulse_channels_from_single_physical_channel(self):
        physical_channel = next(iter(self.model.physical_channels.values()))
        pulse_channels = iter(
            self.model.get_pulse_channels_from_physical_channel(physical_channel)
        )
        return next(pulse_channels), next(pulse_channels)

    def get_two_pulse_channels_from_different_physical_channels(self):
        physical_channels = iter(self.model.physical_channels.values())
        pulse_channel_1 = next(
            iter(
                self.model.get_pulse_channels_from_physical_channel(next(physical_channels))
            )
        )
        pulse_channel_2 = next(
            iter(
                self.model.get_pulse_channels_from_physical_channel(next(physical_channels))
            )
        )
        return pulse_channel_1, pulse_channel_2

    def set_frequency_range(self, target_data, pulse_channel, lower_tol, upper_tol):
        return target_data.model_copy(
            update={
                "QUBIT_DATA": target_data.QUBIT_DATA.model_copy(
                    update={
                        "pulse_channel_lo_freq_max": pulse_channel.frequency + upper_tol,
                        "pulse_channel_lo_freq_min": pulse_channel.frequency - lower_tol,
                    }
                ),
                "RESONATOR_DATA": target_data.RESONATOR_DATA.model_copy(
                    update={
                        "pulse_channel_lo_freq_max": pulse_channel.frequency + upper_tol,
                        "pulse_channel_lo_freq_min": pulse_channel.frequency - lower_tol,
                    }
                ),
            }
        )

    @pytest.mark.parametrize("freq", [-1e-9, -1e-8, 0, 1e8, 1e9])
    def test_raises_no_error_when_freq_shift_in_range(self, freq):
        channel = self.get_single_pulse_channel()
        target_data = self.set_frequency_range(self.target_data, channel, 1e9, 1e9)
        builder = self.model.create_builder()
        builder.frequency_shift(channel, freq)
        FrequencyValidation(self.model, target_data).run(builder, self.res_mgr)

    @pytest.mark.parametrize("freq", [-1e9, 1e9])
    def test_raises_value_error_when_freq_shift_out_of_range(self, freq):
        channel = self.get_single_pulse_channel()
        target_data = self.set_frequency_range(self.target_data, channel, 1e8, 1e8)
        builder = self.model.create_builder()
        builder.frequency_shift(channel, freq)
        with pytest.raises(ValueError):
            FrequencyValidation(self.model, target_data).run(builder, self.res_mgr)

    @pytest.mark.parametrize("freq", [-2e8, 2e8])
    def test_moves_out_and_in_raises_value_error(self, freq):
        channel = self.get_single_pulse_channel()
        target_data = self.set_frequency_range(self.target_data, channel, 1e8, 1e8)
        builder = self.model.create_builder()
        builder.frequency_shift(channel, freq)
        builder.frequency_shift(channel, -freq)
        with pytest.raises(ValueError):
            FrequencyValidation(self.model, target_data).run(builder, self.res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_no_value_error_for_two_channels_same_physical_channel(self, sign):
        channels = self.get_two_pulse_channels_from_single_physical_channel()
        target_data = self.set_frequency_range(self.target_data, channels[0], 1e9, 1e9)
        builder = self.model.create_builder()

        # interweave with random instructions
        builder.phase_shift(channels[0], np.pi)
        builder.frequency_shift(channels[0], sign * 1e8)
        builder.phase_shift(channels[1], -np.pi)
        builder.frequency_shift(channels[1], sign * 2e8)
        builder.frequency_shift(channels[0], sign * 3e8)
        builder.phase_shift(channels[1], -2.54)
        builder.frequency_shift(channels[1], sign * 4e8)
        FrequencyValidation(self.model, target_data).run(builder, self.res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_value_error_for_two_channels_same_physical_channel(self, sign):
        channels = self.get_two_pulse_channels_from_single_physical_channel()
        target_data = self.set_frequency_range(self.target_data, channels[0], 1e9, 1e9)
        builder = self.model.create_builder()

        # interweave with random instructions
        builder.phase_shift(channels[0], np.pi)
        builder.frequency_shift(channels[0], sign * 5e8)
        builder.phase_shift(channels[1], -np.pi)
        builder.frequency_shift(channels[1], sign * 6e8)
        builder.frequency_shift(channels[0], sign * 1e8)
        builder.phase_shift(channels[1], -2.54)
        builder.frequency_shift(channels[1], sign * 5e8)
        with pytest.raises(ValueError):
            FrequencyValidation(self.model, target_data).run(builder, self.res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_value_error_for_two_channels_different_physical_channel(self, sign):
        channels = self.get_two_pulse_channels_from_different_physical_channels()
        target_data = self.set_frequency_range(self.target_data, channels[0], 1e9, 1e9)
        target_data = self.set_frequency_range(self.target_data, channels[1], 5e8, 5e8)
        builder = self.model.create_builder()
        builder.frequency_shift(channels[0], sign * 4e8)
        builder.frequency_shift(channels[1], sign * 1e8)
        builder.frequency_shift(channels[0], sign * 7e8)
        builder.frequency_shift(channels[1], sign * 4e8)
        with pytest.raises(ValueError):
            FrequencyValidation(self.model, target_data).run(builder, self.res_mgr)

    def test_fixed_if_raises_not_implemented_error(self):
        channels = self.get_two_pulse_channels_from_different_physical_channels()
        builder = self.model.create_builder()
        channels[0].fixed_if = True
        builder.frequency_shift(channels[0], 1e8)
        builder.frequency_shift(channels[1], 1e8)
        with pytest.raises(NotImplementedError):
            FrequencyValidation(self.model, self.target_data).run(builder, self.res_mgr)

    def test_fixed_if_does_not_affect_other_channel(self):
        channels = self.get_two_pulse_channels_from_different_physical_channels()
        builder = self.model.create_builder()
        channels[0].fixed_if = True
        builder.frequency_shift(channels[1], 1e8)
        FrequencyValidation(self.model, self.target_data).run(builder, self.res_mgr)


class TestPydHardwareConfigValidity:
    def test_max_shot_limit_exceeded(self):
        hw_model = generate_hw_model(n_qubits=8)
        comp_config = CompilerConfig(repeats=qatconfig.MAX_REPEATS_LIMIT + 1)
        ir = "test"
        res_mgr = ResultManager()

        with pytest.raises(ValueError):
            PydHardwareConfigValidity(hw_model).run(
                ir, res_mgr, compiler_config=comp_config
            )

        comp_config = CompilerConfig(repeats=qatconfig.MAX_REPEATS_LIMIT)
        PydHardwareConfigValidity(hw_model).run(ir, res_mgr, compiler_config=comp_config)

    @pytest.mark.parametrize("n_qubits", [2, 4, 8, 32, 64])
    def test_error_mitigation(self, n_qubits):
        hw_model = generate_hw_model(n_qubits=n_qubits)
        comp_config = CompilerConfig(
            repeats=10,
            error_mitigation=ErrorMitigationConfig.LinearMitigation,
            results_format=QuantumResultsFormat().binary_count(),
        )
        ir = "test"
        res_mgr = ResultManager()

        # Error mitigation not enabled in hw model.
        with pytest.raises(ValueError):
            PydHardwareConfigValidity(hw_model).run(
                ir, res_mgr, compiler_config=comp_config
            )

        qubit_indices = list(hw_model.qubits.keys())
        linear = generate_random_linear(qubit_indices)
        readout_mit = ReadoutMitigation(linear=linear)
        hw_model.error_mitigation = ErrorMitigation(readout_mitigation=readout_mit)
        comp_config = CompilerConfig(
            repeats=10,
            error_mitigation=ErrorMitigationConfig.LinearMitigation,
            results_format=QuantumResultsFormat().binary(),
        )

        # Error mitigation only works with binary count as results format.
        with pytest.raises(ValueError):
            PydHardwareConfigValidity(hw_model).run(
                ir, res_mgr, compiler_config=comp_config
            )

        comp_config = CompilerConfig(
            repeats=10,
            error_mitigation=ErrorMitigationConfig.LinearMitigation,
            results_format=QuantumResultsFormat().binary_count(),
        )
        PydHardwareConfigValidity(hw_model).run(ir, res_mgr, compiler_config=comp_config)
