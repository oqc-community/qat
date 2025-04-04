# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from copy import deepcopy

import numpy as np
import pytest

from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.middleend.passes.legacy.validation import PhysicalChannelAmplitudeValidation
from qat.middleend.passes.validation import PydNoMidCircuitMeasurementValidation
from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.compiler.instructions import CustomPulse, Pulse, PulseShapeType
from qat.utils.hardware_model import generate_hw_model


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
