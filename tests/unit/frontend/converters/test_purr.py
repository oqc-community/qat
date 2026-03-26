# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import numpy as np
import pytest
from compiler_config.config import InlineResultsProcessing

import qat.ir.instructions as Instructions
import qat.ir.measure as MeasureInstructions
import qat.ir.waveforms as Waveforms
import qat.purr.compiler.instructions as PurrInstructions
from qat.frontend.converters.purr import HardwareModelMapper, PurrConverter
from qat.ir.instruction_basetypes import AcquireMode, PostProcessType, ProcessAxis
from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.model.loaders.purr import EchoModelLoader
from qat.model.validators import MismatchingHardwareModelException
from qat.purr.compiler.instructions import PulseShapeType


class TestHardwareModelMapper:
    model = EchoModelLoader(qubit_count=3).load()
    converted_model = convert_purr_echo_hw_to_pydantic(model)

    def test_model_passes_validation(self):
        """Basic smoke test to ensure that converted models pass validation."""
        HardwareModelMapper(self.model, self.converted_model).validate_physical_properties()

    def test_mismatching_qubit_number_fails_validation(self):
        converted_model = convert_purr_echo_hw_to_pydantic(
            EchoModelLoader(qubit_count=4).load()
        )
        with pytest.raises(MismatchingHardwareModelException, match="number of qubits"):
            HardwareModelMapper(self.model, converted_model).validate_physical_properties()

    def test_mismatching_qubit_indices_fails_validation(self):
        jagged_model = EchoModelLoader(qubit_count=3).load()
        jagged_model.qubits[-1].index = 6
        converted_model = convert_purr_echo_hw_to_pydantic(self.model)
        with pytest.raises(MismatchingHardwareModelException, match="qubit indices"):
            HardwareModelMapper(
                jagged_model, converted_model
            ).validate_physical_properties()

    def test_mismatching_couplings_fails_validation(self):
        broken_model = EchoModelLoader(qubit_count=3).load()
        del broken_model.qubit_direction_couplings[-1]
        with pytest.raises(MismatchingHardwareModelException, match="qubit couplings"):
            HardwareModelMapper(
                broken_model, self.converted_model
            ).validate_physical_properties()

    def test_physical_channel_with_mismatching_acquires_allowed_fails_validation(self):
        broken_model = EchoModelLoader(qubit_count=3).load()
        broken_model.qubits[-1].measure_device.physical_channel.acquire_allowed = False
        with pytest.raises(
            MismatchingHardwareModelException, match="acquire allowed property"
        ):
            HardwareModelMapper(
                broken_model, self.converted_model
            ).validate_physical_properties()

    def test_physical_channel_with_mismatching_frequencies_fails_validation(self):
        broken_model = EchoModelLoader(qubit_count=3).load()
        broken_model.qubits[-1].measure_device.physical_channel.baseband.frequency *= 1.05
        with pytest.raises(MismatchingHardwareModelException, match="baseband frequency"):
            HardwareModelMapper(
                broken_model, self.converted_model
            ).validate_physical_properties()

    def test_get_pulse_channel_id(self):
        mapper = HardwareModelMapper(self.model, self.converted_model)

        for qubit in self.model.qubits:
            drive_channel = qubit.get_drive_channel()
            drive_channel_id = self.converted_model.qubit_with_index(
                qubit.index
            ).drive_pulse_channel.uuid
            assert mapper.get_pulse_channel_id(drive_channel) == drive_channel_id

    def test_cross_resonance_channels_get_mapped_to_same_id(self):
        mapper = HardwareModelMapper(self.model, self.converted_model)

        for coupling in self.model.qubit_direction_couplings:
            control_index = coupling.direction[0]
            target_index = coupling.direction[1]
            control_qubit = self.model.get_qubit(control_index)
            target_qubit = self.model.get_qubit(target_index)
            target_cross_resonance_channel = target_qubit.get_cross_resonance_channel(
                control_qubit
            )
            expected_id = (
                self.converted_model.qubit_with_index(target_index)
                .cross_resonance_pulse_channels[control_index]
                .uuid
            )
            cross_resonance_channel_id = mapper.get_pulse_channel_id(
                target_cross_resonance_channel
            )

            assert expected_id == cross_resonance_channel_id

    def test_get_custom_pulse_channel_id(self):
        mapper = HardwareModelMapper(self.model, self.converted_model)

        new_channel = self.model.get_qubit(0).physical_channel.create_pulse_channel(
            "my_channel"
        )
        id_ = mapper.get_pulse_channel_id(new_channel)
        assert id_ == "my_channel"

    def test_pulse_channel_mapping_exists(self):
        mapper = HardwareModelMapper(self.model, self.converted_model)

        drive_channel = self.model.qubits[0].get_drive_channel()
        assert mapper.pulse_channel_mapping_exists(drive_channel) is False
        mapper.get_pulse_channel_id(drive_channel)
        assert mapper.pulse_channel_mapping_exists(drive_channel) is True

    def test_physical_channel_mapping(self):
        mapper = HardwareModelMapper(self.model, self.converted_model)

        for qubit in self.model.qubits:
            pyd_qubit = self.converted_model.qubit_with_index(qubit.index)
            for physical_channel, pyd_physical_channel in [
                (qubit.physical_channel, pyd_qubit.physical_channel),
                (
                    qubit.measure_device.physical_channel,
                    pyd_qubit.resonator.physical_channel,
                ),
            ]:
                assert (
                    mapper.physical_channel_map[physical_channel.full_id()]
                    == pyd_physical_channel.uuid
                )


class TestPurrConverter:
    hw_model = EchoModelLoader(qubit_count=3).load()
    converted_model = convert_purr_echo_hw_to_pydantic(hw_model)
    drive_channel = hw_model.qubits[0].get_drive_channel()
    drive_channel_two = hw_model.qubits[1].get_drive_channel()
    drive_channel_id = converted_model.qubit_with_index(0).drive_pulse_channel.uuid
    drive_channel_two_id = converted_model.qubit_with_index(1).drive_pulse_channel.uuid

    @pytest.mark.parametrize(
        "purr_instruction, pyd_instruction",
        [
            (PurrInstructions.Repeat(2000), Instructions.Repeat(repeat_count=2000)),
            (
                PurrInstructions.Assign("test", [1, 2, 3]),
                Instructions.Assign(name="test", value=[1, 2, 3]),
            ),
            (
                PurrInstructions.PostProcessing(
                    PurrInstructions.Acquire(
                        drive_channel,
                        time=1e-6,
                        mode=AcquireMode.INTEGRATOR,
                        output_variable="test",
                    ),
                    process=PostProcessType.MEAN,
                    axes=[ProcessAxis.TIME],
                    args=[0.5],
                ),
                MeasureInstructions.PostProcessing(
                    output_variable="test",
                    process_type=PostProcessType.MEAN,
                    axes=[ProcessAxis.TIME],
                    args=[0.5],
                ),
            ),
            (
                PurrInstructions.ResultsProcessing("test", InlineResultsProcessing.Program),
                Instructions.ResultsProcessing(
                    variable="test", results_processing=InlineResultsProcessing.Program
                ),
            ),
            (
                PurrInstructions.Return(["test1", "test2"]),
                Instructions.Return(variables=["test1", "test2"]),
            ),
            (
                PurrInstructions.Variable("test", int, 1),
                Instructions.Variable(name="test", var_type=int, value=1),
            ),
            (PurrInstructions.Reset(drive_channel), Instructions.Reset(qubit_target=0)),
            (
                PurrInstructions.FrequencySet(drive_channel, 1.1e9),
                Instructions.FrequencySet(target=drive_channel_id, frequency=1.1e9),
            ),
            (
                PurrInstructions.FrequencyShift(drive_channel, 1.1e9),
                Instructions.FrequencyShift(target=drive_channel_id, frequency=1.1e9),
            ),
            (
                PurrInstructions.PhaseSet(drive_channel, 0.5),
                Instructions.PhaseSet(target=drive_channel_id, phase=0.5),
            ),
            (
                PurrInstructions.PhaseShift(drive_channel, 0.5),
                Instructions.PhaseShift(target=drive_channel_id, phase=0.5),
            ),
            (
                PurrInstructions.PhaseReset(drive_channel),
                Instructions.PhaseSet(target=drive_channel_id, phase=0.0),
            ),
            (
                PurrInstructions.Delay(drive_channel, 1e-6),
                Instructions.Delay(target=drive_channel_id, duration=1e-6),
            ),
            (
                PurrInstructions.Synchronize([drive_channel, drive_channel_two]),
                Instructions.Synchronize(targets=[drive_channel_id, drive_channel_two_id]),
            ),
            (
                PurrInstructions.Pulse(
                    drive_channel, shape=PulseShapeType.SQUARE, width=80e-9, amp=0.5
                ),
                Waveforms.Pulse(
                    target=drive_channel_id,
                    waveform=Waveforms.SquareWaveform(width=80e-9, amp=0.5),
                ),
            ),
            (
                PurrInstructions.CustomPulse(
                    drive_channel,
                    samples=np.linspace(0.1, 0.8, 8),
                ),
                Waveforms.Pulse(
                    target=drive_channel_id,
                    waveform=Waveforms.SampledWaveform(samples=np.linspace(0.1, 0.8, 8)),
                    duration=8e-9,
                ),
            ),
            (
                PurrInstructions.Acquire(
                    drive_channel,
                    time=80e-9,
                    mode=AcquireMode.INTEGRATOR,
                    output_variable="test",
                    filter=PurrInstructions.Pulse(
                        drive_channel, width=80e-9, shape=PulseShapeType.SQUARE, amp=0.5
                    ),
                    delay=0.0,
                ),
                MeasureInstructions.Acquire(
                    target=drive_channel_id,
                    duration=80e-9,
                    mode=AcquireMode.INTEGRATOR,
                    output_variable="test",
                    filter=Waveforms.Pulse(
                        target=drive_channel_id,
                        waveform=Waveforms.SquareWaveform(width=80e-9, amp=0.5),
                    ),
                ),
            ),
        ],
    )
    def test_instruction(self, purr_instruction, pyd_instruction):
        """Test that instructions are correctly parsed."""
        parser = PurrConverter(self.converted_model)

        builder = self.hw_model.create_builder()
        builder.add(purr_instruction)
        assert len(builder.instructions) == 1
        instructions = parser.convert(builder).instructions
        assert len(instructions) == 1
        assert instructions[0] == pyd_instruction

    @pytest.mark.parametrize("attr", ["repetition_period", "passive_reset_time"])
    def test_repeat_with_passive_reset_time_raises_warning(self, attr):
        parser = PurrConverter(self.converted_model)
        reset_instruction = PurrInstructions.Repeat(repeat_count=1000, **{attr: 1e-6})

        builder = self.hw_model.create_builder()
        builder.add(reset_instruction)
        with pytest.warns(
            UserWarning,
            match="Converting of Repeat instructions with repetition periods or passive",
        ):
            parser.convert(builder)

    def test_acquire_with_delay_is_composed_of_delay_and_acquire(self):
        parser = PurrConverter(self.converted_model)

        drive_channel = self.hw_model.qubits[0].get_drive_channel()
        drive_channel_id = self.converted_model.qubit_with_index(0).drive_pulse_channel.uuid
        acquire_instruction = PurrInstructions.Acquire(
            drive_channel,
            time=80e-9,
            mode=AcquireMode.INTEGRATOR,
            output_variable="test",
            delay=100e-9,
        )

        builder = self.hw_model.create_builder()
        builder.add(acquire_instruction)
        assert len(builder.instructions) == 1
        instructions = parser.convert(builder).instructions
        assert len(instructions) == 2
        assert instructions[0] == Instructions.Delay(
            target=drive_channel_id, duration=100e-9
        )
        assert instructions[1] == MeasureInstructions.Acquire(
            target=drive_channel_id,
            duration=80e-9,
            mode=AcquireMode.INTEGRATOR,
            output_variable="test",
            filter=None,
        )

    def test_instructions_with_multiple_targets_split_into_multiple(self):
        parser = PurrConverter(self.converted_model)

        drive_channel = self.hw_model.qubits[0].get_drive_channel()
        drive_channel_two = self.hw_model.qubits[1].get_drive_channel()
        drive_channel_id = self.converted_model.qubit_with_index(0).drive_pulse_channel.uuid
        drive_channel_two_id = self.converted_model.qubit_with_index(
            1
        ).drive_pulse_channel.uuid
        instr = PurrInstructions.PhaseShift([drive_channel, drive_channel_two], 1e-6)

        builder = self.hw_model.create_builder()
        builder.add(instr)
        assert len(builder.instructions) == 1
        instructions = parser.convert(builder).instructions
        assert len(instructions) == 2
        targets = set()
        for instruction in instructions:
            assert isinstance(instruction, Instructions.PhaseShift)
            targets.add(instruction.target)
        assert targets == {drive_channel_id, drive_channel_two_id}

    def test_composite_program(self):
        parser = PurrConverter(self.converted_model)

        drive_channel = self.hw_model.qubits[0].get_drive_channel()
        drive_channel_two = self.hw_model.qubits[1].get_drive_channel()
        acquire_channel = self.hw_model.qubits[1].get_acquire_channel()

        builder = self.hw_model.create_builder()
        builder.phase_shift(drive_channel, 0.5)
        builder.add(PurrInstructions.PhaseSet(drive_channel_two, np.pi))
        builder.pulse(drive_channel, shape=PulseShapeType.SQUARE, width=80e-9, amp=0.5)
        builder.synchronize([drive_channel, drive_channel_two, acquire_channel])
        builder.acquire(
            acquire_channel,
            time=80e-9,
            mode=AcquireMode.INTEGRATOR,
            output_variable="test",
            delay=0.0,
        )
        builder.post_processing(
            builder.instructions[-1], process=PostProcessType.MEAN, axes=[ProcessAxis.TIME]
        )

        new_builder = parser.convert(builder)

        drive_channel_id = self.converted_model.qubit_with_index(0).drive_pulse_channel.uuid
        drive_channel_two_id = self.converted_model.qubit_with_index(
            1
        ).drive_pulse_channel.uuid
        acquire_channel_id = self.converted_model.qubit_with_index(
            1
        ).acquire_pulse_channel.uuid

        new_drive_channel = new_builder.get_pulse_channel(drive_channel_id)
        new_drive_channel_two = new_builder.get_pulse_channel(drive_channel_two_id)
        new_acquire_channel = new_builder.get_pulse_channel(acquire_channel_id)

        for old_channel, new_channel in zip(
            [drive_channel, drive_channel_two, acquire_channel],
            [new_drive_channel, new_drive_channel_two, new_acquire_channel],
        ):
            assert old_channel.scale == new_channel.scale
            assert old_channel.phase_offset == new_channel.phase_iq_offset
            assert old_channel.imbalance == new_channel.imbalance
            assert old_channel.frequency == new_channel.frequency

        assert (
            new_drive_channel.physical_channel_id
            == self.converted_model.qubit_with_index(0).physical_channel.uuid
        )
        assert (
            new_drive_channel_two.physical_channel_id
            == self.converted_model.qubit_with_index(1).physical_channel.uuid
        )
        assert (
            new_acquire_channel.physical_channel_id
            == self.converted_model.qubit_with_index(1).resonator.physical_channel.uuid
        )

        assert len(new_builder.instructions) == 6
        assert new_builder.instructions[0] == Instructions.PhaseShift(
            target=drive_channel_id, phase=0.5
        )
        assert new_builder.instructions[1] == Instructions.PhaseSet(
            target=drive_channel_two_id, phase=np.pi
        )
        assert new_builder.instructions[2] == Waveforms.Pulse(
            target=drive_channel_id,
            waveform=Waveforms.SquareWaveform(width=80e-9, amp=0.5),
        )
        assert new_builder.instructions[3] == Instructions.Synchronize(
            targets=[
                drive_channel_id,
                drive_channel_two_id,
                acquire_channel_id,
            ]
        )
        assert new_builder.instructions[4] == MeasureInstructions.Acquire(
            target=acquire_channel_id,
            duration=80e-9,
            mode=AcquireMode.INTEGRATOR,
            output_variable="test",
            filter=None,
        )
        assert new_builder.instructions[5] == MeasureInstructions.PostProcessing(
            output_variable="test",
            process_type=PostProcessType.MEAN,
            axes=[ProcessAxis.TIME],
            args=[],
        )

    def test_updated_pulse_channel(self):
        qubit = self.hw_model.get_qubit(0)
        drive_channel = qubit.get_drive_channel()
        drive_channel.frequency *= 1.05

        model_drive_channel = self.converted_model.qubit_with_index(0).drive_pulse_channel
        assert np.isclose(drive_channel.frequency, model_drive_channel.frequency * 1.05)

        builder = self.hw_model.create_builder()
        builder.X(qubit)
        parser = PurrConverter(self.converted_model)
        new_builder = parser.convert(builder)

        new_drive_channel = new_builder.get_pulse_channel(self.drive_channel_id)
        assert np.isclose(new_drive_channel.frequency, model_drive_channel.frequency * 1.05)
        assert np.isclose(new_drive_channel.frequency, drive_channel.frequency)
        assert np.isclose(new_drive_channel.scale, drive_channel.scale)
        assert np.isclose(new_drive_channel.phase_iq_offset, drive_channel.phase_offset)

    def test_invalid_waveform_shape(self):
        parser = PurrConverter(self.converted_model)

        drive_channel = self.hw_model.qubits[0].get_drive_channel()
        builder = self.hw_model.create_builder()
        builder.add(
            PurrInstructions.Pulse(
                drive_channel, shape="invalid_shape", width=80e-9, amp=0.5
            )
        )
        with pytest.raises(ValueError, match="Unsupported waveform shape"):
            parser.convert(builder)

    def test_invalid_target(self):
        parser = PurrConverter(self.converted_model)

        builder = self.hw_model.create_builder()
        builder.phase_shift(self.drive_channel, 0.5)
        builder.instructions[-1].quantum_targets = ["test"]
        with pytest.raises(ValueError, match="Expected target to be a PulseChannel"):
            parser.convert(builder)
