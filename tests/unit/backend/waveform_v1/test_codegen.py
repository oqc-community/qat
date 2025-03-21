# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

import pytest

from qat.backend.waveform_v1.codegen import WaveformV1Backend, WaveformV1Executable
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.instructions import CrossResonancePulse


class TestWaveformV1Emitter:

    def test_frequency_shift_raises_value_error_when_outside_allowed_range(self):
        # Set the maximum frequency so that we can trigger the error
        model = get_default_echo_hardware()
        pc = next(iter(model.pulse_channels.values()))
        pc.physical_channel.pulse_channel_max_frequency = pc.frequency + 1e8
        builder = model.create_builder()
        builder.frequency_shift(pc, 2e8)
        emitter = WaveformV1Backend(model)
        with pytest.raises(ValueError):
            emitter.emit(builder)

    def test_frequency_shift_raises_value_error_when_on_fixed_IF(self):
        # Fix the IR on a channel
        model = get_default_echo_hardware()
        pc = next(iter(model.pulse_channels.values()))
        pc.fixed_if = True
        builder = model.create_builder()
        builder.frequency_shift(pc, 1e8)
        emitter = WaveformV1Backend(model)
        with pytest.raises(NotImplementedError):
            emitter.emit(builder)

    def test_if_pass_raises_value_error(self):
        # Fix the IR on two pulse channels that share a physical channel, with diff freqs
        model = get_default_echo_hardware()
        physical_channel = next(iter(model.physical_channels.values()))
        pulse_channels = iter(
            model.get_pulse_channels_from_physical_channel(physical_channel)
        )
        pulse_channels = [next(pulse_channels) for _ in range(2)]
        pulse_channels[0].fixed_if = True
        pulse_channels[1].fixed_if = True
        pulse_channels[0].frequency = 1e8
        pulse_channels[1].frequency = 2e8
        builder = model.create_builder()
        builder.X(pulse_channels[0])

        # this should work as the second channel has no action
        emitter = WaveformV1Backend(model)
        emitter.emit(builder)

        # now add an instruction to the second channel, and check it raises an error...
        builder.add(
            CrossResonancePulse(
                pulse_channels[1], **model.get_qubit(0).pulse_hw_zx_pi_4["Q1"]
            )
        )
        emitter = WaveformV1Backend(model)
        with pytest.raises(ValueError):
            emitter.emit(builder)

    def test_repeats_when_given(self):
        model = get_default_echo_hardware()
        builder = model.create_builder()
        builder.repeat(512, 1e-4)
        emitter = WaveformV1Backend(model)
        executable = emitter.emit(builder)
        assert executable.shots == 512
        assert executable.repetition_time == 1e-4

    def test_repeats_when_not_given(self):
        model = get_default_echo_hardware()
        builder = model.create_builder()
        emitter = WaveformV1Backend(model)
        executable = emitter.emit(builder)
        assert executable.shots == model.default_repeat_count
        assert executable.repetition_time == model.default_repetition_period


class TestWaveformV1Executable:

    def test_same_after_serialize_deserialize_roundtrip(self):
        model = get_default_echo_hardware(10)
        builder = model.create_builder()
        builder.had(model.get_qubit(0))
        for i in range(9):
            builder.cnot(model.get_qubit(i), model.get_qubit(i + 1))

        executable = WaveformV1Backend(model).emit(builder)
        blob = executable.serialize()
        new_executable = WaveformV1Executable.deserialize(blob)
        assert executable == new_executable
