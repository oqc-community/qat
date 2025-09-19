# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.backend.waveform_v1.executable import WaveformV1Executable
from qat.executables import BatchedExecutable
from qat.model.loaders.purr.echo import EchoModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.purr.sweep.compile import CompileSweepPipeline
from qat.pipelines.purr.waveform_v1 import EchoExecutePipeline, WaveformV1CompilePipeline

from .utils import (
    sweep_pulse_scales,
    sweep_pulse_widths,
    sweep_pulse_widths_and_amps,
    sweep_sequential_pulse_widths,
    sweep_zipped_parameters,
)


class TestCompileSweepPipeline:
    @pytest.fixture
    def model(self):
        return EchoModelLoader(qubit_count=4).load()

    @pytest.fixture
    def base_pipeline(self, model):
        return WaveformV1CompilePipeline(
            config=dict(name="test"), model=model, target_data=TargetData.default()
        )

    @pytest.fixture
    def pipeline(self, base_pipeline):
        return CompileSweepPipeline(base_pipeline=base_pipeline)

    def test_init_with_non_updateable_pipeline_raises(self, base_pipeline):
        with pytest.raises(
            TypeError, match="The base pipeline must be an UpdateablePipeline"
        ):
            CompileSweepPipeline(base_pipeline=base_pipeline.pipeline)

    def test_init_with_execute_pipeline_raises(self, model):
        base_pipeline = EchoExecutePipeline(
            config=dict(name="test"), model=model, target_data=TargetData.default()
        )
        with pytest.raises(
            TypeError, match="CompileSweepPipeline can only wrap CompilePipelines."
        ):
            CompileSweepPipeline(base_pipeline=base_pipeline)

    def test_compile_rejects_non_builders(self, pipeline):
        with pytest.raises(
            TypeError,
            match="SweepCompilePipeline expects to see an InstructionBuilder with sweeps.",
        ):
            pipeline.compile(program="not a builder")

    def test_single_sweep(self, pipeline, model):
        """Tests the sweeping over a pulse duration. Checks the buffers to make sure the
        pulse is active at the expected times."""
        times = np.linspace(80e-9, 800e-9, 10)
        builder = sweep_pulse_widths(model, qubit=0, times=times)
        executables, _ = pipeline.compile(builder)
        assert isinstance(executables, BatchedExecutable)
        assert len(executables.executables) == len(times)

        physical_channel = model.get_qubit(0).physical_channel
        for i, time in enumerate(times):
            executable = executables.executables[i].executable
            params = executables.executables[i].parameters
            assert isinstance(executable, WaveformV1Executable)
            assert params["t"] == time

            # test the buffers directly to check the timing is correct
            number_of_samples = int(np.round(time / physical_channel.sample_time))
            waveforms = executable.channel_data[physical_channel.id].buffer
            assert np.all(np.abs(waveforms[0:number_of_samples]) > 0.0)
            assert np.allclose(np.abs(waveforms[number_of_samples:]), 0.0)

    def test_single_sweep_with_device_assigns(self, pipeline, model):
        """Tests sweeping over a pulse amplitude scale, using a device assign to set the
        scale on the drive channel. Checks the buffers to make sure the amplitude scales
        correctly, and that the original scale is restored after compiling."""

        original_scale = model.get_qubit(0).get_drive_channel().scale
        scales = np.linspace(0.5, 1.5, 10)
        builder = sweep_pulse_scales(model, qubit=0, scales=scales)
        executables, _ = pipeline.compile(builder)
        assert isinstance(executables, BatchedExecutable)
        assert len(executables.executables) == len(scales)

        physical_channel = model.get_qubit(0).physical_channel
        max_amps = []
        for i, scale in enumerate(scales):
            executable = executables.executables[i].executable
            params = executables.executables[i].parameters
            assert isinstance(executable, WaveformV1Executable)
            assert params["s"] == scale

            # test the buffers directly to check the amplitude scales correctly
            waveforms = executable.channel_data[physical_channel.id].buffer
            max_amps.append(np.max(np.abs(waveforms)))

        scale = np.asarray(max_amps) / scales
        assert np.allclose(scale, scale[0])
        assert model.get_qubit(0).get_drive_channel().scale == original_scale

    def test_multiple_sweeps(self, pipeline, model):
        """Tests sweeping over both a pulse width and amplitude WITHOUT a device assign.
        Checks the buffers to make sure the timing of the pulse is correct, and the
        amplitude scales correctly."""
        times = np.linspace(80e-9, 800e-9, 5)
        amps = np.linspace(0.5, 1.5, 5)
        builder = sweep_pulse_widths_and_amps(model, qubit=0, times=times, amps=amps)
        executables, _ = pipeline.compile(builder)
        assert isinstance(executables, BatchedExecutable)
        assert len(executables.executables) == len(times) * len(amps)

        physical_channel = model.get_qubit(0).physical_channel
        max_amps = np.zeros((len(times), len(amps)))
        for i, time in enumerate(times):
            for j, amp in enumerate(amps):
                index = i * len(amps) + j
                executable = executables.executables[index].executable
                params = executables.executables[index].parameters
                assert isinstance(executable, WaveformV1Executable)
                assert params["t"] == time
                assert params["a"] == amp

                # test the buffers directly to check the timing is correct
                number_of_samples = int(np.round(time / physical_channel.sample_time))
                waveforms = executable.channel_data[physical_channel.id].buffer
                assert np.all(np.abs(waveforms[0:number_of_samples]) > 0.0)
                assert np.allclose(np.abs(waveforms[number_of_samples:]), 0.0)
                max_amps[i, j] = np.max(np.abs(waveforms))

        for i in range(len(times)):
            scale = max_amps[i, :] / np.linspace(0.5, 1.5, 5)
            assert np.allclose(scale, scale[0])

    def test_one_sweep_on_multiple_instructions(self, pipeline, model):
        """Tests sweeping the duration of a pulse, immediately followed by a second pulse on
        a different qubit. The timing of the second pulse is implemented by the dynamic
        timing of a delay."""
        times = np.linspace(80e-9, 800e-9, 10)
        builder = sweep_sequential_pulse_widths(model, qubit1=0, qubit2=1, times=times)
        executables, _ = pipeline.compile(builder)
        assert isinstance(executables, BatchedExecutable)
        assert len(executables.executables) == len(times)

        physical_channel1 = model.get_qubit(0).physical_channel
        physical_channel2 = model.get_qubit(1).physical_channel
        time_pulse2 = model.get_qubit(1).pulse_hw_x_pi_2["width"]
        number_of_samples2 = int(np.round(time_pulse2 / physical_channel2.sample_time))
        for i, time in enumerate(times):
            executable = executables.executables[i].executable
            params = executables.executables[i].parameters
            assert isinstance(executable, WaveformV1Executable)
            assert params["t"] == time

            # test the buffers directly to check the timing is correct
            number_of_samples1 = int(np.round(time / physical_channel1.sample_time))

            waveforms1 = executable.channel_data[physical_channel1.id].buffer
            assert np.all(np.abs(waveforms1[0:number_of_samples1]) > 0.0)
            assert np.allclose(np.abs(waveforms1[number_of_samples1:]), 0.0)

            waveforms2 = executable.channel_data[physical_channel2.id].buffer
            assert np.allclose(np.abs(waveforms2[0:number_of_samples1]), 0.0)
            assert np.all(
                np.abs(
                    waveforms2[number_of_samples1 : number_of_samples1 + number_of_samples2]
                )
                > 0.0
            )
            assert np.allclose(
                np.abs(waveforms2[number_of_samples1 + number_of_samples2 :]), 0.0
            )

    def test_zipped_parameters(self, pipeline, model):
        """Tests sweeping over two parameters that are zipped together, so that each
        flattened builder only has one of each parameter."""
        times = np.linspace(80e-9, 800e-9, 10)
        builder = sweep_zipped_parameters(model, 0, times, 2 * times)
        executables, _ = pipeline.compile(builder)
        assert isinstance(executables, BatchedExecutable)
        assert len(executables.executables) == len(times)

        physical_channel = model.get_qubit(0).physical_channel
        readout_channel = model.get_qubit(0).measure_device.physical_channel
        for i, time in enumerate(times):
            executable = executables.executables[i].executable
            params = executables.executables[i].parameters
            assert isinstance(executable, WaveformV1Executable)
            assert params["t1"] == time
            assert params["t2"] == 2 * time

            # test the buffers directly to check the timing is correct
            number_of_samples = int(np.round(time / physical_channel.sample_time))
            waveforms = executable.channel_data[physical_channel.id].buffer
            assert np.all(np.abs(waveforms[0:number_of_samples]) > 0.0)
            assert np.allclose(np.abs(waveforms[number_of_samples:]), 0.0)

            readout_waveforms = executable.channel_data[readout_channel.id].buffer
            number_of_readout_samples = int(
                np.round(3 * time / readout_channel.sample_time)
            )
            assert np.allclose(np.abs(readout_waveforms[0:number_of_readout_samples]), 0.0)
            assert np.abs(readout_waveforms[number_of_readout_samples]) > 0.0
