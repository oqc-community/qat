# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from copy import deepcopy

import numpy as np
import pytest
from compiler_config.config import InlineResultsProcessing

from qat.backend.passes.purr.analysis import (
    IntermediateFrequencyResult,
    PulseChannelTimeline,
    TimelineAnalysisResult,
)
from qat.backend.waveform_v1.executable import WaveformV1Executable
from qat.backend.waveform_v1.purr.codegen import (
    WaveformContext,
    WaveformV1Backend,
)
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.executables import AcquireData
from qat.ir.lowered import PartitionedIR
from qat.middleend.passes.purr.transform import (
    LowerSyncsToDelays,
    RepeatTranslation,
)
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.instructions import (
    Acquire,
    Delay,
    PhaseSet,
    PhaseShift,
    PostProcessType,
    ProcessAxis,
    Pulse,
    PulseShapeType,
    ResultsProcessing,
    Return,
)

from tests.unit.utils.pulses import pulse_attributes


class TestWaveformV1Executable:
    def test_same_after_serialize_deserialize_roundtrip(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        builder.repeat(1000, 100e-6)
        builder.had(model.get_qubit(0))
        for i in range(9):
            builder.cnot(model.get_qubit(i), model.get_qubit(i + 1))
        # backend is not expected to see syncs, so lets remove them using this pass for
        # ease
        builder = LowerSyncsToDelays().run(builder)
        executable = WaveformV1Backend(model).emit(builder)
        blob = executable.serialize()
        new_executable = WaveformV1Executable.deserialize(blob)
        assert executable == new_executable


class TestWaveformV1Backend:
    """Tests for the WaveformV1Backend class.

    Includes:

    * Basic smoke tests to ensure the backend pipeline works as expected; note individual
      tests for passes are in the unit tests for the passes.
    * Checks codegen methods for creating buffers and acquires.
    * Checks the emit function outputs a valid executable.
    """

    model: EchoModelLoader = EchoModelLoader(10).load()
    channel = model.qubits[0].get_drive_channel()
    backend = WaveformV1Backend(model)

    @pytest.fixture(scope="class")
    def builder(self):
        drive_chan = self.model.qubits[0].get_drive_channel()
        cr_chan = self.model.qubits[0].get_cross_resonance_channel(self.model.qubits[1])
        measure_chan = self.model.qubits[0].get_measure_channel()
        acquire_chan = self.model.qubits[0].get_acquire_channel()
        builder: QuantumInstructionBuilder = self.model.create_builder()

        builder.repeat(1254, 100e-6)
        builder.add(PhaseSet(drive_chan, 0.0))
        builder.add(PhaseSet(measure_chan, 0.0))
        builder.add(PhaseSet(acquire_chan, 0.0))
        builder.add(PhaseSet(cr_chan, 0.0))
        builder.pulse(drive_chan, shape=PulseShapeType.SQUARE, width=80e-9)
        builder.delay(cr_chan, 80e-9)
        builder.pulse(cr_chan, shape=PulseShapeType.SQUARE, width=40e-9)
        builder.delay(drive_chan, 440e-9)
        builder.delay(cr_chan, 400e-9)
        builder.delay(measure_chan, 120e-9)
        builder.delay(acquire_chan, 160e-9)
        builder.add(PhaseSet(measure_chan, 0.0))
        builder.pulse(measure_chan, width=400e-9, shape=PulseShapeType.SQUARE)
        builder.acquire(acquire_chan, time=360e-9, delay=0.0, output_variable="test_var")
        builder.post_processing(
            builder.instructions[-1],
            process=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
            axes=[ProcessAxis.SEQUENCE],
            args=[0.254, -1.23455e2],
        )
        builder.results_processing("test_var", InlineResultsProcessing.Binary)
        builder.returns("test_var")
        return builder

    @pytest.fixture(scope="class")
    def res_mgr(self):
        return ResultManager()

    @pytest.fixture(scope="class")
    def partitioned_ir(self, builder, res_mgr):
        ir = self.backend.run_pass_pipeline(builder, res_mgr, MetricsManager())
        return ir

    @pytest.fixture(scope="class")
    def executable(self, builder):
        return self.backend.emit(builder)

    @pytest.mark.parametrize("passive_reset_time", [1e-06, None])
    @pytest.mark.parametrize("repeat_translation", [True, False])
    def test_repeat_handling(self, passive_reset_time, repeat_translation):
        builder = self.model.create_builder()
        builder.repetition_period = builder.model.default_repetition_period
        builder.passive_reset_time = None
        builder.repeat(1000, passive_reset_time=passive_reset_time)
        builder.had(self.model.get_qubit(0))
        for i in range(9):
            builder.cnot(self.model.get_qubit(i), self.model.get_qubit(i + 1))
        # backend is not expected to see syncs, so lets remove them using this pass for
        # ease
        builder = LowerSyncsToDelays().run(builder)
        if repeat_translation:
            builder = RepeatTranslation(TargetData.default()).run(builder)

        executable = self.backend.emit(builder)
        assert isinstance(executable, WaveformV1Executable)
        assert executable.shots == 1000
        assert executable.compiled_shots == 1000

    def test_pipeline_gives_partitioned_ir(self, partitioned_ir):
        assert isinstance(partitioned_ir, PartitionedIR)

    def test_partitioned_ir_shots(self, partitioned_ir):
        assert partitioned_ir.shots == 1254
        assert partitioned_ir.passive_reset_time == None

    def test_partitioned_ir_has_returns(self, partitioned_ir):
        assert len(partitioned_ir.returns) == 1
        assert isinstance(partitioned_ir.returns[0], Return)
        assert partitioned_ir.returns[0].variables == ["test_var"]

    def test_partitioned_ir_has_acquires(self, partitioned_ir):
        assert len(partitioned_ir.acquire_map) == 1
        assert len(next(iter(partitioned_ir.acquire_map.values()))) == 1
        acquire_inst = next(iter(partitioned_ir.acquire_map.values()))[0]
        assert acquire_inst.output_variable == "test_var"
        assert acquire_inst.time == 360e-9
        assert acquire_inst.delay == 0.0

    def test_partitioned_ir_has_post_processing(self, partitioned_ir):
        assert len(partitioned_ir.pp_map) == 1
        assert len(next(iter(partitioned_ir.pp_map.values()))) == 1
        pp_inst = next(iter(partitioned_ir.pp_map.values()))[0]
        assert pp_inst.process == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL
        assert pp_inst.axes == [ProcessAxis.SEQUENCE]
        assert pp_inst.args == [0.254, -123.455]

    def test_partitioned_ir_has_results_processing(self, partitioned_ir):
        assert len(partitioned_ir.rp_map) == 1
        assert "test_var" in partitioned_ir.rp_map
        results_processing = partitioned_ir.rp_map["test_var"]
        assert isinstance(results_processing, ResultsProcessing)
        assert results_processing.variable == "test_var"
        assert results_processing.results_processing == InlineResultsProcessing.Binary

    def test_results_manager_includes_timeline_analysis(self, res_mgr):
        result = res_mgr.lookup_by_type(TimelineAnalysisResult)
        assert isinstance(result, TimelineAnalysisResult)

    def test_results_manager_includes_if_analysis(self, res_mgr):
        result = res_mgr.lookup_by_type(IntermediateFrequencyResult)
        assert isinstance(result, IntermediateFrequencyResult)

    def test_create_pulse_channel_buffer(self):
        instructions = [
            PhaseSet(self.channel, np.pi / 4),
            Pulse(self.channel, width=80e-9, shape=PulseShapeType.SQUARE, amp=0.5),
            Delay(self.channel, 40e-9),
            PhaseShift(self.channel, -np.pi / 3),
            Pulse(self.channel, width=120e-9, shape=PulseShapeType.SQUARE, amp=2.0),
        ]
        buffer = self.backend.create_pulse_channel_buffer(
            self.channel, instructions, [0, 160, 80, 0, 240]
        )
        assert buffer.shape == (480,)
        assert np.allclose(buffer[0:160], 0.5 * np.exp(1j * np.pi / 4))
        assert np.allclose(buffer[160:240], 0.0)
        assert np.allclose(buffer[240:320], 2.0 * np.exp(-1j * np.pi / 12))

    def test_create_physical_channel_buffers(self, partitioned_ir, res_mgr):
        timeline_res = res_mgr.lookup_by_type(TimelineAnalysisResult)
        buffers = self.backend.create_physical_channel_buffers(partitioned_ir, timeline_res)
        max_time = 520e-9
        for physical_channel in self.model.physical_channels.values():
            assert physical_channel in buffers
            buffer = buffers[physical_channel]
            assert isinstance(buffer, np.ndarray)
            if physical_channel == self.model.qubits[0].physical_channel:
                assert buffer.size == int(np.round(max_time / physical_channel.sample_time))
                num_samples = int(np.round(120e-9 / physical_channel.sample_time))
                assert np.all(np.logical_not(np.isclose(buffer[0:num_samples], 0.0)))
                assert np.all(np.isclose(buffer[num_samples:], 0.0))
            elif physical_channel == self.model.qubits[0].measure_device.physical_channel:
                assert buffer.size == int(np.round(max_time / physical_channel.sample_time))
                num_samples_before = int(np.round(120e-9 / physical_channel.sample_time))
                assert np.all(np.isclose(buffer[0:num_samples_before], 0.0))
                assert np.all(np.logical_not(np.isclose(buffer[num_samples_before:], 0.0)))
            else:
                assert buffer.size == 0

    def test_create_acquire_dict(self):
        acquire_chan_1 = self.model.qubits[0].get_acquire_channel()
        acquire_chan_2 = self.model.qubits[1].get_acquire_channel()
        acquire_1 = Acquire(acquire_chan_1, 80e-9, delay=0.0, output_variable="test_var1")
        acquire_2 = Acquire(acquire_chan_2, 88e-9, delay=0.0, output_variable="test_var2")
        target_map = {
            acquire_chan_1: [Delay(acquire_chan_1, 40e-9), acquire_1],
            acquire_chan_2: [Delay(acquire_chan_1, 56e-9), acquire_2],
        }
        acquire_map = {acquire_chan_1: [acquire_1], acquire_chan_2: [acquire_2]}
        ir = PartitionedIR(target_map=target_map, acquire_map=acquire_map)
        timeline_res = TimelineAnalysisResult(
            target_map={
                acquire_chan_1: PulseChannelTimeline(samples=[10, 5]),
                acquire_chan_2: PulseChannelTimeline(samples=[11, 7]),
            }
        )
        acquire_dict = self.backend.create_acquire_dict(ir, timeline_res)
        assert len(acquire_dict) == 2
        assert acquire_chan_1.physical_channel in acquire_dict
        assert acquire_chan_2.physical_channel in acquire_dict
        assert acquire_dict[acquire_chan_1.physical_channel][0].length == 5
        assert acquire_dict[acquire_chan_2.physical_channel][0].length == 7
        assert acquire_dict[acquire_chan_1.physical_channel][0].position == 10
        assert acquire_dict[acquire_chan_2.physical_channel][0].position == 11

    def test_emit_gives_executable(self, executable):
        assert isinstance(executable, WaveformV1Executable)

    def test_executable_repeats(self, executable):
        assert executable.shots == 1254

    def test_executable_post_processing(self, executable):
        assert len(executable.post_processing) == 1
        assert "test_var" in executable.post_processing
        pp = executable.post_processing["test_var"]
        assert len(pp) == 1
        pp = pp[0]
        assert pp.process_type == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL
        assert pp.axes == [ProcessAxis.SEQUENCE]
        assert pp.args == [0.254, -123.455]

    def test_executable_results_processing(self, executable):
        assert len(executable.results_processing) == 1
        assert "test_var" in executable.results_processing
        rp = executable.results_processing["test_var"]
        assert rp == InlineResultsProcessing.Binary

    def test_executable_returns(self, executable):
        assert len(executable.returns) == 1
        assert "test_var" in executable.returns

    def test_executable_calibration_id(self, executable):
        assert executable.calibration_id == self.model.calibration_id

    def test_executable_channel_data(self, executable):
        # Both legacy and new backend creates data for every physical channel, even if
        # not used.
        assert len(executable.channel_data) == len(self.model.physical_channels)
        channels = [
            self.model.qubits[0].get_drive_channel().physical_channel.full_id(),
            self.model.qubits[0].get_measure_channel().physical_channel.full_id(),
        ]
        for channel in channels:
            assert len(executable.channel_data[channel].buffer) > 0

    def test_executable_acquires(self, executable):
        assert len(executable.acquires) == 1
        acquire = executable.acquires[0]
        assert isinstance(acquire, AcquireData)
        assert acquire.output_variable == "test_var"
        assert acquire.length == 360
        assert acquire.position == 160

    def test_batched_shots(self):
        builder = self.model.create_builder()
        builder.shots = 15000
        builder.compiled_shots = 1000
        builder.repetition_period = 100e-6
        builder.pulse(self.channel, width=80e-9, shape=PulseShapeType.SQUARE)
        executable = self.backend.emit(builder)
        assert executable.shots == 15000
        assert executable.compiled_shots == 1000


class TestWaveformContext:
    model = EchoModelLoader(2).load()
    channel = model.qubits[0].get_drive_channel()

    def test_delay(self):
        context = WaveformContext(self.channel, 100)
        assert context._duration == 0
        context.process_delay(10)
        assert context._duration == 10

    def test_phaseshift(self):
        context = WaveformContext(self.channel, 100)
        assert context._phase == 0.0
        context.process_phaseshift(2.54)
        assert context._phase == 2.54

    def test_phaseset(self):
        context = WaveformContext(self.channel, 100)
        context._phase = 0.5
        context.process_phaseset(2.54)
        assert context._phase == 2.54

    def test_phasereset(self):
        context = WaveformContext(self.channel, 100)
        context._phase = 0.5
        context.process_phasereset()
        assert context._phase == 0.0

    def test_frequencyshift(self):
        context = WaveformContext(self.channel, 100)
        freq_before = context._frequency
        context.process_frequencyshift(0.1 * freq_before)
        assert np.isclose(context._frequency, 1.1 * freq_before)

    def test_frequencyset(self):
        context = WaveformContext(self.channel, 100)
        freq_before = context._frequency
        new_freq = 0.75 * freq_before
        context.process_frequencyset(new_freq)
        assert np.isclose(context._frequency, new_freq)

    @pytest.mark.parametrize("fixed_if", [True, False])
    def test_upconvert(self, fixed_if):
        # Prepare the channel
        model = EchoModelLoader(2).load()
        channel = model.qubits[0].get_drive_channel()
        channel.fixed_if = fixed_if
        if fixed_if:
            channel.physical_channel.baseband.if_frequency = 0.1e9
        else:
            channel.frequency = channel.baseband_frequency + 0.1e9
        assert channel.phase_offset == 0.0
        assert channel.imbalance == 1.0

        times = np.linspace(0, 800e-9, 101)
        buffer = np.ones((101,), dtype=np.complex128)
        context = WaveformContext(channel, 101)
        new_buffer = context._do_upconvert(deepcopy(buffer), times)

        assert not np.allclose(buffer, new_buffer)
        assert np.allclose(np.abs(new_buffer), 1.0)

    @pytest.mark.parametrize("imbalance", [0.999, 1.001])
    def test_upconvert_with_imbalance(self, imbalance):
        """Tests that the imbalance works by checking key properties, such as normalization,
        is altered."""

        # Prepare the channel
        model = EchoModelLoader(2).load()
        channel = model.qubits[0].get_drive_channel()
        assert channel.fixed_if is False
        channel.frequency = channel.baseband_frequency + 0.1e9
        assert channel.phase_offset == 0.0
        times = np.linspace(0, 800e-9, 101)
        buffer = np.ones((101,), dtype=np.complex128)
        context = WaveformContext(channel, 101)

        # Do without an imbalance
        channel.imbalance = 1.0
        buffer_without_imbalance = context._do_upconvert(deepcopy(buffer), times)

        # Do with an imbalance
        channel.imbalance = imbalance
        buffer_with_imbalance = context._do_upconvert(deepcopy(buffer), times)

        assert not np.allclose(buffer_without_imbalance, buffer_with_imbalance)
        assert not np.allclose(np.abs(buffer_with_imbalance), 1.0)

    def test_upconvert_with_phase_offset(self):
        # Prepare the channel
        model = EchoModelLoader(2).load()
        channel = model.qubits[0].get_drive_channel()
        assert channel.fixed_if is False
        channel.frequency = channel.baseband_frequency + 0.1e9
        assert channel.imbalance == 1.0

        times = np.linspace(0, 800e-9, 101)
        buffer = np.ones((101,), dtype=np.complex128)
        context = WaveformContext(channel, 101)

        # Do without an imbalance
        channel.phase_offset = 0.0
        buffer_without_offset = context._do_upconvert(deepcopy(buffer), times)

        # Do with an imbalance
        channel.phase_offset = 0.1
        buffer_with_offset = context._do_upconvert(deepcopy(buffer), times)

        ratio = buffer_with_offset / buffer_without_offset
        assert np.allclose(ratio, ratio[0])

    @pytest.mark.parametrize(
        "delay",
        [
            0,
            10,
        ],
    )
    @pytest.mark.parametrize("do_upconvert", [True, False])
    @pytest.mark.parametrize("scale", [0.95, 1.0])
    @pytest.mark.parametrize("ignore_scale", [True, False])
    @pytest.mark.parametrize("sample_time", [1e-09, 0.5e-09])
    def test_square_pulse(self, delay, do_upconvert, scale, ignore_scale, sample_time):
        """Tests that a square pulse is processed correctly."""
        model = EchoModelLoader(2).load()
        channel = model.qubits[0].get_drive_channel()
        channel.scale = scale
        assert channel.fixed_if is False
        channel.frequency = channel.baseband_frequency + 0.1e9
        assert channel.imbalance == 1.0

        context = WaveformContext(channel, 20)
        context._duration = delay
        pulse = Pulse(
            channel,
            width=80e-9,
            shape=PulseShapeType.SQUARE,
            ignore_channel_scale=ignore_scale,
        )
        context.process_pulse(pulse, 10, do_upconvert=do_upconvert)
        assert np.allclose(context.buffer[0:delay], 0.0)
        assert np.allclose(context.buffer[delay + 10 :], 0.0)

        pulse = context.buffer[delay : delay + 10]
        amp = 1.0 if ignore_scale else scale
        assert np.allclose(np.abs(pulse), amp)
        if do_upconvert:
            assert not np.allclose(pulse, pulse[0])

    @pytest.mark.parametrize("attributes", pulse_attributes)
    @pytest.mark.parametrize(
        "delay",
        [
            0,
            10,
        ],
    )
    def test_pulse_shapes(self, attributes, delay):
        """Formally the context currently works with all pulse shape, even though we'd like
        to have shapes evaluated as a pass."""

        context = WaveformContext(self.channel, 20)
        context._duration = delay
        pulse = Pulse(self.channel, width=80e-9, **attributes)
        context.process_pulse(pulse, 10, 10e-09)
        assert np.allclose(context.buffer[0:delay], 0.0)
        assert not np.allclose(context.buffer[delay : delay + 10], 0.0)
        assert np.allclose(context.buffer[delay + 10 :], 0.0)

    def test_process_pulse_with_freq_shift(self):
        """Regression test that checks pulses are correctly calculated when there is a phase
        shift."""

        assert self.channel.fixed_if is False
        freq_shift = 100e6
        context = WaveformContext(self.channel, 10)
        pulse = Pulse(
            self.channel,
            width=80e-9,
            shape=PulseShapeType.SQUARE,
        )
        context.process_pulse(pulse, 10, do_upconvert=True)
        waveform = context.buffer

        context = WaveformContext(self.channel, 10)
        context.process_frequencyshift(freq_shift)
        context.process_pulse(pulse, 10, do_upconvert=True)
        waveform_2 = context.buffer
        assert not np.allclose(waveform, waveform_2)

        times = self.channel.sample_time * np.arange(10)
        expected_phase = 2.0 * np.pi * freq_shift * times
        print(expected_phase)
        expected_waveform = np.exp(1.0j * expected_phase) * waveform
        assert np.allclose(waveform_2, expected_waveform)
