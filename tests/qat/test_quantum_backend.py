# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2024 Oxford Quantum Circuits Ltd
import tempfile
from os.path import dirname, join

import numpy as np
import pytest
from compiler_config.config import CompilerConfig
from scipy import fftpack

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.live import LiveDeviceEngine, sync_baseband_frequencies_to_value
from qat.purr.backends.realtime_chip_simulator import (
    get_default_RTCS_hardware,
    qutip_available,
)
from qat.purr.backends.utilities import get_axis_map
from qat.purr.compiler.devices import (
    Calibratable,
    ChannelType,
    MaxPulseLength,
    PhysicalBaseband,
    PhysicalChannel,
    PulseShapeType,
    QubitCoupling,
    add_cross_resonance,
    build_qubit,
    build_resonator,
)
from qat.purr.compiler.emitter import InstructionEmitter, QatFile
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    Delay,
    DrivePulse,
    MeasurePulse,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    Pulse,
    SweepValue,
    Variable,
)
from qat.purr.compiler.runtime import QuantumRuntime, execute_instructions, get_builder
from qat.purr.integrations.qasm import Qasm2Parser
from qat.qat import execute

from tests.qat.qasm_qir_utils import get_qasm2
from tests.qat.test_readout_mitigation import apply_error_mitigation_setup


class FakeBaseQuantumExecution(LiveDeviceEngine):
    baseband_frequencies = {}
    buffers = {}

    def startup(self):
        pass

    def shutdown(self):
        pass

    def _execute_on_hardware(self, sweep_iterator, package: QatFile, interrupt=None):
        self.buffers = {}
        self.baseband_frequencies = {}

        results = {}
        increment = 0
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)

            position_map = self.create_duration_timeline(package.instructions)
            pulse_channel_buffers = self.build_pulse_channel_buffers(position_map, True)
            buffers = self.build_physical_channel_buffers(pulse_channel_buffers)
            baseband_freqs = self.build_baseband_frequencies(pulse_channel_buffers)
            aq_map = self.build_acquire_list(position_map)

            self.buffers[increment] = buffers
            self.baseband_frequencies[increment] = baseband_freqs
            for channel, aqs in aq_map.items():
                for aq in aqs:
                    dt = aq.physical_channel.sample_time
                    start = round(aq.start + aq.delay / dt)
                    response = self.buffers[increment][aq.physical_channel.full_id()][
                        start : start + aq.samples
                    ]

                    response_axis = get_axis_map(aq.mode, response)
                    for pp in package.get_pp_for_variable(aq.output_variable):
                        response, response_axis = self.run_post_processing(
                            pp, response, response_axis
                        )

                    var_result = results.setdefault(
                        aq.output_variable,
                        np.empty(
                            (sweep_iterator.get_results_shape(response.shape)),
                            response.dtype,
                        ),
                    )
                    sweep_iterator.insert_result_at_sweep_position(var_result, response)

            increment += 1
        return results


def apply_setup_to_hardware(model):
    bb1 = PhysicalBaseband("AP1-L1", 4.024e9, 250e6)
    bb2 = PhysicalBaseband("AP1-L2", 8.43135e9, 250e6)
    bb3 = PhysicalBaseband("AP1-L3", 3.6704e9, 250e6)
    bb4 = PhysicalBaseband("AP1-L4", 7.8891e9, 250e6)

    ch1 = PhysicalChannel("Ch1", 0.5e-9, bb1, 1)
    ch2 = PhysicalChannel("Ch2", 1e-09, bb2, 1, acquire_allowed=True)
    ch3 = PhysicalChannel("Ch3", 0.5e-9, bb3, 1)
    ch4 = PhysicalChannel("Ch4", 1e-09, bb4, 1, acquire_allowed=True)

    r0 = build_resonator("R0", ch2, frequency=8.68135e9, measure_fixed_if=True)
    q0 = build_qubit(
        0,
        r0,
        ch1,
        drive_freq=4.274e9,
        second_state_freq=4.085e9,
        measure_amp=10e-3,
        fixed_drive_if=True,
    )

    r1 = build_resonator("R1", ch4, frequency=8.68135e9, measure_fixed_if=True)
    q1 = build_qubit(
        1,
        r1,
        ch3,
        drive_freq=3.9204e9,
        second_state_freq=3.7234e9,
        measure_amp=10e-3,
        fixed_drive_if=True,
    )

    add_cross_resonance(q0, q1)

    model.add_physical_baseband(bb1, bb2, bb3, bb4)
    model.add_physical_channel(ch1, ch2, ch3, ch4)
    model.add_quantum_device(r0, q0, r1, q1)


def get_test_model() -> QuantumHardwareModel:
    """Passing in some hardware applies these settings to that instance."""
    model = QuantumHardwareModel()
    apply_setup_to_hardware(model)

    model.is_calibrated = False
    return model


def get_test_execution_engine(model) -> FakeBaseQuantumExecution:
    return FakeBaseQuantumExecution(model)


def get_test_runtime(model) -> QuantumRuntime:
    return QuantumRuntime(get_test_execution_engine(model))


class TestBaseQuantum:
    def get_qasm2(self, file_name):
        with open(join(dirname(__file__), "files", "qasm", file_name), "r") as qasm_file:
            return qasm_file.read()

    def test_batched_execution(self):
        hw = get_default_echo_hardware()
        hw.shot_limit = 10

        config = CompilerConfig()
        config.repeats = 50
        config.results_format.binary_count()

        results = execute(get_qasm2("basic.qasm"), hw, config)

        # Assert we have a full 50 results at 00.
        assert results["c"]["00"] == 50

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_invalid_qubit_numbers(self):
        qasm_string = get_qasm2("example.qasm")
        with pytest.raises(ValueError):
            Qasm2Parser().parse(get_builder(get_default_RTCS_hardware()), qasm_string)

    def test_load_predefined_qutip_hardware_config(self):
        hw = get_test_model()
        for id_, device in hw.quantum_devices.items():
            assert hw.get_quantum_device(id_) is not None
            for pc_id, pc in device.pulse_channels.items():
                pc_id_decomp = pc_id.split(".")
                assert (
                    hw.get_pulse_channel_from_device(
                        ChannelType[pc_id_decomp[-1]], id_, pc_id_decomp[:-1]
                    )
                    is not None
                )
        for id_ in hw.physical_channels.keys():
            assert hw.get_physical_channel(id_) is not None
        for id_ in hw.basebands.keys():
            assert hw.get_physical_baseband(id_) is not None

    def test_fixed_if_non_regular_pulse_generation(self):
        hw = get_test_model()
        pulse_channel = hw.get_pulse_channel_from_device(ChannelType.drive, "Q0")
        physical_channel = pulse_channel.physical_channel
        baseband = physical_channel.baseband
        pulse_channel.frequency = 5e9
        pulse_channel.fixed_if = True
        baseband.if_frequency = 250e6
        baseband.frequency = 5.5e9
        qubit = hw.get_qubit(0)
        drive_channel = qubit.get_drive_channel()
        drive_rate = 30e-3

        builder = (
            get_builder(hw)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=1e-6, amp=drive_rate)
            .measure_scope_mode(qubit)
        )

        engine = get_test_execution_engine(hw)
        execute_instructions(engine, builder)

        pulses = engine.buffers[0]["Ch1"]
        assert (
            engine.baseband_frequencies[0]["Ch1"]
            == pulse_channel.frequency - baseband.if_frequency
        )
        x = fftpack.fft(pulses)
        if_freq = np.argmax(x) / physical_channel.sample_time / len(pulses)
        assert if_freq == pytest.approx(baseband.if_frequency)

    def test_not_fixed_if_non_regular_pulse_generation(self):
        hw = get_test_model()
        pulse_channel = hw.get_pulse_channel_from_device(ChannelType.drive, "Q0")
        physical_channel = pulse_channel.physical_channel
        baseband = physical_channel.baseband
        pulse_channel.frequency = 5e9
        pulse_channel.fixed_if = False
        baseband.frequency = 4.5e9
        baseband.if_frequency = 250e6

        qubit = hw.get_qubit(0)
        drive_channel = qubit.get_drive_channel()
        drive_rate = 30e-3
        builder = (
            get_builder(hw)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=1e-6, amp=drive_rate)
            .measure_scope_mode(qubit)
        )

        engine = get_test_execution_engine(hw)
        execute_instructions(engine, builder)

        pulses = engine.buffers[0]["Ch1"]
        assert "Ch1" not in engine.baseband_frequencies
        x = fftpack.fft(pulses)
        if_freq = np.argmax(x) / physical_channel.sample_time / len(pulses)
        assert if_freq == pytest.approx(pulse_channel.frequency - baseband.frequency)

    def test_fixed_if_pulse_channel_frequency_priority_with_multiple_pulse_channels(
        self,
    ):
        hw = get_test_model()
        qubit = hw.get_qubit(0)
        drive_channel = qubit.get_drive_channel()
        physical_channel = drive_channel.physical_channel
        baseband = physical_channel.baseband
        drive_channel.frequency = 5e9
        drive_channel.fixed_if = True
        drive_rate = 30e-3
        baseband.if_frequency = 250e6

        second_state_channel = qubit.get_second_state_channel()
        second_state_channel.frequency = 4.8e9
        second_state_channel.fixed_if = False
        hw.plot_buffers = True

        physical_channel = drive_channel.physical_channel
        baseband = physical_channel.baseband
        baseband.if_frequency = 250e6
        baseband.frequency = drive_channel.frequency - baseband.if_frequency

        builder = (
            get_builder(hw)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=1e-6, amp=drive_rate)
            .synchronize([drive_channel, second_state_channel])
            .pulse(second_state_channel, PulseShapeType.SQUARE, width=2e-6, amp=drive_rate)
            .measure_scope_mode(qubit)
        )

        engine = get_test_execution_engine(hw)
        execute_instructions(engine, builder)

        pulses = engine.buffers[0]["Ch1"]
        assert (
            engine.baseband_frequencies[0]["Ch1"]
            == drive_channel.frequency - baseband.if_frequency
        )
        x = fftpack.fft(pulses)
        if_freq_second_state = (
            np.argsort(x)[-1] / physical_channel.sample_time / len(pulses)
        )
        if_freq_drive = np.argsort(x)[-2] / physical_channel.sample_time / len(pulses)
        assert if_freq_second_state == pytest.approx(
            second_state_channel.frequency - baseband.frequency
        )
        assert if_freq_drive == pytest.approx(baseband.if_frequency)

    def test_sync_baseband_frequencies_after_pulse_channel_update(self):
        hw = get_test_model()
        drive_channel_1 = hw.get_qubit(0).get_drive_channel()
        drive_channel_2 = hw.get_qubit(1).get_drive_channel()
        baseband_1 = drive_channel_1.physical_channel.baseband
        baseband_2 = drive_channel_2.physical_channel.baseband
        drive_channel_1.fixed_if = True
        drive_channel_2.fixed_if = True
        drive_channel_1.frequency = 5.0e9 + 1
        drive_channel_2.frequency = 5.5e9 + 1
        baseband_1.frequency = 4.75e9
        baseband_2.frequency = 5.25e9
        baseband_1.if_frequency = 250e6
        baseband_2.if_frequency = 250e6

        common_lo_frequency = 5e9
        sync_baseband_frequencies_to_value(hw, common_lo_frequency, [0, 1])
        assert drive_channel_1.frequency == 5.0e9
        assert drive_channel_2.frequency == 5.5e9
        assert baseband_1.if_frequency == drive_channel_1.frequency - baseband_1.frequency
        assert baseband_1.frequency == common_lo_frequency
        assert baseband_2.frequency == common_lo_frequency
        assert baseband_2.if_frequency == drive_channel_2.frequency - baseband_2.frequency

    def test_setup_hold_measure_pulse(self):
        hw = get_test_model()
        qubit = hw.get_qubit(0)
        qubit.pulse_measure = {
            "shape": PulseShapeType.SETUP_HOLD,
            "width": 3.0e-6,
            "amp": 0.01,
            "amp_setup": 0.02,
            "rise": 100e-9,
        }
        qubit.measure_acquire["delay"] = 0.0
        result, _ = execute_instructions(
            get_test_execution_engine(hw), get_builder(hw).measure_scope_mode(qubit)
        )
        result = result[0]
        setup_length = (
            int(qubit.pulse_measure["rise"] / hw.get_physical_channel("Ch2").sample_time)
            + 1
        )
        full_length = int(
            qubit.pulse_measure["width"] / hw.get_physical_channel("Ch2").sample_time
        )

        assert np.array_equal(
            qubit.pulse_measure["amp_setup"] * np.ones(setup_length),
            np.real(result[:setup_length]),
        )
        assert np.array_equal(
            np.zeros(setup_length), np.round(np.imag(result[:setup_length]), 7)
        )
        assert np.array_equal(
            qubit.pulse_measure["amp"] * np.ones(full_length - setup_length),
            np.real(result[setup_length:full_length]),
        )
        assert np.array_equal(
            np.zeros(full_length - setup_length),
            np.round(np.imag(result[setup_length:full_length]), 7),
        )

    def test_variable_acquire_pulse(self):
        hw = get_test_model()
        qubit = hw.get_qubit(0)
        qubit.pulse_measure = {
            "shape": PulseShapeType.SETUP_HOLD,
            "width": 3.0e-6,
            "amp": 0.01,
            "amp_setup": 0.02,
            "rise": 500e-9,
        }
        qubit.measure_acquire = {"delay": 300e-9, "sync": False, "width": 1e-6}
        result, _ = execute_instructions(
            get_test_execution_engine(hw), get_builder(hw).measure_scope_mode(qubit)
        )
        result = result[0]
        setup_length = int(
            qubit.pulse_measure["rise"] / hw.get_physical_channel("Ch2").sample_time
        )
        delay_length = int(
            qubit.measure_acquire["delay"] / hw.get_physical_channel("Ch2").sample_time
        )
        full_length = int(
            qubit.measure_acquire["width"] / hw.get_physical_channel("Ch2").sample_time
        )
        assert np.array_equal(
            qubit.pulse_measure["amp_setup"] * np.ones(setup_length - delay_length),
            result[: setup_length - delay_length].real,
        )
        assert np.array_equal(
            qubit.pulse_measure["amp"] * np.ones(full_length - setup_length + delay_length),
            result[setup_length - delay_length : full_length].real,
        )

    def test_qubit_coupling_direction_calibration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_hw = get_test_model()
            coupling_direction = (1, 2)
            coupling_quality = 9
            original_hw.qubit_direction_couplings = [
                QubitCoupling(coupling_direction, quality=coupling_quality)
            ]
            cal_file_path = join(tmp_dir, "cal.json")
            original_hw.save_calibration_to_file(cal_file_path, use_cwd=False)

            empty_hw = Calibratable.load_calibration_from_file(cal_file_path)

            assert 1 == len(empty_hw.qubit_direction_couplings)
            assert coupling_direction == empty_hw.qubit_direction_couplings[0].direction
            assert coupling_quality == empty_hw.qubit_direction_couplings[0].quality

    def test_error_mitigation_calibration(self):
        original_hw = get_test_model()
        assert original_hw.error_mitigation is None

        apply_error_mitigation_setup(
            original_hw,
            q0_ro_fidelity_0=0.8,
            q0_ro_fidelity_1=0.7,
            q1_ro_fidelity_0=0.9,
            q1_ro_fidelity_1=0.6,
        )
        assert original_hw.error_mitigation is not None

        original_hw_cal_string = original_hw.get_calibration()
        loaded_hw = Calibratable.load_calibration(original_hw_cal_string)
        assert loaded_hw.error_mitigation is not None

        original_linear_rm = original_hw.error_mitigation.readout_mitigation.linear
        loaded_linear_rm = loaded_hw.error_mitigation.readout_mitigation.linear
        assert loaded_linear_rm == original_linear_rm

    def test_sweep_pulse_length_exceeds_max_throws_error(self):
        hw = get_test_model()
        qubit = hw.get_qubit(0)
        drive_channel = qubit.get_drive_channel()
        nb_points = 11
        time = np.linspace(0.0, 10 * MaxPulseLength, nb_points)
        builder = (
            get_builder(hw)
            .sweep(SweepValue("t", time))
            .pulse(drive_channel, PulseShapeType.SQUARE, width=Variable("t"))
            .measure_scope_mode(qubit)
        )

        engine = get_test_execution_engine(hw)
        with pytest.raises(ValueError):
            execute_instructions(engine, builder)

        time = np.linspace(0.0, 100e-6, nb_points)
        builder = (
            get_builder(hw)
            .sweep(SweepValue("t", time))
            .pulse(drive_channel, PulseShapeType.SQUARE, width=Variable("t"))
            .measure_scope_mode(qubit)
        )
        assert nb_points == execute_instructions(engine, builder)[0].shape[0]

    def test_phase_shift_optimisation_squashes_down_adjacent_phase_shifts(self):
        hw = get_test_model()
        qubit = hw.get_qubit(0)
        phase_shift_1 = 0.2
        phase_shift_2 = 0.1
        builder = (
            get_builder(hw)
            .X(qubit, np.pi / 2.0)
            .phase_shift(qubit, phase_shift_1)
            .phase_shift(qubit, phase_shift_2)
            .X(qubit, np.pi / 2.0)
            .measure_mean_z(qubit)
        )
        engine = get_test_execution_engine(hw)
        optimized_instructions = engine.optimize(builder.instructions)
        phase_shift_list = [
            instr for instr in optimized_instructions if isinstance(instr, PhaseShift)
        ]
        assert len(phase_shift_list) == 1
        assert phase_shift_list[0].phase == phase_shift_1 + phase_shift_2

    def test_phase_shift_optimisation_does_not_squash_down_non_adjacent_phase_shifts(
        self,
    ):
        hw = get_test_model()
        qubit = hw.get_qubit(0)
        phase_shift_1 = 0.2
        phase_shift_2 = 0.1
        builder = (
            get_builder(hw)
            .phase_shift(qubit, phase_shift_1)
            .X(qubit, np.pi / 2.0)
            .phase_shift(qubit, phase_shift_2)
            .X(qubit, np.pi / 2.0)
            .measure_mean_z(qubit)
        )
        engine = get_test_execution_engine(hw)
        optimized_instructions = engine.optimize(builder.instructions)
        print(optimized_instructions)
        phase_shift_list = [
            instr for instr in optimized_instructions if isinstance(instr, PhaseShift)
        ]
        assert len(phase_shift_list) == 2
        assert phase_shift_list[0].phase == phase_shift_1
        assert phase_shift_list[1].phase == phase_shift_2

    def test_phase_shift_optimisation_skips_sweep_variables(self):
        hw = get_test_model()
        qubit = hw.get_qubit(0)
        phase_shift_fixed = 0.2
        phase_shift_sweep = np.linspace(0.0, 1.0, 5)
        builder = (
            get_builder(hw)
            .sweep(SweepValue("p", phase_shift_sweep))
            .X(qubit, np.pi / 2.0)
            .phase_shift(qubit, Variable("p"))
            .phase_shift(qubit, phase_shift_fixed)
            .X(qubit, np.pi / 2.0)
            .measure_mean_z(qubit)
        )
        engine = get_test_execution_engine(hw)
        optimized_instructions = engine.optimize(builder.instructions)
        phase_shift_list = [
            instr for instr in optimized_instructions if isinstance(instr, PhaseShift)
        ]
        assert len(phase_shift_list) == 2
        assert phase_shift_list[1].phase == phase_shift_fixed
        assert isinstance(phase_shift_list[0].phase, Variable)

    def setup_frequency_shift(self, hardware, freq_shift_qubits=[]):
        builder = get_builder(hardware)

        for qid in freq_shift_qubits:
            qubit = hardware.get_qubit(qid)
            freq_channel = qubit.create_pulse_channel(
                ChannelType.freq_shift, amp=1.0, scale=1.0, frequency=8.5e9
            )

        for qubit in hardware.qubits:
            builder.pulse(
                qubit.get_drive_channel(),
                PulseShapeType.SQUARE,
                width=1e-6,
                amp=1,
                ignore_channel_scale=True,
            )

        return builder

    def get_qubit_buffers(self, hw, engine, ids):
        if not isinstance(ids, list):
            ids = [ids]
        return [engine.buffers[0][hw.get_qubit(_id).physical_channel.id] for _id in ids]

    def test_no_freq_shift(self):
        hardware = get_default_echo_hardware(2)
        engine = get_test_execution_engine(hardware)

        builder = self.setup_frequency_shift(hardware)

        execute_instructions(engine, builder)
        qubit1_buffer, qubit2_buffer = self.get_qubit_buffers(hardware, engine, [0, 1])
        assert len(qubit1_buffer) > 0
        assert len(qubit2_buffer) > 0
        assert np.isclose(qubit1_buffer, 1 + 0j).all()
        assert np.isclose(qubit2_buffer, 1 + 0j).all()

    def test_freq_shift(self):
        hardware = get_default_echo_hardware(2)
        engine = get_test_execution_engine(hardware)

        builder = self.setup_frequency_shift(hardware, freq_shift_qubits=[0])

        execute_instructions(engine, builder)
        qubit1_buffer, qubit2_buffer = self.get_qubit_buffers(hardware, engine, [0, 1])

        assert len(qubit1_buffer) > 0
        assert len(qubit2_buffer) > 0
        assert np.isclose([abs(val) for val in qubit1_buffer], 2).all()
        assert np.isclose([abs(val) for val in qubit2_buffer], 1).all()

    @pytest.mark.parametrize(
        "pre_combine, batch_results, post_combine",
        [
            ({}, {}, {}),
            ({}, {"a": [1]}, {"a": [1]}),
            ({"a": [1]}, {"a": [2]}, {"a": [1, 2]}),
        ],
    )
    def test_combining_python_list_results_succeeds(
        self, pre_combine, batch_results, post_combine
    ):
        results = FakeBaseQuantumExecution._accumulate_results(pre_combine, batch_results)
        assert results == post_combine

    @pytest.mark.parametrize(
        "pre_combine, batch_results, post_combine",
        [
            ({"a": np.array([1, 2])}, {"a": np.array([3])}, {"a": np.array([1, 2, 3])}),
        ],
    )
    def test_combining_numpy_array_results_succeeds(
        self, pre_combine, batch_results, post_combine
    ):
        results = FakeBaseQuantumExecution._accumulate_results(pre_combine, batch_results)
        assert results.keys() == post_combine.keys()
        for k in results.keys():
            assert results[k].tolist() == post_combine[k].tolist()

    @pytest.mark.parametrize(
        "pre_combine, batch_results",
        [
            ({"a": [1]}, {"b": [2]}),
            ({"a": None}, {"a": [3]}),
            ({"a": {1, 2}}, {"a": [3]}),
            ({"a": np.array([1, 2])}, {"a": [3]}),
        ],
    )
    def test_combining_results_fails(self, pre_combine, batch_results):
        with pytest.raises(ValueError):
            FakeBaseQuantumExecution._accumulate_results(pre_combine, batch_results)

    @pytest.mark.parametrize(
        "repeat_count, repeat_limit, expected_batches",
        [(123, 5, [5] * 24 + [3]), (124, 5, [5] * 24 + [4]), (125, 5, [5] * 25)],
    )
    def test_execution_batching(self, repeat_count, repeat_limit, expected_batches):
        hw = get_test_model()
        hw.repeat_limit = repeat_limit
        engine = get_test_execution_engine(hw)
        generated_batches = engine._generate_repeat_batches(repeat_count)
        assert generated_batches == expected_batches

    def test_empty_channels_removed(self):
        hw = get_test_model()
        qubit = hw.get_qubit(0)

        builder = (
            get_builder(hw).X(qubit, np.pi / 2.0).measure_mean_z(qubit).synchronize(qubit)
        )
        engine = get_test_execution_engine(hw)
        qat_file = InstructionEmitter().emit(builder.instructions, hw)

        channels_with_pulses = [
            qubit.get_drive_channel(),
            qubit.get_measure_channel(),
            qubit.get_acquire_channel(),
        ]
        channels_removed = [
            qubit.get_second_state_channel(),
            qubit.get_cross_resonance_channel(hw.get_qubit(1)),
            qubit.get_cross_resonance_cancellation_channel(hw.get_qubit(1)),
        ]

        # Check if empty channels are in QAT file
        sync_instr = qat_file.instructions[-1]
        for ch in channels_removed:
            assert ch in sync_instr.quantum_targets

        position_map = engine.create_duration_timeline(qat_file.instructions)

        for ch in channels_with_pulses:
            assert ch in position_map.keys()
        for ch in channels_removed:
            assert ch not in position_map.keys()
        for ch, positions in position_map.items():
            assert not all(
                [
                    isinstance(p.instruction, (Delay, PhaseShift, PhaseReset))
                    for p in positions
                ]
            )

    def test_create_duration_timeline_mapping(self):
        """Test that instructions are correctly mapped to expected channels."""
        hw = get_test_model()
        qubit = hw.get_qubit(0)

        builder = (
            get_builder(hw).X(qubit, np.pi / 2.0).measure_mean_z(qubit).synchronize(qubit)
        )
        engine = get_test_execution_engine(hw)
        qat_file = InstructionEmitter().emit(builder.instructions, hw)

        position_map = engine.create_duration_timeline(qat_file.instructions)

        drive_data = position_map[qubit.get_drive_channel()]
        measure_data = position_map[qubit.get_measure_channel()]
        acquire_data = position_map[qubit.get_acquire_channel()]

        assert [type(pos.instruction) for pos in drive_data] == [
            DrivePulse,
            Delay,
        ]
        assert [type(pos.instruction) for pos in measure_data] == [
            Delay,
            MeasurePulse,
        ]
        assert [type(pos.instruction) for pos in acquire_data] == [
            Delay,
            Acquire,
            *[PostProcessing] * 4,
        ]

    @pytest.mark.parametrize(
        "hw", [get_test_model(), get_default_echo_hardware(), get_default_RTCS_hardware()]
    )
    def test_duration_timeline_aligns(self, hw):
        """
        For circuits where quantum targets are synced, check that instructions align.
        """
        engine = hw.create_engine()
        q1 = hw.get_qubit(0)
        q2 = hw.get_qubit(1)

        # Build a simple circuit
        builder = (
            get_builder(hw)
            .X(q1, np.pi / 2.0)
            .X(q2, np.pi)
            .ECR(q1, q2)
            .measure_mean_z(q1)
            .measure_mean_z(q2)
        )
        pos_map = engine.create_duration_timeline(builder.instructions)

        # Check that pulse instructions start concurrently
        q1_cr = q1.get_pulse_channel(ChannelType.cross_resonance, [q2])
        q2_crc = q2.get_pulse_channel(ChannelType.cross_resonance_cancellation, [q1])
        times_q1_cr = [
            inst.start for inst in pos_map[q1_cr] if isinstance(inst.instruction, Pulse)
        ]
        times_q2_crc = [
            inst.start for inst in pos_map[q2_crc] if isinstance(inst.instruction, Pulse)
        ]
        assert len(times_q1_cr) > 0
        assert len(times_q1_cr) == len(times_q2_crc)
        assert all(np.isclose(times_q1_cr, times_q2_crc))

        # Check that the acquires start concurrently
        q1_acquire = q1.get_acquire_channel()
        q2_acquire = q2.get_acquire_channel()
        times_q1_acquire = [
            inst.start
            for inst in pos_map[q1_acquire]
            if isinstance(inst.instruction, Acquire)
        ]
        times_q2_acquire = [
            inst.start
            for inst in pos_map[q2_acquire]
            if isinstance(inst.instruction, Acquire)
        ]
        assert len(times_q1_acquire) == 1
        assert len(times_q2_acquire) == 1
        assert np.isclose(times_q1_acquire[0], times_q2_acquire[0])

        # Check that the measures start concurrently
        q1_measure = q1.get_measure_channel()
        q2_measure = q2.get_measure_channel()
        times_q1_measure = [
            inst.start
            for inst in pos_map[q1_measure]
            if isinstance(inst.instruction, MeasurePulse)
        ]
        times_q2_measure = [
            inst.start
            for inst in pos_map[q2_measure]
            if isinstance(inst.instruction, MeasurePulse)
        ]
        assert len(times_q1_measure) == 1
        assert len(times_q2_measure) == 1
        assert np.isclose(times_q1_measure[0], times_q2_measure[0])

    @pytest.mark.parametrize(
        "hw", [get_test_model(), get_default_echo_hardware(), get_default_RTCS_hardware()]
    )
    def test_duration_timeline_times(self, hw):
        """
        Tests that the creation of a duration timeline with a two-qubit circuit
        gives a timeline where the position map for each pulse channel has instructions
        that align
        """
        engine = hw.create_engine()
        q1 = hw.get_qubit(0)
        q2 = hw.get_qubit(1)

        # Build a simple circuit
        builder = (
            get_builder(hw)
            .X(q1, np.pi / 2.0)
            .X(q2, np.pi)
            .cnot(q1, q2)
            .measure_mean_z(q1)
            .measure_mean_z(q2)
        )
        res1 = engine.create_duration_timeline(builder.instructions)

        # Check the times match up
        for val in res1.values():
            end = 0
            for pos in val:
                assert end == pos.start
                end = pos.end

    def compare_position_maps(self, res1, res2):
        for key in res1.keys():
            starts1 = [
                inst.start
                for inst in res1[key]
                if (not isinstance(inst.instruction, Delay))
            ]
            starts2 = [
                inst.start
                for inst in res2[key]
                if (not isinstance(inst.instruction, Delay))
            ]
            ends1 = [
                inst.start
                for inst in res1[key]
                if (not isinstance(inst.instruction, Delay))
            ]
            ends2 = [
                inst.start
                for inst in res2[key]
                if (not isinstance(inst.instruction, Delay))
            ]
            assert starts1 == starts2
            assert ends1 == ends2

    @pytest.mark.parametrize(
        "hw", [get_test_model(), get_default_echo_hardware(), get_default_RTCS_hardware()]
    )
    def test_duration_timeline_sync_ecr(self, hw):
        """
        Tests that a redundant sync has no effect on the circuit.
        In this example, the ECR adds a sync itself, so syncing before is redundant.
        """
        engine = hw.create_engine()
        q1 = hw.get_qubit(0)
        q2 = hw.get_qubit(1)

        # Build a simple circuit
        builder = (
            get_builder(hw)
            .X(q1, np.pi / 2.0)
            .X(q2, np.pi)
            .ECR(q1, q2)
            .measure_mean_z(q1)
            .measure_mean_z(q2)
        )
        res1 = engine.create_duration_timeline(builder.instructions)

        # Build the same circuit with an unnecessary sync & check the times match up
        builder = (
            get_builder(hw)
            .X(q1, np.pi / 2.0)
            .X(q2, np.pi)
            .synchronize([q1, q2])
            .ECR(q1, q2)
            .measure_mean_z(q1)
            .measure_mean_z(q2)
        )
        res2 = engine.create_duration_timeline(builder.instructions)
        self.compare_position_maps(res1, res2)

    @pytest.mark.parametrize(
        "hw", [get_test_model(), get_default_echo_hardware(), get_default_RTCS_hardware()]
    )
    def test_duration_timeline_sync_x_pulses(self, hw):
        """
        Tests that a redundant sync has no effect on the circuit.
        In this example, the two x pulses on Q1 take the same time to execute as the single
        pulser in Q2, so the sync is redundant.
        """
        engine = hw.create_engine()
        q1 = hw.get_qubit(0)
        q2 = hw.get_qubit(1)

        # Build a simple circuit
        builder = (
            get_builder(hw)
            .X(q1, np.pi / 2.0)
            .X(q2, np.pi)
            .X(q1, np.pi / 2.0)
            .had(q1)
            .had(q2)
            .measure_mean_z(q1)
            .measure_mean_z(q2)
        )
        res1 = engine.create_duration_timeline(builder.instructions)

        # Build the same circuit with an unnecessary sync & check the times match up
        builder = (
            get_builder(hw)
            .X(q1, np.pi / 2.0)
            .X(q2, np.pi)
            .X(q1, np.pi / 2.0)
            .synchronize([q1, q2])
            .had(q1)
            .had(q2)
            .measure_mean_z(q1)
            .measure_mean_z(q2)
        )
        res2 = engine.create_duration_timeline(builder.instructions)
        self.compare_position_maps(res1, res2)

    def evaluate_circuit_time(self, hw, engine, builder):
        qatfile = InstructionEmitter().emit(builder.instructions, hw)
        res = engine.create_duration_timeline(qatfile.instructions)
        # pulse channels could have different block times
        return max([val[-1].end * pc.block_time for pc, val in res.items()])

    @pytest.mark.parametrize(
        "hw", [get_test_model(), get_default_echo_hardware(), get_default_RTCS_hardware()]
    )
    def test_duration_timeline_compare(self, hw):
        """
        Tests that the duration of individual circuit elements matches that
        of the full circuit.
        """
        engine = hw.create_engine()
        q1 = hw.get_qubit(0)
        q2 = hw.get_qubit(1)

        # Build a simple circuit
        builder = (
            get_builder(hw)
            .X(q1, np.pi / 2.0)
            .X(q2, np.pi)
            .ECR(q1, q2)
            .measure_mean_z(q1)
            .measure_mean_z(q2)
        )
        maxtime = self.evaluate_circuit_time(hw, engine, builder)

        # Check the circuit executes in an expected time
        circs = [
            get_builder(hw).X(q1, np.pi / 2.0),
            get_builder(hw).X(q2, np.pi),
            get_builder(hw).ECR(q1, q2),
            get_builder(hw).measure_mean_z(q1),
            get_builder(hw).measure_mean_z(q2),
        ]

        ts = [self.evaluate_circuit_time(hw, engine, circ) for circ in circs]

        # ECR will sync q1 and q2 before and after, so the total execution time is
        # a maximum of the X pulses plus a maximum of the measures.
        assert np.isclose(max(ts[0], ts[1]) + ts[2] + max(ts[3], ts[4]), maxtime)

    @pytest.mark.parametrize(
        "hw", [get_test_model(), get_default_echo_hardware(), get_default_RTCS_hardware()]
    )
    def test_duration_timeline_compare_sync(self, hw):
        """
        Tests that the duration of individual circuit elements matches that
        of the full circuit when syncs are used.
        """
        engine = hw.create_engine()
        q1 = hw.get_qubit(0)
        q2 = hw.get_qubit(1)

        # Check that synchronize gives the expected times
        builder = (
            get_builder(hw)
            .X(q1, np.pi / 2.0)
            .synchronize([q1, q2])
            .X(q2, np.pi)
            .ECR(q1, q2)
            .measure_mean_z(q1)
            .measure_mean_z(q2)
        )
        maxtime = self.evaluate_circuit_time(hw, engine, builder)

        # Check the circuit executes in an expected time
        circs = [
            get_builder(hw).X(q1, np.pi / 2.0),
            get_builder(hw).X(q2, np.pi),
            get_builder(hw).ECR(q1, q2),
            get_builder(hw).measure_mean_z(q1),
            get_builder(hw).measure_mean_z(q2),
        ]

        ts = [self.evaluate_circuit_time(hw, engine, circ) for circ in circs]

        # start time of measures will be synced because of the ECR, so the total
        # execution time will only depend on the max of the two measures
        assert np.isclose(ts[0] + ts[1] + ts[2] + max(ts[3], ts[4]), maxtime)

    @pytest.mark.parametrize("pre_measures", [[0], [1], [0, 1]])
    def test_mid_circuit_validation(self, pre_measures):
        """Tests that an error is thrown for mid-circuit measurements."""

        hw = get_test_model()
        engine = get_test_execution_engine(hw)
        q0 = hw.get_qubit(0)
        q1 = hw.get_qubit(1)
        builder = get_builder(hw).had(q0).X(q1)
        for qbit in pre_measures:
            builder.measure(hw.get_qubit(qbit))
        builder.cnot(q0, q1).measure(q0).measure(q1)
        with pytest.raises(ValueError):
            engine.validate(builder.instructions)

    @pytest.mark.parametrize("qbit", [0, 1])
    def test_mid_circuit_validation_pass(self, qbit):
        """
        Checks a circuit successfully validates when a measurement occurs on one qubit,
        and subsequent gates on another.
        """

        hw = get_test_model()
        engine = get_test_execution_engine(hw)
        q0 = hw.get_qubit(qbit)
        q1 = hw.get_qubit(1 - qbit)
        builder = get_builder(hw).had(q0).cnot(q0, q1).measure(q1).X(q0)
        engine.validate(builder.instructions)
        assert True
