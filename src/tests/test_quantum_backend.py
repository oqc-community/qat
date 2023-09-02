# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import tempfile
from os.path import dirname, join

import numpy as np
import pytest
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
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import SweepValue, Variable
from qat.purr.compiler.runtime import QuantumRuntime, execute_instructions, get_builder
from qat.purr.integrations.qasm import Qasm2Parser
from scipy import fftpack

from tests.qasm_utils import get_qasm2


class TestBaseQuantumExecution(LiveDeviceEngine):
    baseband_frequencies = {}
    buffers = {}

    def startup(self):
        pass

    def shutdown(self):
        pass

    def _execute_on_hardware(self, sweep_iterator, package: QatFile):
        self.buffers = {}
        self.baseband_frequencies = {}

        results = {}
        increment = 0
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)

            position_map = self.create_duration_timeline(package)
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
                    response = self.buffers[increment][aq.physical_channel.full_id()
                                                      ][start:start + aq.samples]

                    response_axis = get_axis_map(aq.mode, response)
                    for pp in package.get_pp_for_variable(aq.output_variable):
                        response, response_axis = \
                            self.run_post_processing(pp, response, response_axis)

                    var_result = results.setdefault(
                        aq.output_variable,
                        np.empty((sweep_iterator.get_results_shape(response.shape)),
                                 response.dtype)
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
        fixed_drive_if=True
    )

    r1 = build_resonator("R1", ch4, frequency=8.68135e9, measure_fixed_if=True)
    q1 = build_qubit(
        1,
        r1,
        ch3,
        drive_freq=3.9204e9,
        second_state_freq=3.7234e9,
        measure_amp=10e-3,
        fixed_drive_if=True
    )

    add_cross_resonance(q0, q1)

    model.add_physical_baseband(bb1, bb2, bb3, bb4)
    model.add_physical_channel(ch1, ch2, ch3, ch4)
    model.add_quantum_device(r0, q0, r1, q1)


def get_test_model() -> QuantumHardwareModel:
    """ Passing in some hardware applies these settings to that instance. """
    model = QuantumHardwareModel()
    apply_setup_to_hardware(model)

    model.is_calibrated = False
    return model


def get_test_execution_engine(model) -> TestBaseQuantumExecution:
    return TestBaseQuantumExecution(model)


def get_test_runtime(model) -> QuantumRuntime:
    return QuantumRuntime(get_test_execution_engine(model))


class TestBaseQuantum:
    def get_qasm2(self, file_name):
        with open(join(dirname(__file__), "files", "qasm", file_name), "r") as qasm_file:
            return qasm_file.read()

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
                pc_id_decomp = pc_id.split('.')
                assert hw.get_pulse_channel_from_device(
                    ChannelType[pc_id_decomp[-1]], id_, pc_id_decomp[:-1]
                ) is not None
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
            get_builder(hw).pulse(
                drive_channel, PulseShapeType.SQUARE, width=1e-6, amp=drive_rate
            ).measure_scope_mode(qubit)
        )

        engine = get_test_execution_engine(hw)
        execute_instructions(engine, builder)

        pulses = engine.buffers[0]["Ch1"]
        assert engine.baseband_frequencies[0][
            "Ch1"] == pulse_channel.frequency - baseband.if_frequency
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
            get_builder(hw).pulse(
                drive_channel, PulseShapeType.SQUARE, width=1e-6, amp=drive_rate
            ).measure_scope_mode(qubit)
        )

        engine = get_test_execution_engine(hw)
        execute_instructions(engine, builder)

        pulses = engine.buffers[0]["Ch1"]
        assert "Ch1" not in engine.baseband_frequencies
        x = fftpack.fft(pulses)
        if_freq = np.argmax(x) / physical_channel.sample_time / len(pulses)
        assert if_freq == pytest.approx(pulse_channel.frequency - baseband.frequency)

    def test_fixed_if_pulse_channel_frequency_priority_with_multiple_pulse_channels(self):
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
            get_builder(hw).pulse(
                drive_channel, PulseShapeType.SQUARE, width=1e-6, amp=drive_rate
            ).synchronize([drive_channel, second_state_channel]).pulse(
                second_state_channel, PulseShapeType.SQUARE, width=2e-6, amp=drive_rate
            ).measure_scope_mode(qubit)
        )

        engine = get_test_execution_engine(hw)
        execute_instructions(engine, builder)

        pulses = engine.buffers[0]["Ch1"]
        assert engine.baseband_frequencies[0][
            "Ch1"] == drive_channel.frequency - baseband.if_frequency
        x = fftpack.fft(pulses)
        if_freq_second_state = \
          np.argsort(x)[-1] / physical_channel.sample_time / len(pulses)
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
        assert baseband_1.if_frequency == \
            drive_channel_1.frequency - baseband_1.frequency
        assert baseband_1.frequency == common_lo_frequency
        assert baseband_2.frequency == common_lo_frequency
        assert baseband_2.if_frequency == \
            drive_channel_2.frequency - baseband_2.frequency

    def test_setup_hold_measure_pulse(self):
        hw = get_test_model()
        qubit = hw.get_qubit(0)
        qubit.pulse_measure = {
            'shape': PulseShapeType.SETUP_HOLD,
            'width': 3.0e-6,
            'amp': 0.01,
            'amp_setup': 0.02,
            'rise': 100e-9
        }
        qubit.measure_acquire['delay'] = 0.0
        result, _ = execute_instructions(
            get_test_execution_engine(hw), get_builder(hw).measure_scope_mode(qubit)
        )
        result = result[0]
        setup_length = int(
            qubit.pulse_measure['rise'] / hw.get_physical_channel("Ch2").sample_time
        ) + 1
        full_length = int(
            qubit.pulse_measure['width'] / hw.get_physical_channel("Ch2").sample_time
        )

        assert np.array_equal(
            qubit.pulse_measure['amp_setup'] * np.ones(setup_length),
            np.real(result[:setup_length])
        )
        assert np.array_equal(
            np.zeros(setup_length), np.round(np.imag(result[:setup_length]), 7)
        )
        assert np.array_equal(
            qubit.pulse_measure['amp'] * np.ones(full_length - setup_length),
            np.real(result[setup_length:full_length])
        )
        assert np.array_equal(
            np.zeros(full_length - setup_length),
            np.round(np.imag(result[setup_length:full_length]), 7)
        )

    def test_variable_acquire_pulse(self):
        hw = get_test_model()
        qubit = hw.get_qubit(0)
        qubit.pulse_measure = {
            'shape': PulseShapeType.SETUP_HOLD,
            'width': 3.0e-6,
            'amp': 0.01,
            'amp_setup': 0.02,
            'rise': 500e-9
        }
        qubit.measure_acquire = {'delay': 300e-9, 'sync': False, 'width': 1e-6}
        result, _ = execute_instructions(
            get_test_execution_engine(hw), get_builder(hw).measure_scope_mode(qubit)
        )
        result = result[0]
        setup_length = int(
            qubit.pulse_measure['rise'] / hw.get_physical_channel("Ch2").sample_time
        )
        delay_length = int(
            qubit.measure_acquire['delay'] / hw.get_physical_channel("Ch2").sample_time
        )
        full_length = int(
            qubit.measure_acquire['width'] / hw.get_physical_channel("Ch2").sample_time
        )
        assert np.array_equal(
            qubit.pulse_measure['amp_setup'] * np.ones(setup_length - delay_length),
            result[:setup_length - delay_length].real
        )
        assert np.array_equal(
            qubit.pulse_measure['amp'] * np.ones(full_length - setup_length + delay_length),
            result[setup_length - delay_length:full_length].real
        )

    def test_qubit_coupling_direction_calibration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_hw = get_test_model()
            coupling_direction = (1, 2)
            coupling_quality = 9
            original_hw.qubit_direction_couplings = [
                QubitCoupling(coupling_direction, quality=coupling_quality)
            ]
            cal_file_path = join(tmp_dir, 'cal.json')
            original_hw.save_calibration_to_file(cal_file_path, use_cwd=False)

            empty_hw = Calibratable.load_calibration_from_file(cal_file_path)

            assert 1 == len(empty_hw.qubit_direction_couplings)
            assert coupling_direction == empty_hw.qubit_direction_couplings[0].direction
            assert coupling_quality == empty_hw.qubit_direction_couplings[0].quality

    def test_sweep_pulse_length_exceeds_max_throws_error(self):
        hw = get_test_model()
        qubit = hw.get_qubit(0)
        drive_channel = qubit.get_drive_channel()
        nb_points = 11
        time = np.linspace(0.0, 10 * MaxPulseLength, nb_points)
        builder = (
            get_builder(hw).sweep(
                SweepValue("t", time)
            ).pulse(drive_channel, PulseShapeType.SQUARE,
                    width=Variable("t")).measure_scope_mode(qubit)
        )
        engine = get_test_execution_engine(hw)
        try:
            pytest.raises(ValueError, execute_instructions(engine, builder))
        except ValueError:
            pass

        time = np.linspace(0.0, 100e-6, nb_points)
        builder = (
            get_builder(hw).sweep(
                SweepValue("t", time)
            ).pulse(drive_channel, PulseShapeType.SQUARE,
                    width=Variable("t")).measure_scope_mode(qubit)
        )
        assert nb_points == execute_instructions(engine, builder)[0].shape[0]
