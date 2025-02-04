# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

import numpy as np
import pytest
from compiler_config.config import (
    CompilerConfig,
    ErrorMitigationConfig,
    QuantumResultsFormat,
)

from qat import qatconfig
from qat.backend.validation_passes import (
    FrequencyValidation,
    NCOFrequencyVariability,
    NoAcquireWeightsValidation,
    NoMultipleAcquiresValidation,
    PydHardwareConfigValidity,
)
from qat.ir.pass_base import QatIR
from qat.ir.result_base import ResultManager
from qat.model.error_mitigation import ErrorMitigation, ReadoutMitigation
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.instructions import Pulse, PulseShapeType

from tests.qat.utils.builder_nuggets import resonator_spect
from tests.qat.utils.hardware_models import generate_hw_model, generate_random_linear


class TestValidationPasses:
    def test_nco_freq_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()
        ir = QatIR(builder)

        NCOFrequencyVariability().run(ir, res_mgr, model)

        channel = next(iter(model.pulse_channels.values()))
        channel.fixed_if = True

        with pytest.raises(ValueError):
            NCOFrequencyVariability().run(ir, res_mgr, model)

        channel.fixed_if = False
        NCOFrequencyVariability().run(ir, res_mgr, model)


class TestFrequencyValidation:
    res_mgr = ResultManager()
    model = get_default_echo_hardware()

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

    def set_frequency_range(self, pulse_channel, lower_tol, upper_tol):
        phys_chan = pulse_channel.physical_channel
        phys_chan.pulse_channel_max_frequency = pulse_channel.frequency + upper_tol
        phys_chan.pulse_channel_min_frequency = pulse_channel.frequency - lower_tol

    @pytest.mark.parametrize("freq", [-1e-9, -1e-8, 0, 1e8, 1e9])
    def test_raises_no_error_when_freq_shift_in_range(self, freq):
        channel = self.get_single_pulse_channel()
        self.set_frequency_range(channel, 1e9, 1e9)
        builder = self.model.create_builder()
        builder.frequency_shift(channel, freq)
        FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

    @pytest.mark.parametrize("freq", [-1e9, 1e9])
    def test_raises_value_error_when_freq_shift_out_of_range(self, freq):
        channel = self.get_single_pulse_channel()
        self.set_frequency_range(channel, 1e8, 1e8)
        builder = self.model.create_builder()
        builder.frequency_shift(channel, freq)
        with pytest.raises(ValueError):
            FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

    @pytest.mark.parametrize("freq", [-2e8, 2e8])
    def test_moves_out_and_in_raises_value_error(self, freq):
        channel = self.get_single_pulse_channel()
        self.set_frequency_range(channel, 1e8, 1e8)
        builder = self.model.create_builder()
        builder.frequency_shift(channel, freq)
        builder.frequency_shift(channel, -freq)
        with pytest.raises(ValueError):
            FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_no_value_error_for_two_channels_same_physical_channel(self, sign):
        channels = self.get_two_pulse_channels_from_single_physical_channel()
        self.set_frequency_range(channels[0], 1e9, 1e9)
        builder = self.model.create_builder()

        # interweave with random instructions
        builder.phase_shift(channels[0], np.pi)
        builder.frequency_shift(channels[0], sign * 1e8)
        builder.phase_shift(channels[1], -np.pi)
        builder.frequency_shift(channels[1], sign * 2e8)
        builder.frequency_shift(channels[0], sign * 3e8)
        builder.phase_shift(channels[1], -2.54)
        builder.frequency_shift(channels[1], sign * 4e8)
        FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_value_error_for_two_channels_same_physical_channel(self, sign):
        channels = self.get_two_pulse_channels_from_single_physical_channel()
        self.set_frequency_range(channels[0], 1e9, 1e9)
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
            FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_no_value_error_for_two_channels_different_physical_channel(self, sign):
        channels = self.get_two_pulse_channels_from_different_physical_channels()
        self.set_frequency_range(channels[0], 1e9, 1e9)
        self.set_frequency_range(channels[1], 5e8, 5e8)
        builder = self.model.create_builder()
        builder.frequency_shift(channels[0], sign * 4e8)
        builder.frequency_shift(channels[1], sign * 1e8)
        builder.frequency_shift(channels[0], sign * 3e8)
        builder.frequency_shift(channels[1], sign * 4e8)
        FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_value_error_for_two_channels_different_physical_channel(self, sign):
        channels = self.get_two_pulse_channels_from_different_physical_channels()
        self.set_frequency_range(channels[0], 1e9, 1e9)
        self.set_frequency_range(channels[1], 5e8, 5e8)
        builder = self.model.create_builder()
        builder.frequency_shift(channels[0], sign * 4e8)
        builder.frequency_shift(channels[1], sign * 1e8)
        builder.frequency_shift(channels[0], sign * 7e8)
        builder.frequency_shift(channels[1], sign * 4e8)
        with pytest.raises(ValueError):
            FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

    def test_fixed_if_raises_not_implemented_error(self):
        channels = self.get_two_pulse_channels_from_different_physical_channels()
        builder = self.model.create_builder()
        channels[0].fixed_if = True
        builder.frequency_shift(channels[0], 1e8)
        builder.frequency_shift(channels[1], 1e8)
        with pytest.raises(NotImplementedError):
            FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

    def test_fixed_if_does_not_affect_other_channel(self):
        channels = self.get_two_pulse_channels_from_different_physical_channels()
        builder = self.model.create_builder()
        channels[0].fixed_if = True
        builder.frequency_shift(channels[1], 1e8)
        FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)


class TestNoAcquireWeightsValidation:

    def test_acquire_with_filter_raises_error(self):
        model = get_default_echo_hardware()
        res_mgr = ResultManager()
        qubit = model.get_qubit(0)
        channel = qubit.get_acquire_channel()
        builder = model.create_builder()
        builder.acquire(
            channel, delay=0.0, filter=Pulse(channel, PulseShapeType.SQUARE, 1e-6)
        )
        with pytest.raises(NotImplementedError):
            NoAcquireWeightsValidation().run(QatIR(builder), res_mgr)


class TestNoMultipleAcquiresValidation:

    def test_multiple_acquires_on_same_pulse_channel_raises_error(self):
        model = get_default_echo_hardware()
        res_mgr = ResultManager()
        qubit = model.get_qubit(0)
        channel = qubit.get_acquire_channel()
        builder = model.create_builder()
        builder.acquire(channel, delay=0.0)

        # Test should run as there is only one acquire
        NoMultipleAcquiresValidation().run(QatIR(builder), res_mgr)

        # Add another acquire and test it breaks it
        builder.acquire(channel, delay=0.0)
        with pytest.raises(NotImplementedError):
            NoMultipleAcquiresValidation().run(QatIR(builder), res_mgr)

    def test_multiple_acquires_that_share_physical_channel_raises_error(self):
        model = get_default_echo_hardware()
        res_mgr = ResultManager()
        qubit = model.get_qubit(0)
        acquire_channel = qubit.get_acquire_channel()
        # in practice, we shouldn't do this with a measure channel, but convinient for
        # testing
        measure_channel = qubit.get_measure_channel()
        builder = model.create_builder()
        builder.acquire(acquire_channel, delay=0.0)
        builder.acquire(measure_channel, delay=0.0)
        with pytest.raises(NotImplementedError):
            NoMultipleAcquiresValidation().run(QatIR(builder), res_mgr)


# Test passes for the Pydantic hardware model.


class TestPydHardwareConfigValidity:

    def test_max_shot_limit_exceeded(self):
        hw_model = generate_hw_model(n_qubits=8)
        comp_config = CompilerConfig(repeats=qatconfig.MAX_REPEATS_LIMIT + 1)
        ir = QatIR("test")
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
        ir = QatIR("test")
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
