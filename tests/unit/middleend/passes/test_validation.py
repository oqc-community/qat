# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import logging
from copy import deepcopy

import numpy as np
import pytest
from compiler_config.config import (
    CompilerConfig,
    ErrorMitigationConfig,
    QuantumResultsFormat,
)

from qat.core.config.configure import get_config
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import PydQuantumInstructionBuilder
from qat.ir.instructions import FrequencySet
from qat.ir.measure import AcquireMode, ProcessAxis
from qat.ir.waveforms import SquareWaveform
from qat.middleend.passes.analysis import ActivePulseChannelResults
from qat.middleend.passes.purr.validation import (
    HardwareConfigValidity,
)
from qat.middleend.passes.validation import (
    PydDynamicFrequencyValidation,
    PydHardwareConfigValidity,
    PydNoMidCircuitMeasurementValidation,
    PydReadoutValidation,
    PydRepeatSanitisationValidation,
)
from qat.model.error_mitigation import ErrorMitigation, ReadoutMitigation
from qat.model.loaders.converted import EchoModelLoader as PydEchoModelLoader
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.instructions import PostProcessType
from qat.utils.hardware_model import generate_hw_model, generate_random_linear

qatconfig = get_config()


class TestPydDynamicFrequencyValidation:
    target_data = TargetData.default()
    hw = PydEchoModelLoader(qubit_count=4).load()

    def get_single_pulse_channel(self):
        return next(iter(self.hw._ids_to_pulse_channels.values()))

    def get_two_pulse_channels_from_single_physical_channel(self):
        device = next(iter(self.hw.quantum_devices))
        pulse_channels = filter(
            lambda ch: ch.frequency is not np.nan, device.all_pulse_channels
        )
        return next(pulse_channels), next(pulse_channels)

    def get_two_pulse_channels_from_different_physical_channels(self):
        devices = iter(self.hw.quantum_devices)
        pulse_channel_1 = next(
            filter(lambda ch: ch.frequency is not np.nan, next(devices).all_pulse_channels)
        )
        pulse_channel_2 = next(
            filter(lambda ch: ch.frequency is not np.nan, next(devices).all_pulse_channels)
        )
        return pulse_channel_1, pulse_channel_2

    def get_baseband_frequency_of_pulse_channel(self, channel_id):
        return self.hw.physical_channel_for_pulse_channel_id(channel_id).baseband.frequency

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
        is_resonator = PydDynamicFrequencyValidation._create_resonator_map(self.hw)
        assert len([val for val in is_resonator.values() if val]) == len(
            [val for val in is_resonator.values() if not val]
        )
        assert len(is_resonator) == len(self.hw._ids_to_physical_channels.keys())

    @pytest.mark.parametrize("method", ["frequency_shift", "frequency_set"])
    def test_violating_if_is_zero(self, method):
        target_data = self.target_data_with_non_zero_if_min()
        channel = self.get_single_pulse_channel()
        qubit_data = target_data.QUBIT_DATA
        assert qubit_data.pulse_channel_if_freq_min > 0.0
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        if method == "frequency_shift":
            delta_freq = 1.1 * qubit_data.pulse_channel_if_freq_min
            builder.frequency_shift(channel, delta_freq)
            builder.frequency_shift(channel, -delta_freq)
        else:
            builder.add(
                FrequencySet(
                    target=channel.uuid,
                    frequency=self.get_baseband_frequency_of_pulse_channel(channel.uuid),
                )
            )
        res_mgr = ResultManager()
        res_mgr.add(ActivePulseChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            PydDynamicFrequencyValidation(self.hw, target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("scale", [-0.95, -0.5, 0.0, 0.5, 0.95])
    def test_raises_no_error_when_freq_shift_in_range(self, scale):
        """Shifts the frequency in the range [min_if, max_if] to check it does not fail."""
        channel = self.get_single_pulse_channel()
        qubit_data = self.target_data.QUBIT_DATA
        target_freq = (
            self.get_baseband_frequency_of_pulse_channel(channel.uuid)
            + scale * qubit_data.pulse_channel_if_freq_max
            - (1 - scale) * qubit_data.pulse_channel_if_freq_min
        )
        delta_freq = target_freq - channel.frequency
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.frequency_shift(channel, delta_freq)
        res_mgr = ResultManager()
        res_mgr.add(ActivePulseChannelResults(target_map={channel: "doesn't matter"}))
        PydDynamicFrequencyValidation(self.hw, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("scale", [-0.95, -0.5, 0.0, 0.5, 0.95])
    def test_raises_no_error_when_freq_set_in_range(self, scale):
        """Sets the frequency in the range [min_if, max_if] to check it does not fail."""
        channel = self.get_single_pulse_channel()
        qubit_data = self.target_data.QUBIT_DATA
        target_freq = (
            self.get_baseband_frequency_of_pulse_channel(channel.uuid)
            + scale * qubit_data.pulse_channel_if_freq_max
            - (1 - scale) * qubit_data.pulse_channel_if_freq_min
        )
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.add(FrequencySet(target=channel.uuid, frequency=target_freq))
        res_mgr = ResultManager()
        res_mgr.add(ActivePulseChannelResults(target_map={channel: "doesn't matter"}))
        PydDynamicFrequencyValidation(self.hw, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_raises_error_when_shifted_lower_than_min(self, sign):
        target_data = self.target_data_with_non_zero_if_min()
        channel = self.get_single_pulse_channel()
        qubit_data = target_data.QUBIT_DATA
        assert qubit_data.pulse_channel_if_freq_min > 0.0
        target_freq = (
            self.get_baseband_frequency_of_pulse_channel(channel.uuid)
            + sign * 0.5 * qubit_data.pulse_channel_if_freq_min
        )
        delta_freq = target_freq - channel.frequency
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.frequency_shift(channel, delta_freq)
        res_mgr = ResultManager()
        res_mgr.add(ActivePulseChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            PydDynamicFrequencyValidation(self.hw, target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_raises_error_when_set_lower_than_min(self, sign):
        target_data = self.target_data_with_non_zero_if_min()
        channel = self.get_single_pulse_channel()
        qubit_data = target_data.QUBIT_DATA
        assert qubit_data.pulse_channel_if_freq_min > 0.0
        target_freq = (
            self.get_baseband_frequency_of_pulse_channel(channel.uuid)
            + sign * 0.5 * qubit_data.pulse_channel_if_freq_min
        )
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.add(FrequencySet(target=channel.uuid, frequency=target_freq))
        res_mgr = ResultManager()
        res_mgr.add(ActivePulseChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            PydDynamicFrequencyValidation(self.hw, target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_raises_error_when_shifted_higher_than_max(self, sign):
        channel = self.get_single_pulse_channel()
        qubit_data = self.target_data.QUBIT_DATA
        target_freq = (
            self.get_baseband_frequency_of_pulse_channel(channel.uuid)
            + sign * 1.1 * qubit_data.pulse_channel_if_freq_max
        )
        delta_freq = target_freq - channel.frequency
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.frequency_shift(channel, delta_freq)
        res_mgr = ResultManager()
        res_mgr.add(ActivePulseChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            PydDynamicFrequencyValidation(self.hw, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_raises_error_when_set_higher_than_max(self, sign):
        channel = self.get_single_pulse_channel()
        qubit_data = self.target_data.QUBIT_DATA
        target_freq = (
            self.get_baseband_frequency_of_pulse_channel(channel.uuid)
            + sign * 1.1 * qubit_data.pulse_channel_if_freq_max
        )
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.add(FrequencySet(target=channel.uuid, frequency=target_freq))
        res_mgr = ResultManager()
        res_mgr.add(ActivePulseChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            PydDynamicFrequencyValidation(self.hw, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_raises_error_when_shifting_out_of_and_back_in_to_range(self, sign):
        channel = self.get_single_pulse_channel()
        qubit_data = self.target_data.QUBIT_DATA
        target_freq = (
            self.get_baseband_frequency_of_pulse_channel(channel.uuid)
            + sign * 1.1 * qubit_data.pulse_channel_if_freq_max
        )
        delta_freq = target_freq - channel.frequency
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.frequency_shift(channel, delta_freq)
        builder.frequency_shift(channel, -delta_freq)
        res_mgr = ResultManager()
        res_mgr.add(ActivePulseChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            PydDynamicFrequencyValidation(self.hw, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_raises_error_when_set_out_of_and_back_in_range(self, sign):
        channel = self.get_single_pulse_channel()
        qubit_data = self.target_data.QUBIT_DATA
        target_freq = (
            self.get_baseband_frequency_of_pulse_channel(channel.uuid)
            + sign * 1.1 * qubit_data.pulse_channel_if_freq_max
        )
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.add(FrequencySet(target=channel.uuid, frequency=target_freq))
        builder.add(FrequencySet(target=channel.uuid, frequency=0.5 * target_freq))
        res_mgr = ResultManager()
        res_mgr.add(ActivePulseChannelResults(target_map={channel: "doesn't matter"}))
        with pytest.raises(ValueError):
            PydDynamicFrequencyValidation(self.hw, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_no_value_error_for_two_channels_same_physical_channel(self, sign):
        channels = self.get_two_pulse_channels_from_single_physical_channel()
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)

        qubit_data = self.target_data.QUBIT_DATA
        delta_freq = 0.05 * (
            qubit_data.pulse_channel_if_freq_max - qubit_data.pulse_channel_if_freq_min
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
            ActivePulseChannelResults(
                target_map={channels[0]: "doesn't matter", channels[1]: "matters even less"}
            )
        )
        PydDynamicFrequencyValidation(self.hw, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_value_error_for_two_channels_same_physical_channel(self, sign):
        channels = self.get_two_pulse_channels_from_single_physical_channel()
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)

        qubit_data = self.target_data.QUBIT_DATA
        delta_freq = 0.05 * (
            qubit_data.pulse_channel_if_freq_max - qubit_data.pulse_channel_if_freq_min
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
            ActivePulseChannelResults(
                target_map={channels[0]: "doesn't matter", channels[1]: "matters even less"}
            )
        )
        with pytest.raises(ValueError):
            PydDynamicFrequencyValidation(self.hw, self.target_data).run(builder, res_mgr)

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_value_error_for_two_channels_different_physical_channel(self, sign):
        channels = self.get_two_pulse_channels_from_different_physical_channels()
        qubit_data = self.target_data.QUBIT_DATA
        delta_freq = 0.05 * (
            qubit_data.pulse_channel_if_freq_max - qubit_data.pulse_channel_if_freq_min
        )

        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)

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
            ActivePulseChannelResults(
                target_map={channels[0]: "doesn't matter", channels[1]: "matters even less"}
            )
        )
        with pytest.raises(ValueError):
            PydDynamicFrequencyValidation(self.hw, self.target_data).run(builder, res_mgr)

    def test_small_change_to_if_is_valid(self):
        """Lil' regression test to test a bug fix for a bug found in pipeline tests.

        Does a little change to the IF frequency - ensures the current IF value is added.
        """

        target_data = self.target_data_with_non_zero_if_min()
        pulse_channel = self.hw.qubits[0].drive_pulse_channel
        pulse_channel.frequency = (
            self.get_baseband_frequency_of_pulse_channel(pulse_channel.uuid) + 1e7
        )
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.frequency_shift(pulse_channel, 1e3)
        builder.pulse(target=pulse_channel.uuid, duration=80e-9, waveform=SquareWaveform())
        res_mgr = ResultManager()
        res_mgr.add(ActivePulseChannelResults(target_map={pulse_channel: "doesn't matter"}))
        PydDynamicFrequencyValidation(self.hw, target_data).run(builder, res_mgr)


class TestPydNoMidCircuitMeasurementValidation:
    hw = PydEchoModelLoader(qubit_count=4).load()

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


class TestHardwareConfigValidity:
    @staticmethod
    def get_hw(legacy=True):
        if legacy:
            return EchoModelLoader(qubit_count=4).load()
        return PydEchoModelLoader(qubit_count=4).load()

    @pytest.mark.parametrize(
        "mitigation_config",
        [
            ErrorMitigationConfig.LinearMitigation,
            ErrorMitigationConfig.MatrixMitigation,
            ErrorMitigationConfig.Empty,
            None,
        ],
    )
    @pytest.mark.parametrize("legacy", [True, False])
    def test_hardware_config_valid_for_all_error_mitigation_settings(
        self, mitigation_config, legacy
    ):
        hw = self.get_hw(legacy=legacy)
        qubit_indices = [i for i in range(4)]
        linear = generate_random_linear(qubit_indices)
        hw.error_mitigation = ErrorMitigation(
            readout_mitigation=ReadoutMitigation(linear=linear)
        )
        compiler_config = CompilerConfig(
            error_mitigation=mitigation_config,
            results_format=QuantumResultsFormat().binary_count(),
        )
        ir = QuantumInstructionBuilder(hardware_model=hw)
        hw_config_valid = HardwareConfigValidity(hw)
        ir_out = hw_config_valid.run(ir, compiler_config=compiler_config)
        assert ir_out == ir

    @pytest.mark.parametrize("hw_mitigation", [None, ErrorMitigation()])
    @pytest.mark.parametrize(
        "mitigation_config",
        [
            ErrorMitigationConfig.LinearMitigation,
            ErrorMitigationConfig.MatrixMitigation,
        ],
    )
    def test_hardware_config_errors_with_incompatible_mitigation_configs(
        self, mitigation_config, hw_mitigation
    ):
        hw = self.get_hw()
        hw.error_mitigation = hw_mitigation
        compiler_config = CompilerConfig(error_mitigation=mitigation_config)
        ir = QuantumInstructionBuilder(hardware_model=hw)
        hw_config_valid = HardwareConfigValidity(hw)
        with pytest.raises(
            ValueError, match="Error mitigation not calibrated on this device."
        ):
            hw_config_valid.run(ir, compiler_config=compiler_config)

    @pytest.mark.parametrize(
        "results_format",
        [
            QuantumResultsFormat().binary(),
            QuantumResultsFormat().raw(),
            QuantumResultsFormat().squash_binary_result_arrays(),
        ],
    )
    @pytest.mark.parametrize(
        "mitigation_config",
        [
            ErrorMitigationConfig.LinearMitigation,
            ErrorMitigationConfig.MatrixMitigation,
        ],
    )
    @pytest.mark.parametrize("legacy", [True, False])
    def test_hardware_config_errors_with_incompatible_results_format(
        self, results_format, mitigation_config, legacy
    ):
        hw = self.get_hw(legacy=legacy)
        qubit_indices = [i for i in range(4)]
        linear = generate_random_linear(qubit_indices)
        hw.error_mitigation = ErrorMitigation(
            readout_mitigation=ReadoutMitigation(linear=linear)
        )
        compiler_config = CompilerConfig(
            error_mitigation=mitigation_config,
            results_format=results_format,
        )
        ir = QuantumInstructionBuilder(hardware_model=hw)
        hw_config_valid = HardwareConfigValidity(hw)
        with pytest.raises(
            ValueError, match="Binary Count format required for readout error mitigation"
        ):
            hw_config_valid.run(ir, compiler_config=compiler_config)


class TestPydHardwareConfigValidity:
    @pytest.mark.parametrize("max_shots", [qatconfig.MAX_REPEATS_LIMIT, None])
    def test_max_shot_limit_exceeded(self, max_shots):
        hw_model = generate_hw_model(n_qubits=8)
        invalid_shots = (
            qatconfig.MAX_REPEATS_LIMIT + 1 if max_shots is None else max_shots + 1
        )

        comp_config = CompilerConfig(repeats=invalid_shots)
        ir = "test"
        res_mgr = ResultManager()

        validation_pass = PydHardwareConfigValidity(hw_model, max_shots=max_shots)
        with pytest.raises(ValueError):
            validation_pass.run(ir, res_mgr, compiler_config=comp_config)

        comp_config = CompilerConfig(repeats=invalid_shots - 1)
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


class TestReadoutValidation:
    hw = PydEchoModelLoader(qubit_count=4).load()

    def test_valid_readout(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.measure_single_shot_z(target=qubit)

        PydReadoutValidation().run(builder)

    @pytest.mark.parametrize(
        "acquire_mode, process_axis",
        [
            (AcquireMode.SCOPE, ProcessAxis.SEQUENCE),
            (AcquireMode.INTEGRATOR, ProcessAxis.TIME),
            (AcquireMode.RAW, None),
        ],
    )
    def test_acquire_with_invalid_pp(self, acquire_mode, process_axis):
        qubit = self.hw.qubits[0]

        output_variable = (
            "out_" + qubit.uuid + f"_{np.random.randint(np.iinfo(np.int32).max)}"
        )
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.acquire(qubit, mode=acquire_mode, output_variable=output_variable)
        builder.post_processing(
            target=qubit,
            output_variable=output_variable,
            process_type=PostProcessType.MEAN,
            axes=process_axis,
        )

        with pytest.raises(ValueError, match="Invalid"):
            PydReadoutValidation().run(builder)

    def test_acquire_without_matching_output_variable(self):
        qubit = self.hw.qubits[0]

        output_variable_1 = (
            "out_" + qubit.uuid + f"_{np.random.randint(np.iinfo(np.int32).max)}"
        )
        output_variable_2 = output_variable_1 + "_2"
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.acquire(
            qubit, mode=AcquireMode.INTEGRATOR, output_variable=output_variable_1
        )
        builder.post_processing(
            target=qubit,
            output_variable=output_variable_2,
            process_type=PostProcessType.MEAN,
            axes=ProcessAxis.SEQUENCE,
        )

        with pytest.raises(ValueError, match="No AcquireMode found"):
            PydReadoutValidation().run(builder)


class TestRepeatSanitisationValidation:
    hw = PydEchoModelLoader(qubit_count=4).load()
    warning_msg = "Could not find any repeat instructions."

    def test_repeat_sanitisation_validation_doesnt_edit_instructions(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.measure_single_shot_z(target=qubit)

        p = PydRepeatSanitisationValidation()
        builder_before = deepcopy(builder)
        p.run(builder)

        assert builder_before.number_of_instructions == builder.number_of_instructions
        for instr_before, instr_after in zip(builder_before, builder):
            assert instr_before == instr_after

    def test_repeat_sanitisation_validation_logs_warning_when_repeat_is_not_present(
        self, caplog
    ):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        caplog.set_level(logging.WARNING)
        for qubit in self.hw.qubits.values():
            builder.measure_single_shot_z(target=qubit)
        PydRepeatSanitisationValidation().run(builder)

        assert any(cl.message == self.warning_msg for cl in caplog.records)

    def test_repeat_sanitisation_validation_doesnt_log_warning_when_repeat_is_present(
        self, caplog
    ):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        caplog.set_level(logging.WARNING)
        builder.repeat(10)
        for qubit in self.hw.qubits.values():
            builder.measure_single_shot_z(target=qubit)
        PydRepeatSanitisationValidation().run(builder)

        assert all(cl.message != self.warning_msg for cl in caplog.records)
