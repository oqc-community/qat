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

from qat.core.config.configure import get_config
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import PydQuantumInstructionBuilder
from qat.ir.measure import AcquireMode, ProcessAxis
from qat.middleend.passes.purr.validation import (
    HardwareConfigValidity,
)
from qat.middleend.passes.validation import (
    PydHardwareConfigValidity,
    PydNoMidCircuitMeasurementValidation,
    PydReadoutValidation,
)
from qat.model.error_mitigation import ErrorMitigation, ReadoutMitigation
from qat.model.loaders.converted import EchoModelLoader as PydEchoModelLoader
from qat.model.loaders.purr import EchoModelLoader
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.instructions import PostProcessType
from qat.utils.hardware_model import generate_hw_model, generate_random_linear

qatconfig = get_config()


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
