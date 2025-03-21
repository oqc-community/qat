# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from random import random, seed

import numpy as np
import pytest
from compiler_config.config import (
    CompilerConfig,
    ErrorMitigationConfig,
    QuantumResultsFormat,
)
from numpy import identity, isclose
from numpy.random import rand
from qiskit import QuantumCircuit, qasm2

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.backends.realtime_chip_simulator import (
    RealtimeChipSimEngine,
    get_default_RTCS_hardware,
)
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import SweepIterator
from qat.purr.compiler.hardware_models import (
    ErrorMitigation,
    ReadoutMitigation,
    get_cl2qu_index_mapping,
)
from qat.purr.compiler.runtime import get_builder
from qat.purr.integrations.qasm import Qasm2Parser
from qat.purr.qat import execute_qasm

from tests.unit.utils.models import get_jagged_echo_hardware


def apply_error_mitigation_setup(
    hw, q0_ro_fidelity_0, q0_ro_fidelity_1, q1_ro_fidelity_0, q1_ro_fidelity_1
):
    def lin_mit_dict_from_fidelity(ro_fidelity_0, ro_fidelity_1):
        return {
            "0|0": ro_fidelity_0,
            "1|0": 1 - ro_fidelity_0,
            "0|1": 1 - ro_fidelity_1,
            "1|1": ro_fidelity_1,
        }

    lin_mit = {
        "0": lin_mit_dict_from_fidelity(q0_ro_fidelity_0, q0_ro_fidelity_1),
        "1": lin_mit_dict_from_fidelity(q1_ro_fidelity_0, q1_ro_fidelity_1),
    }
    hw.error_mitigation = ErrorMitigation(ReadoutMitigation(linear=lin_mit))

    return hw


@pytest.mark.parametrize(
    "model",
    [get_default_echo_hardware(), get_default_RTCS_hardware(), get_jagged_echo_hardware()],
)
class TestErrorMitigationValidation:

    def apply_mitigation(
        self,
        hw,
        q0_ro_fidelity_0=0.8,
        q0_ro_fidelity_1=0.7,
        q1_ro_fidelity_0=0.9,
        q1_ro_fidelity_1=0.6,
    ):
        hw = apply_error_mitigation_setup(
            hw, q0_ro_fidelity_0, q0_ro_fidelity_1, q1_ro_fidelity_0, q1_ro_fidelity_1
        )
        compiler_config = CompilerConfig(
            results_format=QuantumResultsFormat().binary_count(),
            error_mitigation=ErrorMitigationConfig.LinearMitigation,
        )
        return hw, compiler_config

    def test_hardware_validation(self, model):
        model, compiler_config = self.apply_mitigation(model)
        compiler_config.validate(model)
        assert True

    def test_engine_validation(self, model):
        model, compiler_config = self.apply_mitigation(model)
        engine = model.create_engine()
        compiler_config.validate(engine)
        assert True


class TestReadoutMitigation:
    def get_qasm(self, qubit_count):
        return f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[{qubit_count}];
        creg b[{qubit_count}];
        h q[0];
        cx q[0], q[1];
        measure q -> b;
        """

    def build_error_mitigation(self, linear=None, matrix=None, m3=None):
        readout = ReadoutMitigation(linear=linear, matrix=matrix, m3=m3)
        return ErrorMitigation(readout_mitigation=readout)

    def generate_random_linear(self, qubit_indices, random_data=True):
        output = {}
        for index in qubit_indices:
            random_0 = random() if random_data else 1
            random_1 = random() if random_data else 1
            output[str(index)] = {
                "0|0": random_0,
                "1|0": 1 - random_0,
                "1|1": random_1,
                "0|1": 1 - random_1,
            }
        return output

    def build_config(self, configs):
        config_map = {
            "matrix_readout_mitigation": ErrorMitigationConfig.MatrixMitigation,
            "linear_readout_mitigation": ErrorMitigationConfig.LinearMitigation,
        }
        mitigation_config = ErrorMitigationConfig.Empty

        for config in configs:
            mitigation_config |= config_map[config]
        compiler_config = CompilerConfig(
            results_format=QuantumResultsFormat().binary_count(),
            error_mitigation=mitigation_config,
        )
        return compiler_config

    def apply_hardware_options(self, hardware, random_cal, config_options):
        matrix = None
        linear = None
        qubit_indices = [qubit.index for qubit in hardware.qubits]
        qubit_count = len(qubit_indices)
        if "matrix_readout_mitigation" in config_options:
            if random_cal:
                matrix = rand(2**qubit_count, 2**qubit_count)
            else:
                matrix = identity(2**qubit_count)
        if "linear_readout_mitigation":
            linear = self.generate_random_linear(qubit_indices, random_cal)
        hardware.error_mitigation = self.build_error_mitigation(
            matrix=matrix, linear=linear
        )


@pytest.mark.parametrize(
    "get_hardware",
    [
        get_default_echo_hardware,
        get_default_qiskit_hardware,
        get_jagged_echo_hardware,
    ],
)
@pytest.mark.parametrize(
    "config_options",
    [
        ["linear_readout_mitigation"],
    ],
)
class TestLinearReadoutMitigation(TestReadoutMitigation):
    @pytest.mark.parametrize("qubit_count", [i for i in range(2, 9)])
    @pytest.mark.parametrize("random_cal", [True, False])
    def test_something_changes_qasm(
        self, get_hardware, qubit_count, random_cal, config_options
    ):
        hw = get_hardware(qubit_count)
        self.apply_hardware_options(hw, random_cal, config_options)
        compiler_config = self.build_config(config_options)
        result = execute_qasm(
            self.get_qasm(qubit_count=qubit_count), hw, compiler_config=compiler_config
        )
        for config in config_options:
            if random_cal:
                assert result["b"] != result[config]
                assert all([i > 0 for i in result[config].values()])
            else:
                original = result["b"]
                zero = "0" * qubit_count
                one = "11" + zero[2:]
                assert sum([original.get(zero, 0), original.get(one, 0)]) == 1000.0
                mitigated = result[config]
                for key, value in mitigated.items():
                    if key in original:
                        assert isclose(value, original[key] / 1000.0)
                    else:
                        assert isclose(value, 0.0)

    def test_multiple_creg_fail(self, get_hardware, config_options):
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg b[1];
        creg c[1];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> b[0];
        measure q[1] -> c[0];
        """
        hw = get_hardware(2)
        self.apply_hardware_options(hw, False, config_options)
        compiler_config = self.build_config(config_options)
        with pytest.raises(ValueError):
            execute_qasm(qasm, hw, compiler_config=compiler_config)

    def test_non_binary_count_format_fails(self, get_hardware, config_options):
        hw = get_hardware(2)
        self.apply_hardware_options(hw, False, config_options)
        compiler_config = self.build_config(config_options)
        compiler_config.results_format = QuantumResultsFormat().raw()
        with pytest.raises(ValueError):
            execute_qasm(self.get_qasm(2), hw, compiler_config=compiler_config)

    def test_lack_of_hardware_calibration_fails(self, get_hardware, config_options):
        hw = get_hardware(2)
        self.apply_hardware_options(hw, False, [])
        compiler_config = self.build_config(config_options)
        compiler_config.results_format = QuantumResultsFormat().raw()
        with pytest.raises(ValueError):
            execute_qasm(self.get_qasm(2), hw, compiler_config=compiler_config)


# TODO - Add jagged hardware for Matrix error mitigation tests when they're fixed
@pytest.mark.parametrize(
    "get_hardware",
    [get_default_echo_hardware, get_default_qiskit_hardware],
)
@pytest.mark.parametrize(
    "config_options",
    [
        ["matrix_readout_mitigation"],
        ["matrix_readout_mitigation", "linear_readout_mitigation"],
    ],
)
class TestMatrixReadoutMitigation(TestReadoutMitigation):
    @pytest.mark.parametrize("qubit_count", [i for i in range(2, 9)])
    @pytest.mark.parametrize("random_cal", [True, False])
    def test_something_changes_qasm(
        self, get_hardware, qubit_count, random_cal, config_options
    ):
        hw = get_hardware(qubit_count)
        self.apply_hardware_options(hw, random_cal, config_options)
        compiler_config = self.build_config(config_options)
        result = execute_qasm(
            self.get_qasm(qubit_count=qubit_count), hw, compiler_config=compiler_config
        )
        for config in config_options:
            if random_cal:
                assert result["b"] != result[config]
                assert all([i > 0 for i in result[config].values()])
            else:
                original = result["b"]
                zero = "0" * qubit_count
                one = "11" + zero[2:]
                assert sum([original.get(zero, 0), original.get(one, 0)]) == 1000.0
                mitigated = result[config]
                for key, value in mitigated.items():
                    if key in original:
                        assert isclose(value, original[key] / 1000.0)
                    else:
                        assert isclose(value, 0.0)

    def test_multiple_creg_fail(self, get_hardware, config_options):
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg b[1];
        creg c[1];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> b[0];
        measure q[1] -> c[0];
        """
        hw = get_hardware(2)
        self.apply_hardware_options(hw, False, config_options)
        compiler_config = self.build_config(config_options)
        with pytest.raises(ValueError):
            execute_qasm(qasm, hw, compiler_config=compiler_config)

    def test_non_binary_count_format_fails(self, get_hardware, config_options):
        hw = get_hardware(2)
        self.apply_hardware_options(hw, False, config_options)
        compiler_config = self.build_config(config_options)
        compiler_config.results_format = QuantumResultsFormat().raw()
        with pytest.raises(ValueError):
            execute_qasm(self.get_qasm(2), hw, compiler_config=compiler_config)

    def test_lack_of_hardware_calibration_fails(self, get_hardware, config_options):
        hw = get_hardware(2)
        self.apply_hardware_options(hw, False, [])
        compiler_config = self.build_config(config_options)
        compiler_config.results_format = QuantumResultsFormat().raw()
        with pytest.raises(ValueError):
            execute_qasm(self.get_qasm(2), hw, compiler_config=compiler_config)


@pytest.mark.parametrize(
    "get_hardware",
    [
        get_default_echo_hardware,
        get_default_qiskit_hardware,
        get_jagged_echo_hardware,
    ],
)
@pytest.mark.parametrize(
    "error_mitigation",
    [None, ErrorMitigationConfig.Empty],
)
class TestNoReadoutMitigation:
    def get_qasm(self, qubit_count):
        return f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[{qubit_count}];
        creg b[{qubit_count}];
        h q[0];
        cx q[0], q[1];
        measure q -> b;
        """

    def test_execution(self, get_hardware, error_mitigation):
        hw = get_hardware(2)
        config = CompilerConfig(
            results_format=QuantumResultsFormat().binary_count(),
            error_mitigation=error_mitigation,
            repeats=10000,
        )
        results = execute_qasm(self.get_qasm(2), hw, compiler_config=config)
        assert results is not None
        assert len(results) == 1


class TestOnNoisySimulator:
    config = CompilerConfig(
        results_format=QuantumResultsFormat().binary_count(),
        error_mitigation=ErrorMitigationConfig.LinearMitigation,
        repeats=10000,
    )

    class NoisySimulatorEngine(RealtimeChipSimEngine):
        def __init__(self, model=None, auto_plot=False, sim_qubit_dt=0.25e-10):
            super().__init__(model, auto_plot, sim_qubit_dt)
            self.fidelity_r0 = {qubit.index: 1.0 for qubit in self.model.qubits}
            self.fidelity_r1 = {qubit.index: 1.0 for qubit in self.model.qubits}

        def _execute_on_hardware(
            self, sweep_iterator: SweepIterator, package: QatFile, interrupt=None
        ):
            result = super()._execute_on_hardware(sweep_iterator, package)
            for key in result.keys():
                q = int(key.split("_", 1)[1])
                for array in result[key]:
                    for j in range(len(array)):
                        if (array[j] > 0 and np.random.rand() > self.fidelity_r0[q]) or (
                            array[j] < 0 and np.random.rand() > self.fidelity_r1[q]
                        ):
                            array[j] *= -1
            return result

    @pytest.mark.parametrize("bitstring", ["00", "01", "10", "11"])
    def test_prepare_bitstring_and_measure(
        self,
        bitstring,
        q0_ro_fidelity_0=0.8,
        q0_ro_fidelity_1=0.7,
        q1_ro_fidelity_0=0.9,
        q1_ro_fidelity_1=0.6,
        threshold=0.051,
    ):
        assert len(bitstring) == 2
        # Fix the seed for the experiment
        np.random.seed(0)
        seed(0)

        # Create the circuit
        circuit = QuantumCircuit(2, 2)
        for i, bit in enumerate(bitstring):
            if bit == "1":
                circuit.x(i)
        circuit.measure(0, 0)
        circuit.measure(1, 1)
        qasm = qasm2.dumps(circuit)

        # prepare and add readout mitigation to hardware model
        hw = get_default_RTCS_hardware()
        apply_error_mitigation_setup(
            hw, q0_ro_fidelity_0, q0_ro_fidelity_1, q1_ro_fidelity_0, q1_ro_fidelity_1
        )

        # prepare noisy simulator engine
        eng = self.NoisySimulatorEngine(hw)
        eng.fidelity_r0 = [q0_ro_fidelity_0, q1_ro_fidelity_0]
        eng.fidelity_r1 = [q0_ro_fidelity_1, q1_ro_fidelity_1]

        # Run the simulation
        mitigated_result = execute_qasm(qasm, eng, self.config)["linear_readout_mitigation"]

        # reset the seed
        np.random.seed(None)
        seed(None)
        for output_bits, probability in mitigated_result.items():
            if output_bits == bitstring:
                assert abs(1 - probability) < threshold
            else:
                assert abs(probability) < threshold


mapping_setup1 = (
    """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg b[2];
        measure q[0] -> b[1];
        measure q[1] -> b[0];
        """,
    {"0": 1, "1": 0},
)
mapping_setup2 = (
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg b[3];
    measure q[0] -> b[2];
    measure q[1] -> b[1];
    measure q[2] -> b[0];
    """,
    {"0": 2, "1": 1, "2": 0},
)
mapping_setup3 = (
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg b[3];
    measure q -> b;
    """,
    {"0": 0, "1": 1, "2": 2},
)


@pytest.mark.parametrize(
    "qasm_string, expected_mapping", [mapping_setup1, mapping_setup2, mapping_setup3]
)
def test_cl2qu_index_mapping(qasm_string, expected_mapping):
    hw = get_default_echo_hardware(3)
    parser = Qasm2Parser()
    result = parser.parse(get_builder(hw), qasm_string)
    mapping = get_cl2qu_index_mapping(result.instructions, hw)
    assert mapping == expected_mapping

    blob = result.serialize()
    result2 = InstructionBuilder.deserialize(blob)
    mapping = get_cl2qu_index_mapping(result2.instructions, hw)
    assert mapping == expected_mapping
