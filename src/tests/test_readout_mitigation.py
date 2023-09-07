from random import random

import pytest
from numpy import identity, isclose
from numpy.random import rand
from qat import execute_qasm
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.config import (
    CompilerConfig,
    ErrorMitigationConfig,
    QuantumResultsFormat,
)
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.hardware_models import ErrorMitigation, ReadoutMitigation
from qat.purr.compiler.runtime import get_builder
from qat.purr.error_mitigation.readout_mitigation import ApplyReadoutMitigation
from qat.purr.integrations.qasm import Qasm2Parser


@pytest.mark.parametrize(
    "config_options",
    [[
        "matrix_readout_mitigation",
    ], ["linear_readout_mitigation"],
     ["matrix_readout_mitigation", "linear_readout_mitigation"]]
)
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

    def build_hardware(self, qubit_count, error_mitigation):
        hw = get_default_echo_hardware(qubit_count)
        hw.error_mitigation = error_mitigation
        return hw

    def generate_random_linear(self, qubit_count, random_data=True):
        output = {}
        for qubit in range(qubit_count):
            random_0 = random() if random_data else 1
            random_1 = random() if random_data else 1
            output[qubit] = {
                "0|0": random_0, "1|0": 1 - random_0, "1|1": random_1, "0|1": 1 - random_1
            }
        return output

    def build_config(self, configs):
        config_map = {
            "matrix_readout_mitigation": ErrorMitigationConfig.MatrixMitigation,
            "linear_readout_mitigation": ErrorMitigationConfig.LinearMitigation
        }
        mitigation_config = ErrorMitigationConfig.Empty

        for config in configs:
            mitigation_config |= config_map[config]
        compiler_config = CompilerConfig(
            results_format=QuantumResultsFormat().binary_count(),
            error_mitigation=mitigation_config
        )
        return compiler_config

    def build_hardware_with_options(self, qubit_count, random_cal, config_options):
        matrix = None
        linear = None
        if "matrix_readout_mitigation" in config_options:
            if random_cal:
                matrix = rand(2**qubit_count, 2**qubit_count)
            else:
                matrix = identity(2**qubit_count)
        if "linear_readout_mitigation":
            linear = self.generate_random_linear(qubit_count, random_cal)
        hw = self.build_hardware(
            qubit_count, self.build_error_mitigation(matrix=matrix, linear=linear)
        )
        return hw

    @pytest.mark.parametrize("qubit_count", [i for i in range(2, 9)])
    @pytest.mark.parametrize("random_cal", [True, False])
    def test_something_changes_qasm(self, qubit_count, random_cal, config_options):
        hw = self.build_hardware_with_options(qubit_count, random_cal, config_options)
        compiler_config = self.build_config(config_options)
        result = execute_qasm(
            self.get_qasm(qubit_count=qubit_count), hw, compiler_config=compiler_config
        )
        for config in config_options:
            if random_cal:
                assert result['original'] != result[config]
                assert all([i > 0 for i in result[config].values()])
            else:
                original = result['original']['b']
                assert original["0" * qubit_count] == 1000.0
                mitigated = result[config]
                for key, value in mitigated.items():
                    if key == "0" * qubit_count:
                        assert isclose(value, 1.0)
                    else:
                        assert isclose(value, 0.0)

    def test_multiple_creg_fail(self, config_options):
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
        hw = self.build_hardware_with_options(2, False, config_options)
        compiler_config = self.build_config(config_options)
        with pytest.raises(ValueError):
            execute_qasm(qasm, hw, compiler_config=compiler_config)

    def test_non_binary_count_format_fails(self, config_options):
        hw = self.build_hardware_with_options(2, False, config_options)
        compiler_config = self.build_config(config_options)
        compiler_config.results_format = QuantumResultsFormat().raw()
        with pytest.raises(ValueError):
            execute_qasm(self.get_qasm(2), hw, compiler_config=compiler_config)

    def test_lack_of_hardware_calibration_fails(self, config_options):
        hw = self.build_hardware_with_options(2, False, [])
        compiler_config = self.build_config(config_options)
        compiler_config.results_format = QuantumResultsFormat().raw()
        with pytest.raises(ValueError):
            execute_qasm(self.get_qasm(2), hw, compiler_config=compiler_config)


qasm1 = (
    """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg b[2];
        measure q[0] -> b[1];
        measure q[1] -> b[0];
        """, {
        "0": 1, "1": 0
    }
)
qasm2 = (
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg b[3];
    measure q[0] -> b[2];
    measure q[1] -> b[1];
    measure q[2] -> b[0];
    """, {
        "0": 2, "1": 1, "2": 0
    }
)
qasm3 = (
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg b[3];
    measure q -> b;
    """, {
        "0": 0, "1": 1, "2": 2
    }
)


class TestMapping:
    @pytest.mark.parametrize("qasm_map", [qasm1, qasm2, qasm3])
    def test_mapping(self, qasm_map):
        hw = get_default_echo_hardware(3)
        parser = Qasm2Parser()
        result = parser.parse(get_builder(hw), qasm_map[0])
        instructions = result.instructions
        qat_file = InstructionEmitter().emit(instructions, hw)
        mapping, _ = ApplyReadoutMitigation().get_mapping_and_qubits(qat_file)
        assert mapping == qasm_map[1]
