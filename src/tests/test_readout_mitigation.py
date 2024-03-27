from random import random

import numpy as np
import pytest
from numpy import identity, isclose
from numpy.random import rand
from qiskit import QuantumCircuit

from qat.core import execute_qasm
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import (
    RealtimeChipSimEngine,
    get_default_RTCS_hardware,
)
from qat.purr.compiler.config import (
    CompilerConfig,
    ErrorMitigationConfig,
    QuantumResultsFormat,
)
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import SweepIterator
from qat.purr.compiler.hardware_models import ErrorMitigation, ReadoutMitigation


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
    "config_options",
    [
        ["matrix_readout_mitigation"],
        ["linear_readout_mitigation"],
        ["matrix_readout_mitigation", "linear_readout_mitigation"],
    ],
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
            output[str(qubit)] = {
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
                assert result["b"] != result[config]
                assert all([i > 0 for i in result[config].values()])
            else:
                original = result["b"]
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
    ):
        assert len(bitstring) == 2
        circuit = QuantumCircuit(2, 2)
        for i, bit in enumerate(bitstring):
            if bit == "1":
                circuit.x(i)
        circuit.measure(0, 0)
        circuit.measure(1, 1)
        qasm = circuit.qasm()

        # prepare and add readout mitigation to hardware model
        hw = get_default_RTCS_hardware()
        apply_error_mitigation_setup(
            hw, q0_ro_fidelity_0, q0_ro_fidelity_1, q1_ro_fidelity_0, q1_ro_fidelity_1
        )

        # prepare noisy simulator engine
        eng = self.NoisySimulatorEngine(hw)
        eng.fidelity_r0 = [q0_ro_fidelity_0, q1_ro_fidelity_0]
        eng.fidelity_r1 = [q0_ro_fidelity_1, q1_ro_fidelity_1]

        mitigated_result = execute_qasm(qasm, eng, self.config)["linear_readout_mitigation"]
        for output_bits, probability in mitigated_result.items():
            if output_bits == bitstring:
                assert abs(probability - 1) < 0.05
            else:
                assert abs(probability) < 0.05
