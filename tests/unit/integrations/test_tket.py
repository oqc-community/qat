# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest
from compiler_config.config import Qasm2Optimizations, Qasm3Optimizations

from qat.integrations.tket import run_pyd_tket_optimizations
from qat.utils.hardware_model import generate_hw_model

from tests.unit.utils.qasm_qir import get_qasm2, get_qasm3

qasm2_files = [
    "basic.qasm",
    "basic_results_formats.qasm",
    "basic_single_measures.qasm",
    "ecr_exists.qasm",
    "ecr.qasm",
    "invalid_mid_circuit_measure.qasm",
    "logic_example.qasm",
    "more_basic.qasm",
    "primitives.qasm",
    "valid_custom_gate.qasm",
]
qasm3_files = [
    "basic.qasm",
    "u_gate.qasm",
    "ecr_test.qasm",
    "ghz.qasm",
    "complex_gates_test.qasm",
    "arb_waveform.qasm",
]


@pytest.mark.parametrize("n_qubits", [4, 8, 32, 64])
@pytest.mark.parametrize("seed", [7, 8, 9])
class TestPydTketOptimisation:
    @pytest.mark.parametrize("qasm_file", qasm2_files)
    def test_pyd_tket_qasm2_optimization(self, n_qubits, seed, qasm_file):
        qasm_string = get_qasm2(qasm_file)
        hw_model = generate_hw_model(n_qubits, seed=seed)

        run_pyd_tket_optimizations(
            qasm_string, opts=Qasm2Optimizations(), hardware=hw_model
        )

    @pytest.mark.parametrize("qasm_file", qasm3_files)
    def test_pyd_tket_qasm3_optimization(self, n_qubits, seed, qasm_file):
        qasm_string = get_qasm3(qasm_file)
        hw_model = generate_hw_model(n_qubits, seed=seed)

        run_pyd_tket_optimizations(
            qasm_string, opts=Qasm3Optimizations(), hardware=hw_model
        )
