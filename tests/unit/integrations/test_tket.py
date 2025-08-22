# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import random

import pytest
from compiler_config.config import Qasm2Optimizations, Qasm3Optimizations, Tket

from qat.integrations.tket import TketOptimisationHelper, run_pyd_tket_optimizations
from qat.model.error_mitigation import ErrorMitigation, ReadoutMitigation
from qat.model.loaders.converted import PydEchoModelLoader
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


def test_one_qubit_circuit_optimisation():
    hw = PydEchoModelLoader(4).load()

    linear = {}
    qubit_qualities = {}
    for q_id, qubit in hw.qubits.items():
        p00 = random.uniform(0.8, 1.0)
        p11 = random.uniform(0.8, 1.0)
        linear[q_id] = [[p00, 1 - p11], [1 - p00, p11]]
        qubit_qualities[q_id] = (p00 + p11) / 2
    best_qubit = max(qubit_qualities, key=qubit_qualities.get)

    hw.error_mitigation = ErrorMitigation(
        readout_mitigation=ReadoutMitigation(linear=linear)
    )

    qasm_1q_x_input = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    x q;
    measure q -> c;
    """
    opts = Tket().default()
    optimiser = TketOptimisationHelper(qasm_1q_x_input, opts, hw)
    optimiser.run_one_qubit_optimizations()
    circ = optimiser.circ
    assert best_qubit == circ.qubits[0].index[0]


def test_two_qubit_circuit_optimisation():
    hw = PydEchoModelLoader(4).load()
    linear = {}
    qubit_qualities = {}
    for q_id, qubit in hw.qubits.items():
        p00 = random.uniform(0.8, 1.0)
        p11 = random.uniform(0.8, 1.0)
        linear[q_id] = [[p00, 1 - p11], [1 - p00, p11]]
        qubit_qualities[q_id] = (p00 + p11) / 2

    hw.error_mitigation = ErrorMitigation(
        readout_mitigation=ReadoutMitigation(linear=linear)
    )

    qubit_pair_qualities = {}
    for q, coupled_qs in hw.logical_connectivity.items():
        for coupled_q in coupled_qs:
            quality = hw.logical_connectivity_quality[(q, coupled_q)]
            qubit_quality = hw.qubit_quality(q)
            coupled_qubit_quality = hw.qubit_quality(coupled_q)
            quality *= qubit_quality * coupled_qubit_quality
            qubit_pair_qualities[(q, coupled_q)] = quality

    best_qubit_pair = max(qubit_pair_qualities, key=qubit_pair_qualities.get)

    qasm_2q_bell_input = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];

    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """
    opts = Tket().default()
    optimiser = TketOptimisationHelper(qasm_2q_bell_input, opts, hw)
    optimiser.run_multi_qubit_optimizations(use_1q_quality=True)
    circ = optimiser.circ

    circ_indices = [idx for qubit in circ.qubits for idx in qubit.index]

    for qubit in best_qubit_pair:
        assert qubit in circ_indices
