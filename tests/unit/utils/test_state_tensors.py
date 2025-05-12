# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np

from qat.ir.gates.gates_1q import Hadamard, X, Y
from qat.ir.gates.gates_2q import CNOT
from qat.utils.state_tensors import StateOperator, StateVector


class TestStateVector:
    P = np.array([[1, 0], [0, 0]])
    Q = np.array([[0, 0], [0, 1]])
    Hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def test_init(self):
        psi = StateVector(4)
        assert np.isclose(np.linalg.norm(psi.tensor), 1.0)
        assert np.isclose(psi.tensor[0, 0, 0, 0], 1.0)

    def test_apply_op_with_projective_measurement_1q(self):
        for qubit in range(4):
            psi = StateVector(4)
            tensor = psi._apply_op(self.P, qubit)
            assert np.isclose(np.linalg.norm(tensor), 1.0)
            assert np.isclose(tensor[0, 0, 0, 0], 1.0)

            tensor = psi._apply_op(self.Q, qubit)
            assert np.isclose(np.linalg.norm(tensor), 0.0)

            psi.tensor = psi._apply_op(self.Hadamard, qubit)
            psi.tensor = psi._apply_op(self.Q, qubit)
            assert np.isclose(np.linalg.norm(psi.tensor), 1 / np.sqrt(2))
            assert np.isclose(
                psi.tensor[tuple(0 if i != qubit else 1 for i in range(4))], 1 / np.sqrt(2)
            )

    def test_apply_gate_with_1q_X(self):
        for qubit in range(4):
            psi = StateVector(4).apply_gate(X(qubit=qubit))
            assert np.isclose(np.linalg.norm(psi.tensor), 1.0)
            assert np.isclose(
                psi.tensor[tuple(0 if i != qubit else 1 for i in range(4))], 1.0
            )

    def test_apply_op_with_2q_XX(self):
        for qubit1 in range(4):
            for qubit2 in range(4):
                if qubit1 == qubit2:
                    continue
                psi = StateVector(4)
                psi.apply_gate(X(qubit=qubit1)).apply_gate(X(qubit=qubit2))
                assert np.isclose(np.linalg.norm(psi.tensor), 1.0)
                assert np.isclose(
                    psi.tensor[
                        tuple(0 if i not in (qubit1, qubit2) else 1 for i in range(4))
                    ],
                    1.0,
                )

    def test_apply_gate_with_2q_CNOT(self):
        for qubit1 in range(4):
            for qubit2 in range(4):
                if qubit1 == qubit2:
                    continue
                psi = (
                    StateVector(4)
                    .apply_gate(X(qubit=qubit1))
                    .apply_gate(CNOT(qubit1=qubit1, qubit2=qubit2))
                    .apply_gate(X(qubit=qubit1))
                )
                assert np.isclose(np.linalg.norm(psi.tensor), 1.0)
                assert np.isclose(
                    psi.tensor[tuple(0 if i != qubit2 else 1 for i in range(4))],
                    1.0,
                )

    def test_apply_gate_with_bell_state(self):
        psi = (
            StateVector(3)
            .apply_gate(Hadamard(qubit=0))
            .apply_gate(CNOT(qubit1=0, qubit2=1))
            .apply_gate(CNOT(qubit1=1, qubit2=2))
        )
        assert np.isclose(np.linalg.norm(psi.tensor), 1.0)
        assert np.isclose(psi.tensor[0, 0, 0], 1 / np.sqrt(2))
        assert np.isclose(psi.tensor[1, 1, 1], 1 / np.sqrt(2))


class TestStateOperator:
    def test_init(self):
        U = StateOperator(4)
        assert np.isclose(np.linalg.norm(U.tensor), np.sqrt(2.0**4))

    def test_apply_gate_with_1q_X(self):
        for qubit in range(4):
            U = StateOperator(4).apply_gate(X(qubit=qubit))
            tensor = 1.0
            for i in range(4):
                if i == qubit:
                    op = X(qubit=i).matrix
                else:
                    op = np.eye(2)
                tensor = np.kron(tensor, op)
            assert np.all(np.isclose(U.tensor, np.reshape(tensor, (2,) * 8)))

    def test_apply_op_with_2q_XY(self):
        for qubit1 in range(4):
            for qubit2 in range(4):
                if qubit1 == qubit2:
                    continue
                U = StateOperator(4)
                U.apply_gate(X(qubit=qubit1)).apply_gate(Y(qubit=qubit2))

                tensor = 1.0
                for i in range(4):
                    if i == qubit1:
                        op = X(qubit=i).matrix
                    elif i == qubit2:
                        op = Y(qubit=i).matrix
                    else:
                        op = np.eye(2)
                    tensor = np.kron(tensor, op)
                assert np.all(np.isclose(U.tensor, np.reshape(tensor, (2,) * 8)))

    def test_apply_gate_with_2q_CNOT(self):
        for qubit1 in range(4):
            for qubit2 in range(4):
                if qubit1 == qubit2:
                    continue
                U = StateOperator(4)
                U.apply_gate(CNOT(qubit1=qubit1, qubit2=qubit2))

                tensor1 = 1.0
                for i in range(4):
                    if i == qubit1:
                        op = [[0, 0], [0, 1]]
                    elif i == qubit2:
                        op = X(qubit=i).matrix
                    else:
                        op = np.eye(2)
                    tensor1 = np.kron(tensor1, op)

                tensor2 = 1.0
                for i in range(4):
                    if i == qubit1:
                        op = [[1, 0], [0, 0]]
                    else:
                        op = np.eye(2)
                    tensor2 = np.kron(tensor2, op)
                assert np.all(np.isclose(U.tensor, np.reshape(tensor1 + tensor2, (2,) * 8)))
