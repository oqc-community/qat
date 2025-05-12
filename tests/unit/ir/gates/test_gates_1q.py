# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.ir.gates.gates_1q import Rx, Ry, Rz, S, Sdg, T, Tdg, U, X, Y, Z

from tests.unit.utils.gates import same_up_to_phase, test_angles


class TestRx:
    @pytest.mark.parametrize("angle", [np.pi, -np.pi])
    def test_same_as_X_with_pi(self, angle):
        assert same_up_to_phase(X(qubit=1).matrix, Rx(qubit=1, theta=angle).matrix)


class TestRy:
    @pytest.mark.parametrize("angle", [np.pi, -np.pi])
    def test_same_as_Y_with_pi(self, angle):
        assert same_up_to_phase(Y(qubit=1).matrix, Ry(qubit=1, theta=angle).matrix)


class TestRz:
    @pytest.mark.parametrize("angle", [np.pi, -np.pi])
    def test_same_as_Z_with_pi(self, angle):
        assert same_up_to_phase(Z(qubit=1).matrix, Rz(qubit=1, theta=angle).matrix)

    def test_same_as_S_with_pi_2(self):
        assert same_up_to_phase(S(qubit=1).matrix, Rz(qubit=1, theta=np.pi / 2).matrix)

    def test_same_as_Sdg_with_minus_pi_2(self):
        assert same_up_to_phase(Sdg(qubit=1).matrix, Rz(qubit=1, theta=-np.pi / 2).matrix)

    def test_same_as_T_with_pi_4(self):
        assert same_up_to_phase(T(qubit=1).matrix, Rz(qubit=1, theta=np.pi / 4).matrix)

    def test_same_as_Tdg_with_minus_pi_4(self):
        assert same_up_to_phase(Tdg(qubit=1).matrix, Rz(qubit=1, theta=-np.pi / 4).matrix)


class TestU:
    @pytest.mark.parametrize("theta", test_angles)
    @pytest.mark.parametrize("phi", test_angles)
    @pytest.mark.parametrize("lambd", test_angles)
    def test_from_matrix_round_trip(self, theta, phi, lambd):
        gate = U(qubit=0, theta=theta, lambd=lambd, phi=phi)
        new_gate = U.from_matrix(0, gate.matrix)
        assert same_up_to_phase(gate.matrix, new_gate.matrix)

    @pytest.mark.parametrize("coeff", [1.0, -1.0, np.exp(1j * 2.54), np.exp(0.5j)])
    def test_identity(self, coeff):
        gate = coeff * np.eye(2)
        new_gate = U.from_matrix(0, gate)
        assert same_up_to_phase(gate, new_gate.matrix)

    @pytest.mark.parametrize("theta", test_angles)
    def test_same_as_Ry(self, theta):
        gate = U(qubit=0, theta=theta, phi=0.0, lambd=0.0)
        Ry_gate = Ry(qubit=0, theta=theta)
        assert same_up_to_phase(gate.matrix, Ry_gate.matrix)

    @pytest.mark.parametrize("theta", test_angles)
    def test_same_as_Rx(self, theta):
        gate = U(qubit=0, theta=theta, phi=-np.pi / 2, lambd=np.pi / 2)
        Rx_gate = Rx(qubit=0, theta=theta)
        assert same_up_to_phase(gate.matrix, Rx_gate.matrix)

    @pytest.mark.parametrize("phi", test_angles)
    @pytest.mark.parametrize("lambd", test_angles)
    def test_same_as_Rz(self, phi, lambd):
        gate = U(qubit=0, theta=0.0, phi=phi, lambd=lambd)
        Rz_gate = Rz(qubit=0, theta=phi + lambd)
        assert same_up_to_phase(gate.matrix, Rz_gate.matrix)
