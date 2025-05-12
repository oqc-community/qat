# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.frontend.circuit_builder import CircuitBuilder
from qat.ir.gates.base import is_equal_angle
from qat.ir.gates.gates_1q import Hadamard, Id, U, X, Y, Z
from qat.ir.gates.gates_2q import CNOT, ECR, Gate2Q
from qat.ir.gates.native import X_pi_2, Z_phase, ZX_pi_4
from qat.middleend.passes.gates.transform import (
    Decompose2QToCNOTs,
    DecomposeToNativeGates,
    Squash1QGates,
    SquashCNOTs,
)

from tests.unit.utils.gates import (
    circuit_as_unitary,
    one_q_gate_tests,
    same_up_to_phase,
    two_q_gate_tests,
)


class TestDecompose2QToCNOTs:
    @pytest.mark.parametrize(["gate", "params"], two_q_gate_tests())
    def test_2qs_decompose_to_only_cnots(self, gate, params):
        circ = CircuitBuilder(2)
        qubits = params.pop("qubits")
        circ.add(gate(qubit1=qubits[0], qubit2=qubits[1], **params))
        ir = circ.emit()
        ir = Decompose2QToCNOTs().run(ir)
        assert len(ir.instructions) >= 1
        for gate in ir.instructions:
            if isinstance(gate, Gate2Q):
                assert isinstance(gate, (CNOT, ECR, ZX_pi_4))


class TestDecomposeToNativeGates:
    @pytest.mark.parametrize(["gate", "params"], one_q_gate_tests())
    def test_1qs_decompose_to_only_native(self, gate, params):
        circ = CircuitBuilder(2)
        circ.add(gate(**params))
        ir = circ.emit()
        ir = DecomposeToNativeGates().run(ir)
        assert len(ir.instructions) >= 1 or gate == Id
        for gate in ir.instructions:
            assert isinstance(gate, (Z_phase, X_pi_2))

    @pytest.mark.parametrize(["gate", "params"], two_q_gate_tests())
    def test_2qs_decompose_to_only_native(self, gate, params):
        circ = CircuitBuilder(2)
        qubits = params.pop("qubits")
        circ.add(gate(qubit1=qubits[0], qubit2=qubits[1], **params))
        ir = circ.emit()
        ir = DecomposeToNativeGates().run(ir)
        assert len(ir.instructions) >= 1
        for gate in ir.instructions:
            assert isinstance(gate, (Z_phase, X_pi_2, ZX_pi_4))


involutory_gates = [X, Y, Z, Hadamard]


class TestSquash1QGates:
    @pytest.mark.parametrize(["gate", "params"], one_q_gate_tests())
    def test_squash_gives_same_matrix(self, gate, params):
        gate = gate(**params)
        u_gate = Squash1QGates().squash_gate([gate])
        assert same_up_to_phase(u_gate.matrix, gate.matrix)

    @pytest.mark.parametrize(
        "gates",
        [[gate(qubit=0) for _ in range(n)] for gate in involutory_gates for n in [2, 4]],
    )
    def test_squash_gate_with_identity_products(self, gates):
        u_gate = Squash1QGates().squash_gate(gates)
        assert is_equal_angle(u_gate.theta, 0.0)
        assert is_equal_angle(u_gate.phi + u_gate.lambd, 0.0)
        assert same_up_to_phase(u_gate.matrix, np.eye(2))

    def test_squashes_to_one_gate(self):
        circ = CircuitBuilder(1).X(0).Y(0).Z(0).Hadamard(0)
        ir = circ.emit()
        ir = Squash1QGates().run(ir)
        assert len(ir.instructions) == 1

    def test_one_gate_is_unchanged(self):
        circ = CircuitBuilder(1).X(0)
        ir = circ.emit()
        ir = Squash1QGates().run(ir)
        assert len(ir.instructions) == 1
        assert isinstance(ir.instructions[0], X)

    def test_one_gate_two_qubits_are_unchanged(self):
        circ = CircuitBuilder(2).X(0).Y(1)
        ir = circ.emit()
        ir = Squash1QGates().run(ir)
        assert len(ir.instructions) == 2
        assert isinstance(ir.instructions[0], X)
        assert isinstance(ir.instructions[1], Y)

    def test_1q_gates_interweaved_with_cnot_unchanged(self):
        circ = CircuitBuilder(2).X(0).Y(1).CNOT(0, 1).Z(0).Hadamard(1)
        ir = circ.emit()
        ir = Squash1QGates().run(ir)
        assert len(ir.instructions) == 5
        assert [type(g) for g in ir.instructions] == [X, Y, CNOT, Z, Hadamard]

    def test_1q_gates_interweaved_with_cnot_squashed(self):
        circ = (
            CircuitBuilder(2)
            .X(0)
            .Y(1)
            .Y(0)
            .Z(1)
            .CNOT(0, 1)
            .Z(0)
            .Hadamard(1)
            .X(0)
            .Hadamard(1)
        )
        ir = circ.emit()
        ir = Squash1QGates().run(ir)
        assert len(ir.instructions) == 5
        assert [type(g) for g in ir.instructions] == [U, U, CNOT, U, U]
        assert same_up_to_phase(ir.instructions[0].matrix, Z(qubit=0).matrix)
        assert same_up_to_phase(ir.instructions[1].matrix, X(qubit=1).matrix)
        assert same_up_to_phase(ir.instructions[3].matrix, Y(qubit=0).matrix)
        assert same_up_to_phase(ir.instructions[4].matrix, np.eye(2))


class TestSquashCNOTs:
    def test_two_CNOTs_are_removed(self):
        circ = CircuitBuilder(2).CNOT(0, 1).CNOT(0, 1)
        ir = circ.emit()
        U1 = circuit_as_unitary(2, ir)
        ir = SquashCNOTs().run(ir)
        assert len(ir.instructions) == 0
        U2 = circuit_as_unitary(2, ir)
        assert same_up_to_phase(U1, U2)

    def test_three_CNOTs_leaves_one(self):
        circ = CircuitBuilder(2).CNOT(0, 1).CNOT(0, 1).CNOT(0, 1)
        ir = circ.emit()
        U1 = circuit_as_unitary(2, ir)
        ir = SquashCNOTs().run(ir)
        assert len(ir.instructions) == 1
        U2 = circuit_as_unitary(2, ir)
        assert same_up_to_phase(U1, U2)

    def test_CNOTS_interweaved_is_unchanged(self):
        circ = CircuitBuilder(2).CNOT(0, 1).X(0).CNOT(0, 1)
        ir = circ.emit()
        ir = SquashCNOTs().run(ir)
        assert len(ir.instructions) == 3
        assert isinstance(ir.instructions[0], CNOT)
        assert isinstance(ir.instructions[1], X)
        assert isinstance(ir.instructions[2], CNOT)

    def test_CNOTs_are_removed_when_interweaved_with_other_qubits(self):
        circ = CircuitBuilder(4).CNOT(0, 1).X(2).CNOT(2, 3).CNOT(0, 1)
        ir = circ.emit()
        U1 = circuit_as_unitary(4, ir)
        ir = SquashCNOTs().run(ir)
        assert len(ir.instructions) == 2
        assert ir.instructions[0] == X(qubit=2)
        assert ir.instructions[1] == CNOT(qubit1=2, qubit2=3)
        U2 = circuit_as_unitary(4, ir)
        assert same_up_to_phase(U1, U2)

    def test_multiple_pairs_are_removed(self):
        circ = CircuitBuilder(4).CNOT(0, 1).CNOT(2, 3).CNOT(0, 1).CNOT(2, 3).CNOT(0, 1)
        ir = circ.emit()
        U1 = circuit_as_unitary(4, ir)
        ir = SquashCNOTs().run(ir)
        assert len(ir.instructions) == 1
        assert ir.instructions[0] == CNOT(qubit1=0, qubit2=1)
        U2 = circuit_as_unitary(4, ir)
        assert same_up_to_phase(U1, U2)

    def test_crz_swap_gives_three_cnots(self):
        circ = CircuitBuilder(2).CRz(0, 1, 2.54).SWAP(0, 1)
        ir = circ.emit()
        U1 = circuit_as_unitary(2, ir)
        ir = Decompose2QToCNOTs().run(ir)
        assert len([gate for gate in ir.instructions if isinstance(gate, CNOT)]) == 5
        ir = SquashCNOTs().run(ir)
        assert len([gate for gate in ir.instructions if isinstance(gate, CNOT)]) == 3
        U2 = circuit_as_unitary(2, ir)
        assert same_up_to_phase(U1, U2)

    def test_cnots_cancel_with_z_between_controls(self):
        circ = CircuitBuilder(2).CNOT(0, 1).Z(0).Rz(0, 0.254).CNOT(0, 1)
        ir = circ.emit()
        U1 = circuit_as_unitary(2, ir)
        ir = SquashCNOTs().run(ir)
        assert len([gate for gate in ir.instructions if isinstance(gate, CNOT)]) == 0
        U2 = circuit_as_unitary(2, ir)
        assert same_up_to_phase(U1, U2)

    def test_cnots_cancel_with_x_between_targets(self):
        circ = CircuitBuilder(2).CNOT(0, 1).X(1).Rx(1, 0.254).CNOT(0, 1)
        ir = circ.emit()
        U1 = circuit_as_unitary(2, ir)
        ir = SquashCNOTs().run(ir)
        assert len([gate for gate in ir.instructions if isinstance(gate, CNOT)]) == 0
        U2 = circuit_as_unitary(2, ir)
        assert same_up_to_phase(U1, U2)

    def test_cnots_do_not_cancel_with_z_between_targets(self):
        circ = CircuitBuilder(2).CNOT(0, 1).Z(1).CNOT(0, 1)
        ir = circ.emit()
        ir = SquashCNOTs().run(ir)
        assert len([gate for gate in ir.instructions if isinstance(gate, CNOT)]) == 2

    def test_cnots_do_not_cancel_with_x_between_controls(self):
        circ = CircuitBuilder(2).CNOT(0, 1).X(0).CNOT(0, 1)
        ir = circ.emit()
        ir = SquashCNOTs().run(ir)
        assert len([gate for gate in ir.instructions if isinstance(gate, CNOT)]) == 2
