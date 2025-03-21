# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.ir.gates.gates_1q import Gate1Q
from qat.middleend.decompositions.gates import DefaultGateDecompositions
from qat.utils.state_tensors import StateOperator

from tests.unit.utils.gates import one_q_gate_tests, same_up_to_phase, two_q_gate_tests


class TestDefaultGateDecompositions:
    @pytest.mark.parametrize(["gate", "params"], one_q_gate_tests())
    def test_1q_decompositions_give_same_matrices(self, gate, params):
        """Tests decompositions for all gates inherited from :class:`Gate1Q`."""
        gate = gate(**params)
        decomps = DefaultGateDecompositions()
        decomposed_gates = decomps.decompose(gate)
        assert all([isinstance(g, tuple(decomps.end_nodes)) for g in decomposed_gates])

        # Test the decomposition gives the same unitary operator
        U1 = StateOperator(1).apply_gate(gate)
        U2 = StateOperator(1)
        for g in decomposed_gates:
            U2.apply_gate(g)
        assert same_up_to_phase(U1.tensor, U2.tensor)

    @pytest.mark.parametrize(["gate", "params"], two_q_gate_tests())
    def test_2q_decompositions_give_same_matrices(self, gate, params):
        """Tests decompositions for all gates inherited from :class:`Gate2Q`."""
        qubit1, qubit2 = params.pop("qubits")
        gate = gate(qubit1=qubit1, qubit2=qubit2, **params)
        decomps = DefaultGateDecompositions()
        decomposed_gates = decomps.decompose(gate)
        assert all([isinstance(g, tuple(decomps.end_nodes)) for g in decomposed_gates])

        # Test the decomposition gives the same unitary operator
        U1 = StateOperator(2).apply_gate(gate)
        U2 = StateOperator(2)
        for g in decomposed_gates:
            U2.apply_gate(g)
        assert same_up_to_phase(
            np.reshape(U1.tensor, (4, 4)), np.reshape(U2.tensor, (4, 4))
        )

    def test_fallback_gate_1q(self):
        """For 1Q gates with a decomposition that is not implemented, test that the fallback
        method works."""

        class TestGate(Gate1Q):
            @property
            def matrix(self):
                return np.array([[0, 1], [1, 0]])

        gate = TestGate(qubit=1)
        decomps = DefaultGateDecompositions()
        decomposed_gates = decomps.decompose(gate)

        # Test the decomposition gives the same unitary operator
        U1 = StateOperator(2).apply_gate(gate)
        U2 = StateOperator(2)
        for g in decomposed_gates:
            U2.apply_gate(g)
        assert same_up_to_phase(
            np.reshape(U1.tensor, (4, 4)), np.reshape(U2.tensor, (4, 4))
        )
