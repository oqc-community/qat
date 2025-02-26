# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import inspect

import pytest

from qat.frontend.circuit_builder import CircuitBuilder
from qat.ir.gates.gates_1q import Gate1Q, Hadamard, X
from qat.ir.gates.gates_2q import CNOT, Gate2Q
from qat.ir.gates.operation import Barrier, Measure, Reset

from tests.qat.utils.gates import get_non_abstract_subgates


class TestCircuitBuilder:

    def test_add_raises_value_error(self):
        circuit = CircuitBuilder(4)

        # Test cases that should fail
        with pytest.raises(ValueError):
            circuit.add(X(qubit=-1))
        with pytest.raises(ValueError):
            circuit.add(X(qubit=4))
        with pytest.raises(ValueError):
            circuit.add(CNOT(qubit1=-1, qubit2=1))
        with pytest.raises(ValueError):
            circuit.add(CNOT(qubit1=0, qubit2=4))
        with pytest.raises(ValueError):
            circuit.add(CNOT(qubit1=1, qubit2=1))

        # Test cases that should not fail
        circuit.add(X(qubit=0))
        circuit.add(X(qubit=3))
        circuit.add(CNOT(qubit1=1, qubit2=2))
        circuit.add(CNOT(qubit1=2, qubit2=1))

    def test_barrier(self):
        circuit = CircuitBuilder(4)
        circuit.barrier(2)
        circuit.barrier(1, 3)
        assert len(circuit.instructions) == 2
        assert all([isinstance(inst, Barrier) for inst in circuit.instructions])
        assert circuit.instructions[0].qubits == [
            2,
        ]
        assert circuit.instructions[1].qubits == [1, 3]

    def test_emit(self):
        circ = (
            CircuitBuilder(2)
            .Hadamard(0)
            .CNOT(0, 1)
            .barrier(0, 1)
            .measure(0, 0)
            .measure(1, 1)
            .reset(0)
        )
        ir = circ.emit()
        assert len(ir.instructions) == 6
        assert [type(inst) for inst in ir.instructions] == [
            Hadamard,
            CNOT,
            Barrier,
            Measure,
            Measure,
            Reset,
        ]

    @pytest.mark.parametrize(
        "gate", get_non_abstract_subgates(Gate1Q).union(get_non_abstract_subgates(Gate2Q))
    )
    def test_gates(self, gate):
        method = getattr(CircuitBuilder, gate.__name__, None)
        if not method:
            pytest.skip(f"CircuitBuilder doesn't have method {gate.__class__.__name__}.")

        circ = CircuitBuilder(32)
        params = {}
        qubit = 0
        for param, ty in inspect.getfullargspec(method).annotations.items():
            if ty == int:
                params[param] = qubit
                qubit += 1
            elif ty == float:
                params[param] = 0.254
            elif ty == float | None:
                pass
        method(circ, **params)
        assert len(circ.instructions) == 1
        assert isinstance(circ.instructions[0], gate)
