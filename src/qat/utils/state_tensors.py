# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from abc import ABC

import numpy as np

from qat.ir.gates.base import GateBase


class StateTensor(ABC):
    """A tensorial description of quantum-related objects, such as quantum states and
    unitary operators.

    Quantum many-body states with a discrete Hilbert space can be represented exactly using
    so state vector approaches, where the vector elements describe the probability amplitude
    for a particular state. The same concept can naturally be extended to described mixed
    states and quantum operations (unitary, non-unitary and quantum channels alike).

    This object is a minimal implementation of this type of reasoning. For now it is used
    purely for testing the higher parts of our compilation stack (hence its perhaps
    misplaced location in the `utils` package), such as gate-level optimisations. Its
    capabilities will be extended as required, and it could potentially be used to implement
    an in-house gate-level simulator later down the line...
    """

    pass


class StateVector(StateTensor):
    r"""Represents the wavefunction of a discrete quantum many-body state.

    The wavefunction can be represented as a :math:`d^{N}` vector for a system with
    :math:`N` constituents each with local dimension :math:`d^{N}`, which is :math:`d=2` for
    qubits. However, a more convenient (and entirely equivalent) way to represent this is
    actually as a :math:`(d, d, \dots, d, d)`-tensor (with total rank :math:`N`). This
    allows to do calculations using local operations (such as a 1Q or 2Q gate, or a
    measurement) without calculating the operation for the full Hilbert space, reducing the
    computational complexity of the operation. Note that while the cost is still exponential
    in :math:`N`, it can be done with a smaller exponent, allowing for more performant
    numerics.

    This object currently just represents the state an has implementations for basic
    application of quantum gates: this will be extended as needed.
    """

    def __init__(self, num_qubits: int):
        """Initiate the state with all qubits in their lowest-energy state."""

        self.num_qubits = num_qubits
        self.tensor = np.zeros([2] * num_qubits, dtype=np.complex128)
        self.tensor[(0,) * num_qubits] = 1.0

    def apply_gate(self, gate: GateBase) -> "StateVector":
        """Apply a gate to the quantum state."""
        self.tensor = self._apply_op(gate.matrix, gate.qubits)
        return self

    def _apply_op(self, matrix: np.ndarray, qubits: tuple[int, ...]):
        """Applies a matrix to the tensor, permuting the dimensions back into the correct
        order. Has computational cost O(2^(N+k)) where k is the number of qubits the matrix
        acts on."""

        qubits = (qubits,) if not isinstance(qubits, (tuple, list)) else qubits
        matrix = np.reshape(matrix, (2,) * (2 * len(qubits)))
        matrix_contraction_dims = tuple(len(qubits) + i for i in range(len(qubits)))
        tensor = np.tensordot(self.tensor, matrix, (qubits, matrix_contraction_dims))
        dim_order = tuple(set(range(self.num_qubits)) ^ set(qubits)) + qubits
        permute_order = sorted(range(len(dim_order)), key=lambda k: dim_order[k])
        return np.permute_dims(tensor, permute_order)


class StateOperator(StateTensor):
    r"""Represents a unitary operator for discrete quantum many-body quantum systems.

    A generalisation of :class:`StateVector` to matrix-like objects, which has rank
    :math:`2N`. Is initialized to be an identity matrix, which can be "evolved" by gates
    to represent a quantum circuit.
    """

    def __init__(self, num_qubits: int):
        """Initiate the the operator as the identity matrix."""

        self.num_qubits = num_qubits
        self.tensor = np.reshape(
            np.eye(2**num_qubits), tuple(2 for _ in range(2 * num_qubits))
        )

    def apply_gate(self, gate: GateBase) -> "StateOperator":
        """Apply a gate to the operator."""
        self.tensor = self._apply_op(gate.matrix, gate.qubits)
        return self

    def _apply_op(self, matrix: np.ndarray, qubits: tuple[int, ...]):
        """Applies a matrix to the tensor, permuting the dimensions back into the correct
        order. Has computational cost O(2^(2N+k)) where k is the number of qubits the matrix
        acts on."""

        qubits = (qubits,) if not isinstance(qubits, (tuple, list)) else qubits
        matrix = np.reshape(matrix, (2,) * (2 * len(qubits)))
        matrix_contraction_dims = tuple(len(qubits) + i for i in range(len(qubits)))
        tensor = np.tensordot(matrix, self.tensor, (matrix_contraction_dims, qubits))
        dim_order = (
            qubits
            + tuple(set(range(self.num_qubits)) ^ set(qubits))
            + tuple(self.num_qubits + i for i in range(self.num_qubits))
        )
        permute_order = sorted(range(len(dim_order)), key=lambda k: dim_order[k])
        return np.permute_dims(tensor, permute_order)
