# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from functools import singledispatchmethod

import numpy as np

from qat.ir.gates.base import GateBase
from qat.ir.gates.gates_1q import (
    Gate1Q,
    Hadamard,
    Id,
    Phase,
    Rx,
    Ry,
    Rz,
    S,
    Sdg,
    T,
    Tdg,
    U,
    X,
    Y,
    Z,
)
from qat.ir.gates.gates_2q import CNOT, ECR, SWAP, CPhase, CRx, CRy, CRz
from qat.ir.gates.native import NativeGate, X_pi_2, Z_phase, ZX_pi_4
from qat.middleend.decompositions.base import DecompositionBase


class GateDecompositionBase(DecompositionBase):
    """Base object for implementing decompositions of gates with :class:`NativeGate` for
    end nodes.

    This handles decompositions of gates to gates, and should not be used to lower gates to
    pulse instructions. While it is recommended to use :class:`DefaultGateDecompositons`,
    custom gate decompositions strategies can be implemented via this class. This could be
    used, for example, if targeting machines that have a different native gate set, or
    trying out approximate decompositions that have a cheaper implementation cost.
    """

    end_nodes = (NativeGate,)


class DefaultGateDecompositions(GateDecompositionBase):
    """Implements the standard gate decompositions used in OQC hardware.

    Provides decomposition rules for the native gate set {:class:`Z_phase`,
    :class:`X_pi_2`, :class:`ZX_pi_4`}.
    """

    end_nodes = (Z_phase, X_pi_2, ZX_pi_4)

    @singledispatchmethod
    def decompose_op(self, gate: GateBase):
        """Implements the definition of a decomposition of a gate.

        The definition does not have to be in terms of native gates, but decompositions
        must form a DAG. For example,

        .. code-block:: python

            CNOT -> {ECR, X, Z} -> {ZX_pi_4, Z_phase, X_pi_2}
            U -> {x_pi_2, Z_phase}
        """
        raise NotImplementedError(
            f"Decomposition for gate {gate.__class__.__name__} not implemented."
        )

    @decompose_op.register(Gate1Q)
    def _(self, gate: Gate1Q):
        """Decomposes a generic one qubit gate.

        A fallback method for when a decomposition is not implemented. Converts the gate to
        a :class:`U` gate, and uses that decomposition in its place.

        .. warning::
            Using this is likely to be less performant than a specific implementation. It is
            recommended to register a new decomposition for each new gate that is added.
        """

        new_gate = U.from_matrix(gate.qubit, gate.matrix)
        return self.decompose(new_gate)

    @decompose_op.register(U)
    def _(self, gate: U):
        """Decomposes the U gate into a product of X_pi_2 gates and Rz gates.

        Does not deal with special cases that can reduce the number of :class:`X_pi_2`
        gates: this is dealt with my optimisation passes.
        """

        return [
            Z_phase(qubit=gate.qubit, theta=gate.lambd + np.pi),
            X_pi_2(qubit=gate.qubit),
            Z_phase(qubit=gate.qubit, theta=np.pi - gate.theta),
            X_pi_2(qubit=gate.qubit),
            Z_phase(qubit=gate.qubit, theta=gate.phi),
        ]

    @decompose_op.register(X)
    def _(self, gate: X):
        """Decomposes the Pauli-X gate."""
        return [Rx(qubit=gate.qubit, theta=np.pi)]

    @decompose_op.register(Y)
    def _(self, gate: Y):
        """Decomposes the Pauli-Y gate."""
        return [Ry(qubit=gate.qubit, theta=np.pi)]

    @decompose_op.register(Z)
    def _(self, gate: Z):
        """Decomposes the Pauli-Z gate."""
        return [Rz(qubit=gate.qubit, theta=np.pi)]

    @decompose_op.register(Rx)
    def _(self, gate: Rx):
        """Decomposes the X-rotation gate.

        Does not deal with special cases that can reduce the number of :class:`X_pi_2`
        gates: this is dealt with my optimisation passes.
        """
        return [
            Z_phase(qubit=gate.qubit, theta=-np.pi / 2),
            X_pi_2(qubit=gate.qubit),
            Z_phase(qubit=gate.qubit, theta=np.pi - gate.theta),
            X_pi_2(qubit=gate.qubit),
            Z_phase(qubit=gate.qubit, theta=-np.pi / 2),
        ]

    @decompose_op.register(Ry)
    def _(self, gate: Ry):
        """Decomposes the Y-rotation gate.

        Does not deal with special cases that can reduce the number of :class:`X_pi_2`
        gates: this is dealt with my optimisation passes.
        """
        return [
            Z_phase(qubit=gate.qubit, theta=np.pi),
            X_pi_2(qubit=gate.qubit),
            Z_phase(qubit=gate.qubit, theta=np.pi - gate.theta),
            X_pi_2(qubit=gate.qubit),
        ]

    @decompose_op.register(Rz)
    def _(self, gate: Rz):
        """Decomposes the Z-rotation gate."""
        return [Z_phase(qubit=gate.qubit, theta=gate.theta)]

    @decompose_op.register(Phase)
    def _(self, gate: Phase):
        """Decomposes the Z-rotation gate."""
        return [Z_phase(qubit=gate.qubit, theta=gate.theta)]

    @decompose_op.register(Id)
    def _(self, _: Id):
        """The identity gate does nothing."""
        return []

    @decompose_op.register(Hadamard)
    def _(self, gate: Hadamard):
        r"""Decomposes the Hadamard as :math:`H = R_{y}(\pi/2) Z`."""
        return [Z(qubit=gate.qubit), Ry(qubit=gate.qubit, theta=np.pi / 2)]

    @decompose_op.register(S)
    def _(self, gate: S):
        r"""Decompose the :math:`S` gate as a :math:`R_{z}(\pi/2)` gate (up to global
        phase).
        """
        return [Rz(qubit=gate.qubit, theta=np.pi / 2)]

    @decompose_op.register(Sdg)
    def _(self, gate: Sdg):
        r"""Decompose the :math:`S^{\dagger}` gate as a :math:`R_{z}(-\pi/2)` gate (up to
        global phase).
        """
        return [Rz(qubit=gate.qubit, theta=-np.pi / 2)]

    @decompose_op.register(T)
    def _(self, gate: T):
        r"""Decompose the :math:`T` gate as a :math:`R_{z}(\pi/4)` gate (up to global
        phase).
        """
        return [Rz(qubit=gate.qubit, theta=np.pi / 4)]

    @decompose_op.register(Tdg)
    def _(self, gate: Tdg):
        r"""Decompose the :math:`T^{\dagger}` gate as a :math:`R_{z}(-\pi/4)` gate (up to
        global phase).
        """
        return [Rz(qubit=gate.qubit, theta=-np.pi / 4)]

    @decompose_op.register(ECR)
    def _(self, gate: ECR):
        r"""Decomposes an ECR using :math:`ZX^{0, 1}(-\pi/4) R_{x}^{0} ZX^{0, 1}(\pi/4)`."""
        return [
            ZX_pi_4(qubit1=gate.qubit1, qubit2=gate.qubit2),
            Rx(qubit=gate.qubit1, theta=np.pi),
            Rz(qubit=gate.qubit1, theta=np.pi),
            Rz(qubit=gate.qubit2, theta=np.pi),
            ZX_pi_4(qubit1=gate.qubit1, qubit2=gate.qubit2),
            Rz(qubit=gate.qubit1, theta=np.pi),
            Rz(qubit=gate.qubit2, theta=np.pi),
        ]

    @decompose_op.register(CNOT)
    def _(self, gate: CNOT):
        r"""Decompose a CNOT into an ECR gate as
        :math:`R_{x}^{1}(-\pi/2) R_{Z}^{0}(-\pi/2) R_{x}^{0}(\pi) ECR^{0, 1}`.
        """
        return [
            ECR(qubit1=gate.control, qubit2=gate.target),
            Rx(qubit=gate.control, theta=np.pi),
            Rz(qubit=gate.control, theta=-np.pi / 2),
            Rx(qubit=gate.target, theta=-np.pi / 2),
        ]

    @decompose_op.register(SWAP)
    def _(self, gate: SWAP):
        """Decompose a SWAP gate as three CNOTS."""
        return [
            CNOT(qubit1=gate.qubit1, qubit2=gate.qubit2),
            CNOT(qubit1=gate.qubit2, qubit2=gate.qubit1),
            CNOT(qubit1=gate.qubit1, qubit2=gate.qubit2),
        ]

    @decompose_op.register(CRz)
    def _(self, gate: CRz):
        """Decomposes a controlled-Rz gate into two CNOTs and two Rz gates."""
        return [
            Rz(qubit=gate.target, theta=gate.theta / 2),
            CNOT(qubit1=gate.control, qubit2=gate.target),
            Rz(qubit=gate.target, theta=-gate.theta / 2),
            CNOT(qubit1=gate.control, qubit2=gate.target),
        ]

    @decompose_op.register(CPhase)
    def _(self, gate: CPhase):
        """Decomposes a controlled-Phase gate into a controlled-Rz gate and an Rz gate."""
        return [
            Rz(qubit=gate.control, theta=gate.theta / 2),
            CRz(qubit1=gate.control, qubit2=gate.target, theta=gate.theta),
        ]

    @decompose_op.register(CRx)
    def _(self, gate: CRx):
        """Decomposes a controlled-Rx gate via controlled-Rz gates and Hadamards"""
        return [
            Hadamard(qubit=gate.target),
            CRz(qubit1=gate.control, qubit2=gate.target, theta=gate.theta),
            Hadamard(qubit=gate.target),
        ]

    @decompose_op.register(CRy)
    def _(self, gate: CRy):
        """Decomposes a controlled-Ry gate via cnots and Ry gates."""
        return [
            Ry(qubit=gate.target, theta=gate.theta / 2),
            CNOT(qubit1=gate.control, qubit2=gate.target),
            Ry(qubit=gate.target, theta=-gate.theta / 2),
            CNOT(qubit1=gate.control, qubit2=gate.target),
        ]
