# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from abc import ABC, abstractmethod

import numpy as np
from pydantic import NonNegativeInt

from qat.ir.gates.base import Angle, GateBase


class Gate1Q(GateBase, ABC):
    """Base gate for single qubit operations."""

    qubit: NonNegativeInt

    @property
    @abstractmethod
    def matrix(self):
        pass

    @property
    def qubits(self):
        return (self.qubit,)


class Id(Gate1Q):
    r"""The identity gate for a single qubit.

    Matrix representation:

    .. math:: I = \begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix}.

    :param qubit: Target qubit index.
    """

    @property
    def matrix(self):
        return np.array([[1, 0], [0, 1]])


class X(Gate1Q):
    r"""Implements the Pauli-X gate.

    Matrix representation:

    .. math:: X = \begin{bmatrix}0 & 1 \\ 1 & 0 \end{bmatrix}.

    :param qubit: Target qubit index.

    """

    @property
    def matrix(self):
        return np.array([[0, 1], [1, 0]])


class Y(Gate1Q):
    r"""Implements the Pauli-Y gate.

    Matrix representation:

    .. math:: Y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}.

    :param qubit: Target qubit index.
    """

    @property
    def matrix(self):
        return np.array([[0, -1j], [1j, 0]])


class Z(Gate1Q):
    r"""Implements the Pauli-Z gate.

    Matrix representation:

    .. math:: X = \begin{bmatrix}0 & 1 \\ 1 & 0 \end{bmatrix}.

    :param qubit: Target qubit index.
    """

    @property
    def matrix(self):
        return np.array([[1, 0], [0, -1]])


class Rx(Gate1Q):
    r"""The :math:`R_{x}(\theta) = e^{-i\theta X/2}` gate for some rotation angle
    :math:`\theta`.

    Matrix representation:

    .. math:: R_{x}(\theta) = \begin{bmatrix}
        \cos(\theta/2) & -i\sin(\theta/2) \\ 
        -i\sin(\theta/2) & \cos(\theta/2)
        \end{bmatrix}.

    :param qubit: Target qubit index.
    :param theta: Rotation angle
    """

    theta: Angle = np.pi

    @property
    def matrix(self):
        cos = np.cos(self.theta / 2)
        sin = np.sin(self.theta / 2)
        return np.array([[cos, -1j * sin], [-1j * sin, cos]])


class Ry(Gate1Q):
    r"""The :math:`R_{y}(\theta) = e^{-i\theta Y/2}` gate for some rotation angle
    :math:`\theta`.

    Matrix representation:

    .. math:: R_{y}(\theta) = \begin{bmatrix}
        \cos(\theta/2) & -\sin(\theta/2) \\ 
        \sin(\theta/2) & \cos(\theta/2)
        \end{bmatrix}.

    :param qubit: Target qubit index.
    :param theta: Rotation angle
    """

    theta: Angle = np.pi

    @property
    def matrix(self):
        cos = np.cos(self.theta / 2)
        sin = np.sin(self.theta / 2)
        return np.array([[cos, -sin], [sin, cos]])


class Rz(Gate1Q):
    r"""The :math:`R_{z}(\theta) = e^{-i\theta Z/2}` gate for some rotation angle
    :math:`\theta`.

    Matrix representation:

    .. math:: R_{z}(\theta) = \begin{bmatrix}
        e^{-i\theta/2} & 0 \\ 
        0 & e^{i\theta/2}
        \end{bmatrix}.

    :param qubit: Target qubit index.
    :param theta: Rotation angle
    """

    theta: Angle = np.pi

    @property
    def matrix(self):
        return np.array(
            [[np.exp(-1j * self.theta / 2), 0], [0, np.exp(1j * self.theta / 2)]]
        )


class Phase(Gate1Q):
    r"""The :math:`{\rm Phase}(\theta)` gate for some rotation angle :math:`\theta`.

    Equivalent to the :math:`Rz(\theta)` gate up to a global rotation. Matrix
    representation:

    .. math:: {\rm Phase}(\theta) = \begin{bmatrix}  1 & 0 \\ 0 & e^{i\theta}\end{bmatrix}.

    :param qubit: Target qubit index.
    :param theta: Rotation angle
    """

    theta: Angle = np.pi

    @property
    def matrix(self):
        return np.array([[1, 0], [0, np.exp(1j * self.theta)]])


class U(Gate1Q):
    r"""Implements the :math:`U(\theta, \phi, \lambda)` rotation gate defined by three
    rotation angles.

    The gate can be expressed as

    .. math::
        U(\theta, \phi, \lambda) = 
        e^{\frac{1}{2}i\pi(\lambda+\phi)}R_{z}(\phi)R_{y}(\theta)R_{z}(\lambda)

    Matrix representation:

    .. math::
        U(\theta, \phi, \lambda) = \begin{bmatrix}
        \cos(\theta/2) & -e^{i\lambda} \sin(\theta/2) \\
        e^{i\phi} \sin(\theta/2) & e^{i(\phi+\lambda)} \cos(\theta/2)
        \end{bmatrix}.

    
    :param qubit: Target qubit index.
    :param theta: Rotation angle :math:`\theta` around the Y-axis.
    :param phi: Rotation angle :math:`\phi` around the Z-axis (final).
    :param lambd: Rotation angle :math:`\lambda` around the Z-axis (first).
    """

    theta: Angle = 0.0
    phi: Angle = 0.0
    lambd: Angle = 0.0

    @property
    def matrix(self):
        return np.array(
            [
                [np.cos(self.theta / 2), -np.exp(1j * self.lambd) * np.sin(self.theta / 2)],
                [
                    np.exp(1j * self.phi) * np.sin(self.theta / 2),
                    np.exp(1j * (self.phi + self.lambd)) * np.cos(self.theta / 2),
                ],
            ]
        )

    @classmethod
    def from_matrix(cls, qubit: int, mat: np.ndarray):
        r"""Returns a :math:`U(\theta, \phi, \lambda)` gate directly from a 1Q matrix.

        Determines the angles using simple algebra.
        """
        if np.isclose(mat[0, 0], 0.0):
            theta = np.pi
            lambd = np.angle(-mat[0, 1])
            phi = np.angle(mat[1, 0])
        elif np.isclose(mat[1, 0], 0.0):
            theta = 0.0
            lambd = 0.0
            phi = np.angle(mat[1, 1]) - np.angle(mat[0, 0])
        else:
            alpha = np.angle(mat[0, 0])
            lambd = np.angle(-mat[0, 1]) - alpha
            phi = np.angle(mat[1, 0]) - alpha
            theta = 2 * np.arccos(np.abs(mat[0, 0]))

        return cls(qubit=qubit, theta=theta, phi=phi, lambd=lambd)


class Hadamard(Gate1Q):
    r"""Implements the Hadamard gate :math:`H`.

    Matrix representation:

    .. math::
        H = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}.
    """

    @property
    def matrix(self):
        return 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])


class S(Gate1Q):
    r"""Implements the :math:`S` gate.

    This gate is identical to the :math:`R_{z}(\pi/2)` gate up to global phase. Matrix
    representation:

    .. math:: S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}.
    """

    @property
    def matrix(self):
        return np.array([[1, 0], [0, 1j]])


class Sdg(Gate1Q):
    r"""Implements the adjoint of the :math:`S` gate, :math:`S^{\dagger}`.

    This gate is identical to the :math:`R_{z}(-\pi/2)` gate up to global phase. Matrix
    representation:

    .. math:: S^{\dagger} = \begin{bmatrix} 1 & 0 \\ 0 & -i \end{bmatrix}.
    """

    @property
    def matrix(self):
        return np.array([[1, 0], [0, -1j]])


class T(Gate1Q):
    r"""Implements the :math:`T` gate.

    This gate is identical to the :math:`R_{z}(\pi/4)` gate up to global phase. Matrix
    representation:

    .. math:: T = \begin{bmatrix} 1 & 0 \\ 0 & (1 + i)/\sqrt{2} \end{bmatrix}.
    """

    @property
    def matrix(self):
        return np.array([[1, 0], [0, (1 + 1j) / (np.sqrt(2))]])


class Tdg(Gate1Q):
    r"""Implements the adjoint of the :math:`T` gate, :math:`T^{\dagger}`.

    This gate is identical to the :math:`R_{z}(-\pi/4)` gate up to global phase. Matrix
    representation:

    .. math:: T^{\dagger} = \begin{bmatrix} 1 & 0 \\ 0 & (1 - i)/\sqrt{2} \end{bmatrix}."""

    @property
    def matrix(self):
        return np.array([[1, 0], [0, (1 - 1j) / (np.sqrt(2))]])
