# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from abc import ABC, abstractmethod

import numpy as np
from pydantic import NonNegativeInt

from qat.ir.gates.base import Angle, GateBase


class Gate2Q(GateBase, ABC):
    """Base gate for two qubit operations."""

    qubit1: NonNegativeInt
    qubit2: NonNegativeInt

    @property
    def qubits(self):
        return (self.qubit1, self.qubit2)

    @property
    @abstractmethod
    def matrix(self):
        pass


class ControlGate2Q(Gate2Q, ABC):
    r"""Base class for controlled two-qubit gates. The convention is :attr:`qubit1` is the
    control qubit and :attr:`qubit` is the target qubit. For some single qubit gate
    :math:`G`, the controlled gate is

    .. math:: CG = \ket{1}\bra{1} \otimes G + \ket{0}\bra{0} \otimes I

    Will eventually be replaced by a mechanism for creating control gates from standard
    gates. Used now to mark gates as control gates (which will be considered in future
    optimisation passes).
    """

    @property
    def control(self):
        return self.qubit1

    @property
    def target(self):
        return self.qubit2


class CNOT(ControlGate2Q):
    r"""Implements a CNOT with the control gate on :param:`qubit1` and target gate on
    :param:`qubit2`.
    
    Matrix representation:
    
    .. math:: {\rm CNOT}_{c, t} = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 1 \\
        0 & 0 & 1 & 0
        \end{bmatrix}
    """

    @property
    def matrix(self):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


CX = CNOT


class ECR(Gate2Q):
    r"""Implements the ECR gate.
    
    Matrix representation:
    
    .. math:: {\rm ECR} = \begin{bmatrix}
        0 & 0 & 1 & i \\
        0 & 0 & i & 1 \\
        1 & -i & 0 & 0 \\
        -i & 1 & 0 & 0
        \end{bmatrix}
    """

    @property
    def matrix(self):
        return (1 / np.sqrt(2)) * np.array(
            [[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]]
        )


class SWAP(Gate2Q):
    r"""Implements the SWAP gate.
    
    Matrix representation:
    
    .. math:: {\rm SWAP} = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 1
        \end{bmatrix}
    """

    @property
    def matrix(self):
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


class CPhase(ControlGate2Q):
    r"""Implements the controlled-:math:`Phase` gate.

    Unlike the non-controlled gate, note that this is not equivalent to the
    :math:`CR_{z}(\theta)` gate.
    """

    theta: Angle = np.pi

    @property
    def matrix(self):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * self.theta)],
            ]
        )


class CRz(ControlGate2Q):
    r"""Implements the controlled-:math:`R_{z}(\theta)` gate."""

    theta: Angle = np.pi

    @property
    def matrix(self):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * self.theta / 2), 0],
                [0, 0, 0, np.exp(1j * self.theta / 2)],
            ]
        )


class CRx(ControlGate2Q):
    r"""Implements the controlled-:math:`R_{x}(\theta)` gate."""

    theta: Angle = np.pi

    @property
    def matrix(self):
        cos = np.cos(self.theta / 2)
        sin = np.sin(self.theta / 2)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, cos, -1j * sin],
                [0, 0, -1j * sin, cos],
            ]
        )


class CRy(ControlGate2Q):
    r"""Implements the controlled-:math:`R_{y}(\theta)` gate."""

    theta: Angle = np.pi

    @property
    def matrix(self):
        cos = np.cos(self.theta / 2)
        sin = np.sin(self.theta / 2)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, cos, -sin],
                [0, 0, sin, cos],
            ]
        )
