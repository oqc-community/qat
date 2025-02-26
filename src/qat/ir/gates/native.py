# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from abc import ABC
from typing import Literal

import numpy as np

from qat.ir.gates.base import GateBase
from qat.ir.gates.gates_1q import Gate1Q, Rz
from qat.ir.gates.gates_2q import Gate2Q


class NativeGate(GateBase, ABC):
    """Marks a gate as a native gate."""


class Z_phase(Rz, NativeGate):
    r"""A native implementation of :math:`R_{z}(\theta)` gate via a phase shift.

    Derived from the :class:`Rz` gate as its mathematically no different. However, this
    class is used to mark it as a :class:`NativeGate` for hardware that support virtual-Z
    gates.

    :param qubit: Target qubit index.
    :param theta: Rotation angle
    """

    inst: Literal["Z_phase"] = "Z_phase"


class X_pi_2(Gate1Q, NativeGate):
    r"""A native :math:`X(\pi/2)` gate.

    Mathematically equivalent to :math:`R_{x}(\pi/2)`, but semantically represents part of
    a decomposed gate.

    :param qubit: Index for the qubit that the gate acts on.
    """

    inst: Literal["X_pi_2"] = "X_pi_2"

    @property
    def matrix(self):
        return (1 / np.sqrt(2)) * np.array([[1, -1j], [-1j, 1]])


class ZX_pi_4(Gate2Q, NativeGate):
    r"""A native :math:`ZX(\pi/4)` gate.

    .. warning:: While this gate mathematically represents a :math:`ZX(\pi/4)` gate, it
        is frequently used as a block within an :class:`ECR` gate, and is calibrated for
        this use. Using this gate on its own might yield unexpected results.

    :param qubit1: Target qubit for the Z gate.
    :param qubit2: Target qubit for the X gate.
    """

    inst: Literal["ZX_pi_4"] = "ZX_pi_4"

    @property
    def matrix(self):
        cos = np.cos(np.pi / 8)
        sin = np.sin(np.pi / 8)
        return np.array(
            [
                [cos, -1j * sin, 0, 0],
                [-1j * sin, cos, 0, 0],
                [0, 0, cos, 1j * sin],
                [0, 0, 1j * sin, cos],
            ]
        )
