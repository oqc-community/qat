# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from abc import ABC
from numbers import Number

import numpy as np
from pydantic import AfterValidator
from typing_extensions import Annotated

from qat.ir.instructions import Instruction


class QubitInstruction(Instruction, ABC):
    """Denotes that an instruction is done at the level of a qubit (as opposed to a pulse
    channel level instruction).
    """


class GateBase(QubitInstruction, ABC):
    """Base implementation of a gate in a quantum circuit that acts on some qubits, and if
    needed, defined by some parameters.
    """

    def __repr__(self):
        params = tuple(
            [field for field in self.model_dump().values() if isinstance(field, float)]
        )
        return f"{self.__class__.__name__}(qubits={self.qubits}, params={params})"


def constrain_angle(theta: float) -> float:
    """Constrains a rotation angle between [-np.pi, np.pi)."""
    if not isinstance(theta, Number) or np.iscomplex(theta):
        raise ValueError("Angle must be a real number.")
    return (theta + np.pi) % (2 * np.pi) - np.pi


def is_equal_angle(angle1: float, angle2: float) -> bool:
    diff = (angle1 - angle2) % (2 * np.pi)
    return np.isclose(diff, 0.0) or np.isclose(diff, 2 * np.pi)


Angle = Annotated[float, AfterValidator(constrain_angle)]
