# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from typing import Any

from pydantic import Field

from qat.model.device import Qubit
from qat.utils.pydantic import NoExtraFieldsModel, ValidatedDict, ValidatedList


class QubitRegister(NoExtraFieldsModel):
    qubits: ValidatedList[Qubit] = Field(default_factory=lambda: ValidatedList[Qubit]())

    @property
    def contents(self):
        return list(self.qubits)

    def __repr__(self):
        return "QubitRegister: " + str(self.qubits)


class CregIndexValue(NoExtraFieldsModel):
    """
    Used to reference when we are looking at a particular index in a creg variable.
    """

    register_name: str
    index: int
    value: Any

    @property
    def variable(self):
        return f"{self.register_name}[{self.index}]"

    def __repr__(self):
        return self.variable


class BitRegister(NoExtraFieldsModel):
    bits: ValidatedList[CregIndexValue] = Field(
        default_factory=lambda: ValidatedList[CregIndexValue]()
    )

    @property
    def contents(self):
        return list(self.bits)

    def __repr__(self):
        return "BitRegister: " + str(self.bits)


class Registers(NoExtraFieldsModel):
    quantum: ValidatedDict[str, QubitRegister] = Field(
        default_factory=lambda: ValidatedDict[str, QubitRegister]()
    )
    classic: ValidatedDict[str, BitRegister] = Field(
        default_factory=lambda: ValidatedDict[str, BitRegister]()
    )

    def __repr__(self):
        return "quantum: " + str(self.quantum) + "\nclassic: " + str(self.classic)
