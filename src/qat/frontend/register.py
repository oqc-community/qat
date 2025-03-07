from typing import Any

from qat.model.device import Qubit
from qat.utils.pydantic import NoExtraFieldsModel, ValidatedDict, ValidatedList


class QubitRegister(NoExtraFieldsModel):
    qubits: ValidatedList[Qubit] = ValidatedList[Qubit]()

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
    bits: ValidatedList[CregIndexValue] = ValidatedList[CregIndexValue]()

    @property
    def contents(self):
        return list(self.bits)

    def __repr__(self):
        return "BitRegister: " + str(self.bits)


class Registers(NoExtraFieldsModel):
    quantum: ValidatedDict[str, QubitRegister] = ValidatedDict[str, QubitRegister]()
    classic: ValidatedDict[str, BitRegister] = ValidatedDict[str, BitRegister]()

    def __repr__(self):
        return "quantum: " + str(self.quantum) + "\nclassic: " + str(self.classic)
