from typing import Any, List, Optional, Set

from src.QAT.qat.purr.compiler.hardware_models import HardwareModel
import numpy as np

class LogicalQubit:
    def __init__(self, index):
        self.index = index
        self.coupled_qubits: Set[int] = set()

class LogicalQubitCoupling:
    def __init__(self, direction):
        """
        Direction of coupling stated in a tuple: (4,5) means we have a  4 -> 5 coupling.
        """
        self.direction = tuple(direction)

class LogicalHardware(HardwareModel):
    def __init__(self, qubits: Optional[List[LogicalQubit]]=None):
        super().__init__()
        self.qubits = qubits


    def get_hw_x_pi_2(self, qubit): ...

    def get_hw_z(self, qubit, phase): ...

    def get_hw_zx_pi_4(self, qubit: LogicalQubit, target_qubit: LogicalQubit) -> List[Any]:

    def get_gate_U(self, qubit, theta, phi, lamb):
        return [
            *self.get_hw_z(qubit, lamb + np.pi),
            *self.get_hw_x_pi_2(qubit),
            *self.get_hw_z(qubit, np.pi - theta),
            *self.get_hw_x_pi_2(qubit),
            *self.get_hw_z(qubit, phi),
        ]