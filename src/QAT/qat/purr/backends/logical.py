import ast

import math
from typing import Any, List, Optional, Set, Union
from enum import Enum, auto

from qat.purr.compiler.builders import Axis, InstructionBuilder
from qat.purr.compiler.config import InlineResultsProcessing
from qat.purr.compiler.instructions import Acquire, ProcessAxis, ResultsProcessing
from qat.purr.compiler.runtime import get_builder

from src.QAT.qat.purr.compiler.hardware_models import HardwareModel, QuantumHardwareModel
import numpy as np

class LogicalQubit:
    def __init__(self, index):
        self.index = index

class LogicalQubitCoupling:
    def __init__(self, direction):
        """
        Direction of coupling stated in a tuple: (4,5) means we have a  4 -> 5 coupling.
        """
        self.direction = tuple(direction)
        self.quality = 1

class Gate(Enum):
    x = auto()
    rz = auto()
    zx = auto()
    meas = auto()


class LogicalGate:
    def __init__(self, qubit:int, gate:Gate, angle:float=None, target_qubit:int=None):
        self.qubit = qubit
        self.gate = gate
        self.angle = angle
        self.target_qubit=target_qubit
    
    def make_dict(self):
        return {"qubit":self.qubit, "gate": self.gate.name, "angle": self.angle, "target_qubit": self.target_qubit}

    def __str__(self):
        return f"{self.gate}[{self.qubit},{self.target_qubit} , {self.angle}]"

class LogicalHardware():
    def __init__(self, qubits: Optional[List[LogicalQubit]]=None, couplings: Optional[List[LogicalQubitCoupling]]=None):
        super().__init__()
        self.qubits = qubits
        self.couping = couplings
        self.qubit_direction_couplings = couplings


    def get_hw_x_pi_2(self, qubit:LogicalQubit)->LogicalGate:
        return LogicalGate(qubit=qubit.index, gate=Gate.x)

    def get_hw_z(self, qubit:LogicalQubit, phase:float)->LogicalGate:
        return LogicalGate(qubit=qubit.index, gate=Gate.rz, angle=phase)

    def get_hw_zx_pi_4(self, qubit: LogicalQubit, target_qubit: LogicalQubit) -> LogicalGate:
        return LogicalGate(qubit=qubit.index, gate=Gate.zx, target_qubit=target_qubit.index)
    
    def get_gate_X(self, qubit, theta, ):
        return self.get_gate_U(qubit, theta, -np.pi / 2.0, np.pi / 2.0)

    def get_gate_ZX(self, qubit, theta, target_qubit):
        if np.isclose(theta, 0.0):
            return []
        elif np.isclose(theta, np.pi / 4.0):
            return [self.get_hw_zx_pi_4(qubit, target_qubit)]
        elif np.isclose(theta, -np.pi / 4.0):
            return [
                self.get_hw_z(qubit, np.pi),
                self.get_hw_z(target_qubit, np.pi),
                self.get_hw_zx_pi_4(qubit, target_qubit),
                self.get_hw_z(qubit, np.pi),
                self.get_hw_z(target_qubit, np.pi),
            ]

    def get_gate_U(self, qubit, theta, phi, lamb):
        return [
            self.get_hw_z(qubit, lamb + np.pi),
            self.get_hw_x_pi_2(qubit),
            self.get_hw_z(qubit, np.pi - theta),
            self.get_hw_x_pi_2(qubit),
            self.get_hw_z(qubit, phi),
        ]
    
    # def 

    def get_qubit(self, qubit):
        return self.qubits[qubit]
    
    def create_builder(self):
        return LogicalBuilder(self)

def get_default_logical_hardware(qubit_count):
    qubits = [LogicalQubit(index=i) for i in range(qubit_count)]
    connectivity = [LogicalQubitCoupling((i, (i + 1) % qubit_count)) for i in range(qubit_count)]
    return LogicalHardware(qubits=qubits, couplings=connectivity)

class LogicalBuilder(InstructionBuilder):
    def __init__(self, hardware_model: QuantumHardwareModel):
        # super().__init__()
        self.model = hardware_model
        self._instructions = []
    
    def serialise(self):
        # output = []
        # for inst in self._instructions:
        #     if isinstance(inst, LogicalGate):
        #         output.append(inst.make_dict())
        return ";".join(str(inst.make_dict()) for inst in self._instructions if isinstance(inst, LogicalGate))

    def add(self, inst: Union[List[Gate], Gate]):
        if isinstance(inst, List):
            self._instructions.extend(inst)
        else:
            self._instructions.append(inst)
        return self

    def R(self, axis: Axis, target: LogicalQubit, radii=None):
        if radii is None:
            radii = np.pi

        if axis == Axis.X:
            self._instructions.append(self.model.get_hw_z(qubit=target, phase=radii))
        elif axis == Axis.Y:
            self._instructions.append(self.model.get_hw_z(qubit=target, phase=radii))
        elif axis == Axis.Z:
            self._instructions.append(self.model.get_hw_z(qubit=target, phase=radii))
        return self

    def results_processing(self, variable: str, res_format: InlineResultsProcessing):
        return self.add(ResultsProcessing(variable, res_format))

    def measure_single_shot_z(
        self, target: LogicalQubit, axis: ProcessAxis = None, output_variable: str = None
    ):
        return self.measure(target, axis, output_variable)

    def measure_single_shot_signal(
        self, target: LogicalQubit, axis: ProcessAxis = None, output_variable: str = None
    ):
        return self.measure(target, axis, output_variable)

    def measure_mean_z(
        self, target: LogicalQubit, axis: ProcessAxis = None, output_variable: str = None
    ):
        return self.measure(target, axis, output_variable)

    def measure_mean_signal(self, target: LogicalQubit, output_variable: str = None):
        return self.measure(target, output_variable=output_variable)

    def measure(
        self, target: LogicalQubit, axis: ProcessAxis = None, output_variable: str = None
    ) -> "InstructionBuilder":
        self.add(LogicalGate(qubit=target.index, gate=Gate.meas))


    def X(self, target: LogicalQubit, radii=None):
        return self.R(Axis.X, target, radii)

    def Y(self, target: LogicalQubit, radii=None):
        return self.R(Axis.Y, target, radii)

    def Z(self, target: LogicalQubit, radii=None):
        return self.R(Axis.Z, target, radii)

    def U(self, target: LogicalQubit, theta, phi, lamb):
        return self.Z(target, lamb).Y(target, theta).Z(target, phi)

    def swap(self, target: LogicalQubit, destination: LogicalQubit):
        raise ValueError("Not available on this hardware model.")

    def had(self, qubit: LogicalQubit):
        self.Z(qubit)
        return self.Y(qubit, math.pi / 2)

    def post_processing(
        self, acq: Acquire, process, axes=None, target: LogicalQubit = None, args=None
    ):
        raise ValueError("Not available on this hardware model.")

    def sweep(self, variables_and_values):
        raise ValueError("Not available on this hardware model.")

    def pulse(self, *args, **kwargs):
        raise ValueError("Not available on this hardware model.")

    def acquire(self, *args, **kwargs):
        raise ValueError("Not available on this hardware model.")

    def delay(self, target: LogicalQubit, time: float):
        raise ValueError("Not available on this hardware model.")

    def synchronize(self, targets: Union[LogicalQubit, List[LogicalQubit]]):
        raise ValueError("Not available on this hardware model.")

    def phase_shift(self, target: LogicalQubit, phase):
        raise ValueError("Not available on this hardware model.")

    def SX(self, target):
        return self.X(target, np.pi / 2)

    def SXdg(self, target):
        return self.X(target, -(np.pi / 2))

    def S(self, target):
        return self.Z(target, np.pi / 2)

    def Sdg(self, target):
        return self.Z(target, -(np.pi / 2))

    def T(self, target):
        return self.Z(target, np.pi / 4)

    def Tdg(self, target):
        return self.Z(target, -(np.pi / 4))

    def cR(
        self,
        axis: Axis,
        controllers: Union[LogicalQubit, List[LogicalQubit]],
        target: LogicalQubit,
        theta: float,
    ):
        raise ValueError("Generic controlled rotations not available.")

    def cX(self, controllers: Union[LogicalQubit, List[LogicalQubit]], target: LogicalQubit, radii=None):
        return self.cnot(Axis.X, controllers, target, radii)

    def cY(self, controllers: Union[LogicalQubit, List[LogicalQubit]], target: LogicalQubit, radii=None):
        return self.cR(Axis.Y, controllers, target, radii)

    def cZ(self, controllers: Union[LogicalQubit, List[LogicalQubit]], target: LogicalQubit, radii=None):
        return self.cR(Axis.Z, controllers, target, radii)

    def cnot(self, control: LogicalQubit, target_qubit: LogicalQubit):
        self.ECR(control, target_qubit)
        self.X(control)
        self.Z(control, -np.pi / 2)
        return self.X(target_qubit, -np.pi / 2)

    def ccnot(self, cone: LogicalQubit, ctwo: LogicalQubit, target_qubit: LogicalQubit):
        raise self.cX([cone, ctwo], target_qubit, np.pi)

    def cswap(self, controllers: Union[LogicalQubit, List[LogicalQubit]], target, destination):
        raise ValueError("Not available on this hardware model.")

    def ECR(self, control: LogicalQubit, target: LogicalQubit):
        instructions = []
        instructions.extend(self.model.get_gate_ZX(control, np.pi / 4.0, target))
        instructions.extend(self.model.get_gate_X(control, np.pi))
        instructions.extend(self.model.get_gate_ZX(control, -np.pi / 4.0, target))
        self.add(instructions)


def convert_to_other(logical_string, hardware:QuantumHardwareModel):
    gates =[ast.literal_eval(string) for string in logical_string.split(";")]

    builder = get_builder(hardware)

    for gate in gates:
        logical_gate = gate['gate']
        qubit = hardware.get_qubit(gate['qubit'])
        if logical_gate == Gate.x.name:
            builder._instructions.append(hardware.get_hw_x_pi_2(qubit=qubit))
        elif logical_gate == Gate.rz.name:
            builder._instructions.append(hardware.get_hw_z(qubit=qubit, phase=gate['angle']))
        elif logical_gate == Gate.zx.name:
            builder._instructions.append(hardware.get_hw_zx_pi_4(qubit=qubit, target_qubit=hardware.get_qubit(gate['target_qubit'])))
    return builder