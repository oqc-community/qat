import ast

import math
from typing import Any, List, Optional, Set, Union
from enum import Enum, auto

from qat.purr.backends.qiskit_simulator import QiskitHardwareModel
from qat.purr.compiler.builders import Axis, FluidBuilderWrapper, InstructionBuilder, QuantumInstructionBuilder
from qat.purr.compiler.config import InlineResultsProcessing
from qat.purr.compiler.instructions import Acquire, ProcessAxis, ResultsProcessing
from qat.purr.compiler.runtime import get_builder

from qat.purr.compiler.hardware_models import HardwareModel, QuantumHardwareModel
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
        angle = self.angle
        if self.angle is not None:
            if np.isclose(self.angle, np.pi):
                angle = "p"
            elif np.isclose(self.angle, -np.pi):
                angle = "-p"
            if np.isclose(self.angle, np.pi/2):
                angle = "p2"
            elif np.isclose(self.angle, -np.pi/2):
                angle = "-p2"
        return {"q":self.qubit, "g": self.gate.name, "a": angle, "t": self.target_qubit}

    def __str__(self):
        return f"{self.gate}[{self.qubit},{self.target_qubit} , {self.angle}]"

class LogicalHardware(QuantumHardwareModel):
    def __init__(self, qubits: Optional[List[LogicalQubit]]=None, couplings: Optional[List[LogicalQubitCoupling]]=None):
        super().__init__()
        self._qubits = qubits
        self.qubit_direction_couplings = couplings

    @property
    def qubits(self):
        return self._qubits

    def get_hw_x_pi_2(self, qubit:LogicalQubit, *args)->LogicalGate:
        return [LogicalGate(qubit=qubit.index, gate=Gate.x)]

    def get_hw_z(self, qubit:LogicalQubit, phase:float, *args)->LogicalGate:
        return [LogicalGate(qubit=qubit.index, gate=Gate.rz, angle=phase)]

    def get_hw_zx_pi_4(self, qubit: LogicalQubit, target_qubit: LogicalQubit, *args) -> LogicalGate:
        return [LogicalGate(qubit=qubit.index, gate=Gate.zx, target_qubit=target_qubit.index)]
    
    def get_gate_X(self, qubit, theta, *args):
        return [self.get_gate_U(qubit, theta, -np.pi / 2.0, np.pi / 2.0)]

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

    def get_gate_U(self, qubit, theta, phi, lamb, *args):
        return [
            self.get_hw_z(qubit, lamb + np.pi),
            self.get_hw_x_pi_2(qubit),
            self.get_hw_z(qubit, np.pi - theta),
            self.get_hw_x_pi_2(qubit),
            self.get_hw_z(qubit, phi),
        ]
    
    # def 
    def _resolve_qb_pulse_channel(self, target):
        return target, None

    def get_qubit(self, qubit):
        return self.qubits[qubit]
    
    def create_builder(self):
        return LogicalBuilder(self)

def get_default_logical_hardware(qubit_count):
    qubits = [LogicalQubit(index=i) for i in range(qubit_count)]
    connectivity = [LogicalQubitCoupling((i, (i + 1) % qubit_count)) for i in range(qubit_count)]
    return LogicalHardware(qubits=qubits, couplings=connectivity)

class LogicalBuilder(QuantumInstructionBuilder):
    def __init__(self, hardware_model: QuantumHardwareModel):
        # super().__init__()
        self.model = hardware_model
        self._instructions = []
    
    def repeat(self, *args, **kwargs):
        return self

    def serialise(self):
        return ";".join(str(inst.make_dict()) for inst in self._instructions[0].instructions if isinstance(inst, LogicalGate))

    def add(self, inst: Union[List[Gate], Gate]):
        if isinstance(inst, List):
            for ins in inst:
                self.add(ins)
            # self._instructions.extend(inst)
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

    def measure(
        self, target: LogicalQubit, axis: ProcessAxis = None, output_variable: str = None
    ) -> "InstructionBuilder":
        self.add(LogicalGate(qubit=target.index, gate=Gate.meas))
        return FluidBuilderWrapper(self, None)


    def cX(self, controllers: Union[LogicalQubit, List[LogicalQubit]], target: LogicalQubit, radii=None):
        return self.cnot(Axis.X, controllers, target, radii)

    def cnot(self, control: LogicalQubit, target_qubit: LogicalQubit):
        self.ECR(control, target_qubit)
        self.X(control)
        self.Z(control, -np.pi / 2)
        return self.X(target_qubit, -np.pi / 2)

    def ECR(self, control: LogicalQubit, target: LogicalQubit):
        instructions = []
        instructions.extend(self.model.get_gate_ZX(control, np.pi / 4.0, target))
        instructions.extend(self.model.get_gate_X(control, np.pi))
        instructions.extend(self.model.get_gate_ZX(control, -np.pi / 4.0, target))
        return self.add(instructions)

def get_angle(angle):
    if isinstance(angle, float):
        return angle
    if angle == "p":
        return np.pi
    elif angle == "-p":
        return -np.pi
    elif angle == "p2":
        return np.pi/2
    elif angle == "-p2":
        return -np.pi/2
    return None

def convert_to_other(logical_string, hardware:QuantumHardwareModel):
    gates =[ast.literal_eval(string) for string in logical_string.split(";")]

    builder = get_builder(hardware)
    if not isinstance(hardware, QiskitHardwareModel):
        for gate in gates:
            logical_gate = gate['g']
            qubit = hardware.get_qubit(gate['q'])
            if logical_gate == Gate.x.name:
                builder._instructions.append(hardware.get_hw_x_pi_2(qubit=qubit))
            elif logical_gate == Gate.rz.name:
                builder._instructions.append(hardware.get_hw_z(qubit=qubit, phase=get_angle(gate['a'])))
            elif logical_gate == Gate.zx.name:
                builder._instructions.append(hardware.get_hw_zx_pi_4(qubit=qubit, target_qubit=hardware.get_qubit(gate['t'])))
            # elif logical_gate == Gate.meas.name:
                # builder._instructions.append()
    else:
        for gate in gates:
            logical_gate = gate['g']
            qubit = hardware.get_qubit(gate['q'])
            if logical_gate == Gate.x.name:
                builder.R(axis=Axis.X, target=qubit, radii=math.pi/2)
            elif logical_gate == Gate.rz.name:
                builder.R(axis=Axis.Z, target=qubit, radii=get_angle(gate['a']))
            elif logical_gate == Gate.zx.name:
                builder.ECR(control=qubit, target=hardware.get_qubit(gate['t']))
            # elif logical_gate == Gate.meas.name:
        builder.circuit.measure_all()
    return builder