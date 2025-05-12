# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np

from qat.ir.gates.gates_1q import Hadamard, Phase, Rx, Ry, Rz, S, Sdg, T, Tdg, U, X, Y, Z
from qat.ir.gates.gates_2q import CNOT, ECR, SWAP, CPhase, CRx, CRy, CRz
from qat.ir.gates.operation import Barrier, Measure, Reset
from qat.ir.instructions import Instruction
from qat.ir.qat_ir import QatIR


class CircuitBuilder:
    """The :class:`CircuitBuilder` is used to construct high-level quantum programs.

    It can be used to implement high-level operations such as quantum gates and measure
    operations.
    """

    # TODO: support pulse instructions (COMPILER-304)
    # TODO: support named classical registers (COMPILER-285)
    # TODO: support control-flow; requires control flow instructions to be established
    # across all IR (COMPILER-146)

    def __init__(self, num_qubits: int, num_clbits: int = None):
        """
        :param num_qubits: The number of qubits in the circuit.
        :param num_clbits: The number of classical registers to save measurements too.
        """

        self.num_qubits = num_qubits
        self.num_clbits = num_clbits if num_clbits is not None else num_qubits
        self.instructions: list[Instruction] = []

    def add(self, gate):
        """Adds an operation to the circuit, validating the qubits indices are valid."""
        if not all([qubit < self.num_qubits for qubit in gate.qubits]):
            raise ValueError(
                f"The operation {gate} contains qubits with indices outside the valid "
                "range."
            )
        if len(gate.qubits) != len(set(gate.qubits)):
            raise ValueError(
                f"The operation {gate} contains qubits with duplicate indices."
            )
        self.instructions.append(gate)
        return self

    def X(self, qubit: int):
        r"""Adds an :math:`X` gate."""
        return self.add(X(qubit=qubit))

    def Y(self, qubit: int):
        r"""Adds a :math:`Y` gate."""
        return self.add(Y(qubit=qubit))

    def Z(self, qubit: int):
        r"""Adds a :math:`Z` gate."""
        return self.add(Z(qubit=qubit))

    def Rx(self, qubit: int, theta: float = np.pi):
        """Rotates a qubit around the x-axis by angle `theta`."""
        return self.add(Rx(qubit=qubit, theta=theta))

    def Ry(self, qubit: int, theta: float = np.pi):
        """Rotates a qubit around the y-axis by angle `theta`."""
        return self.add(Ry(qubit=qubit, theta=theta))

    def Rz(self, qubit: int, theta: float = np.pi):
        """Rotates a qubit around the z-axis by angle `theta`."""
        return self.add(Rz(qubit=qubit, theta=theta))

    def U(self, qubit: int, theta: float = 0.0, phi: float = 0.0, lambd: float = 0.0):
        """Apply a full rotation to a qubit around the Bloch sphere given three rotation
        angles.
        """
        return self.add(U(qubit=qubit, theta=theta, phi=phi, lambd=lambd))

    def Phase(self, qubit: int, theta: float = 0.0):
        r"""Applies a phase gate to a qubit, which can be considered equivalent to an
        :math:`R_{z}(\theta)` gate up to a global phase."""
        return self.add(Phase(qubit=qubit, theta=theta))

    def Hadamard(self, qubit: int):
        return self.add(Hadamard(qubit=qubit))

    def S(self, qubit: int):
        return self.add(S(qubit=qubit))

    def Sdg(self, qubit: int):
        return self.add(Sdg(qubit=qubit))

    def T(self, qubit: int):
        return self.add(T(qubit=qubit))

    def Tdg(self, qubit: int):
        return self.add(Tdg(qubit=qubit))

    def ECR(self, qubit1: int, qubit2: int):
        return self.add(ECR(qubit1=qubit1, qubit2=qubit2))

    def CNOT(self, control: int, target: int):
        return self.add(CNOT(qubit1=control, qubit2=target))

    def CX(self, control: int, target: int):
        return self.CNOT(control, target)

    def SWAP(self, qubit1: int, qubit2: int):
        return self.add(SWAP(qubit1=qubit1, qubit2=qubit2))

    def CRx(self, control: int, target: int, theta: float = np.pi):
        """Rotates the target qubit around the x-axis by angle `theta`, controlled by the
        state of another qubit."""
        return self.add(CRx(qubit1=control, qubit2=target, theta=theta))

    def CRy(self, control: int, target: int, theta: float = np.pi):
        """Rotates the target qubit around the y-axis by angle `theta`, controlled by the
        state of another qubit."""
        return self.add(CRy(qubit1=control, qubit2=target, theta=theta))

    def CRz(self, control: int, target: int, theta: float = np.pi):
        """Rotates the target qubit around the z-axis by angle `theta`, controlled by the
        state of another qubit."""
        return self.add(CRz(qubit1=control, qubit2=target, theta=theta))

    def CPhase(self, control: int, target: int, theta: float = np.pi):
        """Applies a phase rotation to a target qubit by angle `theta`, controlled by the
        state of another qubit."""
        return self.add(CPhase(qubit1=control, qubit2=target, theta=theta))

    def measure(self, qubit: int, clbit: int = None):
        """Measures a qubit and saves the result in the specified classical bit. If the
        classical bit isn't given, it will be decided automatically."""
        # TODO: implement a more sophisticed allocation procedure (COMPILER-305)
        if clbit is None:
            pass
        return self.add(Measure(qubit=qubit, clbit=clbit))

    def reset(self, qubit: int):
        """Reset a qubit to its :math:`|0>` state."""
        return self.add(Reset(qubit=qubit))

    def barrier(self, *qubits: int):
        """Adds a barrier between circuit operations to prevent optimisation over the
        barrier and sync qubits."""
        return self.add(Barrier(qubits=qubits))

    def emit(self):
        """Exports the circuit as :class:`QatIR`.

        Currently just wraps the instructions in :class:`QatIR`. Later the emitter could
        take on more responsibilities (e.g. some form of validation checks). It could prove
        benefitical to move this responsibility to a front end that wraps the builder as
        Qat IR complexity increases.
        """
        return QatIR(instructions=self.instructions)
