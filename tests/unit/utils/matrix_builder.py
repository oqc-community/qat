# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from itertools import product
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import expm

from qat.purr.backends.echo import (
    Connectivity,
    apply_setup_to_hardware,
    generate_connectivity,
)
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Instruction


def get_default_matrix_hardware(
    qubit_count=4,
    connectivity: Optional[Union[Connectivity, List[Tuple[int, int]]]] = None,
) -> "MatrixHardwareModel":
    model = MatrixHardwareModel()
    if isinstance(connectivity, Connectivity):
        connectivity = generate_connectivity(connectivity, qubit_count)
    return apply_setup_to_hardware(model, qubit_count, connectivity)


class Gates:
    """
    Standard definition of common (and some not-so-common) unitary gates.
    Used to compare gate decompositions to their definitions.
    """

    @staticmethod
    def id():
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def _p0():
        """Projection operator onto |0>."""
        return np.array([[1, 0], [0, 0]])

    @staticmethod
    def _p1():
        """Projection operator onto |1>."""
        return np.array([[0, 0], [0, 1]])

    @staticmethod
    def rx(theta=np.pi):
        return np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )

    @staticmethod
    def ry(theta=np.pi):
        return np.array(
            [
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )

    @staticmethod
    def rz(theta=np.pi):
        return np.array(
            [
                [np.exp(-1j * theta / 2), 0.0],
                [0.0, np.exp(1j * theta / 2)],
            ]
        )

    @staticmethod
    def z():
        return np.array([[1, 0], [0, -1]])

    @staticmethod
    def y():
        return np.array([[0, -1j], [1j, 0]])

    @staticmethod
    def x():
        return np.array([[0, 1], [1, 0]])

    @staticmethod
    def sx():
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])

    @staticmethod
    def sxdg():
        return np.transpose(np.conj(Gates.sx()))

    @staticmethod
    def s():
        return np.array([[1, 0], [0, 1j]])

    @staticmethod
    def sdg():
        return np.conj(np.transpose(Gates.s()))

    @staticmethod
    def h():
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])

    @staticmethod
    def t():
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

    @staticmethod
    def p(lamb):
        return np.array([[1, 0], [0, np.exp(1j * lamb)]])

    @staticmethod
    def phase(lamb):
        return Gates().p(lamb)

    @staticmethod
    def tdg():
        return np.conj(np.transpose(Gates.t()))

    @staticmethod
    def U(theta, phi, lamb):
        return np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lamb) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lamb)) * np.cos(theta / 2),
                ],
            ]
        )

    @staticmethod
    def u(theta, phi, lamb):
        return Gates.U(theta, phi, lamb)

    @staticmethod
    def u1(lamb):
        return Gates.p(lamb)

    @staticmethod
    def u2(phi, lamb):
        return Gates.U(np.pi / 2, phi, lamb)

    @staticmethod
    def u3(theta, phi, lamb):
        return Gates.U(theta, phi, lamb)

    @staticmethod
    def rzx(theta):
        gate = np.kron(Gates.z(), Gates.x())
        return expm((-1j * theta / 2) * gate)

    @staticmethod
    def rxz(theta):
        gate = np.kron(Gates.x(), Gates.z())
        return expm((-1j * theta / 2) * gate)

    @staticmethod
    def rxx(theta):
        return expm((-1j * theta / 2) * np.kron(Gates.x(), Gates.x()))

    @staticmethod
    def rzz(theta):
        return expm((-1j * theta / 2) * np.kron(Gates.z(), Gates.z()))

    @staticmethod
    def ecr():
        return np.sqrt(1 / 2) * (
            np.kron(Gates.x(), np.eye(2)) - np.kron(Gates.y(), Gates.x())
        )

    @staticmethod
    def ecr_rev():
        return np.sqrt(1 / 2) * (
            np.kron(np.eye(2), Gates.x()) - np.kron(Gates.x(), Gates.y())
        )

    @staticmethod
    def swap():
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    @staticmethod
    def _control_gate(U, qubits=2, pos=1):
        """
        Calculates a control gate on some unitary U, where all other qubits act
        as control gates.
        """
        # First find all combinations of operators for the control qubits
        num_qubits_gate = int(np.log2(np.shape(U)[0]))
        control_combs = product(*[[0, 1] for _ in range(qubits - num_qubits_gate)])

        # We then need to loop through each combination and determine the action
        gate = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
        for comb in control_combs:
            _gate = 1
            ctr = 0
            for i in range(qubits + 1 - num_qubits_gate):
                if i == pos:
                    target = U if all(comb) else np.eye(2**num_qubits_gate)
                else:
                    target = Gates._p1() if comb[ctr] == 1 else Gates._p0()
                    ctr += 1
                _gate = np.kron(_gate, target)
            gate += _gate
        return gate

    @staticmethod
    def cnot(target=1):
        return Gates._control_gate(Gates.x(), qubits=2, pos=target)

    @staticmethod
    def cx(target=1):
        return Gates.cnot(target)

    @staticmethod
    def CX(target=1):
        return Gates.cnot(target)

    @staticmethod
    def cz(target=1):
        return Gates._control_gate(Gates.z(), qubits=2, pos=target)

    @staticmethod
    def cy(target=1):
        return Gates._control_gate(Gates.y(), qubits=2, pos=target)

    @staticmethod
    def csx(target=1):
        return Gates._control_gate(Gates.sx(), qubits=2, pos=target)

    @staticmethod
    def ch(target=1):
        return Gates._control_gate(Gates.h(), qubits=2, pos=target)

    @staticmethod
    def cp(lamb, target=1):
        return Gates._control_gate(Gates.p(lamb), qubits=2, pos=target)

    @staticmethod
    def cphase(lamb, target=1):
        return Gates._control_gate(Gates.phase(lamb), qubits=2, pos=target)

    @staticmethod
    def crx(theta, target=1):
        return Gates._control_gate(Gates.rx(theta), qubits=2, pos=target)

    @staticmethod
    def cry(theta, target=1):
        return Gates._control_gate(Gates.ry(theta), qubits=2, pos=target)

    @staticmethod
    def crz(theta, target=1):
        return Gates._control_gate(Gates.rz(theta), qubits=2, pos=target)

    @staticmethod
    def cu1(lamb, target=1):
        return Gates._control_gate(Gates.u1(lamb), qubits=2, pos=target)

    @staticmethod
    def cu2(phi, lamb, target=1):
        return Gates._control_gate(Gates.u2(phi, lamb), qubits=2, pos=target)

    @staticmethod
    def cu3(theta, phi, lamb, target=1):
        return Gates._control_gate(Gates.u3(theta, phi, lamb), qubits=2, pos=target)

    @staticmethod
    def cu(theta, phi, lamb, gamma, target=1):
        return Gates._control_gate(
            np.exp(1j * gamma) * Gates.U(theta, phi, lamb), qubits=2, pos=target
        )

    @staticmethod
    def ccx():
        return Gates._control_gate(Gates.x(), 3, 2)

    @staticmethod
    def cswap():
        return Gates._control_gate(Gates.swap(), 3, 1)

    @staticmethod
    def c3x():
        return Gates._control_gate(Gates.x(), 4, 3)

    @staticmethod
    def c3sqrtx():
        # why qelib1.inc calls this c3sqrtx and not c3sx is confusing...
        return Gates._control_gate(Gates.sx(), 4, 3)

    @staticmethod
    def c4x():
        return Gates._control_gate(Gates.x(), 5, 4)


class MatrixHardwareModel(QuantumHardwareModel):
    """
    A hardware model that overloads the instructions generated by native
    gates with their unitary matrix equivalents. Used along side the
    MatrixInstructionBuilder to construct exact unitary matrix descriptions
    of circuits.
    """

    def create_builder(self):
        return MatrixInstructionBuilder(self)

    def get_hw_x_pi_2(self, qubit, *args):
        qubit = qubit if isinstance(qubit, int) else qubit.index
        x_gate = 1.0
        for i in range(len(self.qubits)):
            if i == qubit:
                x_gate = np.kron(x_gate, Gates.rx(np.pi / 2))
            else:
                x_gate = np.kron(x_gate, np.eye(2))

        return [x_gate]

    def get_hw_z(self, qubit, phase, *args):
        qubit = qubit if isinstance(qubit, int) else qubit.index
        z_gate = 1.0
        for i in range(len(self.qubits)):
            if i == qubit:
                z_gate = np.kron(z_gate, Gates.rz(phase))
            else:
                z_gate = np.kron(z_gate, np.eye(2))

        return [z_gate]

    def get_hw_zx_pi_4(self, qubit, target, phase=np.pi / 4, *args):
        qubit = qubit if isinstance(qubit, int) else qubit.index
        target = target if isinstance(target, int) else target.index
        zx_gate = 1.0
        for i in range(len(self.qubits)):
            if i == qubit:
                zx_gate = np.kron(zx_gate, Gates.z())
            elif i == target:
                zx_gate = np.kron(zx_gate, Gates.x())
            else:
                zx_gate = np.kron(zx_gate, np.eye(2))

        return [expm((-1j * phase / 2) * zx_gate)]

    def get_gate_ZX(self, qubit, theta, target_qubit):
        return [*self.get_hw_zx_pi_4(qubit, target_qubit, theta)]


class MatrixInstructionBuilder(QuantumInstructionBuilder):
    """
    A QuantumInstructionBuilder that uses native gates from MatrixHardwareModel
    to construct a unitary matrix description of quantum circuits.

    It essentially overloads the add operation to instead perform sequential
    matrix multiplication.
    """

    def __init__(self, hardware_model, instructions=[]):
        self.matrix = np.eye(2 ** len(hardware_model.qubits))
        super().__init__(hardware_model, instructions)

    def add(self, matrices):
        if not isinstance(matrices, List):
            matrices = [matrices]

        for matrix in matrices:
            if not isinstance(matrix, Instruction):
                self.matrix = matrix @ self.matrix

        return self

    def ZX(self, control, target, theta):
        # only added for testing purposes
        return self.add(self.model.get_gate_ZX(control, theta, target))


def assert_same_up_to_phase(gate1, gate2):
    """
    If gates U_{1} and U_{2} are equivalent upto a global phase (U_{2} = e^{-i\\gamma} U_{1}), then

        U_{2}^{\\dagger} x U_{1} = e^{-i\\gamma} * identity.

    This fact can be used to test that this is true.
    """
    gate = np.conj(np.transpose(gate1)) @ gate2
    if np.isclose(gate[0, 0], 0.0 + 0.0j):
        assert False
    assert np.isclose(gate / gate[0, 0], np.eye(np.shape(gate1)[0])).all()


def extend_gate(gate, num_qubits, pos=0):
    """
    Used to extend a gate on a connected subsystem of qubits to a large number of qubits,
    num_qubits. The position of the first gate is given by pos.
    """

    num_qubits_gate = int(np.log2(np.size(gate, 0)))
    id_before = np.eye(2**pos)
    id_after = np.eye(2 ** (num_qubits - num_qubits_gate - pos))
    return np.kron(np.kron(id_before, gate), id_after)


def single_gate_list():
    func_to_gates = {
        "SX": ["SX", Gates.sx(), ()],
        "SXdg": ["SXdg", np.conj(np.transpose(Gates.sx())), ()],
        "T": ["T", Gates.t(), ()],
        "Tdg": ["Tdg", np.conj(np.transpose(Gates.t())), ()],
        "hadamard": ["had", Gates.h(), ()],
    }

    thetas = [0.0, np.pi / 2, np.pi, -np.pi / 2, 0.321, -1.58]
    for theta in thetas:
        func_to_gates[f"Rx({theta})"] = ["X", Gates.rx(theta), (theta,)]
        func_to_gates[f"Ry({theta})"] = ["Y", Gates.ry(theta), (theta,)]
        func_to_gates[f"Rz({theta})"] = ["Z", Gates.rz(theta), (theta,)]

    params = product(thetas, thetas, thetas)
    for args in params:
        func_to_gates[f"U({args[0]}, {args[1]}, {args[2]})"] = ["U", Gates.U(*args), args]

    return func_to_gates


def double_gate_list():
    func_to_gates = {
        "ZX(pi/4)": ["ZX", Gates.rzx(np.pi / 4), (np.pi / 4,)],
        "ZX(-pi/4)": ["ZX", Gates.rzx(-np.pi / 4), (-np.pi / 4,)],
        "ECR": ["ECR", Gates.ecr(), ()],
        "cnot": ["cnot", Gates.cnot(), ()],
    }
    return func_to_gates


def double_gate_rev_list():
    func_to_gates = {
        "XZ(pi/4)": ["ZX", Gates.rxz(np.pi / 4), (np.pi / 4,)],
        "XZ(-pi/4)": ["ZX", Gates.rxz(-np.pi / 4), (-np.pi / 4,)],
        "ECR": ["ECR", Gates.ecr_rev(), ()],
        "cnot": ["cnot", Gates.cnot(target=0), ()],
    }
    return func_to_gates
