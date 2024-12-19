# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from itertools import product

import numpy as np
import pytest

from qat.purr.integrations.qasm import (
    Qasm2Parser,
    Qasm3Parser,
    Qasm3ParserBase,
    QasmContext,
)

from tests.qat.utils.matrix_builder import Gates, get_default_matrix_hardware


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


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
class TestSingleGates:

    @pytest.mark.parametrize(
        ["func_name", "args", "gate"],
        [(val[0], val[2], val[1]) for val in single_gate_list().values()],
    )
    def test_gates(self, func_name, args, gate, num_qubits):
        """Tests that the decomposition of single gates matches their definition."""
        model = get_default_matrix_hardware(num_qubits)

        for pos in range(num_qubits):
            builder = model.create_builder()
            gate_method = getattr(builder, func_name)
            gate_method(model.get_qubit(pos), *args)
            assert_same_up_to_phase(builder.matrix, extend_gate(gate, num_qubits, pos))

    def test_hadamard(self, num_qubits):
        """The Hadamard has various decompositions - test them also."""
        model = get_default_matrix_hardware(num_qubits)
        decompositions = [
            Gates.z() @ Gates.ry(-np.pi / 2),
            Gates.ry(np.pi / 2) @ Gates.z(),
            Gates.x() @ Gates.ry(np.pi / 2),
            Gates.ry(-np.pi / 2) @ Gates.x(),
        ]

        for pos in range(num_qubits):
            builder = model.create_builder()
            builder.had(model.get_qubit(pos))
            for gate in decompositions:
                assert_same_up_to_phase(builder.matrix, extend_gate(gate, num_qubits, pos))


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


@pytest.mark.parametrize("num_qubits", [2, 3, 4])
class TestDoubleGates:
    @pytest.mark.parametrize(
        ["func_name", "args", "gate"],
        [(val[0], val[2], val[1]) for val in double_gate_list().values()],
    )
    def test_two_gates(self, func_name, args, gate, num_qubits):
        """Test the various two gates supported by our builders."""
        model = get_default_matrix_hardware(num_qubits)
        for pos in range(num_qubits - 1):
            builder = model.create_builder()
            gate_method = getattr(builder, func_name)
            gate_method(model.get_qubit(pos), model.get_qubit(pos + 1), *args)
            assert_same_up_to_phase(builder.matrix, extend_gate(gate, num_qubits, pos))

    @pytest.mark.parametrize(
        ["func_name", "args", "gate"],
        [(val[0], val[2], val[1]) for val in double_gate_rev_list().values()],
    )
    def test_two_gates_rev(self, func_name, args, gate, num_qubits):
        """
        Test the various two gates supported by our builders with qubit order
        reversed.
        """
        model = get_default_matrix_hardware(num_qubits)
        for pos in range(num_qubits - 1):
            builder = model.create_builder()
            gate_method = getattr(builder, func_name)
            gate_method(model.get_qubit(pos + 1), model.get_qubit(pos), *args)
            assert_same_up_to_phase(builder.matrix, extend_gate(gate, num_qubits, pos))


class TestQasm2:

    qasm2_base = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[{N}];
    {gate_strings}
    """

    @pytest.mark.parametrize(
        "gate", Qasm2Parser()._get_intrinsics(), ids=lambda val: val.name
    )
    def test_qasm2_gates(self, gate):
        """Check that each QASM2 gate can be parsed individually."""
        # Skip some gates
        if gate.name == "rccx" or gate.name == "rc3x":
            pytest.skip("Gate is defined by its decompositions. Difficult to compare.")
        elif gate.name == "delay":
            pytest.skip("Delay not implemented.")
        elif gate.name == "id" or gate.name == "u0":
            pytest.skip(f"Gate {gate.name} isn't intrinsic and has no body.")

        # Create an assortment of parameters
        thetas = [0.0, np.pi / 2, np.pi, -np.pi / 2, 0.321, -1.58]

        if gate.num_params > 0:
            args_list = product(*[thetas for _ in range(gate.num_params)])
        else:
            args_list = [tuple()]

        for args in args_list:
            # contruct the qasm qate
            gate_string = gate.name
            if len(args) > 0:
                gate_string += f"(" + ", ".join([str(arg) for arg in args]) + ")"
            gate_string += (
                " " + ", ".join([f"q[{i}]" for i in range(gate.num_qubits)]) + ";"
            )
            qasm = self.qasm2_base.format(N=gate.num_qubits, gate_strings=gate_string)

            # parse it through the hardware and verify result
            hw = get_default_matrix_hardware(gate.num_qubits)
            parser = Qasm2Parser()
            builder = parser.parse(hw.create_builder(), qasm)
            gate_method = getattr(Gates, gate.name)
            actual_gate = gate_method(*args)
            assert_same_up_to_phase(builder.matrix, actual_gate)


def qasm3_gates():
    context = QasmContext()
    Qasm3ParserBase().load_default_gates(context)
    return context.gates


class TestQasm3:

    qasm3_base = """
    OPENQASM 3.0;
    qreg q[{N}];
    {gate_strings}
    """

    @pytest.mark.parametrize(
        ["name", "gate"],
        qasm3_gates().items(),
    )
    def test_qasm3_gates(self, name, gate):
        """Check that each QASM3 gate can be parsed individually."""
        # Create an assortment of parameters
        thetas = [0.0, np.pi / 2, np.pi, -np.pi / 2, 0.321, -1.58]
        num_params = len(gate.arguments)
        qubits = len(gate.qubits)
        if num_params > 0:
            args_list = product(*[thetas for _ in range(num_params)])
        else:
            args_list = [tuple()]

        for args in args_list:
            # contruct the qasm qate
            gate_string = name
            if len(args) > 0:
                gate_string += f"(" + ", ".join([str(arg) for arg in args]) + ")"
            gate_string += " " + ", ".join([f"q[{i}]" for i in range(qubits)]) + ";"
            qasm = self.qasm3_base.format(N=qubits, gate_strings=gate_string)

            # parse it through the hardware and verify result
            hw = get_default_matrix_hardware(qubits)
            parser = Qasm3Parser()
            builder = parser.parse(hw.create_builder(), qasm)
            gate_method = getattr(Gates, name)
            actual_gate = gate_method(*args)
            assert_same_up_to_phase(builder.matrix, actual_gate)
