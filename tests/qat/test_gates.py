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
    gate = np.conj(np.transpose(gate1)) @ gate2
    gate /= gate[0, 0]
    assert np.isclose(gate, np.eye(np.shape(gate1)[0])).all()


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


class TestSingleGates:

    model = get_default_matrix_hardware(1)

    @pytest.mark.parametrize(
        ["func_name", "args", "gate"],
        [(val[0], val[2], val[1]) for val in single_gate_list().values()],
    )
    def test_gates(self, func_name, args, gate):
        """Tests that the decomposition of single gates matches their definition."""
        if func_name == "had":
            pytest.skip("Hadamard is ill defined")
        builder = self.model.create_builder()
        gate_method = getattr(builder, func_name)
        gate_method(self.model.get_qubit(0), *args)
        assert_same_up_to_phase(builder.matrix, gate)

    @pytest.mark.skip("Hadamard gates currently have a non-standard implementation")
    def test_hadamard(self):
        """The Hadamard has various decompositions - test them also."""
        builder = self.model.create_builder()
        builder.h(self.model.get_qubit(0))

        assert_same_up_to_phase(builder.matrix, Gates.z() @ Gates.ry(-np.pi / 2))
        assert_same_up_to_phase(builder.matrix, Gates.ry(np.pi / 2) @ Gates.z())
        assert_same_up_to_phase(builder.matrix, Gates.x() @ Gates.ry(np.pi / 2))
        assert_same_up_to_phase(builder.matrix, Gates.ry(-np.pi / 2) @ Gates.x())


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


class TestDoubleGates:

    model = get_default_matrix_hardware(2)

    @pytest.mark.parametrize(
        ["func_name", "args", "gate"],
        [(val[0], val[2], val[1]) for val in double_gate_list().values()],
    )
    def test_two_gates(self, func_name, args, gate):
        """Test the various two gates supported by our builders."""
        builder = self.model.create_builder()
        gate_method = getattr(builder, func_name)
        gate_method(self.model.get_qubit(0), self.model.get_qubit(1), *args)
        assert_same_up_to_phase(builder.matrix, gate)

    @pytest.mark.parametrize(
        ["func_name", "args", "gate"],
        [(val[0], val[2], val[1]) for val in double_gate_rev_list().values()],
    )
    def test_two_gates_rev(self, func_name, args, gate):
        """
        Test the various two gates supported by our builders with qubit order
        reversed.
        """
        builder = self.model.create_builder()
        gate_method = getattr(builder, func_name)
        gate_method(self.model.get_qubit(1), self.model.get_qubit(0), *args)
        assert_same_up_to_phase(builder.matrix, gate)

    @pytest.mark.parametrize(
        ["func_name", "args", "gate", "qubit"],
        [
            (val[0], val[2], val[1], qubit)
            for val in single_gate_list().values()
            for qubit in [0, 1]
        ],
    )
    def test_single_gates(self, func_name, args, gate, qubit):
        """
        Tests that the decomposition of single gates matches their definition when
        applied to a multi-qubit model. Basically a sanity check that if gates are
        only applied to a single qubit, then the resulting circuit is a tensor
        product of this gate with the identity matrix.

        """
        if func_name == "had":
            pytest.skip("Hadamard is ill defined")
        builder = self.model.create_builder()
        gate_method = getattr(builder, func_name)
        gate_method(self.model.get_qubit(qubit), *args)
        if qubit == 0:
            gate = np.kron(gate, np.eye(2))
        else:
            gate = np.kron(np.eye(2), gate)
        assert_same_up_to_phase(builder.matrix, gate)


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
            pytest.skip("Gates are defined by their decompositions. Difficult to compare.")
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
