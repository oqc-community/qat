import numpy as np


def random_qasm_two_qubits(num_cnots: int, measurements: bool = True):
    """
    Generate a random QASM script of CNOTs sandwiched between random single-
    body unitaries.
    """

    # First string
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    """

    # Add some unitaries
    qasm = _add_random_unitary(qasm, 0)
    qasm = _add_random_unitary(qasm, 1)

    # Add CNOT and randoms
    for i in range(num_cnots):
        qasm += """cx q[0], q[1]; \n"""
        qasm = _add_random_unitary(qasm, 0)
        qasm = _add_random_unitary(qasm, 1)

    # Add measurements
    if measurements:
        qasm += """measure q -> c; \n"""
    return qasm


def random_qasm(num_qubits: int, num_cnots: int, measurements: bool = True):
    """
    Generate a random QASM script for many qubits.
    The quantum circuit has CNOT layers that alternate between odd and even bonds.
    In between each CNOT layer are random single-body unitaries.
    """

    # First string
    qasm = f"""
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[{num_qubits}];
    creg c[{num_qubits}];
    """

    # Add some unitaries
    for i in range(num_qubits):
        qasm = _add_random_unitary(qasm, i)

    # Add CNOT and randoms
    for i in range(num_cnots):
        rng = range(1 if i % 2 == 0 else 0, num_qubits - 1, 2)
        for j in rng:
            qasm += f"""cx q[{j}], q[{j+1}]; \n"""
        for j in range(num_qubits):
            qasm = _add_random_unitary(qasm, j)

    # Add measurements
    if measurements:
        qasm += """measure q -> c; \n"""
    return qasm


def _add_random_unitary(qasm: str, qubit: int):
    rands = np.random.rand(3) * 2 * np.pi
    qasm += f"""u3({rands[0]}, {rands[1]}, {rands[2]}) q[{qubit}]; \n"""
    return qasm
