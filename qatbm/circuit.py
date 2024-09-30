import numpy as np

from qat.purr.compiler.runtime import get_builder


def random_circuit_two_qubits(hw, num_cnots: int, measurements: bool = True):
    # Create a builder
    builder = get_builder(hw)
    q0 = hw.get_qubit(0)
    q1 = hw.get_qubit(1)

    # Add unitaries
    builder = _add_random_unitary(builder, q0)
    builder = _add_random_unitary(builder, q1)

    # Add CNOTS and random unitaries
    for _ in range(num_cnots):
        builder.cnot(q0, q1)
        builder = _add_random_unitary(builder, q0)
        builder = _add_random_unitary(builder, q1)

    if measurements:
        builder.measure_single_shot_z(q0)
        builder.measure_single_shot_z(q1)

    return builder


def random_circuit(hw, num_cnots: int, measurements: bool = True):
    # Create a builder
    builder = get_builder(hw)
    qubits = hw.qubits

    # Add unitaries
    for qubit in qubits:
        _add_random_unitary(builder, qubit)

    # Add CNOTs and random unitaries
    for i in range(num_cnots):
        rng = range(1 if i % 2 == 0 else 0, len(qubits) - 1, 2)
        for j in rng:
            builder.cnot(qubits[j], qubits[j + 1])

        for qubit in qubits:
            _add_random_unitary(builder, qubit)

    # Measurements
    if measurements:
        for qubit in qubits:
            builder.measure_single_shot_z(qubit)

    return builder


def _add_random_unitary(builder, qubit):
    rands = np.random.rand(3) * 2 * np.pi
    builder.U(qubit, *rands)
    return builder
