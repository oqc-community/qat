import numpy as np


def bell_state(model, builder, measurements: bool = True):
    """
    Create a bell state |00> + |11> using QATs circuit builder.
    """
    q1 = model.get_qubit(0)
    q2 = model.get_qubit(1)
    builder.had(q1).cnot(q1, q2)
    if measurements:
        builder.measure(q1).measure(q2)
    return builder


def ghz_state(model, builder, measurements: bool = True):
    """
    Create a GHZ state |0...0> + |1...1> using QATs circuit builder.
    """
    qubits = model.qubits
    builder.had(qubits[0])
    for i in range(len(qubits) - 1):
        builder.cnot(qubits[i], qubits[1])
    if measurements:
        for qubit in qubits:
            builder.measure(qubit)
    return builder


def random_circuit_two_qubits(hw, builder, num_cnots: int, measurements: bool = True):
    """
    Generate a circuit of CNOTs interweved with random single body unitaries using
    QAT's circuit builder for two qubits.
    """
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


def random_circuit(hw, builder, num_cnots: int, measurements: bool = True):
    """
    Generate a circuit of CNOTs interweved with random single body unitaries using
    QAT's circuit builder.
    """
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
