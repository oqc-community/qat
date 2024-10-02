from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware

num_qubits = 2
hw = get_default_qiskit_hardware(num_qubits)
circ = hw.create_builder()
engine = hw.create_engine()

circ.had(hw.get_qubit(0))
for i in range(num_qubits - 1):
    circ.cnot(hw.get_qubit(i), hw.get_qubit(i + 1))
for i in range(num_qubits):
    circ.measure(hw.get_qubit(i))

results = engine.execute(circ)
# simulator = engine._create_simulator(builder)._configuration_options
# print(result.result().results[0].metadata['method'])
