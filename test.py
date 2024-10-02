from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

num_qubits = 60
er = 1e-3

circ = QuantumCircuit(num_qubits, num_qubits)
for i in range(num_qubits):
    circ.rx(np.random.rand(), i)
circ.h(0)
for i in range(num_qubits - 1):
    circ.cx(i, i + 1)
circ.measure(range(num_qubits), range(num_qubits))

noise = NoiseModel()
noise.add_all_qubit_readout_error([[1 - er, er], [er, 1 - er]])

sim = AerSimulator(noise_model=noise, method="matrix_product_state")
sim.set_max_qubits(num_qubits)
circ = transpile(circ, sim)
t = perf_counter()
result = sim.run(circ)
print(perf_counter() - t)
print(result.result().results[0].metadata["method"])
# counts = result.get_counts(circ)
# print(counts)
