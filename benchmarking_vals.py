from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware

from tests.benchmarking.qatbm.circuits.qasm import bell_state as bell_state_qasm
from tests.benchmarking.qatbm.circuits.qasm import random_qasm_two_qubits
from tests.benchmarking.qatbm.circuits.qat import bell_state as bell_state_qat
from tests.benchmarking.qatbm.circuits.qat import random_circuit_two_qubits
from tests.benchmarking.qatbm.pipeline.pass_manager import default_benchmarking

# Storing the hardware and experiments as dicts to make the saved benchmark
# names understandable
hardware_two_qubits = {
    "echo": get_default_echo_hardware(2),
    "rtcs": get_default_RTCS_hardware(),
}
experiments_two_qubits = {
    "qasm_bell_state": bell_state_qasm(),
    "qasm_random_two_qubits": random_qasm_two_qubits(10),
    "qat_bell_state": bell_state_qat,
    "qat_random_two_qubits": lambda a, b: random_circuit_two_qubits(a, b, 10),
}


for hw in list(hardware_two_qubits.keys()):
    for circuit in list(experiments_two_qubits.keys()):
        pm = default_benchmarking(
            hardware_two_qubits[hw],
            experiments_two_qubits[circuit],
        )
        pm.run(100)
        pm.max.dump(f"{circuit}-{hw}.json")
