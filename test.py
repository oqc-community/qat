from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.devices import Calibratable
from qat.purr.compiler.runtime import execute_instructions, get_builder

# serialize and deserialize a model
model = get_default_RTCS_hardware()
for c in model.couplings:
    c.is_calibrated = True
model.save_calibration_to_file("test.json", True)
model = Calibratable().load_calibration_from_file("test.json")

# execute a circuit
circuit = (
    get_builder(model)
    .had(model.qubits[0])
    .cnot(model.qubits[0], model.qubits[1])
    .measure(model.qubits[0])
    .measure(model.qubits[1])
)
res = execute_instructions(model, circuit)
