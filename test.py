from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.devices import Calibratable
from qat.purr.compiler.runtime import execute_instructions, get_builder


def deserialize_hw_model(version, model):
    hw_dir = "tests/qat/files/hw_models/"
    return Calibratable().load_calibration_from_file(
        hw_dir + version + "/" + model + ".json"
    )


model = deserialize_hw_model("2.3.0", "echo")
model = get_default_RTCS_hardware()

print(model.couplings)
# create a circuit
circuit = (
    get_builder(model)
    .had(model.qubits[0])
    .cnot(model.qubits[0], model.qubits[1])
    .measure(model.qubits[0])
    .measure(model.qubits[1])
)
res = execute_instructions(model, circuit)
