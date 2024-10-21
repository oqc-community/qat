import os

import pytest
from packaging.version import Version

from qat.purr.compiler.devices import Calibratable, Qubit, Resonator
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.runtime import execute_instructions, get_builder

hw_dir = "tests/qat/files/hw_models/"
versions = [name for name in os.listdir(hw_dir)]
versions.sort(key=Version)
test_models = {
    version: [os.path.splitext(name)[0] for name in os.listdir(hw_dir + version)]
    for version in versions
}


@pytest.mark.parametrize(
    ["version", "model"],
    [(version, model) for version, models in test_models.items() for model in models],
)
class TestHardwareModels:

    def deserialize_hw_model(self, version, model):
        return Calibratable().load_calibration_from_file(
            hw_dir + version + "/" + model + ".json"
        )

    def test_deserialize_hw_model(self, version, model):
        # check that the model deserializes correctly
        model = self.deserialize_hw_model(version, model)
        assert isinstance(model, QuantumHardwareModel)

        # check that the model has some expected properties
        for qubit in model.qubits:
            assert isinstance(qubit, Qubit)
            assert isinstance(qubit.measure_device, Resonator)
            assert qubit.physical_channel in model.physical_channels.values()
            assert all([pc in qubit.pulse_channels for pc in qubit.pulse_channels])

        for pc in model.physical_channels.values():
            assert pc.baseband in model.basebands.values()

    def test_execute_hw_model(self, version, model):
        # checks that the hw models correctly execute
        model = self.deserialize_hw_model(version, model)
        print(model.physical_channels.keys())

        # create a circuit
        circuit = (
            get_builder(model)
            .had(model.qubits[0])
            .cnot(model.qubits[0], model.qubits[1])
            .measure(model.qubits[0])
            .measure(model.qubits[1])
        )
        execute_instructions(model, circuit)
        assert True
