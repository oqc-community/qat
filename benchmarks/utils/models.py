from qat.purr.compiler.hardware_models import QuantumHardwareModel

from tests.qat.test_quantum_backend import FakeBaseQuantumExecution
from tests.qat.utils.models import apply_setup_to_hardware


class MockLiveHardwareModel(QuantumHardwareModel):
    """
    A hardware model that will return a mock live execution engine to allow us to
    test and benchmark features exclusive to the live engine.
    """

    def create_engine(self):
        return FakeBaseQuantumExecution(self)


def get_mock_live_hardware(num_qubits):
    """
    Returns a general hardware model that is set to create a mock live execution engine.
    """
    return apply_setup_to_hardware(MockLiveHardwareModel(), list(range(num_qubits)))
