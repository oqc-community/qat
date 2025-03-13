from qat.core.pipeline import HardwareLoaders
from qat.model.loaders.legacy.echo import EchoModelLoader
from qat.model.loaders.legacy.qiskit import QiskitModelLoader
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class TestHardwareLoaders:
    def test_init(self):
        loaders_dict = {"a": EchoModelLoader(), "b": QiskitModelLoader()}
        loaders = HardwareLoaders(loaders_dict)

        model_a = loaders.load("a")
        model_b = loaders.load("b")
        assert isinstance(model_a, QuantumHardwareModel)
        assert isinstance(model_b, QuantumHardwareModel)
