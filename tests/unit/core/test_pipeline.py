from qat.core.pipeline import HardwareLoaders, PipelineSet
from qat.model.loaders.legacy.echo import EchoModelLoader
from qat.model.loaders.legacy.qiskit import QiskitModelLoader
from qat.purr.compiler.hardware_models import QuantumHardwareModel

from tests.unit.utils.loaders import MockModelLoader
from tests.unit.utils.pipelines import MockPipelineConfig, MockUpdateablePipeline


class TestHardwareLoaders:
    def test_init(self):
        loaders_dict = {"a": EchoModelLoader(), "b": QiskitModelLoader()}
        loaders = HardwareLoaders(loaders_dict)

        model_a = loaders.load("a")
        model_b = loaders.load("b")
        assert isinstance(model_a, QuantumHardwareModel)
        assert isinstance(model_b, QuantumHardwareModel)

    def test_reload_model(self):
        """Checks the ability to reload a single model in the loader, ensuring other models
        remained unchanged."""
        loaders = HardwareLoaders({"a": MockModelLoader(), "b": MockModelLoader()})
        model = loaders.load("a")
        assert len(model.qubits) == 2
        model = loaders.load("b")
        assert len(model.qubits) == 2
        loaders.reload_model("a")
        assert len(loaders.load("a").qubits) == 3
        assert len(loaders.load("b").qubits) == 2

    def test_reload_all_models(self):
        """Checks that all models are reloaded."""
        loaders = HardwareLoaders({"a": MockModelLoader(), "b": MockModelLoader()})
        model = loaders.load("a")
        assert len(model.qubits) == 2
        model = loaders.load("b")
        assert len(model.qubits) == 2
        loaders.reload_all_models()
        assert len(loaders.load("a").qubits) == 3
        assert len(loaders.load("b").qubits) == 3


class TestPipelineSet:
    def test_reload_model(self):
        """Checks the ability to reload a single pipeline, ensuring other pipelines remained
        unchanged."""

        pipelines = [
            MockUpdateablePipeline(
                config=MockPipelineConfig(name=f"test{i}"), loader=MockModelLoader()
            )
            for i in range(2)
        ]
        pipelines = PipelineSet(pipelines)
        assert all([len(pipelines.get(f"test{i}").model.qubits) == 2 for i in range(2)])
        pipelines.reload_model("test0")
        assert len(pipelines.get("test0").model.qubits) == 3
        assert len(pipelines.get("test1").model.qubits) == 2

    def test_reload_all_pipelines(self):
        """Checks that all pipelines with a loader are reloaded."""
        pipelines = [
            MockUpdateablePipeline(
                config=MockPipelineConfig(name=f"test{i}"), loader=MockModelLoader()
            )
            for i in range(2)
        ]
        pipelines = PipelineSet(pipelines)
        pipelines.add(
            MockUpdateablePipeline(
                config=MockPipelineConfig(name="test3"), model=MockModelLoader().load()
            )
        )
        assert all([len(pipelines.get(f"test{i}").model.qubits) == 2 for i in range(2)])
        pipelines.reload_all_models()
        assert all([len(pipelines.get(f"test{i}").model.qubits) == 3 for i in range(2)])
        assert len(pipelines.get("test3").model.qubits) == 2
