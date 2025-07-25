# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.core.config.descriptions import EngineDescription
from qat.core.pipeline import EngineSet, HardwareLoaders, PipelineSet
from qat.model.loaders.purr.echo import EchoModelLoader
from qat.model.loaders.purr.qiskit import QiskitModelLoader
from qat.purr.compiler.hardware_models import QuantumHardwareModel

from tests.unit.utils.engines import InitableEngine, MockEngineWithModel
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


class TestEngineSet:
    def test_instantiation(self):
        """Checks the instantiation of EngineSet with a list of engines."""
        loader = MockModelLoader()
        loaders = {"engine2": loader}
        engines = {
            "engine1": InitableEngine(),
            "engine2": MockEngineWithModel(loader.load()),
        }
        engine_set = EngineSet(engines, loaders)

        assert len(engine_set._loaders) == 1
        assert "engine2" in engine_set._loaders
        assert engine_set._loaders["engine2"] == loader

        assert len(engine_set._engines) == 2
        assert engine_set.get("engine1") is engines["engine1"]
        assert engine_set.get("engine2") is engines["engine2"]

    def test_reload_model(self):
        loader = MockModelLoader()
        engines = {
            "engine": MockEngineWithModel(loader.load()),
        }
        engine_set = EngineSet(engines, {"engine": loader})

        assert len(engine_set.get("engine").model.qubits) == 2

        engine_set.reload_model("engine")
        assert len(engine_set.get("engine").model.qubits) == 3

    def test_reload_model_with_no_model(self):
        """Checks that reloading a model for an engine without a model does not raise an error."""
        engine = InitableEngine()
        engine_set = EngineSet(dict(engine=engine), {})
        assert not hasattr(engine, "model")

        # This should not raise an error
        engine_set.reload_model("engine")
        assert engine_set.get("engine") is engine

    def test_reload_all_models(self):
        """Checks that all models are reloaded."""
        loader = MockModelLoader()
        model = loader.load()
        engines = {
            "engine1": MockEngineWithModel(model),
            "engine2": InitableEngine(),
            "engine3": MockEngineWithModel(model),
        }
        engine_set = EngineSet(engines, {"engine1": loader})

        assert len(engine_set.get("engine1").model.qubits) == 2
        assert not hasattr(engine_set.get("engine2"), "model")
        assert len(engine_set.get("engine3").model.qubits) == 2

        engine_set.reload_all_models()

        assert len(engine_set.get("engine1").model.qubits) == 3
        assert len(engine_set.get("engine3").model.qubits) == 2

    def test_from_descriptions(self):
        """Checks the creation of EngineSet from engine descriptions."""
        loader = MockModelLoader()
        loader2 = EchoModelLoader(qubit_count=2)
        engine_descriptions = [
            EngineDescription(
                name="engine1",
                type="tests.unit.utils.engines.InitableEngine",
                hardware_loader=None,
            ),
            EngineDescription(
                name="engine2",
                type="tests.unit.utils.engines.MockEngineWithModel",
                hardware_loader="mock",
            ),
            EngineDescription(
                name="engine3",
                type="tests.unit.utils.engines.MockEngineWithModel",
                hardware_loader="mock",
            ),
            EngineDescription(
                name="engine4",
                type="tests.unit.utils.engines.MockEngineWithModel",
                hardware_loader="mock2",
            ),
        ]
        available_hardware = HardwareLoaders({"mock": loader, "mock2": loader2})

        engine_set = EngineSet.from_descriptions(engine_descriptions, available_hardware)

        assert len(engine_set._loaders) == 3
        for i in range(3):
            assert f"engine{i + 2}" in engine_set._loaders

        assert len(engine_set._engines) == 4
        assert isinstance(engine_set.get("engine1"), InitableEngine)

        for i in range(3):
            engine = engine_set.get(f"engine{i + 2}")
            assert isinstance(engine, MockEngineWithModel)
            assert len(engine.model.qubits) == 2

        available_hardware.reload_all_models()
        engine_set.reload_all_models()
        assert len(engine_set.get("engine2").model.qubits) == 3
        assert len(engine_set.get("engine3").model.qubits) == 3
        assert len(engine_set.get("engine4").model.qubits) == 2

        assert isinstance(engine_set.get("engine1"), InitableEngine)
        assert isinstance(engine_set.get("engine2"), MockEngineWithModel)


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
