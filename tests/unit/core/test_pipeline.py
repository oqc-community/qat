# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.core.config.descriptions import EngineDescription
from qat.core.pipeline import EngineSet, HardwareLoaders, PipelineManager, PipelineSet
from qat.model.loaders.purr.echo import EchoModelLoader
from qat.model.loaders.purr.qiskit import QiskitModelLoader
from qat.purr.compiler.hardware_models import QuantumHardwareModel

from tests.unit.utils.engines import InitableEngine, MockEngineWithModel
from tests.unit.utils.loaders import MockModelLoader
from tests.unit.utils.pipelines import (
    MockCompileUpdateablePipeline,
    MockExecuteUpdateablePipeline,
    MockPipelineConfig,
    MockUpdateablePipeline,
)


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

    def test_models_up_to_date_single_loader(self):
        loader = MockModelLoader()
        loaders = HardwareLoaders({"a": loader})
        assert not loaders.models_up_to_date  # model is not yet loaded
        loaders.load("a")
        assert loaders.models_up_to_date
        loader.num_qubits += 1  # Simulate the model being out of date
        assert not loaders.models_up_to_date
        loaders.reload_model("a")
        assert loaders.models_up_to_date

    def test_models_up_to_date_multiple_loaders(self):
        loader1 = MockModelLoader()
        loader2 = MockModelLoader()
        loaders = HardwareLoaders({"a": loader1, "b": loader2})
        assert not loaders.models_up_to_date  # models are not yet loaded
        loaders.load("a")
        assert not loaders.models_up_to_date  # only one model is loaded
        loaders.load("b")
        assert loaders.models_up_to_date
        loader1.num_qubits += 1  # Simulate one model being out of date
        assert not loaders.models_up_to_date
        loaders.reload_model("a")
        assert loaders.models_up_to_date


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


class TestPipelineManager:
    def setup_manager(self):
        """Helper function to set up a PipelineManager with mock pipelines."""
        loader = MockModelLoader()
        compiles = PipelineSet(
            [
                MockCompileUpdateablePipeline(
                    config=MockPipelineConfig(name="compile"), loader=loader
                )
            ]
        )
        executes = PipelineSet(
            [
                MockExecuteUpdateablePipeline(
                    config=MockPipelineConfig(name="execute"), loader=loader
                )
            ]
        )
        compiles.set_default("compile")
        executes.set_default("execute")
        return PipelineManager(
            compile_pipelines=compiles,
            execute_pipelines=executes,
            full_pipelines=PipelineSet(),
        )

    @pytest.mark.parametrize("default", [True, False])
    def test_add_full_pipeline(self, default):
        manager = self.setup_manager()
        loader = MockModelLoader()

        full_pipeline = MockUpdateablePipeline(
            config=MockPipelineConfig(name="full_pipeline"), loader=loader
        )
        manager.add(full_pipeline, default=default)
        assert set(manager.list_compile_pipelines) == {"compile", "full_pipeline"}
        assert set(manager.list_execute_pipelines) == {"execute", "full_pipeline"}
        assert "full_pipeline" in manager._full_pipelines
        assert (manager.default_compile_pipeline == "full_pipeline") is default
        assert (manager.default_execute_pipeline == "full_pipeline") is default

    @pytest.mark.parametrize("default", [True, False])
    def test_add_compile_pipeline(self, default):
        manager = self.setup_manager()
        loader = MockModelLoader()

        compile_pipeline = MockCompileUpdateablePipeline(
            config=MockPipelineConfig(name="new_compile"), loader=loader
        )
        manager.add(compile_pipeline, default=default)
        assert set(manager.list_compile_pipelines) == {"compile", "new_compile"}
        assert set(manager.list_execute_pipelines) == {"execute"}
        assert (manager.default_compile_pipeline == "new_compile") is default
        assert manager.default_execute_pipeline == "execute"

    @pytest.mark.parametrize("default", [True, False])
    def test_add_execute_pipeline(self, default):
        manager = self.setup_manager()
        loader = MockModelLoader()

        execute_pipeline = MockExecuteUpdateablePipeline(
            config=MockPipelineConfig(name="new_execute"), loader=loader
        )
        manager.add(execute_pipeline, default=default)
        assert set(manager.list_compile_pipelines) == {"compile"}
        assert set(manager.list_execute_pipelines) == {"execute", "new_execute"}
        assert (manager.default_execute_pipeline == "new_execute") is default
        assert manager.default_compile_pipeline == "compile"

    def test_add_compile_pipeline_with_duplicate_name_raises_error(self):
        manager = self.setup_manager()
        loader = MockModelLoader()

        compile_pipeline = MockCompileUpdateablePipeline(
            config=MockPipelineConfig(name="compile"), loader=loader
        )
        with pytest.raises(
            ValueError, match="already exists in the manager as a compile pipeline."
        ):
            manager.add(compile_pipeline)

    def test_add_execute_pipeline_with_duplicate_name_raises_error(self):
        manager = self.setup_manager()
        loader = MockModelLoader()

        execute_pipeline = MockExecuteUpdateablePipeline(
            config=MockPipelineConfig(name="execute"), loader=loader
        )
        with pytest.raises(
            ValueError, match="already exists in the manager as an execute pipeline."
        ):
            manager.add(execute_pipeline)

    def test_add_full_pipeline_with_duplicate_name_raises_error(self):
        manager = self.setup_manager()
        loader = MockModelLoader()

        full_pipeline = MockUpdateablePipeline(
            config=MockPipelineConfig(name="compile"), loader=loader
        )
        with pytest.raises(
            ValueError,
            match="already exists in the manager as a compile or execute pipeline.",
        ):
            manager.add(full_pipeline)

    def test_remove_compile_pipeline(self):
        manager = self.setup_manager()
        assert "compile" in manager.list_compile_pipelines
        manager.remove("compile")
        assert "compile" not in manager.list_compile_pipelines

    def test_remove_execute_pipeline(self):
        manager = self.setup_manager()
        assert "execute" in manager.list_execute_pipelines
        manager.remove("execute")
        assert "execute" not in manager.list_execute_pipelines

    def test_remove_full_pipeline(self):
        manager = self.setup_manager()
        loader = MockModelLoader()

        full_pipeline = MockUpdateablePipeline(
            config=MockPipelineConfig(name="full_pipeline"), loader=loader
        )
        manager.add(full_pipeline)
        assert "full_pipeline" in manager.compile_pipelines
        assert "full_pipeline" in manager.execute_pipelines
        manager.remove("full_pipeline")
        assert "full_pipeline" not in manager.compile_pipelines
        assert "full_pipeline" not in manager.execute_pipelines
        assert "full_pipeline" not in manager._full_pipelines

    def test_remove_full_pipeline_with_only_compile_raises_error(self):
        manager = self.setup_manager()
        loader = MockModelLoader()

        full_pipeline = MockUpdateablePipeline(
            config=MockPipelineConfig(name="full_pipeline"), loader=loader
        )
        manager.add(full_pipeline)
        with pytest.raises(ValueError, match="You must remove it from both."):
            manager.remove("full_pipeline", execute=False)

    def test_remove_execute_from_compile_raises_error(self):
        manager = self.setup_manager()
        with pytest.raises(ValueError):
            manager.remove("execute", compile=True)

    def test_set_default_compile_pipeline(self):
        manager = self.setup_manager()
        new_compile = MockCompileUpdateablePipeline(
            config=MockPipelineConfig(name="new_compile"), loader=MockModelLoader()
        )
        manager.add(new_compile)
        manager.set_default("new_compile")
        assert manager.default_compile_pipeline == "new_compile"

    def test_set_default_execute_pipeline(self):
        manager = self.setup_manager()
        new_execute = MockExecuteUpdateablePipeline(
            config=MockPipelineConfig(name="new_execute"), loader=MockModelLoader()
        )
        manager.add(new_execute)
        manager.set_default("new_execute")
        assert manager.default_execute_pipeline == "new_execute"

    def test_set_default_full_pipeline(self):
        manager = self.setup_manager()
        loader = MockModelLoader()

        full_pipeline = MockUpdateablePipeline(
            config=MockPipelineConfig(name="full_pipeline"), loader=loader
        )
        manager.add(full_pipeline)
        manager.set_default("full_pipeline")
        assert manager.default_compile_pipeline == "full_pipeline"
        assert manager.default_execute_pipeline == "full_pipeline"

    @pytest.mark.parametrize("compile", [True, False])
    @pytest.mark.parametrize("execute", [True, False])
    def test_set_default_granular_checks(self, compile, execute):
        manager = self.setup_manager()
        new_compile = MockCompileUpdateablePipeline(
            config=MockPipelineConfig(name="new"), loader=MockModelLoader()
        )
        manager.add(new_compile)
        new_execute = MockExecuteUpdateablePipeline(
            config=MockPipelineConfig(name="new"), loader=MockModelLoader()
        )
        manager.add(new_execute)
        manager.set_default("new", compile=compile, execute=execute)
        assert manager.default_compile_pipeline == ("new" if compile else "compile")
        assert manager.default_execute_pipeline == ("new" if execute else "execute")

    def test_set_default_full_with_only_compile_raises_error(self):
        """Checks that setting a full pipeline as default with only compile raises an error."""
        manager = self.setup_manager()
        loader = MockModelLoader()

        full_pipeline = MockUpdateablePipeline(
            config=MockPipelineConfig(name="full_pipeline"), loader=loader
        )
        manager.add(full_pipeline)
        with pytest.raises(ValueError, match=" It must be set for both."):
            manager.set_default("full_pipeline", compile=True, execute=False)

    def test_set_default_on_non_existent_compile_raises_error(self):
        """Checks that setting a default compile pipeline that does not exist raises an error."""
        manager = self.setup_manager()
        with pytest.raises(ValueError, match="as default for compile pipelines"):
            manager.set_default("non_existent_compile", compile=True)

    def test_set_default_on_non_existent_execute_raises_error(self):
        """Checks that setting a default execute pipeline that does not exist raises an error."""
        manager = self.setup_manager()
        with pytest.raises(ValueError, match="as default for execute pipelines"):
            manager.set_default("non_existent_execute", execute=True)

    def test_reload_models(self):
        """Checks that reloading models in the manager reloads all pipelines."""

        full_pipeline = MockUpdateablePipeline(
            config=MockPipelineConfig(name="full_pipeline"),
            loader=MockModelLoader(num_qubits=1),
        )
        compile_pipeline = MockCompileUpdateablePipeline(
            config=MockPipelineConfig(name="compile"), loader=MockModelLoader(num_qubits=3)
        )
        execute_pipeline = MockExecuteUpdateablePipeline(
            config=MockPipelineConfig(name="execute"), loader=MockModelLoader(num_qubits=5)
        )
        manager = PipelineManager(
            compile_pipelines=PipelineSet([compile_pipeline]),
            execute_pipelines=PipelineSet([execute_pipeline]),
            full_pipelines=PipelineSet([full_pipeline]),
        )

        assert len(manager.get_compile_pipeline("full_pipeline").model.qubits) == 2
        assert len(manager.get_compile_pipeline("compile").model.qubits) == 4
        assert len(manager.get_execute_pipeline("execute").model.qubits) == 6

        manager.reload_all_models()

        assert len(manager.get_compile_pipeline("full_pipeline").model.qubits) == 3
        assert len(manager.get_compile_pipeline("compile").model.qubits) == 5
        assert len(manager.get_execute_pipeline("execute").model.qubits) == 7
