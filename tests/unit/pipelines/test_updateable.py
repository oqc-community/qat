# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.backend import WaveformV1Backend
from qat.engines import ZeroEngine
from qat.engines.waveform_v1 import EchoEngine
from qat.frontend import AutoFrontend
from qat.middleend import DefaultMiddleend
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import TargetData
from qat.model.validators import MismatchingHardwareModelException
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.pipelines.updateable import PipelineConfig
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime import SimpleRuntime

from tests.unit.utils.engines import MockEngineWithModel
from tests.unit.utils.loaders import MockModelLoader
from tests.unit.utils.pipelines import (
    MockCompileUpdateablePipeline,
    MockExecuteUpdateablePipeline,
    MockPipelineConfig,
    MockUpdateablePipeline,
)


class TestUpdateablePipeline:
    """Designed to test the infrastructure of the UpdateablePipeline. The implementation
    of how the pipeline is built is the responsibility of the subclass, and should be tested
    there."""

    def test_initialization_with_loader(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )

        assert isinstance(pipeline, MockUpdateablePipeline)
        assert pipeline._loader == model_loader
        assert pipeline.name == "test"
        assert pipeline.engine == engine
        assert pipeline.target_data == target_data
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, DefaultMiddleend)
        assert isinstance(pipeline.backend, WaveformV1Backend)
        assert isinstance(pipeline.runtime, SimpleRuntime)
        assert isinstance(pipeline.model, QuantumHardwareModel)
        assert pipeline.config == config

    def test_initialization_with_model(self):
        model = EchoModelLoader(qubit_count=4).load()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, model=model, target_data=target_data, engine=engine
        )

        assert isinstance(pipeline, MockUpdateablePipeline)
        assert pipeline.name == "test"
        assert pipeline._loader is None
        assert pipeline.engine == engine
        assert pipeline.target_data == target_data
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, DefaultMiddleend)
        assert isinstance(pipeline.backend, WaveformV1Backend)
        assert isinstance(pipeline.runtime, SimpleRuntime)
        assert pipeline.model is model
        assert pipeline.config == config

    def test_initialization_with_config_as_dict(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = {"name": "testing"}
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )
        assert pipeline.name == "testing"
        assert isinstance(pipeline.config, PipelineConfig)

    def test_initialization_with_no_model_or_loader_raises_error(self):
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        with pytest.raises(ValueError):
            MockUpdateablePipeline(config=config, target_data=target_data, engine=engine)

    def test_initialization_with_model_and_loader_takes_precedence(self):
        model_loader = MockModelLoader()
        model = model_loader.load()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")

        pipeline = MockUpdateablePipeline(
            config=config,
            model=model,
            loader=model_loader,
            target_data=target_data,
            engine=engine,
        )

        assert pipeline._loader == model_loader
        assert pipeline.model is model
        assert len(pipeline.model.qubits) == 2  # it would be 3 if the loader was used again
        pipeline.update(reload_model=True)
        assert len(pipeline.model.qubits) == 3

    def test_initalization_with_engine_that_requires_model(self):
        """Tests that the engine doesn't fail validation."""
        loader = EchoModelLoader(qubit_count=4)
        model = loader.load()
        target_data = TargetData.default()
        config = MockPipelineConfig(name="test")
        engine = MockEngineWithModel(model=model)
        pipeline = MockUpdateablePipeline(
            config=config,
            loader=loader,
            model=model,
            target_data=target_data,
            engine=engine,
        )
        assert pipeline.engine.model is model

    def test_initalization_with_engine_that_requires_model_raises_error_with_loader(self):
        loader = EchoModelLoader(qubit_count=4)
        model = loader.load()
        target_data = TargetData.default()
        config = MockPipelineConfig(name="test")
        engine = MockEngineWithModel(model=model)
        with pytest.raises(MismatchingHardwareModelException):
            MockUpdateablePipeline(
                config=config, loader=loader, target_data=target_data, engine=engine
            )

    def test_update_with_config(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )

        new_config = MockPipelineConfig(name="test", test_attr=True)
        pipeline.update(config=new_config)
        assert pipeline.config == new_config

    def test_update_with_config_as_dict(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )

        new_config_dict = {"test_attr": True}
        pipeline.update(config=new_config_dict)
        assert pipeline.config.name == "test"
        assert pipeline.config.test_attr is True

    def test_update_with_model(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )
        assert len(pipeline.model.qubits) == 2

        new_model = EchoModelLoader(qubit_count=5).load()
        pipeline.update(model=new_model)
        assert len(pipeline.model.qubits) == 5

    def test_update_with_loader(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )
        assert pipeline._loader is model_loader
        assert len(pipeline.model.qubits) == 2

        new_loader = EchoModelLoader(qubit_count=5)
        pipeline.update(loader=new_loader)
        assert pipeline._loader is new_loader
        assert len(pipeline.model.qubits) == 5

    def test_update_with_reload(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )
        assert len(pipeline.model.qubits) == 2

        pipeline.update(reload_model=True)
        assert len(pipeline.model.qubits) == 3

    def test_update_with_target_data(self):
        model_loader = MockModelLoader()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")

        target_data = TargetData.default()
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )
        assert pipeline.target_data == target_data

        new_target_data = TargetData.random()
        assert new_target_data != target_data
        model = pipeline.model
        pipeline.update(target_data=new_target_data)
        assert pipeline.target_data == new_target_data
        assert pipeline.model is model

    def test_update_with_engine(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        config = MockPipelineConfig(name="test")

        engine = EchoEngine()
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )
        assert pipeline.engine == engine

        new_engine = ZeroEngine()
        model = pipeline.model
        pipeline.update(engine=new_engine)
        assert pipeline.engine == new_engine
        assert pipeline.model is model

    def test_update_model_with_engine_that_requires_model(self):
        model = EchoModelLoader(qubit_count=4).load()
        target_data = TargetData.default()
        config = MockPipelineConfig(name="test")

        engine = MockEngineWithModel(model=model)
        pipeline = MockUpdateablePipeline(
            config=config, model=model, target_data=target_data, engine=engine
        )
        assert pipeline.engine == engine
        assert pipeline.engine.model == model

        new_model = EchoModelLoader(qubit_count=5).load()
        pipeline.update(model=new_model)
        assert pipeline.engine.model == new_model

    def test_update_with_model_and_reload_model_raises_error(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )

        new_model = EchoModelLoader(qubit_count=5).load()
        with pytest.raises(ValueError):
            pipeline.update(model=new_model, reload_model=True)

    def test_copy(self):
        loader = EchoModelLoader(qubit_count=4)
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=loader, target_data=target_data, engine=engine
        )

        copied_pipeline = pipeline.copy()

        assert copied_pipeline is not pipeline
        assert copied_pipeline.name == pipeline.name
        assert copied_pipeline._loader == pipeline._loader
        assert copied_pipeline.engine == pipeline.engine
        assert copied_pipeline.config == pipeline.config

    def test_copy_with_config(self):
        loader = EchoModelLoader(qubit_count=4)
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=loader, target_data=target_data, engine=engine
        )

        new_config = MockPipelineConfig(name="new_test", test_attr=True)
        copied_pipeline = pipeline.copy_with(config=new_config)

        assert copied_pipeline is not pipeline
        assert copied_pipeline.name == "new_test"
        assert copied_pipeline._loader == pipeline._loader
        assert copied_pipeline.engine == pipeline.engine
        assert copied_pipeline.target_data == pipeline.target_data
        assert copied_pipeline.config == new_config
        assert new_config != config

    def test_copy_with_model(self):
        loader = EchoModelLoader(qubit_count=4)
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=loader, target_data=target_data, engine=engine
        )

        new_model = EchoModelLoader(qubit_count=6).load()
        copied_pipeline = pipeline.copy_with(model=new_model)

        assert copied_pipeline is not pipeline
        assert copied_pipeline.name == pipeline.name
        assert copied_pipeline._loader is loader
        assert copied_pipeline.engine == pipeline.engine
        assert copied_pipeline.target_data == pipeline.target_data
        assert copied_pipeline.model is new_model

    def test_copy_with_loader(self):
        loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=loader, target_data=target_data, engine=engine
        )

        new_loader = EchoModelLoader(qubit_count=6)
        copied_pipeline = pipeline.copy_with(loader=new_loader)

        assert copied_pipeline is not pipeline
        assert copied_pipeline.name == pipeline.name
        assert copied_pipeline._loader == new_loader
        assert copied_pipeline.engine == pipeline.engine
        assert copied_pipeline.target_data == pipeline.target_data
        assert len(copied_pipeline.model.qubits) == 6

    def test_copy_with_target_data(self):
        loader = EchoModelLoader(qubit_count=4)
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        target_data = TargetData.default()
        pipeline = MockUpdateablePipeline(
            config=config, loader=loader, target_data=target_data, engine=engine
        )

        new_target_data = TargetData.random()
        copied_pipeline = pipeline.copy_with(target_data=new_target_data)

        assert target_data is not new_target_data
        assert copied_pipeline is not pipeline
        assert copied_pipeline.name == pipeline.name
        assert copied_pipeline._loader == pipeline._loader
        assert copied_pipeline.engine == pipeline.engine
        assert copied_pipeline.target_data == new_target_data

    def test_copy_with_engine(self):
        loader = EchoModelLoader(qubit_count=4)
        target_data = TargetData.default()
        config = MockPipelineConfig(name="test")
        engine = EchoEngine()
        pipeline = MockUpdateablePipeline(
            config=config, loader=loader, target_data=target_data, engine=engine
        )

        new_engine = ZeroEngine()
        copied_pipeline = pipeline.copy_with(engine=new_engine)

        assert copied_pipeline is not pipeline
        assert copied_pipeline.name == pipeline.name
        assert copied_pipeline._loader == pipeline._loader
        assert copied_pipeline.target_data == pipeline.target_data
        assert copied_pipeline.engine == new_engine

    def test_copy_with_engine_that_requires_model(self):
        model = EchoModelLoader(qubit_count=4).load()
        target_data = TargetData.default()
        config = MockPipelineConfig(name="test")

        engine = MockEngineWithModel(model=model)
        pipeline = MockUpdateablePipeline(
            config=config, model=model, target_data=target_data, engine=engine
        )
        assert pipeline.engine == engine
        assert pipeline.engine.model == model

        new_model = EchoModelLoader(qubit_count=5).load()
        copied_pipeline = pipeline.copy_with(model=new_model)
        assert copied_pipeline.engine.model == new_model

    def test_has_loader(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )
        assert pipeline.has_loader

        pipeline = MockUpdateablePipeline(
            config=config, model=model_loader.load(), target_data=target_data, engine=engine
        )
        assert pipeline.has_loader is False

    def test_full_pipeline_inherits_properties(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )

        assert pipeline.is_subtype_of(Pipeline)
        assert pipeline.is_subtype_of(CompilePipeline)
        assert pipeline.is_subtype_of(ExecutePipeline)
        assert hasattr(pipeline, "compile")
        assert hasattr(pipeline, "frontend")
        assert hasattr(pipeline, "middleend")
        assert hasattr(pipeline, "backend")
        assert hasattr(pipeline, "execute")
        assert hasattr(pipeline, "runtime")

    def test_compile_pipeline_inherits_properties(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockCompileUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )

        assert pipeline.is_subtype_of(CompilePipeline)
        assert not pipeline.is_subtype_of(Pipeline)
        assert not pipeline.is_subtype_of(ExecutePipeline)
        assert hasattr(pipeline, "compile")
        assert hasattr(pipeline, "frontend")
        assert hasattr(pipeline, "middleend")
        assert hasattr(pipeline, "backend")
        assert not hasattr(pipeline, "execute")
        assert not hasattr(pipeline, "runtime")

    def test_execute_pipeline_inherits_properties(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        engine = EchoEngine()
        config = MockPipelineConfig(name="test")
        pipeline = MockExecuteUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data, engine=engine
        )

        assert pipeline.is_subtype_of(ExecutePipeline)
        assert not pipeline.is_subtype_of(Pipeline)
        assert not pipeline.is_subtype_of(CompilePipeline)
        assert not hasattr(pipeline, "compile")
        assert not hasattr(pipeline, "frontend")
        assert not hasattr(pipeline, "middleend")
        assert not hasattr(pipeline, "backend")
        assert hasattr(pipeline, "execute")
        assert hasattr(pipeline, "runtime")

    @pytest.mark.parametrize(
        "model, loader, reload_model",
        [
            (EchoModelLoader().load(), EchoModelLoader(), True),
            (EchoModelLoader().load(), None, True),
            (None, None, True),
        ],
    )
    def test_resolve_model_raises_error(self, model, loader, reload_model):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, model=model_loader.load(), target_data=target_data
        )

        with pytest.raises(ValueError):
            pipeline._resolve_model(model, loader, reload_model)

    @pytest.mark.parametrize("has_loader", [True, False])
    def test_resolve_model_with_new_model(self, has_loader):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        config = MockPipelineConfig(name="test")
        if has_loader:
            pipeline = MockUpdateablePipeline(
                config=config, loader=model_loader, target_data=target_data
            )
        else:
            pipeline = MockUpdateablePipeline(
                config=config, model=model_loader.load(), target_data=target_data
            )

        new_model = EchoModelLoader(qubit_count=5).load()
        model, loader, refresh = pipeline._resolve_model(new_model, None, False)
        assert model is new_model
        assert refresh == True
        assert loader is (model_loader if has_loader else None)

    @pytest.mark.parametrize("has_loader", [True, False])
    def test_resolve_with_new_loader(self, has_loader):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        config = MockPipelineConfig(name="test")
        if has_loader:
            pipeline = MockUpdateablePipeline(
                config=config, loader=model_loader, target_data=target_data
            )
        else:
            pipeline = MockUpdateablePipeline(
                config=config, model=model_loader.load(), target_data=target_data
            )

        new_loader = EchoModelLoader(qubit_count=5)
        model, loader, refresh = pipeline._resolve_model(None, new_loader, False)
        assert len(model.qubits) == 5
        assert refresh == True
        assert loader is new_loader

    def test_resolve_with_model_and_loader(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data
        )

        new_loader = EchoModelLoader(qubit_count=5)
        new_model = new_loader.load()
        model, loader, refresh = pipeline._resolve_model(new_model, new_loader, False)
        assert model is not pipeline.model
        assert len(model.qubits) == 5
        assert refresh == True
        assert loader is new_loader

    def test_resolve_with_reload(self):
        model_loader = MockModelLoader()
        target_data = TargetData.default()
        config = MockPipelineConfig(name="test")
        pipeline = MockUpdateablePipeline(
            config=config, loader=model_loader, target_data=target_data
        )

        model, loader, refresh = pipeline._resolve_model(None, None, True)
        assert model is not pipeline.model
        assert len(pipeline.model.qubits) == 2
        assert len(model.qubits) == 3
        assert refresh == True
        assert loader is model_loader
