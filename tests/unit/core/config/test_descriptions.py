# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest
from pydantic import ImportString, ValidationError

import qat
import qat.frontend
from qat.backend.waveform_v1 import WaveformV1Backend
from qat.core.config.descriptions import (
    ClassDescription,
    CompilePipelineDescription,
    EngineDescription,
    ExecutePipelineDescription,
    HardwareLoaderDescription,
    ImportFrontend,
    PipelineClassDescription,
    PipelineFactoryDescription,
    PipelineInstanceDescription,
    UpdateablePipelineDescription,
)
from qat.core.pipelines.configurable import (
    ConfigurableCompilePipeline,
    ConfigurableExecutePipeline,
    ConfigurablePipeline,
)
from qat.engines import DefaultEngine, ZeroEngine
from qat.frontend import FallthroughFrontend
from qat.middleend import FallthroughMiddleend
from qat.model.loaders.base import BaseModelLoader
from qat.model.loaders.cache import CacheAccessLoader
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.loaders.purr import EchoModelLoader
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.pipelines.waveform_v1 import (
    EchoExecutePipeline,
    EchoPipeline,
    WaveformV1CompilePipeline,
)
from qat.runtime import SimpleRuntime

from tests.unit.utils.engines import InitableEngine, MockEngineWithModel


class TestEngineDescription:
    def test_valid_description(self):
        desc = EngineDescription(name="zero", type="qat.engines.ZeroEngine")
        assert desc.name == "zero"
        assert desc.type is ZeroEngine

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            EngineDescription(
                name="invalid", type="qat.backend.fallthrough.FallthroughBackend"
            )

    def test_extra_fields_raises_validation_error(self):
        with pytest.raises(ValidationError):
            EngineDescription(
                name="invalid", type="qat.engines.ZeroEngine", extra_field="value"
            )

    def test_construct_engine_without_model(self):
        desc = EngineDescription(name="zero", type="qat.engines.ZeroEngine")
        engine = desc.construct()
        assert isinstance(engine, ZeroEngine)

    def test_construct_engine_with_model(self):
        desc = EngineDescription(
            name="zero", type="tests.unit.utils.engines.MockEngineWithModel"
        )
        model = EchoModelLoader().load()
        engine = desc.construct(model=model)
        assert isinstance(engine, MockEngineWithModel)
        assert engine.model == model

    def test_construct_engine_with_config(self):
        desc = EngineDescription(
            name="initable",
            type="tests.unit.utils.engines.InitableEngine",
            config={"x": 9, "cblam": {"host": "someurl.com", "timeout": 60}},
        )
        engine = desc.construct()
        assert isinstance(engine, InitableEngine)
        assert engine.x == 9
        assert engine.y == "default_y"
        assert engine.z == 3
        assert engine.cblam.host == "someurl.com"
        assert engine.cblam.timeout == 60


mock_compile_pipeline = WaveformV1CompilePipeline(
    config=dict(name="mock_compile"), loader=LucyModelLoader()
).pipeline
mock_execute_pipeline = EchoExecutePipeline(
    config=dict(name="mock_execute"), loader=LucyModelLoader()
).pipeline
mock_full_pipeline = EchoPipeline(
    config=dict(name="mock_full"), loader=LucyModelLoader()
).pipeline


class TestPipelineInstanceDescription:
    def test_valid_description(self):
        desc = PipelineInstanceDescription(
            name="echo8a", pipeline="qat.pipelines.echo.echo8", default=True
        )

        assert desc.name == "echo8a"

    def test_invalid_pipeline_class_raises(self):
        with pytest.raises(ValidationError):
            PipelineInstanceDescription(
                name="invalid",
                pipeline="qat.backend.fallthrough.FallthroughBackend",
                default=True,
            )

    def test_extra_fields_raises_error(self):
        with pytest.raises(ValidationError):
            PipelineInstanceDescription(
                name="invalid",
                pipeline="qat.pipelines.echo.echo8",
                default=True,
                backend="qat.backend.fallthrough.FallthroughBackend",
            )

    def test_construct_pipeline(self):
        from qat.pipelines.echo import echo8

        desc = PipelineInstanceDescription(
            name="echo8a", pipeline="qat.pipelines.echo.echo8", default=True
        )

        P = desc.construct()
        assert P.model == echo8.model
        assert P.name == "echo8a"
        assert P.frontend == echo8.frontend
        assert P.middleend == echo8.middleend
        assert P.backend == echo8.backend
        assert P.runtime == echo8.runtime
        assert P.engine == echo8.engine

    @pytest.mark.parametrize(
        "instance, type",
        [
            (
                "tests.unit.core.config.test_descriptions.mock_compile_pipeline",
                CompilePipeline,
            ),
            (
                "tests.unit.core.config.test_descriptions.mock_execute_pipeline",
                ExecutePipeline,
            ),
            ("tests.unit.core.config.test_descriptions.mock_full_pipeline", Pipeline),
            (
                "tests.unit.core.config.test_descriptions.mock_full_pipeline",
                CompilePipeline,
            ),
            (
                "tests.unit.core.config.test_descriptions.mock_full_pipeline",
                ExecutePipeline,
            ),
        ],
    )
    def test_is_subclass_of_is_true(self, instance, type):
        mock_desc = PipelineInstanceDescription(name="mock_instance", pipeline=instance)
        assert mock_desc.is_subtype_of(type)

    @pytest.mark.parametrize(
        "instance, type",
        [
            (
                "tests.unit.core.config.test_descriptions.mock_compile_pipeline",
                ExecutePipeline,
            ),
            (
                "tests.unit.core.config.test_descriptions.mock_execute_pipeline",
                CompilePipeline,
            ),
        ],
    )
    def test_is_subclass_of_is_false(self, instance, type):
        mock_desc = PipelineInstanceDescription(name="mock_instance", pipeline=instance)
        assert not mock_desc.is_subtype_of(type)


class TestPipelineFactoryDescription:
    def test_valid_description(self):
        desc = PipelineFactoryDescription(
            name="echo8a",
            pipeline="tests.unit.utils.pipelines.get_mock_pipeline",
            default=True,
            hardware_loader="echo8a",
        )

        assert desc.name == "echo8a"
        assert callable(desc.pipeline)

    def test_invalid_pipeline_class_raises(self):
        with pytest.raises(ValidationError):
            PipelineFactoryDescription(
                name="invalid",
                pipeline="qat.backend.fallthrough.FallthroughBackend",
                default=True,
            )

    def test_extra_fields_raises_error(self):
        with pytest.raises(ValidationError):
            PipelineFactoryDescription(
                name="invalid",
                pipeline="tests.unit.utils.pipelines.get_mock_pipeline",
                default=True,
                backend="qat.backend.fallthrough.FallthroughBackend",
            )

    def test_construct_pipeline(self):
        from qat.core.pipelines.factory import PipelineFactory

        desc = PipelineFactoryDescription(
            name="mock_factory",
            pipeline="tests.unit.utils.pipelines.get_mock_pipeline",
            default=True,
            hardware_loader="echo8a",
        )

        from qat.model.loaders.purr import EchoModelLoader

        loader = EchoModelLoader()
        factory = desc.construct(loader=loader)
        assert isinstance(factory, PipelineFactory)
        assert isinstance(factory.engine, ZeroEngine)

    def test_construct_pipeline_with_engine(self):
        from qat.core.pipelines.factory import PipelineFactory
        from qat.engines.waveform_v1 import EchoEngine
        from qat.model.loaders.purr import EchoModelLoader

        desc = PipelineFactoryDescription(
            name="mock_factory",
            pipeline="tests.unit.utils.pipelines.get_mock_pipeline",
            default=True,
            hardware_loader="echo8a",
            engine="qat.engines.waveform_v1.EchoEngine",
        )

        loader = EchoModelLoader()
        factory = desc.construct(loader=loader, engine=EchoEngine())
        assert isinstance(factory, PipelineFactory)
        assert isinstance(factory.engine, EchoEngine)

    @pytest.mark.parametrize(
        "factory, type",
        [
            ("tests.unit.utils.pipelines.get_mock_compile_pipeline", CompilePipeline),
            ("tests.unit.utils.pipelines.get_mock_execute_pipeline", ExecutePipeline),
            ("tests.unit.utils.pipelines.get_mock_pipeline", Pipeline),
            ("tests.unit.utils.pipelines.get_mock_pipeline", CompilePipeline),
            ("tests.unit.utils.pipelines.get_mock_pipeline", ExecutePipeline),
        ],
    )
    def test_is_subtype_of_is_true(self, factory, type):
        mock_desc = PipelineFactoryDescription(name="mock_factory", pipeline=factory)
        assert mock_desc.is_subtype_of(type)

    @pytest.mark.parametrize(
        "factory, type",
        [
            ("tests.unit.utils.pipelines.get_mock_compile_pipeline", ExecutePipeline),
            ("tests.unit.utils.pipelines.get_mock_execute_pipeline", CompilePipeline),
        ],
    )
    def test_is_subtype_of_is_false(self, factory, type):
        mock_desc = PipelineFactoryDescription(name="mock_factory", pipeline=factory)
        assert not mock_desc.is_subtype_of(type)


class TestUpdateablePipelineDescription:
    def test_valid_description(self):
        desc = UpdateablePipelineDescription(
            name="echo8a",
            pipeline="qat.pipelines.echo.EchoPipeline",
            default=True,
            hardware_loader="echo8a",
        )

        assert desc.name == "echo8a"

    def test_invalid_pipeline_class_raises(self):
        with pytest.raises(ValidationError):
            UpdateablePipelineDescription(
                name="invalid",
                pipeline="qat.backend.fallthrough.FallthroughBackend",
                default=True,
            )

    def test_extra_fields_raises_error(self):
        with pytest.raises(ValidationError):
            UpdateablePipelineDescription(
                name="invalid",
                pipeline="qat.pipelines.echo.EchoPipeline",
                default=True,
                backend="qat.backend.fallthrough.FallthroughBackend",
            )

    def test_construct_pipeline(self):
        from tests.unit.utils.pipelines import MockPipeline

        desc = UpdateablePipelineDescription(
            name="mock_updateable",
            pipeline="tests.unit.utils.pipelines.MockPipeline",
            default=True,
            hardware_loader="echo8a",
            engine="qat.engines.waveform_v1.EchoEngine",
        )

        from qat.engines.waveform_v1 import EchoEngine
        from qat.model.loaders.purr import EchoModelLoader

        loader = EchoModelLoader()
        pipeline = desc.construct(loader=loader, engine=EchoEngine())
        assert isinstance(pipeline, MockPipeline)
        assert isinstance(pipeline.engine, EchoEngine)

    @pytest.mark.parametrize(
        "updateable, type",
        [
            ("tests.unit.utils.pipelines.MockUpdateablePipeline", Pipeline),
            ("tests.unit.utils.pipelines.MockUpdateablePipeline", CompilePipeline),
            ("tests.unit.utils.pipelines.MockUpdateablePipeline", ExecutePipeline),
            ("tests.unit.utils.pipelines.MockCompileUpdateablePipeline", CompilePipeline),
            ("tests.unit.utils.pipelines.MockExecuteUpdateablePipeline", ExecutePipeline),
        ],
    )
    def test_is_subtype_of_is_true(self, updateable, type):
        desc = UpdateablePipelineDescription(
            name="mock_updateable",
            pipeline=updateable,
            hardware_loader="test",
        )
        assert desc.is_subtype_of(type)

    @pytest.mark.parametrize(
        "updateable, type",
        [
            ("tests.unit.utils.pipelines.MockCompileUpdateablePipeline", ExecutePipeline),
            ("tests.unit.utils.pipelines.MockExecuteUpdateablePipeline", CompilePipeline),
        ],
    )
    def test_is_subtype_of_is_false(self, updateable, type):
        desc = UpdateablePipelineDescription(
            name="mock_updateable",
            pipeline=updateable,
            hardware_loader="test",
        )
        assert not desc.is_subtype_of(type)


class TestHardwareLoaderDescription:
    def test_valid_description(self):
        desc = HardwareLoaderDescription(
            name="somehardware", type="qat.model.loaders.purr.EchoModelLoader"
        )
        assert isinstance(desc.construct(), BaseModelLoader)

    def test_custom_config(self):
        desc = HardwareLoaderDescription(
            name="somehardware",
            type="qat.model.loaders.purr.EchoModelLoader",
            config={"qubit_count": 8},
        )

        loader = desc.construct()
        assert isinstance(loader, BaseModelLoader)
        assert loader.qubit_count == 8


class TestPipelineClassDescription:
    def test_valid_description(self):
        from qat.model.loaders.purr import EchoModelLoader

        loader = EchoModelLoader()

        desc = PipelineClassDescription(
            name="somepipeline",
            hardware_loader=None,
            frontend="qat.frontend.FallthroughFrontend",
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
            runtime="qat.runtime.SimpleRuntime",
            engine=None,
        )
        pipeline = desc.construct(loader=loader)
        assert isinstance(pipeline, ConfigurablePipeline)
        assert isinstance(pipeline.engine, DefaultEngine)

    def test_end_from_dict(self):
        from qat.model.loaders.purr import EchoModelLoader

        loader = EchoModelLoader()

        desc = PipelineClassDescription(
            name="somepipeline",
            hardware_loader=None,
            frontend={"type": "qat.frontend.FallthroughFrontend"},
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
            runtime="qat.runtime.SimpleRuntime",
            engine=None,
        )
        assert isinstance(desc.construct(loader=loader), ConfigurablePipeline)

    def test_custom_config(self):
        from qat.model.loaders.purr import EchoModelLoader
        from qat.runtime.connection import ConnectionMode

        loader = EchoModelLoader()

        desc = PipelineClassDescription(
            name="somepipeline",
            hardware_loader=None,
            frontend="qat.frontend.FallthroughFrontend",
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
            runtime=dict(
                type="qat.runtime.SimpleRuntime",
                config={"connection_mode": ConnectionMode.ALWAYS_ON_EXECUTE},
            ),
            engine=None,
        )
        P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurablePipeline)
        assert P.runtime.connection_mode == ConnectionMode.ALWAYS_ON_EXECUTE

    def test_extra_field_raises_error(self):
        with pytest.raises(ValidationError):
            PipelineClassDescription(
                name="somepipeline",
                hardware_loader=None,
                frontend="qat.frontend.FallthroughFrontend",
                middleend="qat.middleend.FallthroughMiddleend",
                backend="qat.backend.WaveformV1Backend",
                runtime="qat.runtime.SimpleRuntime",
                engine="qat.engines.ZeroEngine",
                pipeline="qat.pipelines.echo.EchoPipeline",
            )

    def test_construct_pipeline(self):
        desc = PipelineClassDescription(
            name="echo8a",
            hardware_loader=None,
            frontend="qat.frontend.FallthroughFrontend",
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
            runtime="qat.runtime.SimpleRuntime",
            engine=None,
        )

        loader = EchoModelLoader()
        P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurablePipeline)
        assert P.name == "echo8a"
        assert isinstance(P.frontend, FallthroughFrontend)
        assert isinstance(P.middleend, FallthroughMiddleend)
        assert isinstance(P.backend, WaveformV1Backend)
        assert isinstance(P.runtime, SimpleRuntime)
        assert isinstance(P.engine, DefaultEngine)

    def test_construct_pipeline_with_engine(self):
        from qat.engines.waveform_v1 import EchoEngine

        desc = PipelineClassDescription(
            name="echo8a",
            hardware_loader=None,
            frontend="qat.frontend.FallthroughFrontend",
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
            runtime="qat.runtime.SimpleRuntime",
            engine="qat.engines.waveform_v1.EchoEngine",
        )

        loader = EchoModelLoader()
        P = desc.construct(loader=loader, engine=EchoEngine())
        assert isinstance(P, ConfigurablePipeline)
        assert isinstance(P.engine, EchoEngine)

    def test_is_subtype_of(self):
        assert PipelineClassDescription.is_subtype_of(Pipeline)
        assert PipelineClassDescription.is_subtype_of(CompilePipeline)
        assert PipelineClassDescription.is_subtype_of(ExecutePipeline)


class TestCompilePipelineDescription:
    def test_valid_description(self):
        desc = CompilePipelineDescription(
            name="mock_compile",
            hardware_loader="echo8",
            frontend="qat.frontend.FallthroughFrontend",
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
        )

        loader = EchoModelLoader()
        pipeline = desc.construct(loader=loader)
        assert isinstance(pipeline, ConfigurableCompilePipeline)
        assert pipeline.is_subtype_of(CompilePipeline)
        assert pipeline.name == "mock_compile"

    def test_end_from_dict(self):
        desc = CompilePipelineDescription(
            name="mock_compile",
            hardware_loader="echo8",
            frontend={"type": "qat.frontend.FallthroughFrontend"},
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
        )

        loader = EchoModelLoader()
        pipeline = desc.construct(loader=loader)
        assert isinstance(pipeline.frontend, FallthroughFrontend)

    def test_extra_field_raises_error(self):
        with pytest.raises(ValidationError):
            CompilePipelineDescription(
                name="mock_compile",
                hardware_loader="echo8",
                frontend="qat.frontend.FallthroughFrontend",
                middleend="qat.middleend.FallthroughMiddleend",
                backend="qat.backend.WaveformV1Backend",
                runtime="qat.runtime.SimpleRuntime",
            )

    def test_construct_pipeline(self):
        desc = CompilePipelineDescription(
            name="mock_compile",
            hardware_loader="echo8",
            frontend="qat.frontend.FallthroughFrontend",
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
        )

        loader = EchoModelLoader()
        P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurableCompilePipeline)
        assert P.name == "mock_compile"
        assert isinstance(P.frontend, FallthroughFrontend)
        assert isinstance(P.middleend, FallthroughMiddleend)
        assert isinstance(P.backend, WaveformV1Backend)

    def test_is_subtype_of(self):
        assert CompilePipelineDescription.is_subtype_of(CompilePipeline)
        assert not CompilePipelineDescription.is_subtype_of(ExecutePipeline)


class TestExecutePipelineDescription:
    def test_valid_description(self):
        desc = ExecutePipelineDescription(
            name="mock_execute",
            hardware_loader="echo8",
            runtime="qat.runtime.SimpleRuntime",
            engine="echo",
        )

        loader = EchoModelLoader()
        engine = ZeroEngine()
        pipeline = desc.construct(loader=loader, engine=engine)
        assert isinstance(pipeline, ConfigurableExecutePipeline)
        assert pipeline.is_subtype_of(ExecutePipeline)
        assert pipeline.name == "mock_execute"

    def test_runtime_from_dict(self):
        desc = ExecutePipelineDescription(
            name="mock_execute",
            hardware_loader="echo8",
            runtime={"type": "qat.runtime.SimpleRuntime"},
            engine="echo",
        )

        loader = EchoModelLoader()
        engine = ZeroEngine()
        pipeline = desc.construct(loader=loader, engine=engine)
        assert isinstance(pipeline.runtime, SimpleRuntime)

    def test_extra_field_raises_error(self):
        with pytest.raises(ValidationError):
            ExecutePipelineDescription(
                name="mock_execute",
                hardware_loader="echo8",
                runtime="qat.runtime.SimpleRuntime",
                engine="echo",
                frontend="qat.frontend.FallthroughFrontend",
            )

    def test_construct_pipeline(self):
        desc = ExecutePipelineDescription(
            name="mock_execute",
            hardware_loader="echo8",
            runtime="qat.runtime.SimpleRuntime",
            engine="echo",
        )

        loader = EchoModelLoader()
        engine = ZeroEngine()
        P = desc.construct(loader=loader, engine=engine)
        assert isinstance(P, ConfigurableExecutePipeline)
        assert P.name == "mock_execute"
        assert isinstance(P.runtime, SimpleRuntime)
        assert isinstance(P.engine, ZeroEngine)

    def test_construct_pipeline_with_engine_that_requires_model(self):
        desc = ExecutePipelineDescription(
            name="mock_execute",
            hardware_loader="mock",
            runtime="qat.runtime.SimpleRuntime",
            engine="qat.engines.waveform_v1.EchoEngine",
        )

        loader = EchoModelLoader()
        models = {"mock": loader.load()}
        engine = MockEngineWithModel(model=models["mock"])
        cache_loader = CacheAccessLoader(models, "mock")
        P = desc.construct(loader=cache_loader, engine=engine)
        assert isinstance(P, ConfigurableExecutePipeline)
        assert P.engine == engine
        assert P.model == models["mock"]

    def test_is_subtype_of(self):
        assert ExecutePipelineDescription.is_subtype_of(ExecutePipeline)
        assert not ExecutePipelineDescription.is_subtype_of(CompilePipeline)


class TestClassConfig:
    def test_class_description(self):
        from tests.unit.utils.some_classes import SomeClass

        SomeClassDescription = ClassDescription[ImportString]
        desc = SomeClassDescription(type="tests.unit.utils.some_classes.SomeClass")
        assert desc.type is SomeClass
        cls_partial = desc.partial()
        ob = cls_partial()
        assert isinstance(ob, SomeClass)
        assert ob.x == "default_x"

    @pytest.mark.parametrize("config", [{}, {"x": 5}, {"x": 7}])
    def test_class_description_config(self, config):
        from tests.unit.utils.some_classes import SomeClass

        SomeClassDescription = ClassDescription[ImportString]
        desc = SomeClassDescription(
            type="tests.unit.utils.some_classes.SomeClass", config=config
        )

        desc.type is SomeClass
        cls_partial = desc.partial()
        ob = cls_partial()

        assert isinstance(ob, SomeClass)
        assert ob.x == config.get("x", "default_x")
        assert ob.y == config.get("y", "default_y")

    def test_frontend_description(self):
        from qat.model.loaders.lucy import LucyModelLoader

        model = LucyModelLoader().load()

        FrontendDescription = ClassDescription[ImportFrontend]
        frontend_desc = FrontendDescription(type="qat.frontend.AutoFrontend")
        frontend_desc.type is qat.frontend.AutoFrontend
        frontend_partial = frontend_desc.partial()

        frontend = frontend_partial(model=model)
        assert isinstance(frontend, qat.frontend.AutoFrontend)
