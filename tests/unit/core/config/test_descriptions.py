import pytest
from pydantic import ImportString, ValidationError

import qat
import qat.frontend
import qat.pipelines
from qat.core.config.descriptions import (
    ClassDescription,
    HardwareLoaderDescription,
    ImportFrontend,
    PipelineClassDescription,
    PipelineFactoryDescription,
    PipelineInstanceDescription,
)
from qat.core.config.session import QatSessionConfig
from qat.core.pipeline import Pipeline
from qat.model.loaders.base import BaseModelLoader


@pytest.fixture
def qatconfig_testfiles(testpath):
    return testpath / "files" / "qatconfig"


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


class TestPipelineBuilderDescription:
    def test_valid_description(self):
        desc = PipelineFactoryDescription(
            name="echo8a",
            pipeline="qat.pipelines.echo.get_pipeline",
            default=True,
            hardware_loader="echo8a",
        )

        assert desc.name == "echo8a"

    def test_invalid_pipeline_class_raises(self):
        with pytest.raises(ValidationError):
            PipelineFactoryDescription(
                name="invalid",
                pipeline="qat.backend.fallthrough.FallthroughBackend",
                default=True,
            )


class TestHardwareLoaderDescription:
    def test_valid_description(self):
        desc = HardwareLoaderDescription(
            name="somehardware", type="qat.model.loaders.legacy.EchoModelLoader"
        )
        assert isinstance(desc.construct(), BaseModelLoader)

    def test_custom_config(self):
        desc = HardwareLoaderDescription(
            name="somehardware",
            type="qat.model.loaders.legacy.EchoModelLoader",
            config={"qubit_count": 8},
        )

        loader = desc.construct()
        assert isinstance(loader, BaseModelLoader)
        assert loader.qubit_count == 8


class TestPipelineClassDescription:
    def test_valid_description(self):
        from qat.model.loaders.legacy import EchoModelLoader

        model = EchoModelLoader().load()

        desc = PipelineClassDescription(
            name="somepipeline",
            hardware_loader=None,
            frontend="qat.frontend.FallthroughFrontend",
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
            runtime="qat.runtime.SimpleRuntime",
            engine="qat.engines.ZeroEngine",
        )
        assert isinstance(desc.construct(model=model), Pipeline)

    def test_end_from_dict(self):
        from qat.model.loaders.legacy import EchoModelLoader

        model = EchoModelLoader().load()

        desc = PipelineClassDescription(
            name="somepipeline",
            hardware_loader=None,
            frontend={"type": "qat.frontend.FallthroughFrontend"},
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
            runtime="qat.runtime.SimpleRuntime",
            engine="qat.engines.ZeroEngine",
        )
        assert isinstance(desc.construct(model=model), Pipeline)

    def test_custom_config(self):
        from qat.model.loaders.legacy import EchoModelLoader

        model = EchoModelLoader().load()

        desc = PipelineClassDescription(
            name="somepipeline",
            hardware_loader=None,
            frontend="qat.frontend.FallthroughFrontend",
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
            runtime="qat.runtime.SimpleRuntime",
            engine=dict(type="tests.unit.utils.engines.InitableEngine", config={"x": 9}),
        )
        P = desc.construct(model=model)
        assert isinstance(P, Pipeline)
        assert P.runtime.engine.x == 9


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
        from qat.model.loaders.converted import EchoModelLoader as ConvertedEchoModelLoader

        model = ConvertedEchoModelLoader().load()

        FrontendDescription = ClassDescription[ImportFrontend]
        frontend_desc = FrontendDescription(type="qat.frontend.AutoFrontend")
        frontend_desc.type is qat.frontend.AutoFrontend
        frontend_partial = frontend_desc.partial()

        frontend = frontend_partial(model=model)
        assert isinstance(frontend, qat.frontend.AutoFrontend)


class TestQATConfig:
    """These are tests of QAT config features only used by QAT.pipelines"""

    def test_make_qatconfig_list(self):
        pipelines = [
            PipelineInstanceDescription(
                name="echo8a", pipeline="qat.pipelines.echo.echo8", default=True
            ),
            PipelineInstanceDescription(
                name="echo16a", pipeline="qat.pipelines.echo.echo16"
            ),
            PipelineInstanceDescription(
                name="echo32a", pipeline="qat.pipelines.echo.echo32"
            ),
            PipelineFactoryDescription(
                name="echo6",
                pipeline="qat.pipelines.echo.get_pipeline",
                hardware_loader="echo6loader",
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="echo6loader",
                type="qat.model.loaders.legacy.EchoModelLoader",
                config={"qubit_count": 6},
            )
        ]

        qc = QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware)

        pipes = {P.name: P.pipeline for P in qc.PIPELINES}
        assert pipes == {
            "echo8a": qat.pipelines.echo.echo8,
            "echo16a": qat.pipelines.echo.echo16,
            "echo32a": qat.pipelines.echo.echo32,
            "echo6": qat.pipelines.echo.get_pipeline,
        }

        assert len(qc.HARDWARE) == 1
        assert qc.HARDWARE[0].name == qc.PIPELINES[-1].hardware_loader

    def test_make_qatconfig_class_desc(self):
        pipelines = [
            PipelineClassDescription(
                name="echocustom",
                hardware_loader="echo6loader",
                frontend="qat.frontend.FallthroughFrontend",
                middleend="qat.middleend.FallthroughMiddleend",
                backend="qat.backend.WaveformV1Backend",
                runtime="qat.runtime.SimpleRuntime",
                engine="qat.engines.ZeroEngine",
                default=True,
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="echo6loader",
                type="qat.model.loaders.legacy.EchoModelLoader",
                config={"qubit_count": 6},
            )
        ]

        assert pipelines[0].hardware_loader == "echo6loader"
        qc = QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware)

        pipe_names = {P.name for P in qc.PIPELINES}
        assert pipe_names == {"echocustom"}
        assert len(qc.HARDWARE) == 1
        assert qc.HARDWARE[0].name == qc.PIPELINES[-1].hardware_loader

    def test_mismatching_hardware_loader_raises(self):
        pipelines = [
            PipelineFactoryDescription(
                name="echo6",
                pipeline="qat.pipelines.echo.get_pipeline",
                hardware_loader="echo6loader",
                default=True,
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="notwhatwearelookingfor",
                type="qat.model.loaders.legacy.EchoModelLoader",
                config={"qubit_count": 6},
            )
        ]

        with pytest.raises(ValidationError):
            QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware)

    def test_yaml_custom_config(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "customconfig.yaml")

        from qat.model.loaders.legacy import EchoModelLoader

        model = EchoModelLoader().load()

        desc = qatconfig.PIPELINES[0]
        P = desc.construct(model=model)
        assert isinstance(P, Pipeline)
        assert P.runtime.engine.x == 10
        assert P.runtime.engine.cblam.host == "someurl.com"
        assert P.runtime.engine.cblam.timeout == 60

    def test_yaml_legacy_pipeline(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "legacyengine.yaml")

        from qat.model.loaders.legacy import EchoModelLoader

        model = EchoModelLoader().load()

        desc = qatconfig.PIPELINES[0]
        P = desc.construct(model=model)
        assert isinstance(P, Pipeline)
        assert P.runtime.engine.model is model

    def test_yaml_factory_with_engine(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(
            qatconfig_testfiles / "pipelinefactorywithengine.yaml"
        )

        from qat.model.loaders.legacy import EchoModelLoader

        model = EchoModelLoader().load()

        desc = qatconfig.PIPELINES[0]
        P = desc.construct(model=model)
        assert isinstance(P, Pipeline)
        assert P.runtime.engine.x == 10
        assert P.runtime.engine.cblam.host == "someurl.com"
        assert P.runtime.engine.cblam.timeout == 60

    def test_yaml_custom_result_pipeline(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(
            qatconfig_testfiles / "customresultspipeline.yaml"
        )

        from qat.model.loaders.legacy import EchoModelLoader

        model = EchoModelLoader().load()

        desc = qatconfig.PIPELINES[0]
        P = desc.construct(model=model)
        assert isinstance(P, Pipeline)
        results_pipeline = P.runtime.results_pipeline
        assert len(results_pipeline.passes) == 1
        dummypass = results_pipeline.passes[0]._pass
        assert dummypass.model is model
        assert dummypass.some_int == 12

    def test_yaml_custom_config_from_env(self, monkeypatch, qatconfig_testfiles):
        from qat.model.loaders.legacy import EchoModelLoader

        monkeypatch.setenv("SOME_ENV_VAR", "A_VALUE")
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "envvar.yaml")

        model = EchoModelLoader().load()

        desc = qatconfig.PIPELINES[0]
        P = desc.construct(model=model)
        assert isinstance(P, Pipeline)
        assert P.runtime.engine.x == "A_VALUE"

    def test_yaml_defaults(self, qatconfig_testfiles):
        from qat.backend import DefaultBackend
        from qat.engines import DefaultEngine
        from qat.frontend import DefaultFrontend
        from qat.middleend import DefaultMiddleend
        from qat.model.loaders.legacy import EchoModelLoader
        from qat.runtime import DefaultRuntime

        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "defaults.yaml")

        model = EchoModelLoader().load()

        desc = qatconfig.PIPELINES[0]
        P = desc.construct(model=model)
        assert isinstance(P, Pipeline)
        assert type(P.frontend) is DefaultFrontend
        assert type(P.middleend) is DefaultMiddleend
        assert type(P.backend) is DefaultBackend
        assert type(P.runtime) is DefaultRuntime
        assert type(P.runtime.engine) is DefaultEngine

    def test_yaml_custom_config_missing_from_env(self, monkeypatch, qatconfig_testfiles):
        monkeypatch.delenv("SOME_ENV_VAR", raising=False)
        with pytest.raises(ValueError):
            QatSessionConfig.from_yaml(qatconfig_testfiles / "envvar.yaml")

    def test_yaml_custom_config_invalid_type(self, qatconfig_testfiles):
        """Check that invalid config (types) raise exceptions

        Config is only validated on pipeline construction"""
        from qat.model.loaders.legacy import EchoModelLoader

        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "invalidtype.yaml")
        desc = qatconfig.PIPELINES[0]
        model = EchoModelLoader().load()

        with pytest.raises(ValueError):
            desc.construct(model=model)

    def test_yaml_custom_config_invalid_arg(self, qatconfig_testfiles):
        """Check that invalid config (types) raise exceptions

        Config is only validated on pipeline construction"""
        from qat.model.loaders.legacy import EchoModelLoader

        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "invalidarg.yaml")
        desc = qatconfig.PIPELINES[0]
        model = EchoModelLoader().load()

        with pytest.raises(ValueError):
            desc.construct(model=model)
