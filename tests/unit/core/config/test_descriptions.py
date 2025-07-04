import pytest
from pydantic import ImportString, ValidationError

import qat
import qat.frontend
import qat.pipelines
from qat.backend.waveform_v1 import WaveformV1Backend
from qat.core.config.descriptions import (
    ClassDescription,
    HardwareLoaderDescription,
    ImportFrontend,
    PipelineClassDescription,
    PipelineFactoryDescription,
    PipelineInstanceDescription,
    UpdateablePipelineDescription,
)
from qat.core.config.session import QatSessionConfig
from qat.core.pipelines.configurable import ConfigurablePipeline
from qat.engines import ZeroEngine
from qat.frontend import FallthroughFrontend
from qat.middleend import FallthroughMiddleend
from qat.model.loaders.base import BaseModelLoader
from qat.model.loaders.purr import EchoModelLoader
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime import SimpleRuntime


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
            engine="qat.engines.ZeroEngine",
        )

        from qat.model.loaders.purr import EchoModelLoader

        loader = EchoModelLoader()
        pipeline = desc.construct(loader=loader)
        assert isinstance(pipeline, MockPipeline)
        assert isinstance(pipeline.engine, ZeroEngine)


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
            engine="qat.engines.ZeroEngine",
        )
        assert isinstance(desc.construct(loader=loader), ConfigurablePipeline)

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
            engine="qat.engines.ZeroEngine",
        )
        assert isinstance(desc.construct(loader=loader), ConfigurablePipeline)

    def test_custom_config(self):
        from qat.model.loaders.purr import EchoModelLoader

        loader = EchoModelLoader()

        desc = PipelineClassDescription(
            name="somepipeline",
            hardware_loader=None,
            frontend="qat.frontend.FallthroughFrontend",
            middleend="qat.middleend.FallthroughMiddleend",
            backend="qat.backend.WaveformV1Backend",
            runtime="qat.runtime.SimpleRuntime",
            engine=dict(type="tests.unit.utils.engines.InitableEngine", config={"x": 9}),
        )
        P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurablePipeline)
        assert P.runtime.engine.x == 9

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
            engine="qat.engines.ZeroEngine",
        )

        loader = EchoModelLoader()
        P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurablePipeline)
        assert P.name == "echo8a"
        assert isinstance(P.frontend, FallthroughFrontend)
        assert isinstance(P.middleend, FallthroughMiddleend)
        assert isinstance(P.backend, WaveformV1Backend)
        assert isinstance(P.runtime, SimpleRuntime)
        assert isinstance(P.engine, ZeroEngine)


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
            UpdateablePipelineDescription(
                name="echo6",
                pipeline="qat.pipelines.echo.EchoPipeline",
                hardware_loader="echo6loader",
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="echo6loader",
                type="qat.model.loaders.purr.EchoModelLoader",
                config={"qubit_count": 6},
            )
        ]

        qc = QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware)

        pipes = {P.name: P.pipeline for P in qc.PIPELINES}
        assert pipes == {
            "echo8a": qat.pipelines.echo.echo8,
            "echo16a": qat.pipelines.echo.echo16,
            "echo32a": qat.pipelines.echo.echo32,
            "echo6": qat.pipelines.echo.EchoPipeline,
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
                type="qat.model.loaders.purr.EchoModelLoader",
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
            UpdateablePipelineDescription(
                name="echo6",
                pipeline="qat.pipelines.echo.EchoPipeline",
                hardware_loader="echo6loader",
                default=True,
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="notwhatwearelookingfor",
                type="qat.model.loaders.purr.EchoModelLoader",
                config={"qubit_count": 6},
            )
        ]

        with pytest.raises(ValidationError):
            QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware)

    def test_yaml_custom_config(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "customconfig.yaml")

        from qat.model.loaders.purr import EchoModelLoader

        loader = EchoModelLoader()

        desc = qatconfig.PIPELINES[0]
        P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurablePipeline)
        assert P.runtime.engine.x == 10
        assert P.runtime.engine.cblam.host == "someurl.com"
        assert P.runtime.engine.cblam.timeout == 60

    def test_yaml_legacy_pipeline(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "legacyengine.yaml")

        from qat.model.loaders.purr import EchoModelLoader

        loader = EchoModelLoader()

        desc = qatconfig.PIPELINES[0]
        P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurablePipeline)
        assert isinstance(P.runtime.engine.model, QuantumHardwareModel)

    def test_yaml_factory_with_engine(self, qatconfig_testfiles):
        from qat.core.config.descriptions import UpdateablePipelineDescription
        from qat.model.loaders.purr import EchoModelLoader

        from tests.unit.utils.engines import InitableEngine
        from tests.unit.utils.pipelines import MockPipeline

        qatconfig = QatSessionConfig.from_yaml(
            qatconfig_testfiles / "pipelinefactorywithengine.yaml"
        )

        loader = EchoModelLoader()

        desc = qatconfig.PIPELINES[0]
        assert isinstance(desc, UpdateablePipelineDescription)
        P = desc.construct(loader=loader)
        assert isinstance(P, MockPipeline)
        assert isinstance(P.runtime.engine, InitableEngine)
        assert P.runtime.engine.x == 10
        assert P.runtime.engine.cblam.host == "someurl.com"
        assert P.runtime.engine.cblam.timeout == 60

    def test_yaml_custom_result_pipeline(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(
            qatconfig_testfiles / "customresultspipeline.yaml"
        )

        from qat.model.loaders.purr import EchoModelLoader

        loader = EchoModelLoader()

        desc = qatconfig.PIPELINES[0]
        P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurablePipeline)
        results_pipeline = P.runtime.results_pipeline
        assert len(results_pipeline.passes) == 1
        dummypass = results_pipeline.passes[0]._pass
        assert isinstance(dummypass.model, QuantumHardwareModel)
        assert dummypass.some_int == 12

    def test_yaml_custom_config_from_env(self, monkeypatch, qatconfig_testfiles):
        from qat.model.loaders.purr import EchoModelLoader

        monkeypatch.setenv("SOME_ENV_VAR", "A_VALUE")
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "envvar.yaml")

        loader = EchoModelLoader()

        desc = qatconfig.PIPELINES[0]
        P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurablePipeline)
        assert P.runtime.engine.x == "A_VALUE"

    def test_yaml_defaults(self, qatconfig_testfiles):
        from qat.backend import DefaultBackend
        from qat.engines import DefaultEngine
        from qat.frontend import DefaultFrontend
        from qat.middleend import DefaultMiddleend
        from qat.model.loaders.purr import EchoModelLoader
        from qat.runtime import DefaultRuntime

        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "defaults.yaml")

        loader = EchoModelLoader()

        desc = qatconfig.PIPELINES[0]
        P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurablePipeline)
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
        from qat.model.loaders.purr import EchoModelLoader

        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "invalidtype.yaml")
        desc = qatconfig.PIPELINES[0]
        loader = EchoModelLoader()

        with pytest.raises(ValueError):
            desc.construct(loader=loader)

    def test_yaml_custom_config_invalid_arg(self, qatconfig_testfiles):
        """Check that invalid config (types) raise exceptions

        Config is only validated on pipeline construction"""
        from qat.model.loaders.purr import EchoModelLoader

        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "invalidarg.yaml")
        desc = qatconfig.PIPELINES[0]
        loader = EchoModelLoader()

        with pytest.raises(ValueError):
            desc.construct(loader=loader)
