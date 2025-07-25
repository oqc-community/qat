# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import shutil
from typing import Dict

import numpy as np
import pytest
from pydantic import ValidationError

from qat import QAT
from qat.backend.fallthrough import FallthroughBackend
from qat.backend.waveform_v1 import WaveformV1Backend
from qat.core.config.configure import get_config
from qat.core.config.descriptions import (
    HardwareLoaderDescription,
    PipelineInstanceDescription,
    UpdateablePipelineDescription,
)
from qat.core.config.session import QatSessionConfig
from qat.core.pipeline import HardwareLoaders, PipelineSet
from qat.engines import NativeEngine
from qat.engines.waveform_v1 import EchoEngine
from qat.executables import ChannelExecutable, Executable
from qat.frontend import DefaultFrontend, FallthroughFrontend
from qat.middleend.middleends import FallthroughMiddleend
from qat.model.loaders.purr import EchoModelLoader
from qat.pipelines.base import AbstractPipeline
from qat.pipelines.echo import EchoPipeline, PipelineConfig
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.purr.qatconfig import QatConfig
from qat.runtime import SimpleRuntime

from tests.unit.utils.engines import InitableEngine, MockEngineWithModel
from tests.unit.utils.pipelines import (
    MockCompileUpdateablePipeline,
    MockExecuteUpdateablePipeline,
    MockPipelineConfig,
    MockUpdateablePipeline,
)


class MockEngine(NativeEngine):
    def execute(self, package: Executable) -> Dict[str, np.ndarray]:
        return {}

    def startup(self): ...
    def shutdown(self): ...


@pytest.fixture
def echo_pipeline(qubit_count=32):
    loader = EchoModelLoader(qubit_count=qubit_count)
    yield EchoPipeline(config=PipelineConfig(name="echo"), loader=loader)


@pytest.fixture
def fallthrough_pipeline(qubit_count=32):
    model = EchoModelLoader(qubit_count=qubit_count).load()
    yield Pipeline(
        name="fallthrough",
        frontend=FallthroughFrontend(),
        middleend=FallthroughMiddleend(),
        backend=FallthroughBackend(),
        runtime=SimpleRuntime(engine=MockEngine()),
        model=model,
    )


class TestQATPipelineSetup:
    def test_make_qat(self):
        q = QAT()
        assert set(q.pipelines.list()) == {"echo8", "echo16", "echo32"}
        assert q.pipelines.default == "echo32"
        assert q.pipelines.get("default").name == "echo32"

    def test_qat_session_respects_globalqat_config(self):
        qatconfig = get_config()
        OLD_LIMIT = qatconfig.MAX_REPEATS_LIMIT
        NEW_LIMIT = 324214
        assert OLD_LIMIT != NEW_LIMIT
        qatconfig.MAX_REPEATS_LIMIT = NEW_LIMIT

        q = QAT()
        assert set(q.pipelines.list()) == {"echo8", "echo16", "echo32"}
        assert q.pipelines.default == "echo32"
        assert q.pipelines.get("default").name == "echo32"
        assert q.config.MAX_REPEATS_LIMIT == NEW_LIMIT
        qatconfig.MAX_REPEATS_LIMIT = OLD_LIMIT

    def test_qat_session_extends_qatconfig_instance(self):
        qatconfig = QatConfig()
        qatconfig.MAX_REPEATS_LIMIT = 44325
        assert get_config().MAX_REPEATS_LIMIT != qatconfig.MAX_REPEATS_LIMIT

        q = QAT(qatconfig=qatconfig)
        assert set(q.pipelines.list()) == {"echo8", "echo16", "echo32"}
        assert q.pipelines.default == "echo32"
        assert q.pipelines.get("default").name == "echo32"
        assert q.config.MAX_REPEATS_LIMIT == qatconfig.MAX_REPEATS_LIMIT

    def test_make_qatconfig_list(self):
        pipelines = [
            PipelineInstanceDescription(
                name="echo8i", pipeline="qat.pipelines.echo.echo8", default=True
            ),
            PipelineInstanceDescription(
                name="echo16i", pipeline="qat.pipelines.echo.echo16"
            ),
            PipelineInstanceDescription(
                name="echo32i", pipeline="qat.pipelines.echo.echo32"
            ),
            UpdateablePipelineDescription(
                name="echo6b",
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

        q = QAT(qatconfig=QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware))
        assert set(q.pipelines.list()) == {"echo8i", "echo16i", "echo32i", "echo6b"}

    def test_make_invalid_qatconfig_brokenengine_yaml(self, testpath):
        path = str(testpath / "files/qatconfig/invalid_brokenengine.yaml")
        with pytest.raises(ValueError, match="This engine is broken intentionally."):
            QAT(qatconfig=path)

    def test_make_invalid_qatconfig_brokenloader_yaml(self, testpath):
        path = str(testpath / "files/qatconfig/invalid_brokenloader.yaml")
        with pytest.raises(
            ValueError, match="This loader is broken intentionally on init."
        ):
            QAT(qatconfig=path)

    def test_make_invalid_qatconfig_nohardware_yaml(self, testpath):
        path = str(testpath / "files/qatconfig/invalid_nonexistenthardware.yaml")
        with pytest.raises(ValidationError, match="No module named"):
            QAT(qatconfig=path)

    def test_make_invalid_qatconfig_brokenloader_onload_yaml(self, testpath):
        path = str(testpath / "files/qatconfig/invalid_brokenloader_onload.yaml")
        with pytest.raises(
            ValueError, match="This loader is broken intentionally on load."
        ):
            QAT(qatconfig=path)

    def test_make_qatconfig_yaml(self, testpath):
        path = str(testpath / "files/qatconfig/pipelines.yaml")
        q = QAT(qatconfig=path)
        assert set(q.pipelines.list()) == {
            "echo8i",
            "echo16i",
            "echo6b",
            "echo6factory",
            "echo-defaultfrontend",
            "echocustomconfig",
        }
        assert type(q.pipelines.get("echo-defaultfrontend").frontend) is DefaultFrontend
        assert q.pipelines.get("echocustomconfig").runtime.engine.x == 10

    def test_make_qatconfig_yaml_curdir(self, testpath, tmpdir, monkeypatch):
        config_file = testpath / "files/qatconfig/pipelines.yaml"
        q1 = QAT()
        monkeypatch.chdir(tmpdir)
        shutil.copy(config_file, tmpdir / "qatconfig.yaml")
        q2 = QAT()
        assert set(q2.pipelines.list()) != set(q1.pipelines.list())
        assert set(q2.pipelines.list()) == {
            "echo8i",
            "echo16i",
            "echo6b",
            "echo6factory",
            "echo-defaultfrontend",
            "echocustomconfig",
        }

        assert type(q2.pipelines.get("echo-defaultfrontend").frontend) is DefaultFrontend
        assert q2.pipelines.get("echocustomconfig").runtime.engine.x == 10

    def test_FallthroughFrontend(self):
        frontend = FallthroughFrontend()
        assert frontend.model is None

    def test_FallthroughMiddleend(self):
        middleend = FallthroughMiddleend()
        assert middleend.model is None

    def test_FallthroughBackend(self):
        backend = FallthroughBackend()
        assert backend.model is None
        assert backend.emit(ir="blah") == "blah"

    def test_make_pipeline(self):
        qubit_count = 32
        model = EchoModelLoader(qubit_count=qubit_count).load()
        frontend = FallthroughFrontend()
        middleend = FallthroughMiddleend()
        backend = WaveformV1Backend(model)
        runtime = SimpleRuntime(engine=EchoEngine())

        Pipeline(
            name="default",
            frontend=frontend,
            middleend=middleend,
            backend=backend,
            runtime=runtime,
            model=model,
        )

    def test_add_pipeline(self, echo_pipeline):
        q = QAT()
        q.pipelines.add(echo_pipeline)
        assert echo_pipeline.name in q.pipelines.list()

    def test_remove_pipeline(self, echo_pipeline):
        q = QAT()
        q.pipelines.add(echo_pipeline)
        assert echo_pipeline.name in q.pipelines.list()
        q.pipelines.remove(echo_pipeline)
        assert echo_pipeline.name not in q.pipelines.list()

    def test_add_pipeline_set_default(self, fallthrough_pipeline):
        pipe1 = fallthrough_pipeline.copy_with_name("pipe1")
        pipe2 = fallthrough_pipeline.copy_with_name("pipe2")
        q = QAT()
        q.pipelines.add(pipe1, default=True)
        assert q.pipelines.default == pipe1.name
        q.pipelines.add(pipe2)
        assert q.pipelines.default == pipe1.name

    def test_set_default(self, fallthrough_pipeline):
        pipe1 = fallthrough_pipeline.copy_with_name("pipe1")
        pipe2 = fallthrough_pipeline.copy_with_name("pipe2")
        pipe3 = fallthrough_pipeline.copy_with_name("pipe3")
        q = QAT()
        q.pipelines.add(pipe1)
        q.pipelines.add(pipe2)
        q.pipelines.add(pipe3)
        q.pipelines.set_default("pipe1")
        assert q.pipelines.default == "pipe1"
        q.pipelines.set_default("pipe2")
        assert q.pipelines.default == "pipe2"
        q.pipelines.set_default("pipe1")
        assert q.pipelines.default == "pipe1"

    def test_compile(self, fallthrough_pipeline):
        src = "test"
        q = QAT()
        q.pipelines.add(fallthrough_pipeline)
        pkg, _ = q.compile(src, pipeline="fallthrough")
        assert pkg == src

    def test_execute(self, fallthrough_pipeline):
        pkg = ChannelExecutable()
        q = QAT()
        q.pipelines.add(fallthrough_pipeline)
        res, _ = q.execute(pkg, pipeline="fallthrough")
        assert res == {}


class TestQATHardwareModelReloading:
    @pytest.fixture(autouse=True)
    def qat(self):
        """Fixture to create a QAT instance with a specific configuration."""
        return QAT("tests/files/qatconfig/modelreloading.yaml")

    def test_models_are_instantiated_correctly(self, qat):
        hardware = qat._available_hardware
        assert isinstance(hardware, HardwareLoaders)
        assert len(hardware._loaded_models) == 2
        assert ("loader1", "loader2") == tuple(hardware._loaded_models.keys())
        assert len(hardware["loader1"].qubits) == 2
        assert len(hardware["loader2"].qubits) == 6

    def test_engines_are_instantiated_correctly(self, qat):
        engines = qat._engines
        assert len(engines._engines) == 3
        assert ("InitableEngine", "model_engine1", "model_engine2") == tuple(
            engines._engines.keys()
        )
        assert ("model_engine1", "model_engine2") == tuple(engines._loaders.keys())
        assert isinstance(engines.get("InitableEngine"), InitableEngine)
        assert isinstance(engines.get("model_engine1"), MockEngineWithModel)
        assert isinstance(engines.get("model_engine2"), MockEngineWithModel)

    def test_pipelines_are_instantiated_correctly(self, qat):
        pipelines = qat.pipelines
        hardware = qat._available_hardware
        engines = qat._engines
        assert isinstance(pipelines, PipelineSet)
        assert len(pipelines._pipelines) == 6
        expected_pipelines = (
            "standard_echo1",
            "standard_echo2",
            "standard_echo3",
            "custom_pipeline1",
            "custom_pipeline2",
            "custom_pipeline3",
        )
        assert expected_pipelines == tuple(pipelines.list())

        assert pipelines.get("standard_echo1").model == hardware.load("loader1")
        assert pipelines.get("standard_echo2").model == hardware.load("loader1")
        assert pipelines.get("standard_echo3").model == hardware.load("loader2")
        assert pipelines.get("custom_pipeline1").model == hardware.load("loader1")
        assert pipelines.get("custom_pipeline2").model == hardware.load("loader1")
        assert pipelines.get("custom_pipeline3").model == hardware.load("loader2")

        assert pipelines.get("custom_pipeline1").engine == engines.get("InitableEngine")
        assert pipelines.get("custom_pipeline2").engine == engines.get("model_engine1")
        assert pipelines.get("custom_pipeline3").engine == engines.get("model_engine2")

    def test_reload_models(self, qat):
        """This tests that hardware models in the pipeline are the expected instances,
        and the number of qubits is only incremented by, the number of loads is the number
        of hardware loaders, and not the number of pipelines."""
        qat.reload_all_models()
        hardware = qat._available_hardware
        pipelines = qat.pipelines
        engines = qat._engines
        assert pipelines.get("standard_echo1").model == hardware.load("loader1")
        assert pipelines.get("standard_echo2").model == hardware.load("loader1")
        assert pipelines.get("standard_echo3").model == hardware.load("loader2")
        assert pipelines.get("custom_pipeline1").model == hardware.load("loader1")
        assert pipelines.get("custom_pipeline2").model == hardware.load("loader1")
        assert pipelines.get("custom_pipeline3").model == hardware.load("loader2")

        assert engines.get("model_engine1").model == hardware.load("loader1")
        assert engines.get("model_engine2").model == hardware.load("loader2")

        assert len(hardware["loader1"].qubits) == 3
        assert len(hardware["loader2"].qubits) == 7


class TestQAT:
    """Tests the methods of the QAT class."""

    @classmethod
    def setup_class(cls):
        """Setup method to create a QAT instance with the default pipelines."""
        qat = QAT()
        model = EchoModelLoader().load()
        qat.pipelines.add(
            Pipeline(
                name="fallthrough",
                frontend=FallthroughFrontend(),
                middleend=FallthroughMiddleend(),
                backend=FallthroughBackend(),
                runtime=SimpleRuntime(engine=MockEngine()),
                model=model,
            )
        )
        qat.pipelines.add(
            CompilePipeline(
                name="fallthrough_compile",
                frontend=FallthroughFrontend(),
                middleend=FallthroughMiddleend(),
                backend=FallthroughBackend(),
                model=model,
            )
        )
        qat.pipelines.add(
            ExecutePipeline(
                name="fallthrough_execute",
                runtime=SimpleRuntime(engine=MockEngine()),
                model=model,
            )
        )
        qat.pipelines.add(
            MockUpdateablePipeline(
                config=MockPipelineConfig(name="mock_updateable"), model=model
            )
        )
        qat.pipelines.add(
            MockCompileUpdateablePipeline(
                config=MockPipelineConfig(name="mock_compile_updateable"), model=model
            )
        )
        qat.pipelines.add(
            MockExecuteUpdateablePipeline(
                config=MockPipelineConfig(name="mock_execute_updateable"), model=model
            )
        )
        cls.qat = qat

    @pytest.mark.parametrize(
        "pipeline",
        [
            "fallthrough",
            "fallthrough_compile",
            "mock_compile_updateable",
            "mock_updateable",
        ],
    )
    def test_validate_pipeline_can_compile(self, pipeline):
        """Tests that the method correctly identifies pipelines that can compile."""
        pipeline = self.qat.pipelines.get(pipeline)
        assert isinstance(pipeline, AbstractPipeline)
        self.qat._validate_pipeline_can_compile(pipeline)

    @pytest.mark.parametrize(
        "pipeline",
        [
            "fallthrough_execute",
            "mock_execute_updateable",
        ],
    )
    def test_validate_pipeline_can_compile_fails(self, pipeline):
        """Tests that the method raises an error for pipelines that cannot compile."""
        pipeline = self.qat.pipelines.get(pipeline)
        with pytest.raises(TypeError):
            self.qat._validate_pipeline_can_compile(pipeline)

    @pytest.mark.parametrize(
        "pipeline",
        [
            "fallthrough_execute",
            "fallthrough",
            "mock_execute_updateable",
            "mock_updateable",
        ],
    )
    def test_validate_pipeline_can_execute(self, pipeline):
        """Tests that the method correctly identifies pipelines that can execute."""
        pipeline = self.qat.pipelines.get(pipeline)
        assert isinstance(pipeline, AbstractPipeline)
        self.qat._validate_pipeline_can_execute(pipeline)

    @pytest.mark.parametrize(
        "pipeline",
        [
            "fallthrough_compile",
            "mock_compile_updateable",
        ],
    )
    def test_validate_pipeline_can_execute_fails(self, pipeline):
        """Tests that the method raises an error for pipelines that cannot execute."""
        pipeline = self.qat.pipelines.get(pipeline)
        with pytest.raises(TypeError):
            self.qat._validate_pipeline_can_execute(pipeline)

    @pytest.mark.parametrize("compile", [None, "fallthrough_compile"])
    @pytest.mark.parametrize("execute", [None, "fallthrough_execute"])
    def test_resolve_pipelines_uses_correct_pipelines(self, compile, execute):
        """If specified, the compile and execute pipelines should be used,
        otherwise the default pipeline should be used."""

        compile_pipeline, execute_pipeline = self.qat._resolve_pipelines(
            compile,
            execute,
            "fallthrough",
        )
        assert compile_pipeline.name == ("fallthrough" if compile is None else compile)
        assert execute_pipeline.name == ("fallthrough" if execute is None else execute)
