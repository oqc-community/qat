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
from qat.core.pipeline import HardwareLoaders, PipelineManager
from qat.engines import NativeEngine
from qat.engines.waveform_v1 import EchoEngine
from qat.executables import ChannelExecutable, Executable
from qat.frontend import DefaultFrontend, FallthroughFrontend
from qat.middleend.middleends import FallthroughMiddleend
from qat.model.loaders.purr import EchoModelLoader
from qat.pipelines.echo import EchoPipeline, PipelineConfig
from qat.pipelines.pipeline import Pipeline
from qat.purr.qatconfig import QatConfig
from qat.runtime import SimpleRuntime

from tests.unit.utils.engines import InitableEngine, MockEngineWithModel


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
        assert set(q.pipelines.list_compile_pipelines) == {"echo8", "echo16", "echo32"}
        assert set(q.pipelines.list_execute_pipelines) == {"echo8", "echo16", "echo32"}
        assert q.pipelines.default_compile_pipeline == "echo32"
        assert q.pipelines.default_execute_pipeline == "echo32"
        assert q.pipelines.get_compile_pipeline("default").name == "echo32"
        assert q.pipelines.get_execute_pipeline("default").name == "echo32"

    def test_qat_session_respects_globalqat_config(self):
        qatconfig = get_config()
        OLD_LIMIT = qatconfig.MAX_REPEATS_LIMIT
        NEW_LIMIT = 324214
        assert OLD_LIMIT != NEW_LIMIT
        qatconfig.MAX_REPEATS_LIMIT = NEW_LIMIT

        q = QAT()
        assert set(q.pipelines.list_compile_pipelines) == {"echo8", "echo16", "echo32"}
        assert set(q.pipelines.list_execute_pipelines) == {"echo8", "echo16", "echo32"}
        assert q.pipelines.default_compile_pipeline == "echo32"
        assert q.pipelines.default_execute_pipeline == "echo32"
        assert q.pipelines.get_compile_pipeline("default").name == "echo32"
        assert q.pipelines.get_execute_pipeline("default").name == "echo32"
        assert q.config.MAX_REPEATS_LIMIT == NEW_LIMIT
        qatconfig.MAX_REPEATS_LIMIT = OLD_LIMIT

    def test_qat_session_extends_qatconfig_instance(self):
        qatconfig = QatConfig()
        qatconfig.MAX_REPEATS_LIMIT = 44325
        assert get_config().MAX_REPEATS_LIMIT != qatconfig.MAX_REPEATS_LIMIT

        q = QAT(qatconfig=qatconfig)
        assert set(q.pipelines.list_compile_pipelines) == {"echo8", "echo16", "echo32"}
        assert set(q.pipelines.list_execute_pipelines) == {"echo8", "echo16", "echo32"}
        assert q.pipelines.default_compile_pipeline == "echo32"
        assert q.pipelines.default_execute_pipeline == "echo32"
        assert q.pipelines.get_compile_pipeline("default").name == "echo32"
        assert q.pipelines.get_execute_pipeline("default").name == "echo32"
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
        assert set(q.pipelines.list_compile_pipelines) == {
            "echo8i",
            "echo16i",
            "echo32i",
            "echo6b",
        }
        assert set(q.pipelines.list_execute_pipelines) == {
            "echo8i",
            "echo16i",
            "echo32i",
            "echo6b",
        }

    def test_make_invalid_qatconfig_brokenengine_yaml(self, testpath):
        path = str(testpath / "files/qatconfig/invalid/brokenengine.yaml")
        with pytest.raises(ValueError, match="This engine is broken intentionally."):
            QAT(qatconfig=path)

    def test_make_invalid_qatconfig_brokenloader_yaml(self, testpath):
        path = str(testpath / "files/qatconfig/invalid/brokenloader.yaml")
        with pytest.raises(
            ValueError, match="This loader is broken intentionally on init."
        ):
            QAT(qatconfig=path)

    def test_make_invalid_qatconfig_nohardware_yaml(self, testpath):
        path = str(testpath / "files/qatconfig/invalid/nonexistenthardware.yaml")
        with pytest.raises(ValidationError, match="No module named"):
            QAT(qatconfig=path)

    def test_make_invalid_qatconfig_brokenloader_onload_yaml(self, testpath):
        path = str(testpath / "files/qatconfig/invalid/brokenloader_onload.yaml")
        with pytest.raises(
            ValueError, match="This loader is broken intentionally on load."
        ):
            QAT(qatconfig=path)

    def test_make_qatconfig_yaml(self, testpath):
        path = str(testpath / "files/qatconfig/pipelines.yaml")
        q = QAT(qatconfig=path)
        expected_pipelines = {
            "echo8i",
            "echo16i",
            "echo6b",
            "echo6factory",
            "echo-defaultfrontend",
            "echocustomconfig",
        }
        assert set(q.pipelines.list_compile_pipelines) == expected_pipelines
        assert set(q.pipelines.list_execute_pipelines) == expected_pipelines
        assert (
            type(q.pipelines.get_compile_pipeline("echo-defaultfrontend").frontend)
            is DefaultFrontend
        )
        assert q.pipelines.get_execute_pipeline("echocustomconfig").runtime.engine.x == 10

    def test_make_qatconfig_yaml_curdir(self, testpath, tmpdir, monkeypatch):
        config_file = testpath / "files/qatconfig/pipelines.yaml"
        q1 = QAT()
        monkeypatch.chdir(tmpdir)
        shutil.copy(config_file, tmpdir / "qatconfig.yaml")
        q2 = QAT()
        assert set(q2.pipelines.list_compile_pipelines) != set(
            q1.pipelines.list_compile_pipelines
        )
        assert set(q2.pipelines.list_compile_pipelines) == {
            "echo8i",
            "echo16i",
            "echo6b",
            "echo6factory",
            "echo-defaultfrontend",
            "echocustomconfig",
        }

        assert (
            type(q2.pipelines.get_compile_pipeline("echo-defaultfrontend").frontend)
            is DefaultFrontend
        )
        assert q2.pipelines.get_execute_pipeline("echocustomconfig").runtime.engine.x == 10

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
        assert echo_pipeline.name in q.pipelines.list_compile_pipelines
        assert echo_pipeline.name in q.pipelines.list_execute_pipelines

    def test_remove_pipeline(self, echo_pipeline):
        q = QAT()
        q.pipelines.add(echo_pipeline)
        assert echo_pipeline.name in q.pipelines.list_compile_pipelines
        assert echo_pipeline.name in q.pipelines.list_execute_pipelines
        q.pipelines.remove(echo_pipeline)
        assert echo_pipeline.name not in q.pipelines.list_compile_pipelines
        assert echo_pipeline.name not in q.pipelines.list_execute_pipelines

    def test_add_pipeline_set_default(self, fallthrough_pipeline):
        pipe1 = fallthrough_pipeline.copy_with_name("pipe1")
        pipe2 = fallthrough_pipeline.copy_with_name("pipe2")
        q = QAT()
        q.pipelines.add(pipe1, default=True)
        assert q.pipelines.default_compile_pipeline == pipe1.name
        assert q.pipelines.default_execute_pipeline == pipe1.name
        q.pipelines.add(pipe2)

        assert q.pipelines.default_compile_pipeline == pipe1.name
        assert q.pipelines.default_execute_pipeline == pipe1.name

    def test_set_default(self, fallthrough_pipeline):
        pipe1 = fallthrough_pipeline.copy_with_name("pipe1")
        pipe2 = fallthrough_pipeline.copy_with_name("pipe2")
        pipe3 = fallthrough_pipeline.copy_with_name("pipe3")
        q = QAT()
        q.pipelines.add(pipe1)
        q.pipelines.add(pipe2)
        q.pipelines.add(pipe3)
        q.pipelines.set_default("pipe1")
        assert q.pipelines.default_compile_pipeline == "pipe1"
        assert q.pipelines.default_execute_pipeline == "pipe1"
        q.pipelines.set_default("pipe2")
        assert q.pipelines.default_compile_pipeline == "pipe2"
        assert q.pipelines.default_execute_pipeline == "pipe2"
        q.pipelines.set_default("pipe1")
        assert q.pipelines.default_compile_pipeline == "pipe1"
        assert q.pipelines.default_execute_pipeline == "pipe1"

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

    @pytest.fixture(autouse=True)
    def full_pipelines(self):
        return {
            "standard_echo1": "loader1",
            "standard_echo2": "loader1",
            "standard_echo3": "loader2",
            "custom_pipeline1": "loader1",
            "custom_pipeline2": "loader1",
            "custom_pipeline3": "loader2",
        }

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

    def test_compile_pipelines_are_instantiated_correctly(self, qat, full_pipelines):
        pipelines = qat.pipelines
        hardware = qat._available_hardware

        assert isinstance(pipelines, PipelineManager)
        assert set(pipelines.list_compile_pipelines) == {
            *list(full_pipelines.keys()),
        }

        for pipeline, model in full_pipelines.items():
            assert pipelines.get_compile_pipeline(pipeline).model == hardware.load(model)

    def test_execute_pipelines_are_instantiated_correctly(self, qat, full_pipelines):
        pipelines = qat.pipelines
        hardware = qat._available_hardware
        engines = qat._engines

        assert isinstance(pipelines, PipelineManager)
        assert set(pipelines.list_execute_pipelines) == {
            *list(full_pipelines.keys()),
        }

        for pipeline, model in full_pipelines.items():
            assert pipelines.get_execute_pipeline(pipeline).model == hardware.load(model)

        assert pipelines.get_execute_pipeline("custom_pipeline1").engine == engines.get(
            "InitableEngine"
        )
        assert pipelines.get_execute_pipeline("custom_pipeline2").engine == engines.get(
            "model_engine1"
        )
        assert pipelines.get_execute_pipeline("custom_pipeline3").engine == engines.get(
            "model_engine2"
        )

    def test_reload_models(self, qat, full_pipelines):
        """This tests that hardware models in the pipeline are the expected instances,
        and the number of qubits is only incremented by, the number of loads is the number
        of hardware loaders, and not the number of pipelines."""
        qat.reload_all_models()
        hardware = qat._available_hardware
        pipelines = qat.pipelines
        engines = qat._engines
        for pipeline, model in full_pipelines.items():
            assert pipelines.get_compile_pipeline(pipeline).model == hardware.load(model)
            assert pipelines.get_execute_pipeline(pipeline).model == hardware.load(model)

        assert engines.get("model_engine1").model == hardware.load("loader1")
        assert engines.get("model_engine2").model == hardware.load("loader2")

        assert len(hardware["loader1"].qubits) == 3
        assert len(hardware["loader2"].qubits) == 7
