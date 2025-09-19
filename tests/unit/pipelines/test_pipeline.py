# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.backend import WaveformV1Backend
from qat.backend.base import BaseBackend
from qat.backend.waveform_v1.codegen import PydWaveformV1Backend
from qat.core.metrics_base import MetricsManager
from qat.engines import ZeroEngine
from qat.executables import AcquireData, AcquireMode, Executable
from qat.frontend import AutoFrontend, FallthroughFrontend
from qat.middleend import (
    DefaultMiddleend,
    ExperimentalDefaultMiddleend,
    FallthroughMiddleend,
)
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import TargetData
from qat.model.validators import MismatchingHardwareModelException
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.runtime import SimpleRuntime

from tests.unit.utils.engines import MockEngineWithModel


class MockExecutable(Executable):
    returns: list[str] = ["test"]

    @property
    def acquires(self):
        return [
            AcquireData(
                output_variable="test", mode=AcquireMode.INTEGRATOR, length=254, position=0
            )
        ]


class MockBackend(BaseBackend):
    def emit(self, ir, res_mgr=None, met_mgr=None, compiler_config=None):
        return MockExecutable()


class TestCompilePipeline:
    @pytest.fixture()
    def mock_pipeline(self):
        model = EchoModelLoader(qubit_count=4).load()
        return CompilePipeline(
            name="TestCompilePipeline",
            model=model,
            frontend=FallthroughFrontend(model=model),
            middleend=FallthroughMiddleend(model=model),
            backend=MockBackend(model=model),
            target_data=TargetData.default(),
        )

    def test_pipeline_initialization(self, mock_pipeline):
        assert mock_pipeline.name == "TestCompilePipeline"
        assert isinstance(mock_pipeline.frontend, FallthroughFrontend)
        assert isinstance(mock_pipeline.middleend, FallthroughMiddleend)
        assert isinstance(mock_pipeline.backend, MockBackend)
        assert isinstance(mock_pipeline.target_data, TargetData)

    def test_pipeline_with_inconsistent_model_raises_error(self):
        model1 = EchoModelLoader(qubit_count=4).load()
        model2 = EchoModelLoader(qubit_count=6).load()
        with pytest.raises(MismatchingHardwareModelException):
            CompilePipeline(
                name="TestCompilePipeline",
                model=model1,
                frontend=AutoFrontend(model=model1),
                middleend=DefaultMiddleend(model=model1),
                backend=MockBackend(model=model2),
                target_data=TargetData.default(),
            )

    def test_copy_returns_new_instance_same_components(self, mock_pipeline):
        copied_pipeline = mock_pipeline.copy()
        assert copied_pipeline is not mock_pipeline
        assert copied_pipeline.name == mock_pipeline.name
        assert copied_pipeline.model is mock_pipeline.model
        assert copied_pipeline.frontend is mock_pipeline.frontend
        assert copied_pipeline.middleend is mock_pipeline.middleend
        assert copied_pipeline.backend is mock_pipeline.backend
        assert copied_pipeline.target_data is mock_pipeline.target_data

    def test_copy_with_name(self, mock_pipeline):
        copied_pipeline = mock_pipeline.copy_with_name(name="NewCompilePipelineName")
        assert mock_pipeline.name == "TestCompilePipeline"
        assert copied_pipeline.name == "NewCompilePipelineName"
        assert copied_pipeline.model is mock_pipeline.model
        assert copied_pipeline.frontend is mock_pipeline.frontend
        assert copied_pipeline.middleend is mock_pipeline.middleend
        assert copied_pipeline.backend is mock_pipeline.backend
        assert copied_pipeline.target_data is mock_pipeline.target_data

    def test_compile(self, mock_pipeline):
        program = "test"
        assert hasattr(mock_pipeline, "compile")
        package, metrics = mock_pipeline.compile(program)
        assert isinstance(package, MockExecutable)
        assert isinstance(metrics, MetricsManager)

    def test_execute_is_not_supported(self, mock_pipeline):
        assert not hasattr(mock_pipeline, "execute")
        with pytest.raises(AttributeError):
            mock_pipeline.execute(MockExecutable())

    def test_validate_model(self):
        model = LucyModelLoader(qubit_count=4).load()

        # Should not raise
        CompilePipeline(
            name="TestCompilePipeline",
            model=model,
            frontend=AutoFrontend(model=model),
            middleend=ExperimentalDefaultMiddleend(model=model),
            backend=PydWaveformV1Backend(model=model),
            target_data=TargetData.default(),
        )

        # will raise
        new_model = LucyModelLoader(qubit_count=6).load()
        with pytest.raises(MismatchingHardwareModelException):
            CompilePipeline(
                name="TestCompilePipeline",
                model=model,
                frontend=AutoFrontend(model=model),
                middleend=ExperimentalDefaultMiddleend(model=model),
                backend=PydWaveformV1Backend(model=new_model),
                target_data=TargetData.default(),
            )


class TestExecutePipeline:
    @pytest.fixture()
    def mock_pipeline(self):
        model = EchoModelLoader(qubit_count=4).load()
        return ExecutePipeline(
            name="TestExecutePipeline",
            model=model,
            runtime=SimpleRuntime(engine=ZeroEngine()),
            target_data=TargetData.default(),
        )

    def test_pipeline_initialization(self, mock_pipeline):
        assert mock_pipeline.name == "TestExecutePipeline"
        assert isinstance(mock_pipeline.runtime, SimpleRuntime)
        assert isinstance(mock_pipeline.target_data, TargetData)
        assert isinstance(mock_pipeline.engine, ZeroEngine)

    def test_pipeline_with_inconsistent_model_raises_error(self):
        model1 = EchoModelLoader(qubit_count=4).load()
        model2 = EchoModelLoader(qubit_count=6).load()
        with pytest.raises(MismatchingHardwareModelException):
            ExecutePipeline(
                name="TestExecutePipeline",
                model=model1,
                runtime=SimpleRuntime(engine=MockEngineWithModel(model2)),
                target_data=TargetData.default(),
            )

    def test_copy_returns_new_instance_same_components(self, mock_pipeline):
        copied_pipeline = mock_pipeline.copy()
        assert copied_pipeline is not mock_pipeline
        assert copied_pipeline.name == mock_pipeline.name
        assert copied_pipeline.model is mock_pipeline.model
        assert copied_pipeline.runtime is mock_pipeline.runtime
        assert copied_pipeline.target_data is mock_pipeline.target_data
        assert copied_pipeline.engine is mock_pipeline.engine

    def test_copy_with_name(self, mock_pipeline):
        copied_pipeline = mock_pipeline.copy_with_name(name="NewExecutePipelineName")
        assert mock_pipeline.name == "TestExecutePipeline"
        assert copied_pipeline.name == "NewExecutePipelineName"
        assert copied_pipeline.model is mock_pipeline.model
        assert copied_pipeline.runtime is mock_pipeline.runtime
        assert copied_pipeline.target_data is mock_pipeline.target_data
        assert copied_pipeline.engine is mock_pipeline.engine

    def test_compile_is_not_supported(self, mock_pipeline):
        assert not hasattr(mock_pipeline, "compile")
        with pytest.raises(AttributeError):
            mock_pipeline.compile("test")

    def test_execute(self, mock_pipeline):
        mock_executable = MockExecutable()
        assert hasattr(mock_pipeline, "execute")
        results, metrics = mock_pipeline.execute(mock_executable)
        assert len(mock_executable.acquires) == 1
        assert "test" in mock_executable.returns
        assert isinstance(results, dict)
        assert "test" in results
        assert len(results["test"]) == 1000
        assert isinstance(metrics, MetricsManager)

    def test_validate_model(self):
        model = LucyModelLoader(qubit_count=4).load()

        # Should not raise
        ExecutePipeline(
            name="TestExecutePipeline",
            model=model,
            runtime=SimpleRuntime(engine=MockEngineWithModel(model)),
            target_data=TargetData.default(),
        )

        # will raise
        new_model = LucyModelLoader(qubit_count=6).load()
        with pytest.raises(MismatchingHardwareModelException):
            ExecutePipeline(
                name="TestExecutePipeline",
                model=model,
                runtime=SimpleRuntime(engine=MockEngineWithModel(new_model)),
                target_data=TargetData.default(),
            )


class TestPipeline:
    @pytest.fixture()
    def mock_pipeline(self):
        model = EchoModelLoader(qubit_count=4).load()
        return Pipeline(
            name="TestPipeline",
            model=model,
            frontend=FallthroughFrontend(model),
            middleend=FallthroughMiddleend(model),
            backend=MockBackend(model),
            runtime=SimpleRuntime(engine=ZeroEngine()),
            target_data=TargetData.default(),
        )

    def test_correct_pipeline_initialization(self, mock_pipeline):
        assert mock_pipeline.name == "TestPipeline"
        assert isinstance(mock_pipeline.frontend, FallthroughFrontend)
        assert isinstance(mock_pipeline.middleend, FallthroughMiddleend)
        assert isinstance(mock_pipeline.backend, MockBackend)
        assert isinstance(mock_pipeline.runtime, SimpleRuntime)
        assert isinstance(mock_pipeline.engine, ZeroEngine)
        assert isinstance(mock_pipeline.target_data, TargetData)

    def test_pipeline_with_inconsistent_model_raises_error(self):
        model1 = EchoModelLoader(qubit_count=4).load()
        model2 = EchoModelLoader(qubit_count=6).load()
        with pytest.raises(MismatchingHardwareModelException):
            Pipeline(
                name="TestPipeline",
                model=model1,
                frontend=AutoFrontend(model=model1),
                middleend=DefaultMiddleend(model=model1),
                backend=WaveformV1Backend(model=model2),
                runtime=SimpleRuntime(engine=ZeroEngine()),
                target_data=TargetData.default(),
            )

    def test_copy_returns_new_instance_same_components(self, mock_pipeline):
        copied_pipeline = mock_pipeline.copy()
        assert copied_pipeline is not mock_pipeline
        assert copied_pipeline.name == mock_pipeline.name
        assert copied_pipeline.model is mock_pipeline.model
        assert copied_pipeline.frontend is mock_pipeline.frontend
        assert copied_pipeline.middleend is mock_pipeline.middleend
        assert copied_pipeline.backend is mock_pipeline.backend
        assert copied_pipeline.runtime is mock_pipeline.runtime
        assert copied_pipeline.target_data is mock_pipeline.target_data
        assert copied_pipeline.engine is mock_pipeline.engine

    def test_copy_with_name(self, mock_pipeline):
        copied_pipeline = mock_pipeline.copy_with_name(name="NewPipelineName")
        assert mock_pipeline.name == "TestPipeline"
        assert copied_pipeline.name == "NewPipelineName"
        assert copied_pipeline.model is mock_pipeline.model
        assert copied_pipeline.frontend is mock_pipeline.frontend
        assert copied_pipeline.middleend is mock_pipeline.middleend
        assert copied_pipeline.backend is mock_pipeline.backend
        assert copied_pipeline.runtime is mock_pipeline.runtime
        assert copied_pipeline.target_data is mock_pipeline.target_data

    def test_compile(self, mock_pipeline):
        program = "test"
        assert hasattr(mock_pipeline, "compile")
        package, metrics = mock_pipeline.compile(program)
        assert isinstance(package, MockExecutable)
        assert isinstance(metrics, MetricsManager)

    def test_execute(self, mock_pipeline):
        mock_executable = MockExecutable()
        assert hasattr(mock_pipeline, "execute")
        results, metrics = mock_pipeline.execute(mock_executable)
        assert len(mock_executable.acquires) == 1
        assert "test" in mock_executable.returns
        assert isinstance(results, dict)
        assert "test" in results
        assert len(results["test"]) == 1000
        assert isinstance(metrics, MetricsManager)
