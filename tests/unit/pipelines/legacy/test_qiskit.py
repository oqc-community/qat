# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat import QAT
from qat.backend import FallthroughBackend
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.model.loaders.purr import QiskitModelLoader
from qat.pipelines.legacy.qiskit import (
    LegacyQiskitCompilePipeline,
    LegacyQiskitExecutePipeline,
    LegacyQiskitPipeline,
    PipelineConfig,
    legacy_qiskit8,
)
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.purr.backends.qiskit_simulator import QiskitEngine
from qat.runtime import LegacyRuntime

from tests.unit.utils.qasm_qir import get_qasm2


class TestQiskitPipelines:
    """Tests legacy qiskit pipelines.

    The tests are not extensive, as most of the testing for qiskit engines and builders are
    found in the `purr` package, and any adapted passes are tested by their respective unit
    tests. These tests are just some sanity checks to ensure the pipelines run smoothly.

    Qiskit builders in `purr` cannot be serialized, so we don't test any of that here.
    """

    def test_build_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = QiskitModelLoader(qubit_count=8).load()
        pipeline = LegacyQiskitPipeline._build_pipeline(
            config=PipelineConfig(name="legacy_qiskit"),
            model=model,
            target_data=None,
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "legacy_qiskit"
        assert pipeline.model == model
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, CustomMiddleend)
        assert isinstance(pipeline.backend, FallthroughBackend)
        assert isinstance(pipeline.runtime, LegacyRuntime)
        assert isinstance(pipeline.engine, QiskitEngine)

    def test_build_compile_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = QiskitModelLoader(qubit_count=8).load()
        pipeline = LegacyQiskitCompilePipeline._build_pipeline(
            config=PipelineConfig(name="legacy_qiskit_compile"),
            model=model,
            target_data=None,
        )
        assert isinstance(pipeline, CompilePipeline)
        assert pipeline.name == "legacy_qiskit_compile"
        assert pipeline.model == model
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, CustomMiddleend)
        assert isinstance(pipeline.backend, FallthroughBackend)

    def test_build_execute_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = QiskitModelLoader(qubit_count=8).load()
        pipeline = LegacyQiskitExecutePipeline._build_pipeline(
            config=PipelineConfig(name="legacy_qiskit_execute"),
            model=model,
            target_data=None,
        )
        assert isinstance(pipeline, ExecutePipeline)
        assert pipeline.name == "legacy_qiskit_execute"
        assert pipeline.model == model
        assert isinstance(pipeline.runtime, LegacyRuntime)
        assert isinstance(pipeline.engine, QiskitEngine)

    def test_ghz_gives_expected_results(self):
        """Basic execution of a QASM file to test the pipeline works as expected."""

        qasm_str = get_qasm2("ghz.qasm")
        results, _ = QAT().run(qasm_str, pipeline=legacy_qiskit8)
        assert len(results) == 1
        assert "b" in results
        assert len(results["b"]) == 2
        assert "0000" in results["b"]
        assert "1111" in results["b"]
