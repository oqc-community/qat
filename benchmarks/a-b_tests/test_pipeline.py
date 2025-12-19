# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import pytest

from qat import QAT
from qat.pipelines import DefaultCompile, DefaultExecute, DefaultPostProcessing
from qat.purr.compiler.frontends import QASMFrontend

from benchmarks.utils.helpers import load_experiments, load_qasm

experiments = load_experiments(rtcs_hardware=False)


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10, group="Pipeline:")
@pytest.mark.parametrize("key", experiments.keys())
@pytest.mark.parametrize("mode", ["Legacy", "Pipeline"])
class TestPipeline:
    def test_compile_qasm(self, benchmark, key, mode):
        hw = experiments[key]["hardware"]
        circuit = load_qasm(experiments[key]["circuit"])

        # Wrapper functions for benchmarking
        if mode == "Legacy":
            run = lambda: QASMFrontend().parse(circuit, hw)
        else:
            qat = QAT()
            qat.add_pipeline(
                "test",
                compile_pipeline=DefaultCompile(hw),
                execute_pipeline=DefaultExecute(hw),
                postprocess_pipeline=DefaultPostProcessing(hw),
                engine=hw.create_engine(),
            )
            run = lambda: qat.compile(circuit, pipeline="test")
        benchmark(run)
        assert True

    def test_execute_qasm(self, benchmark, key, mode):
        hw = experiments[key]["hardware"]
        builder = experiments[key]["builder"]

        # Wrapper functions for benchmarking
        if mode == "Legacy":
            run = lambda: QASMFrontend().execute(builder, hw)
        else:
            qat = QAT()
            qat.add_pipeline(
                "test",
                compile_pipeline=DefaultCompile(hw),
                execute_pipeline=DefaultExecute(hw),
                postprocess_pipeline=DefaultPostProcessing(hw),
                engine=hw.create_engine(),
            )
            run = lambda: qat.execute(builder, pipeline="test")
        benchmark(run)
        assert True
