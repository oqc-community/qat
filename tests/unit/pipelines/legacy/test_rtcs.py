# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from compiler_config.config import CompilerConfig, QuantumResultsFormat

from qat import QAT
from qat.backend import FallthroughBackend
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.model.loaders.legacy import RTCSModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.legacy.rtcs import (
    LegacyRTCSPipeline,
    LegacyRTCSPipelineConfig,
    legacy_rtcs2,
)
from qat.pipelines.pipeline import Pipeline
from qat.purr.backends.realtime_chip_simulator import RealtimeChipSimEngine
from qat.runtime import LegacyRuntime

from tests.unit.utils.qasm_qir import get_qasm2


class TestRTCSPipelines:
    """Tests legacy RTCS pipelines.

    The tests here are not extensive: the RTCS engine is already tested in the `purr`
    package. The purpose here it just to test that it integrates smoothly in the pipelines,
    and test the integration points that have some friction (e.g. "optimize" function in the
    legacy engine becomes a pass in the pipeline).
    """

    def test_build_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = RTCSModelLoader().load()
        pipeline = LegacyRTCSPipeline._build_pipeline(
            config=LegacyRTCSPipelineConfig(),
            model=model,
            target_data=None,
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "legacy_rtcs"
        assert pipeline.model == model
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, CustomMiddleend)
        assert isinstance(pipeline.backend, FallthroughBackend)
        assert isinstance(pipeline.runtime, LegacyRuntime)
        assert isinstance(pipeline.engine, RealtimeChipSimEngine)

    def execute_bell_state(self, config=None):
        qasm_str = get_qasm2("ghz_2.qasm")
        results, _ = QAT().run(qasm_str, compiler_config=config, pipeline=legacy_rtcs2)
        return results

    def test_bell_with_binary_count(self):
        """Without the sanitization pass on Acquires, this will not pass."""

        config = CompilerConfig(
            results_format=QuantumResultsFormat().binary_count(),
            repeats=TargetData.default().default_shots,
            repetition_period=TargetData.default().QUBIT_DATA.passive_reset_time,
        )
        results = self.execute_bell_state(config)
        assert len(results) == 1
        assert "b" in results
        assert "00" in results["b"]
        assert "11" in results["b"]
        assert results["b"]["00"] + results["b"]["11"] > config.repeats // 2
