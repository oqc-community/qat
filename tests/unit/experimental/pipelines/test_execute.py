# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for the experimental Qblox execute pipeline."""

import pytest

from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.engines.qblox.execution import QbloxEngine
from qat.engines.qblox.live import QbloxLeafInstrument
from qat.experimental.pipelines.execute import (
    ExperimentalQbloxExecutePipeline,
    ExperimentalQbloxExecutePipelineConfig,
)
from qat.pipelines.pipeline import ExecutePipeline
from qat.runtime import SimpleRuntime
from qat.runtime.aggregator import QBloxAggregator

pytest_plugins = ("tests.unit.experimental.utils.canonical",)


class TestExperimentalQbloxExecutePipeline:
    def test_can_be_instantiated_with_canonical_model(self, canonical_model):
        """The public execute pipeline factory accepts canonical system data directly."""
        pipeline = ExperimentalQbloxExecutePipeline(
            config=ExperimentalQbloxExecutePipelineConfig(host="127.0.0.1"),
            model=canonical_model,
        )

        assert pipeline.model is canonical_model
        assert isinstance(pipeline.pipeline, ExecutePipeline)

    def test_build_pipeline_uses_canonical_model_and_qblox_runtime(self, canonical_model):
        """The execute pipeline wires a Qblox engine and aggregator to canonical data."""
        config = ExperimentalQbloxExecutePipelineConfig(
            host="127.0.0.1",
            name="test_qblox_execute",
        )

        pipeline = ExperimentalQbloxExecutePipeline._build_pipeline(
            config=config,
            model=canonical_model,
            target_data=None,
        )

        assert isinstance(pipeline, ExecutePipeline)
        assert pipeline.name == config.name
        assert pipeline.model is canonical_model
        assert pipeline.target_data is TARGET_DATA
        assert isinstance(pipeline.runtime, SimpleRuntime)
        assert isinstance(pipeline.runtime.aggregator, QBloxAggregator)
        assert isinstance(pipeline.engine, QbloxEngine)
        assert isinstance(pipeline.engine.instrument, QbloxLeafInstrument)
        assert pipeline.engine.instrument.address == config.host

    def test_build_pipeline_uses_supplied_target_data(self, canonical_model):
        """The execute builder preserves explicitly supplied target data."""
        target_data = QbloxTargetData()
        config = ExperimentalQbloxExecutePipelineConfig(host="127.0.0.1")

        pipeline = ExperimentalQbloxExecutePipeline._build_pipeline(
            config=config,
            model=canonical_model,
            target_data=target_data,
        )

        assert pipeline.target_data is target_data

    def test_build_pipeline_ignores_supplied_engine(self, canonical_model, caplog):
        """The execute builder warns and creates its configured Qblox engine."""
        config = ExperimentalQbloxExecutePipelineConfig(host="127.0.0.1")
        supplied_engine = QbloxEngine(QbloxLeafInstrument("supplied", "supplied"))

        with caplog.at_level("WARNING"):
            pipeline = ExperimentalQbloxExecutePipeline._build_pipeline(
                config=config,
                model=canonical_model,
                target_data=None,
                engine=supplied_engine,
            )

        assert "provided to the ExperimentalQbloxExecutePipeline" in caplog.text
        assert pipeline.engine is not supplied_engine
        assert isinstance(pipeline.engine, QbloxEngine)
        assert pipeline.engine.instrument.address == config.host

    # TODO(COMPILER-1416): Replace this skipped contract test with the production pass pipeline.
    @pytest.mark.skip(
        reason="COMPILER-1416: dedicated experimental Qblox results pipeline is not available"
    )
    def test_results_pipeline_processes_qblox_acquisitions(self):
        """The execute pipeline should use the dedicated Qblox results pipeline."""
        raise AssertionError("Implement when COMPILER-1416 is complete")
