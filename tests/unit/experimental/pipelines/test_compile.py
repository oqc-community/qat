# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for the experimental Qblox compile pipeline."""

import pytest

from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.experimental.backend.qblox.backend import ExperimentalQbloxBackend
from qat.experimental.frontend.purr import PurrFrontend
from qat.experimental.middleend.middleend import PulseLevelMiddleend
from qat.experimental.pipelines.compile import (
    ExperimentalQbloxCompilePipeline,
    ExperimentalQbloxCompilePipelineConfig,
)
from qat.pipelines.pipeline import CompilePipeline

pytest_plugins = ("tests.unit.experimental.utils.canonical",)


class TestExperimentalQbloxCompilePipeline:
    def test_can_be_instantiated_with_canonical_model(self, canonical_model):
        """The public pipeline factory accepts canonical system data directly."""
        pipeline = ExperimentalQbloxCompilePipeline(
            config=ExperimentalQbloxCompilePipelineConfig(),
            model=canonical_model,
        )

        assert pipeline.model is canonical_model
        assert isinstance(pipeline.pipeline, CompilePipeline)

    def test_build_pipeline_uses_canonical_model_and_experimental_components(
        self, canonical_model
    ):
        """The compile pipeline wires canonical data into all experimental components."""
        pipeline = ExperimentalQbloxCompilePipeline._build_pipeline(
            config=ExperimentalQbloxCompilePipelineConfig(),
            model=canonical_model,
            target_data=None,
        )

        assert isinstance(pipeline, CompilePipeline)
        assert pipeline.model is canonical_model
        assert pipeline.target_data is TARGET_DATA
        assert isinstance(pipeline.frontend, PurrFrontend)
        assert pipeline.frontend.model is canonical_model
        assert isinstance(pipeline.middleend, PulseLevelMiddleend)
        assert pipeline.middleend.model is canonical_model
        assert isinstance(pipeline.backend, ExperimentalQbloxBackend)
        assert pipeline.backend.model is canonical_model

    def test_build_pipeline_uses_supplied_target_data(self, canonical_model):
        """The compile builder preserves explicitly supplied target data."""
        target_data = QbloxTargetData()

        pipeline = ExperimentalQbloxCompilePipeline._build_pipeline(
            config=ExperimentalQbloxCompilePipelineConfig(),
            model=canonical_model,
            target_data=target_data,
        )

        assert pipeline.target_data is target_data

    # TODO(COMPILER-1422): Replace this skipped contract test when Qblox code generation lands.
    @pytest.mark.skip(
        reason="COMPILER-1422: experimental Qblox backend code generation is not implemented"
    )
    def test_compile_emits_qblox_program(self):
        """The backend should emit an executable containing Qblox programs."""
        raise AssertionError("Implement when COMPILER-1422 is complete")
