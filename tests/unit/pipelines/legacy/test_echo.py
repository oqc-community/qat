# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend import FallthroughBackend
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.model.loaders.legacy import EchoModelLoader
from qat.pipelines.legacy.echo import LegacyEchoPipeline, LegacyEchoPipelineConfig
from qat.pipelines.pipeline import Pipeline
from qat.purr.backends.echo import EchoEngine
from qat.runtime import LegacyRuntime


class TestEchoPipelines:
    def test_build_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = EchoModelLoader().load()
        pipeline = LegacyEchoPipeline._build_pipeline(
            config=LegacyEchoPipelineConfig(),
            model=model,
            target_data=None,
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "legacy_echo"
        assert pipeline.model == model
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, CustomMiddleend)
        assert isinstance(pipeline.backend, FallthroughBackend)
        assert isinstance(pipeline.runtime, LegacyRuntime)
        assert isinstance(pipeline.engine, EchoEngine)
