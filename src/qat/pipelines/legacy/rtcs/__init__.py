# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.loaders.purr import RTCSModelLoader
from qat.pipelines.legacy.base import LegacyExecutePipeline as LegacyRTCSExecutePipeline
from qat.pipelines.updateable import PipelineConfig

from .compile import LegacyRTCSCompilePipeline
from .full import LegacyRTCSPipeline

legacy_rtcs2 = LegacyRTCSPipeline(
    config=PipelineConfig(name="legacy_rtcs2"),
    loader=RTCSModelLoader(),
).pipeline

__all__ = [
    "LegacyRTCSPipeline",
    "LegacyRTCSCompilePipeline",
    "LegacyRTCSExecutePipeline",
    "PipelineConfig",
    "legacy_rtcs2",
]
