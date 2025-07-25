# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from inspect import signature

from qat.core.config.descriptions import PipelineFactoryDescription
from qat.engines import NativeEngine
from qat.model.target_data import AbstractTargetData
from qat.pipelines.base import BasePipeline
from qat.pipelines.updateable import Model, UpdateablePipeline


class PipelineFactory(UpdateablePipeline):
    """An updateable pipeline that allows a pipeline factory (function) to be called.
    This will use the factory as a proxy for the _build_pipeline method, allowing the
    pipeline to be refreshed with a new hardware model and/or config."""

    @staticmethod
    def _build_pipeline(
        config: PipelineFactoryDescription,
        model: Model,
        target_data: AbstractTargetData | None = None,
        engine: NativeEngine | None = None,
    ) -> BasePipeline:
        """Wraps the pipeline factory function defined in the config, passing the
        model and target data if required."""

        factory = config.pipeline
        kwargs = {}
        if PipelineFactory._has_argument(factory, "engine"):
            kwargs["engine"] = engine
        if PipelineFactory._has_argument(factory, "target_data"):
            kwargs["target_data"] = target_data
        kwargs.update(config.config)
        return factory(model=model, **kwargs)

    @staticmethod
    def _has_argument(factory: callable, arg: str):
        """Checks if the factory accepts the requested argument."""
        return arg in signature(factory).parameters
