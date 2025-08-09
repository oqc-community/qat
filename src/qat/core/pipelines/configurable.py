# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from typing import Callable

from qat.core.config.descriptions import (
    CompilePipelineDescription,
    ExecutePipelineDescription,
    PipelineClassDescription,
)
from qat.core.config.validators import requires_model, requires_target_data
from qat.engines import NativeEngine
from qat.model.target_data import AbstractTargetData
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.pipelines.updateable import Model, UpdateablePipeline


def _inject_model_and_target_data(
    func: Callable | None, model: Model, target_data: AbstractTargetData, **kwargs
):
    """Injects the model and target data into the function if required.

    Hopefully this will eventually be replaced with a dependency injection solution.
    """
    if func is None:
        return None
    if requires_model(func):
        kwargs["model"] = model
    if requires_target_data(func):
        kwargs["target_data"] = target_data
    return func(**kwargs)


class ConfigurableCompilePipeline(UpdateablePipeline):
    """A pipeline that allows each constituent to be defined in a granular fashion, and
    updated. The config is provided via the :class:`CompilePipelineDescription`.

    Allows a config to be provided that configures each component of the pipeline via
    the qatconfig.
    """

    @staticmethod
    def _build_pipeline(
        config: CompilePipelineDescription,
        model: Model,
        target_data: AbstractTargetData,
        engine: NativeEngine | None = None,
    ) -> CompilePipeline:
        """Constructs a pipeline with each component defined in the config in a granular
        fashion."""

        frontend = _inject_model_and_target_data(config.frontend, model, target_data)
        middleend = _inject_model_and_target_data(config.middleend, model, target_data)
        backend = _inject_model_and_target_data(config.backend, model, target_data)

        return CompilePipeline(
            name=config.name,
            model=model,
            frontend=frontend,
            middleend=middleend,
            backend=backend,
            target_data=target_data,
        )


class ConfigurableExecutePipeline(UpdateablePipeline):
    """A pipeline that allows each constituent to be defined in a granular fashion, and
    updated. The config is provided via the :class:`ExecutePipelineDescription`.

    Allows a config to be provided that configures each component of the pipeline via
    the qatconfig.
    """

    @staticmethod
    def _build_pipeline(
        config: ExecutePipelineDescription,
        model: Model,
        target_data: AbstractTargetData,
        engine: NativeEngine,
    ) -> Pipeline:
        """Constructs a pipeline with each component defined in the config in a granular
        fashion."""

        results_pipeline = _inject_model_and_target_data(
            config.results_pipeline, model, target_data
        )
        runtime = _inject_model_and_target_data(
            config.runtime,
            model,
            target_data,
            engine=engine,
            results_pipeline=results_pipeline,
        )
        return ExecutePipeline(
            name=config.name,
            model=model,
            runtime=runtime,
            target_data=target_data,
        )


class ConfigurablePipeline(UpdateablePipeline):
    """A pipeline that allows each constituent to be defined in a granular fashion, and
    updated. The config is provided via the :class:`PipelineClassDescription`.

    Allows a config to be provided that configures each component of the pipeline via
    the qatconfig.
    """

    @staticmethod
    def _build_pipeline(
        config: PipelineClassDescription,
        model: Model,
        target_data: AbstractTargetData,
        engine: NativeEngine,
    ) -> Pipeline:
        """Constructs a pipeline with each component defined in the config in a granular
        fashion."""

        frontend = _inject_model_and_target_data(config.frontend, model, target_data)
        middleend = _inject_model_and_target_data(config.middleend, model, target_data)
        backend = _inject_model_and_target_data(config.backend, model, target_data)
        results_pipeline = _inject_model_and_target_data(
            config.results_pipeline, model, target_data
        )
        runtime = _inject_model_and_target_data(
            config.runtime,
            model,
            target_data,
            engine=engine,
            results_pipeline=results_pipeline,
        )

        return Pipeline(
            name=config.name,
            model=model,
            frontend=frontend,
            middleend=middleend,
            backend=backend,
            runtime=runtime,
            target_data=target_data,
        )
