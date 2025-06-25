# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from typing import Callable

from qat.core.config.descriptions import PipelineClassDescription
from qat.core.config.validators import requires_model, requires_target_data
from qat.model.target_data import AbstractTargetData
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import Model, UpdateablePipeline
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


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
        engine: None = None,
    ) -> Pipeline:
        """Constructs a pipeline with each component defined in the config in a granular
        fashion."""

        if engine is not None:
            log.warning(
                "The engine for the ConfigurablePipeline is expected to be provided using "
                "the config, and the provided engine will be ignored."
            )

        frontend = ConfigurablePipeline._inject_model_and_target_data(
            config.frontend, model, target_data
        )
        middleend = ConfigurablePipeline._inject_model_and_target_data(
            config.middleend, model, target_data
        )
        backend = ConfigurablePipeline._inject_model_and_target_data(
            config.backend, model, target_data
        )
        engine = ConfigurablePipeline._create_engine(config.engine, model)
        results_pipeline = ConfigurablePipeline._create_results_pipeline(
            config.results_pipeline, model
        )
        runtime = ConfigurablePipeline._create_runtime(
            config.runtime, engine, results_pipeline, model
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

    @staticmethod
    def _inject_model_and_target_data(
        func: Callable | None, model: Model, target_data: AbstractTargetData
    ):
        """Injects the model and target data into the function if required.

        Hopefully this will eventually be replaced with a dependency injection solution.
        """
        if func is None:
            return None
        kwargs = {}
        if requires_model(func):
            kwargs["model"] = model
        if requires_target_data(func):
            kwargs["target_data"] = target_data
        return func(**kwargs)

    @staticmethod
    def _create_engine(engine, model):
        """Instantiates the engine if provided."""
        if engine is None:
            return None
        if requires_model(engine):
            return engine(model=model)
        return engine()

    @staticmethod
    def _create_results_pipeline(results_pipeline, model):
        """Instantiates the results pipeline if provided."""
        if results_pipeline is None:
            return None
        if requires_model(results_pipeline):
            return results_pipeline(model=model)
        return results_pipeline()

    @staticmethod
    def _create_runtime(runtime, engine, results_pipeline, model):
        """Instantiates the runtime if provided."""
        if runtime is None:
            return None
        if requires_model(runtime):
            return runtime(engine=engine, results_pipeline=results_pipeline, model=model)
        return runtime(engine=engine, results_pipeline=results_pipeline)
