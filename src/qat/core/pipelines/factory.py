# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from inspect import signature

from qat.core.config.descriptions import PipelineFactoryDescription
from qat.core.config.validators import requires_model
from qat.model.target_data import AbstractTargetData
from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import Model, UpdateablePipeline
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class PipelineFactory(UpdateablePipeline):
    """An updateable pipeline that allows a pipeline factory (function) to be called.
    This will use the factory as a proxy for the _build_pipeline method, allowing the
    pipeline to be refreshed with a new hardware model and/or config."""

    @staticmethod
    def _build_pipeline(
        config: PipelineFactoryDescription,
        model: Model,
        target_data: AbstractTargetData | None = None,
        engine: None = None,
    ) -> Pipeline:
        """Wraps the pipeline factory function defined in the config, passing the
        model and target data if required."""

        if engine is not None:
            log.warning(
                "An engine was provided to the pipeline factory, which should be provided "
                "via the PipelineFactoryDescription. It will be ignored."
            )

        factory = config.pipeline
        kwargs = {}
        engine = PipelineFactory._create_engine(factory, config.engine, model)
        if engine is not None:
            kwargs["engine"] = engine
        if PipelineFactory._check_target_data(factory):
            kwargs["target_data"] = target_data
        kwargs.update(config.config)
        return factory(model=model, **kwargs)

    @staticmethod
    def _create_engine(factory: callable, engine: type, model=None):
        """Instantiates the engine if provided, and throws a warning if the factory does
        not accept an engine but one is given."""

        if "engine" in signature(factory).parameters:
            if engine is not None:
                if requires_model(engine):
                    return engine(model=model)
                else:
                    return engine()
        elif engine is not None:
            log.warning(
                "An engine was provided for the pipeline factory, but the factory does not "
                "accept an engine argument. The provided engine will be ignored."
            )
        return None

    @staticmethod
    def _check_target_data(factory: callable):
        """Checks if the factory accepts target data."""
        return "target_data" in signature(factory).parameters
