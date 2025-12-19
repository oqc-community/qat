# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from abc import abstractmethod
from typing import get_type_hints

from pydantic import BaseModel, TypeAdapter

from qat.engines.model import requires_hardware_model
from qat.engines.native import NativeEngine
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.loaders.base import BaseModelLoader
from qat.model.target_data import TargetData
from qat.model.validators import MismatchingHardwareModelException
from qat.pipelines.base import AbstractPipeline
from qat.pipelines.pipeline import BasePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class PipelineConfig(BaseModel):
    """Base class for configuring updateable pipelines. Subclasses of
    :class:`UpdateablePipeline` should be paried with their own configuration class which
    specifies custom configuration parameters, and/or sets custom defaults."""

    name: str


Model = QuantumHardwareModel | PhysicalHardwareModel


class UpdateablePipeline(AbstractPipeline):
    """A factory for creating and updating pipelines.

    Pipelines are designed to be immutable to maintain performance and prevent errors from
    moving parts. :class:`UpdateablePipeline` provide a mechanism to update and refresh
    pipelines with updated components, such as an updated hardware model, without
    invalidating other components in the pipeline by reconstructing the entire pipeline.
    It will also inherit the properties and methods of the pipeline it creates, allowing
    it to be used as a drop-in replacement for the original pipeline.

    :class:`UpdateablePipeline` can be instantiated with either a hardware model or a
    loader, or both. If both are provided, the model will take precedence for the initial
    pipeline constuction, but the loader can be used to refresh the model later on. If a
    loader is provided but not a model, the loader will be used to load the model
    during the initial pipeline construction.
    """

    def __init__(
        self,
        config: PipelineConfig,
        model: Model | None = None,
        loader: BaseModelLoader | None = None,
        target_data: TargetData | None = None,
        engine: NativeEngine | None = None,
    ):
        """
        :param config: The pipeline configuration with the name of the pipeline, and any
            additional parameters that can be configured in the pipeline.
        :param model: The hardware model to feed into the pipeline. Defaults to None.
        :param loader: The hardware loader used to load the hardware model which can be used
            to later refresh the hardware model. Defaults to None.
        :param target_data: The data concerning the target device, defaults to None
        :param engine: The engine to use for the pipeline, defaults to None.
        :raises ValueError: If neither model nor loader is provided.
        """

        config = TypeAdapter(self.config_type()).validate_python(config)

        if model is None and loader is None:
            raise ValueError("Model or loader must be provided.")

        if model is None:
            model = loader.load()

        # TODO: Change to just check against RequiresHardwareModelMixin (COMPILER-662)
        if requires_hardware_model(engine):
            if engine.model != model:
                raise MismatchingHardwareModelException(
                    f"Engine model {engine.model} does not match the provided model "
                    f"{model}. If the UpdateablePipeline is only provided a loader, and "
                    "not a model, please consider also instantiating with a model."
                )

        self._loader = loader
        self._engine = engine
        self._pipeline_config = config
        self._pipeline = self.__class__._build_pipeline(config, model, target_data, engine)

    @staticmethod
    @abstractmethod
    def _build_pipeline(
        config: PipelineConfig,
        model: Model,
        target_data: TargetData | None,
        engine: NativeEngine | None = None,
    ) -> BasePipeline:
        """Build the pipeline based on the configuration."""

        raise NotImplementedError(
            "Custom pipelines that are subclasses of the `UpdateablePipeline` are required "
            "to implement their own `_build_pipeline` method that returns a `Pipeline` "
            "object."
        )

    def is_subtype_of(self, cls):
        """Matches the type against the pipeline produced by the factory."""
        return isinstance(self, cls) or self._pipeline.is_subtype_of(cls)

    @classmethod
    def config_type(cls) -> type:
        """Inspects the type of the config attribute in :meth:`_build_pipeline`."""
        type_hints = get_type_hints(cls._build_pipeline)
        if "config" not in type_hints:
            return PipelineConfig
        return get_type_hints(cls._build_pipeline)["config"]

    @property
    def name(self) -> str:
        """Return the name of the pipeline."""
        return self._pipeline_config.name

    @property
    def model(self) -> Model:
        """Return the model associated with the pipeline."""
        return self._pipeline.model

    @model.setter
    def model(self, value: Model):
        """Updates the pipeline with some set model."""
        self.update(model=value)

    @property
    def target_data(self) -> TargetData | None:
        """Return the target data associated with the pipeline."""
        return self._pipeline.target_data

    @property
    def pipeline(self) -> BasePipeline:
        """Return the pipeline instance."""
        return self._pipeline

    @property
    def config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._pipeline_config

    @property
    def has_loader(self) -> bool:
        """Check if the pipeline has a loader."""
        return self._loader is not None

    def update(
        self,
        config: PipelineConfig | None = None,
        model: Model | None = None,
        loader: BaseModelLoader | None = None,
        target_data: TargetData | None = None,
        engine: NativeEngine | None = None,
        reload_model: bool = False,
    ):
        """Update the pipeline configuration and rebuild the pipeline with updated
        arguments. The whole pipeline is reinstantiated to avoid conflicts with changing
        components."""

        model, self._loader, reload_model = self._resolve_model(model, loader, reload_model)

        if config is not None:
            config = TypeAdapter(self.config_type()).validate_python(config)
            if (config.name is not None) and config.name != self._pipeline_config.name:
                raise ValueError("Cannot change the name of the pipeline during an update.")

            # Update the pipeline configuration with the new config
            self._pipeline_config = self._pipeline_config.model_copy(
                update=config.model_dump(exclude_unset=True, exclude_defaults=True)
            )

        if target_data is None:
            target_data = self.target_data

        engine = self._engine if engine is None else engine
        # TODO: Change to just check against RequiresHardwareModelMixin (COMPILER-662)
        if requires_hardware_model(engine) and reload_model:
            engine.model = model
            self._engine = engine

        self._pipeline = self.__class__._build_pipeline(
            config=self._pipeline_config,
            model=model,
            target_data=target_data,
            engine=engine,
        )

    def copy(self) -> "UpdateablePipeline":
        """Create a copy of the pipeline factory."""
        return self.__class__(
            config=self._pipeline_config,
            model=self.model,
            loader=self._loader,
            target_data=self.target_data,
            engine=self.engine,
        )

    def copy_with(
        self,
        config: PipelineConfig | None = None,
        model: Model | None = None,
        loader: BaseModelLoader | None = None,
        target_data: TargetData | None = None,
        engine: NativeEngine | None = None,
        reload_model: bool = False,
    ) -> "UpdateablePipeline":
        """Create a copy of the pipeline factory with updated parameters."""

        model, loader, reload_model = self._resolve_model(model, loader, reload_model)

        if config is not None:
            config = TypeAdapter(self.config_type()).validate_python(config)
            config = self._pipeline_config.model_copy(
                update=config.model_dump(exclude_unset=True, exclude_defaults=True)
            )

        engine = self._engine if engine is None else engine
        # TODO: Change to just check against RequiresHardwareModelMixin (COMPILER-662)
        if requires_hardware_model(engine) and reload_model:
            engine.model = model

        return self.__class__(
            config=config if config is not None else self._pipeline_config,
            model=model,
            loader=loader,
            target_data=target_data if target_data is not None else self.target_data,
            engine=engine,
        )

    def _resolve_model(
        self, model: Model, loader: BaseModelLoader | None, reload_model: bool
    ) -> tuple[Model, BaseModelLoader | None, bool]:
        """Resolves the model from the model, loader and reload_model parameters.

        The model is returned, along with the loader and a boolean indicating if the model
        is updated.
        """

        if model is not None and reload_model:
            raise ValueError("Both model and reload_model cannot be used together.")

        if model is None and loader is None and not reload_model:
            return self._pipeline.model, self._loader, False

        if loader is not None and model is None:
            reload_model = True

        if loader is None:
            loader = self._loader

        if reload_model:
            if loader is None:
                raise ValueError("Factory has no Model loader, cannot reload model.")
            model = loader.load()

        return model, loader, True

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying pipeline instance."""

        if name in self.__dict__:
            return self.__dict__[name]
        if hasattr(self._pipeline, name):
            return getattr(self._pipeline, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
