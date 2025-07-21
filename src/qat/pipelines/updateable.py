from abc import abstractmethod
from typing import get_type_hints

from pydantic import BaseModel, TypeAdapter

from qat.engines.native import NativeEngine
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.loaders.base import BaseModelLoader
from qat.model.target_data import TargetData
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
        :param model: The hardware model to feed into the pipeline, cannot be used
            simultaneously with loader, defaults to None.
        :param loader: The hardware loader used to load the hardware model which can be used
            to later refresh the hardware model. Cannot be used simultaneously with the
            model, defaults to None.
        :param target_data: The data concerning the target device, defaults to None
        :param engine: The engine to use for the pipeline, defaults to None.
        :raises ValueError: If neither model nor loader is provided, or if both are
            provided.
        """

        config = TypeAdapter(self.config_type()).validate_python(config)

        if model is None and loader is None:
            raise ValueError("Model or loader must be provided.")

        if model is not None and loader is not None:
            raise ValueError("Either model or loader must be provided, not both.")

        self._loader = loader
        if self._loader is not None:
            model = self._loader.load()
            if engine is not None and hasattr(engine, "model"):
                engine.model = model

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
    def engine(self) -> NativeEngine | None:
        """Return the engine of the pipeline."""
        if hasattr(self._pipeline, "engine"):
            return self._pipeline.engine
        return None

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

        if model is not None and loader is not None:
            raise ValueError("Either model or loader must be provided, not both.")

        if model is not None and reload_model:
            raise ValueError("Model and reload_model cannot be used together.")

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

        if loader is not None:
            self._loader = loader
            reload_model = True

        if reload_model:
            if self._loader is not None:
                model = self._loader.load()
            else:
                raise ValueError("Factory has no Model loader, cannot reload model.")

        if model is None:
            # If no model is provided, use the existing model from the pipeline
            model = self._pipeline.model

        # Update the engine if needed
        if engine is None:
            engine = self.engine
        if hasattr(engine, "model"):
            engine.model = model

        self._pipeline = self.__class__._build_pipeline(
            config=self._pipeline_config,
            model=model,
            target_data=target_data,
            engine=engine,
        )

    def copy(self) -> "UpdateablePipeline":
        """Create a copy of the pipeline factory.

        .. warning::

            This creates a copy of the class which reinstantiates the builder. As such, if
            the pipeline has a loader, this will force the model to be reloaded.
        """
        return self.__class__(
            config=self._pipeline_config,
            model=self.model if self._loader is None else None,
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
    ) -> "UpdateablePipeline":
        """Create a copy of the pipeline factory with updated parameters.

        .. warning::

            If the pipeline has a loader, this will force the model to be reloaded.
            Additionally, if a model is provided, but the pipeline was previously using a
            hardware loader, the loader will not be copied.
        """

        if model is not None and loader is not None:
            raise ValueError("Either model or loader must be provided, not both.")

        if config is not None:
            # Update the pipeline configuration with the new config
            config = TypeAdapter(self.config_type()).validate_python(config)
            config = self._pipeline_config.model_copy(
                update=config.model_dump(exclude_unset=True, exclude_defaults=True)
            )

        if loader is None and model is None:
            loader = self._loader

        if model is None and loader is None:
            model = self.model

        # Update the engine if needed
        if engine is None:
            engine = self.engine
        if hasattr(engine, "model"):
            engine.model = model if model is not None else loader.load()

        return self.__class__(
            config=config if config is not None else self._pipeline_config,
            model=model,
            loader=loader,
            target_data=target_data if target_data is not None else self.target_data,
            engine=engine,
        )

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying pipeline instance."""

        if name in self.__dict__:
            return self.__dict__[name]
        if hasattr(self._pipeline, name):
            return getattr(self._pipeline, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
