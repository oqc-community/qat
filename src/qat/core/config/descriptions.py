# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from functools import partial
from typing import TYPE_CHECKING, Annotated, Generic, TypeVar

from pydantic import AfterValidator, ConfigDict, ImportString

from qat.utils.pydantic import NoExtraFieldsModel

if TYPE_CHECKING:
    from qat.core.pipelines.configurable import ConfigurablePipeline
    from qat.core.pipelines.factory import PipelineFactory
    from qat.engines.native import NativeEngine
    from qat.model.hardware_model import PhysicalHardwareModel
    from qat.model.loaders.base import BaseModelLoader
    from qat.pipelines.pipeline import Pipeline
    from qat.pipelines.updateable import UpdateablePipeline
    from qat.purr.compiler.hardware_models import QuantumHardwareModel

from .validators import (
    is_backend,
    is_engine,
    is_frontend,
    is_hardwareloader,
    is_middleend,
    is_passmanager_factory,
    is_pipeline_factory,
    is_pipeline_instance,
    is_runtime,
    is_target_data,
    is_updateable_pipeline,
)

ImportPipeline = Annotated[ImportString, AfterValidator(is_pipeline_instance)]
ImportPipelineFactory = Annotated[ImportString, AfterValidator(is_pipeline_factory)]
ImportUpdateablePipeline = Annotated[ImportString, AfterValidator(is_updateable_pipeline)]
ImportHardwareLoader = Annotated[ImportString, AfterValidator(is_hardwareloader)]

ImportFrontend = Annotated[ImportString, AfterValidator(is_frontend)]
ImportMiddleend = Annotated[ImportString, AfterValidator(is_middleend)]
ImportBackend = Annotated[ImportString, AfterValidator(is_backend)]
ImportEngine = Annotated[ImportString, AfterValidator(is_engine)]
ImportRuntime = Annotated[ImportString, AfterValidator(is_runtime)]
ImportTargetData = Annotated[ImportString, AfterValidator(is_target_data)]

ImportPassManagerFactory = Annotated[ImportString, AfterValidator(is_passmanager_factory)]


class PipelineInstanceDescription(NoExtraFieldsModel):
    name: str
    pipeline: ImportPipeline
    default: bool = False

    def construct(self) -> "Pipeline":
        """Returns the requested Pipeline instance"""
        return self.pipeline.copy_with_name(self.name)


class HardwareLoaderDescription(NoExtraFieldsModel):
    name: str
    config: dict = {}
    type: ImportHardwareLoader

    def construct(self) -> "BaseModelLoader":
        """Returns the described Hardware Loader"""
        return self.type(**self.config)


class EngineDescription(NoExtraFieldsModel):
    name: str
    hardware_loader: str | None = None
    config: dict = {}
    type: ImportEngine

    def construct(
        self, model: "None | QuantumHardwareModel | PhysicalHardwareModel" = None
    ) -> "NativeEngine":
        """Returns the described Engine instance, injecting the model if provided."""
        if model is None:
            return self.type(**self.config)
        return self.type(model=model, **self.config)


T = TypeVar("T")


class ClassDescription(NoExtraFieldsModel, Generic[T]):
    type: T
    config: dict = {}

    def partial(self):
        """Returns a partially configured class"""
        try:
            return partial(self.type, **self.config)
        except TypeError as t:
            raise ValueError(f"Validation error {str(t)}")


ToPartialValidator = AfterValidator(lambda v: v.partial())

FrontendDescription = (
    ImportFrontend | Annotated[ClassDescription[ImportFrontend], ToPartialValidator]
)

MiddleendDescription = (
    ImportMiddleend | Annotated[ClassDescription[ImportMiddleend], ToPartialValidator]
)

BackendDescription = (
    ImportBackend | Annotated[ClassDescription[ImportBackend], ToPartialValidator]
)

RuntimeDescription = (
    ImportRuntime | Annotated[ClassDescription[ImportRuntime], ToPartialValidator]
)


TargetDataDescription = (
    ImportTargetData | Annotated[ClassDescription[ImportTargetData], ToPartialValidator]
)


PassManagerFactoryDescription = (
    ImportPassManagerFactory
    | Annotated[ClassDescription[ImportPassManagerFactory], ToPartialValidator]
)


class PipelineFactoryDescription(NoExtraFieldsModel):
    """A description pointing to a function that produces a pipeline, which is configured by
    a model, target data, an engine, and custom configuration."""

    name: str
    hardware_loader: str | None = None
    engine: str | None = None
    target_data: ImportTargetData | None = None
    pipeline: ImportPipelineFactory
    config: dict = {}
    default: bool = False

    def construct(
        self, loader: "BaseModelLoader", engine: "NativeEngine | None" = None
    ) -> "PipelineFactory":
        """Constructs and returns a Pipeline from its description

        :param loader: The hardware model loader to fetch the hardware model.
        :param engine: The engine to use for the pipeline, optional.
        :return: The constructed pipeline
        """

        from qat.core.pipelines.factory import PipelineFactory

        target_data = self.target_data() if self.target_data is not None else None
        return PipelineFactory(
            config=self, loader=loader, target_data=target_data, engine=engine
        )


class UpdateablePipelineDescription(NoExtraFieldsModel):
    """A description pointing to an updateable pipeline, which can be configured with
    a custom hardware model (loader), target data, and an engine. It also always custom
    configuration (given by the concrete updateable pipeline class)."""

    name: str
    pipeline: ImportUpdateablePipeline
    hardware_loader: str | None = None
    engine: str | None = None
    target_data: ImportTargetData | None = None
    config: dict = {}
    default: bool = False

    def construct(
        self, loader: "BaseModelLoader", engine: "NativeEngine | None" = None
    ) -> "UpdateablePipeline":
        """Constructs and returns a Pipeline from its description

        :param loader: The hardware model loader to fetch the hardware model.
        :param engine: The engine to use for the pipeline.
        :return: The updateable pipeline with a constructed pipeline instance.
        """

        config = self.config
        config["name"] = self.name

        target_data = self.target_data() if self.target_data is not None else None
        return self.pipeline(
            config=config, loader=loader, target_data=target_data, engine=engine
        )


class PipelineClassDescription(NoExtraFieldsModel):
    """Allows pipelines to be specified in a granular way and constructed as an updateable
    pipeline, allowing the hardware model to be refreshed."""

    name: str
    hardware_loader: str | None = None
    engine: str | None = None
    frontend: FrontendDescription = "qat.frontend.DefaultFrontend"
    middleend: MiddleendDescription = "qat.middleend.DefaultMiddleend"
    backend: BackendDescription = "qat.backend.DefaultBackend"
    runtime: RuntimeDescription = "qat.runtime.DefaultRuntime"
    target_data: TargetDataDescription = "qat.model.target_data.DefaultTargetData"
    results_pipeline: PassManagerFactoryDescription = (
        "qat.runtime.results_pipeline.get_default_results_pipeline"
    )
    config: dict = {}
    default: bool = False
    model_config = ConfigDict(validate_default=True)

    def construct(
        self, loader: "BaseModelLoader", engine: "NativeEngine | None" = None
    ) -> "ConfigurablePipeline":
        """Constructs and returns a Pipeline from its description

        :param loader: The hardware model loader to fetch the hardware model.
        :param engine: The engine to use for the pipeline.
        :return: The pipeline as a :class:`ConfigurablePipeline` instance.
        """

        from qat.core.pipelines.configurable import ConfigurablePipeline
        from qat.engines import DefaultEngine

        if engine is None:
            engine = DefaultEngine()

        return ConfigurablePipeline(
            config=self, loader=loader, target_data=self.target_data(), engine=engine
        )
