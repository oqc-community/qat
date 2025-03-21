# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from functools import partial
from typing import TYPE_CHECKING, Annotated, Generic, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, ImportString

if TYPE_CHECKING:
    from qat.core.pipeline import Pipeline
    from qat.model.hardware_model import PydLogicalHardwareModel
    from qat.purr.compiler.hardware_models import LegacyHardwareModel
    from qat.model.loaders.base import BaseModelLoader

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
    requires_model,
)

ImportPipeline = Annotated[ImportString, AfterValidator(is_pipeline_instance)]
ImportPipelineFactory = Annotated[ImportString, AfterValidator(is_pipeline_factory)]
ImportHardwareLoader = Annotated[ImportString, AfterValidator(is_hardwareloader)]

ImportFrontend = Annotated[ImportString, AfterValidator(is_frontend)]
ImportMiddleend = Annotated[ImportString, AfterValidator(is_middleend)]
ImportBackend = Annotated[ImportString, AfterValidator(is_backend)]
ImportEngine = Annotated[ImportString, AfterValidator(is_engine)]
ImportRuntime = Annotated[ImportString, AfterValidator(is_runtime)]

ImportPassManagerFactory = Annotated[ImportString, AfterValidator(is_passmanager_factory)]


class PipelineInstanceDescription(BaseModel):
    name: str
    pipeline: ImportPipeline
    default: bool = False

    def construct(self) -> "Pipeline":
        """Returns the requested Pipeline instance"""
        return self.pipeline.model_copy(update={"name": self.name})


class HardwareLoaderDescription(BaseModel):
    name: str
    config: dict = {}
    type: ImportHardwareLoader

    def construct(self) -> "BaseModelLoader":
        """Returns the described Hardware Loader"""
        return self.type(**self.config)


T = TypeVar("T")


class ClassDescription(BaseModel, Generic[T]):
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

EngineDescription = (
    ImportEngine | Annotated[ClassDescription[ImportEngine], ToPartialValidator]
)


PassManagerFactoryDescription = (
    ImportPassManagerFactory
    | Annotated[ClassDescription[ImportPassManagerFactory], ToPartialValidator]
)


class PipelineFactoryDescription(BaseModel):
    name: str
    hardware_loader: str | None = None
    pipeline: ImportPipelineFactory
    engine: EngineDescription | None = None
    config: dict = {}
    default: bool = False

    def construct(
        self, model: "PydLogicalHardwareModel | LegacyHardwareModel"
    ) -> "Pipeline":
        """Constructs and returns a Pipeline from its description

        :param model: The instantiated hardware model, this is required to be the model provided by the associated hardware_loader
        :return: The constructed pipeline
        """
        kw_args = {**self.config}

        if self.engine is not None:
            if requires_model(self.engine):
                engine = self.engine(model=model)
            else:
                engine = self.engine()

            kw_args["engine"] = engine

        return self.pipeline(model=model, name=self.name, **kw_args)


class PipelineClassDescription(BaseModel):
    name: str
    hardware_loader: str | None = None
    frontend: FrontendDescription = "qat.frontend.DefaultFrontend"
    middleend: MiddleendDescription = "qat.middleend.DefaultMiddleend"
    backend: BackendDescription = "qat.backend.DefaultBackend"
    runtime: RuntimeDescription = "qat.runtime.DefaultRuntime"
    engine: EngineDescription = "qat.engines.DefaultEngine"
    results_pipeline: PassManagerFactoryDescription | None = None
    config: dict = {}
    default: bool = False
    model_config = ConfigDict(validate_default=True)

    def construct(
        self, model: "PydLogicalHardwareModel | LegacyHardwareModel"
    ) -> "Pipeline":
        """Constructs and returns a Pipeline from its description

        :param model: The instantiated hardware model, this is required to be the model provided by the associated hardware_loader
        :return: The constructed pipeline
        """
        from qat.core.pipeline import Pipeline

        frontend = self.frontend(model=model) if self.frontend else None
        middleend = self.middleend(model=model) if self.middleend else None
        backend = self.backend(model=model) if self.backend else None

        if self.results_pipeline is None:
            results_pipeline = None
        elif requires_model(self.results_pipeline):
            results_pipeline = self.results_pipeline(model=model)
        else:
            results_pipeline = self.results_pipeline()

        if self.engine is None:
            engine = None
        elif requires_model(self.engine):
            engine = self.engine(model=model)
        else:
            engine = self.engine()

        if self.runtime is None:
            runtime = None
        elif requires_model(self.runtime):
            runtime = self.runtime(
                engine=engine, model=model, results_pipeline=results_pipeline
            )
        else:
            runtime = self.runtime(engine=engine, results_pipeline=results_pipeline)

        return Pipeline(
            name=self.name,
            model=model,
            frontend=frontend,
            middleend=middleend,
            backend=backend,
            runtime=runtime,
            **self.config,
        )
