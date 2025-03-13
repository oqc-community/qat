# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import inspect
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ImportString


def is_pipeline_instance(value: type):
    from qat.core.pipeline import Pipeline

    if not isinstance(value, Pipeline):
        raise ValueError(f"{value} is not a valid Pipeline instance")
    return value


def is_hardwareloader(value: type):
    from qat.model.loaders.base import BaseModelLoader

    if not issubclass(value, BaseModelLoader):
        raise ValueError(f"{value} is not a valid Hardware Model Loader class")
    return value


def is_pipeline_builder(value: type):
    from qat.core.pipeline import Pipeline

    if not (callable(value) and inspect.signature(value).return_annotation is Pipeline):
        raise ValueError(f"{value} is not a valid Pipeline builder")

    return value


ImportPipeline = Annotated[ImportString, AfterValidator(is_pipeline_instance)]
ImportPipelineBuilder = Annotated[ImportString, AfterValidator(is_pipeline_builder)]
ImportHardwareLoader = Annotated[ImportString, AfterValidator(is_hardwareloader)]


class PipelineInstanceDescription(BaseModel):
    name: str
    pipeline: ImportPipeline
    default: bool = False

    def construct(self):
        return self.pipeline.model_copy(update={"name": self.name})


class PipelineBuilderDescription(BaseModel):
    name: str
    hardware_loader: str | None = None
    pipeline: ImportPipelineBuilder
    init: dict = {}
    default: bool = False

    def construct(self, model):
        return self.pipeline(model=model, name=self.name, **self.init)


class HardwareLoaderDescription(BaseModel):
    name: str
    init: dict = {}
    loader: ImportHardwareLoader

    def construct(self):
        return self.loader(**self.init)
