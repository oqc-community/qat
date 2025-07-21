# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.core.config.descriptions import (
    HardwareLoaderDescription,
    PipelineFactoryDescription,
    PipelineInstanceDescription,
    UpdateablePipelineDescription,
)
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.loaders.base import BaseModelLoader
from qat.model.loaders.cache import CacheAccessLoader
from qat.pipelines.base import AbstractPipeline
from qat.pipelines.updateable import UpdateablePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class HardwareLoaders:
    def __init__(self, hardware_loaders: dict[str, BaseModelLoader] = {}):
        self._loaders = dict(**hardware_loaders)
        self._loaded_models = {}

    def __getitem__(self, model_name: str) -> QuantumHardwareModel | PhysicalHardwareModel:
        """Allows indexing to get a hardware model by name."""
        return self.load(model_name, allow_cache=True)

    def get_loader(self, loader_name: str, default=None) -> BaseModelLoader:
        """Returns a hardware model loader by name."""
        loader = self._loaders.get(loader_name)
        if loader is None:
            return default
        return loader

    def load(self, loader_name: str, default=None, allow_cache=True):
        """Loads a hardware model, using the internal cache unless `allow_cache=False`."""
        model = self._loaded_models.get(loader_name) if allow_cache else None
        if model is None:
            loader = self._loaders.get(loader_name)
            if loader is None:
                return default
            model = loader.load()
            self._loaded_models[loader_name] = model

        return model

    def clear_cache(self):
        self._loaded_models = {}

    def reload_model(
        self, loader_name: str
    ) -> QuantumHardwareModel | PhysicalHardwareModel:
        """Reloads a hardware model from its loader, updating the model in the cache."""
        if loader_name not in self._loaders:
            raise Exception(f"Hardware Model Loader {loader_name} not found")
        return self.load(loader_name, allow_cache=False)

    def reload_all_models(self):
        """Reloads all hardware models from their respective loaders, updating the cache."""
        for loader_name in self._loaders:
            self.reload_model(loader_name)

    @classmethod
    def from_descriptions(
        cls,
        hardware_loader_descriptions: list[HardwareLoaderDescription],
    ):
        return cls({hld.name: hld.construct() for hld in hardware_loader_descriptions})


class PipelineSet:
    def __init__(self, pipelines: list[AbstractPipeline] = []):
        self._pipelines = {}

        for P in pipelines:
            self.add(P)

    @classmethod
    def from_descriptions(
        cls,
        pipeline_descriptions: list[
            PipelineInstanceDescription
            | PipelineFactoryDescription
            | UpdateablePipelineDescription
        ],
        available_hardware: HardwareLoaders,
    ):
        default = next(
            (Pdesc.name for Pdesc in pipeline_descriptions if Pdesc.default), None
        )

        pipes = []
        for Pdesc in pipeline_descriptions:
            if hasattr(Pdesc, "hardware_loader"):
                loader = CacheAccessLoader(available_hardware, Pdesc.hardware_loader)
                if loader is None:
                    raise Exception(
                        f"Hardware Model Loader {Pdesc.hardware_loader} not found"
                    )

                P = Pdesc.construct(loader)
            else:
                P = Pdesc.construct()

            pipes.append(P)

        pipelinesset = cls(pipes)
        if default:
            pipelinesset.set_default(default)

        return pipelinesset

    def set_default(self, pipeline: AbstractPipeline | str):
        if isinstance(pipeline, AbstractPipeline):
            name = pipeline.name
            if name not in self._pipelines:
                raise Exception("Add pipeline using add_pipeline before setting default")
        else:
            name = pipeline
            if name not in self._pipelines:
                raise Exception(f"Cannot set default pipeline to unknown pipeline {name}")

        self._default_pipeline = name

    @property
    def default(self) -> str:
        """Returns the name of the current default pipeline"""
        return self._default_pipeline

    def add(self, pipeline: AbstractPipeline, default=False):
        """Adds a pipeline for subsequent use for compilation and execution

        :param pipeline: A pipeline instance to add, indexed by pipeline.name
        :type pipeline: AbstractPipeline
        :param default: Set the added pipeline as the default, defaults to False
        :type default: bool, optional
        """
        self._pipelines[pipeline.name] = pipeline
        if default:
            self.set_default(pipeline)

    def remove(self, pipeline: AbstractPipeline | str):
        """Remove a pipeline

        :param pipeline: The name of a pipeline or a pipeline instance to remove
        :type pipeline: AbstractPipeline | str
        """
        name = pipeline.name if isinstance(pipeline, AbstractPipeline) else pipeline

        if name not in self._pipelines:
            raise Exception(f"Pipeline {pipeline.name} not found")

        if isinstance(pipeline, AbstractPipeline):
            if pipeline is not self._pipelines[name]:
                raise Exception(
                    f"Pipeline {pipeline.name} is not the same as stored pipeline with the same name"
                )

        if self._default_pipeline == name:
            self._default_pipeline = None

        del self._pipelines[name]

    def get(self, pipeline: AbstractPipeline | str) -> AbstractPipeline:
        """Gets a stored pipeline by name (str) or passes through a pipeline instance

        :param pipeline: A pipeline instance or the string name of a stored pipeline
        :type pipeline: AbstractPipeline | str
        """

        if isinstance(pipeline, str):
            if pipeline == "default":
                if self._default_pipeline is None:
                    raise Exception("No Default Pipeline Set")
                pipeline = self._default_pipeline
            elif pipeline not in self._pipelines:
                raise Exception(f"Pipeline {pipeline} not found")

            pipeline = self._pipelines[pipeline]

        return pipeline

    def reload_model(self, pipeline: str):
        """Refreshes a pipeline by updating the models from its cache.

        :param pipeline: The name of the pipeline to refresh.
        """

        pipeline = self._pipelines[pipeline]
        if not (isinstance(pipeline, UpdateablePipeline) and pipeline.has_loader):
            raise Exception(
                f"The pipeline {pipeline} is not an Updateable pipelines equipped with a "
                "loader."
            )
        pipeline.update(reload_model=True)

    def reload_all_models(self):
        """Refreshes all :class:`UpdateablePipeline` instances with hardware model loaders
        by updating the models from their caches."""

        for pipeline in self._pipelines.values():
            if isinstance(pipeline, UpdateablePipeline) and pipeline.has_loader:
                pipeline.update(reload_model=True)

    def list(self) -> list[str]:
        """Returns a list of available pipeline names"""
        return list(self._pipelines.keys())

    def __repr__(self):
        return str(self.list())
