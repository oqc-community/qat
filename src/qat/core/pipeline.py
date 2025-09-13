# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.core.config.descriptions import (
    CompilePipelineDescription,
    EngineDescription,
    ExecutePipelineDescription,
    HardwareLoaderDescription,
    PipelineClassDescription,
    PipelineFactoryDescription,
    PipelineInstanceDescription,
    UpdateablePipelineDescription,
)
from qat.engines import NativeEngine
from qat.engines.model import requires_hardware_model
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.loaders.base import BaseModelLoader
from qat.model.loaders.cache import CacheAccessLoader
from qat.model.loaders.update import ModelUpdateChecker
from qat.pipelines.base import AbstractPipeline
from qat.pipelines.cache import CompilePipelineCache, ExecutePipelineCache
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
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

    @property
    def models_up_to_date(self) -> bool:
        """Checks if all models are up-to-date, for loaders that implement
        :class:`ModelUpdateChecker`.
        """
        for key, loader in self._loaders.items():
            model = self._loaded_models.get(key, None)
            if model is None:
                return False
            if isinstance(loader, ModelUpdateChecker) and not loader.is_up_to_date(model):
                return False
        return True

    @classmethod
    def from_descriptions(
        cls,
        hardware_loader_descriptions: list[HardwareLoaderDescription],
    ):
        return cls({hld.name: hld.construct() for hld in hardware_loader_descriptions})


class EngineSet:
    """Stores a set of engines, allowing identification by name.

    Also stores corresponding loaders which can be used to update the models within engines
    for engines that have a :class:`RequiresHardwareModelMixin` mixin.
    """

    def __init__(
        self,
        engines: dict[str, NativeEngine] = {},
        loaders: dict[str, BaseModelLoader] = {},
    ):
        self._engines: dict[str, NativeEngine] = dict(**engines)
        self._loaders: dict[str, BaseModelLoader] = dict(**loaders)

    def __getitem__(self, name: str) -> NativeEngine:
        """Allows indexing to get an engine by name."""
        return self.get(name)

    def get(self, name: str, default=None) -> NativeEngine:
        """Returns an engine by name, or a default if not found."""
        return self._engines.get(name, default)

    def reload_model(self, name: str):
        """Reloads the model for an engine with the given identifier."""
        if name not in self._engines:
            raise ValueError(f"Engine {name} not found")

        engine = self._engines[name]
        # TODO: Change to just check against RequiresHardwareModelMixin (COMPILER-662)
        if not requires_hardware_model(engine):
            return

        loader = self._loaders.get(name, None)
        if loader is None:
            return

        model = loader.load()
        engine.model = model

    def reload_all_models(self):
        """Reloads all models for engines that require a hardware model."""
        for name in self._engines:
            self.reload_model(name)

    @classmethod
    def from_descriptions(
        cls,
        engine_descriptions: list[EngineDescription],
        available_hardware: HardwareLoaders,
    ) -> "EngineSet":
        """Creates an :class:`EngineSet` from a list of engine descriptions."""

        engines = {}
        loaders = {}
        for desc in engine_descriptions:
            if desc.hardware_loader:
                loader = CacheAccessLoader(available_hardware, desc.hardware_loader)
                loaders[desc.name] = loader
                engines[desc.name] = desc.construct(model=loader.load())
            else:
                engines[desc.name] = desc.construct(model=None)
        return cls(engines, loaders)


class PipelineSet:
    def __init__(self, pipelines: list[AbstractPipeline] = []):
        self._pipelines = {}
        self._default_pipeline = None

        for P in pipelines:
            self.add(P)

    @classmethod
    def from_descriptions(
        cls,
        pipeline_descriptions: list[
            PipelineInstanceDescription
            | PipelineFactoryDescription
            | UpdateablePipelineDescription
            | ExecutePipelineDescription
            | CompilePipelineDescription
            | PipelineClassDescription
        ],
        available_hardware: HardwareLoaders,
        available_engines: EngineSet,
    ):
        pipeline_descriptions = (
            [] if pipeline_descriptions is None else pipeline_descriptions
        )
        default = next(
            (Pdesc.name for Pdesc in pipeline_descriptions if Pdesc.default), None
        )

        pipes = []
        for Pdesc in pipeline_descriptions:
            attrs = dict()
            if hasattr(Pdesc, "hardware_loader"):
                attrs["loader"] = CacheAccessLoader(
                    available_hardware, Pdesc.hardware_loader
                )
            if hasattr(Pdesc, "engine"):
                attrs["engine"] = available_engines[Pdesc.engine]
            P = Pdesc.construct(**attrs)
            pipes.append(P)

        pipelinesset = cls(pipes)
        if default:
            pipelinesset.set_default(default)
        else:
            pipelinesset._default_pipeline = None

        return pipelinesset

    def set_default(self, pipeline: AbstractPipeline | str):
        if isinstance(pipeline, AbstractPipeline):
            name = pipeline.name
            if name not in self._pipelines:
                raise Exception("Add pipeline using add before setting default")
        else:
            name = pipeline
            if name not in self._pipelines:
                raise Exception(f"Cannot set default pipeline to unknown pipeline {name}")

        self._default_pipeline = name

    @property
    def default(self) -> str:
        """Returns the name of the current default pipeline"""
        return self._default_pipeline

    def add(self, pipeline: AbstractPipeline, default: bool = False):
        """Adds a pipeline for subsequent use for compilation and execution

        :param pipeline: A pipeline instance to add, indexed by pipeline.name
        :param default: Set the added pipeline as the default, defaults to False
        """
        self._pipelines[pipeline.name] = pipeline
        if default:
            self.set_default(pipeline)

    def remove(self, pipeline: AbstractPipeline | str):
        """Remove a pipeline

        :param pipeline: The name of a pipeline or a pipeline instance to remove
        """
        name = pipeline.name if isinstance(pipeline, AbstractPipeline) else pipeline

        if name not in self._pipelines:
            raise ValueError(f"Pipeline {name} not found")

        if isinstance(pipeline, AbstractPipeline):
            if pipeline is not self._pipelines[name]:
                raise ValueError(
                    f"Pipeline {name} is not the same as stored pipeline with the same name"
                )

        if self._default_pipeline == name:
            self._default_pipeline = None

        del self._pipelines[name]

    def get(self, pipeline: AbstractPipeline | str) -> AbstractPipeline:
        """Gets a stored pipeline by name (str) or passes through a pipeline instance

        :param pipeline: A pipeline instance or the string name of a stored pipeline
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

    def __contains__(self, pipeline: str) -> bool:
        """Checks if a pipeline is in the set by name."""

        if pipeline == "default":
            return self._default_pipeline is not None
        return pipeline in self._pipelines


class PipelineManager:
    """A manager for pipelines, storing compilation and execution pipelines.

    Currently, pipelines are catergorised by their type (or the type produced by
    factories):

    * :class:`CompilePipeline` - pipelines for compilation
    * :class:`ExecutePipeline` - pipelines for execution
    * :class:`Pipeline` - pipelines that handle both compilation and execution.

    As such, when making calls to pipelines by name, we need to handle the logic for
    retreiving the correct pipeline given the task.
    """

    def __init__(
        self,
        compile_pipelines: PipelineSet,
        execute_pipelines: PipelineSet,
        full_pipelines: PipelineSet,
    ):
        self._full_pipelines = full_pipelines
        self.compile_pipelines = compile_pipelines
        self.execute_pipelines = execute_pipelines

        for pipeline in self._full_pipelines.list():
            P = self._full_pipelines.get(pipeline)
            self._add_full_pipeline_to_compile_and_execute(
                P, default=P.name == self._full_pipelines.default
            )

    def get_compile_pipeline(self, pipeline: str | AbstractPipeline) -> AbstractPipeline:
        """Returns a compile pipeline by name or instance."""
        return self.compile_pipelines.get(pipeline)

    def get_execute_pipeline(self, pipeline: str | AbstractPipeline) -> AbstractPipeline:
        """Returns an execute pipeline by name or instance."""
        return self.execute_pipelines.get(pipeline)

    @property
    def list_compile_pipelines(self) -> list[str]:
        """Returns a list of compile pipeline names."""
        return self.compile_pipelines.list()

    @property
    def list_execute_pipelines(self) -> list[str]:
        """Returns a list of execute pipeline names."""
        return self.execute_pipelines.list()

    @property
    def default_compile_pipeline(self) -> str:
        """Returns the name of the default compile pipeline."""
        return self.compile_pipelines.default

    @property
    def default_execute_pipeline(self) -> str:
        """Returns the name of the default execute pipeline."""
        return self.execute_pipelines.default

    def add(self, pipeline: AbstractPipeline, default: bool = False):
        """Adds a pipeline to the manager.

        :param pipeline: The pipeline to add.
        :param default: If True, sets this pipeline as the default.
        """
        name = pipeline.name

        if pipeline.is_subtype_of(Pipeline):
            if name in self.list_compile_pipelines or name in self.list_execute_pipelines:
                raise ValueError(
                    f"Pipeline {name} already exists in the manager as a compile or "
                    "execute pipeline."
                )
            self._full_pipelines.add(pipeline)
            self._add_full_pipeline_to_compile_and_execute(pipeline, default)

        elif pipeline.is_subtype_of(CompilePipeline):
            if name in self.list_compile_pipelines:
                raise ValueError(
                    f"Pipeline {name} already exists in the manager as a compile pipeline."
                )
            self.compile_pipelines.add(pipeline, default)

        elif pipeline.is_subtype_of(ExecutePipeline):
            if name in self.list_execute_pipelines:
                raise ValueError(
                    f"Pipeline {name} already exists in the manager as an execute pipeline."
                )
            self.execute_pipelines.add(pipeline, default)

        else:
            raise ValueError(
                f"Pipeline {pipeline.name} is not a CompilePipeline or ExecutePipeline."
            )

    def remove(
        self,
        pipeline: AbstractPipeline | str,
        compile: bool | None = None,
        execute: bool | None = None,
    ):
        """Removes a pipeline from the manager.

        :param pipeline: The pipeline to remove, either by name or instance.
        :param compile: Determines if the pipeline should be removed from compile pipelines.
            If None, the manager will decide given the pipeline type.
        :param execute: Determines if the pipeline should be removed from execute pipelines.
            If None, the manager will decide given the pipeline type.
        """
        pipeline = pipeline.name if isinstance(pipeline, AbstractPipeline) else pipeline

        if compile is False and execute is False:
            return

        full = pipeline in self._full_pipelines
        if full and (compile is False or execute is False):
            raise ValueError(
                f"Cannot remove the full pipeline {pipeline} from just compile or execute "
                "pipelines. You must remove it from both."
            )

        compile = compile if compile is not None else pipeline in self.compile_pipelines
        if compile:
            self.compile_pipelines.remove(pipeline)

        execute = execute if execute is not None else pipeline in self.execute_pipelines
        if execute:
            self.execute_pipelines.remove(pipeline)

        if full:
            self._full_pipelines.remove(pipeline)

    def set_default(
        self,
        pipeline: AbstractPipeline | str,
        compile: bool | None = None,
        execute: bool | None = None,
    ):
        """Sets the default pipeline for compilation or execution.

        If the pipeline is a full pipeline, then it must be set as the default for both
        compilation and execution. Else, if there exists a compile or execute pipeline with
        the same name, it can be specified for which the default should be set for. By
        default, it will be both.
        """
        if compile is False and execute is False:
            return

        if isinstance(pipeline, AbstractPipeline):
            pipeline = pipeline.name

        if pipeline in self._full_pipelines:
            if compile is False or execute is False:
                raise ValueError(
                    f"Cannot set the full pipeline {pipeline} as default for just "
                    "compile or execute pipelines. It must be set for both."
                )
            compile = True if compile is None else compile
            execute = True if execute is None else execute

        if pipeline in self.compile_pipelines:
            compile = True if compile is None else compile
        elif compile is True:
            raise ValueError(
                f"Cannot set the pipeline {pipeline} as default for compile pipelines, as "
                "it cannot be found."
            )

        if pipeline in self.execute_pipelines:
            execute = True if execute is None else execute
        elif execute is True:
            raise ValueError(
                f"Cannot set the pipeline {pipeline} as default for execute pipelines, as "
                "it cannot be found."
            )

        if compile:
            self.compile_pipelines.set_default(pipeline)
        if execute:
            self.execute_pipelines.set_default(pipeline)

    def reload_all_models(self):
        """Reloads all models in all pipelines that have a loader."""
        self._full_pipelines.reload_all_models()
        self.compile_pipelines.reload_all_models()
        self.execute_pipelines.reload_all_models()

    @classmethod
    def from_descriptions(
        cls,
        compile_pipelines: list[
            PipelineInstanceDescription
            | PipelineFactoryDescription
            | UpdateablePipelineDescription
            | CompilePipelineDescription
        ],
        execute_pipelines: list[
            PipelineInstanceDescription
            | PipelineFactoryDescription
            | UpdateablePipelineDescription
            | ExecutePipelineDescription
        ],
        full_pipelines: list[
            PipelineInstanceDescription
            | PipelineFactoryDescription
            | UpdateablePipelineDescription
            | PipelineClassDescription
        ],
        available_hardware: HardwareLoaders,
        available_engines: EngineSet,
    ) -> "PipelineManager":
        """Creates a PipelineManager from a list of pipeline descriptions."""
        compile_set = PipelineSet.from_descriptions(
            compile_pipelines, available_hardware, available_engines
        )
        execute_set = PipelineSet.from_descriptions(
            execute_pipelines, available_hardware, available_engines
        )
        full_set = PipelineSet.from_descriptions(
            full_pipelines, available_hardware, available_engines
        )

        return cls(compile_set, execute_set, full_set)

    def _add_full_pipeline_to_compile_and_execute(self, P: Pipeline, default: bool = False):
        """Adds a full pipeline to the manager, updating both compile and execute pipelines."""
        self.compile_pipelines.add(CompilePipelineCache(P), default)
        self.execute_pipelines.add(ExecutePipelineCache(P), default)
