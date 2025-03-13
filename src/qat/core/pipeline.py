# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from pydantic import BaseModel, ValidationInfo, field_validator

from qat.backend.base import BaseBackend
from qat.core.config import PipelineImportDescription
from qat.frontend import BaseFrontend
from qat.middleend.middleends import BaseMiddleend
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.runtimes.base import BaseRuntime


class Pipeline(BaseModel, arbitrary_types_allowed=True, frozen=True):
    """
    Pipeline that compiles high-level language specific, but target-agnostic, input (QASM, QIR, ...)
    to target-specific instructions that are executed on our hardware.
    :param frontend: Compiles a high-level language-specific, but target-agnostic,
                    input :class:`QatInput` to a target-agnostic intermediate representation (IR)
                    :class:`QatIR`.
    :param middleend: Takes an intermediate representation (IR) :class:`QatIR` and alters it based
                    on optimisation and/or validation passes within this pipeline.
    :param backend: Converts an intermediate representation (IR) to code for a given target machine.
    """

    name: str
    frontend: BaseFrontend
    middleend: BaseMiddleend
    backend: BaseBackend
    runtime: BaseRuntime
    model: QuantumHardwareModel

    @field_validator("model", mode="before")
    @classmethod
    def consistent_model(cls, model: QuantumHardwareModel, info: ValidationInfo):
        """Validates that the hardware model supplied to the Pipeline matches the hardware model embedded in other fields.

        Currently, this does not check hardware embedded in the runtime.
        """
        for end_name in ["frontend", "middleend", "backend"]:
            end = info.data[end_name]
            if end.model not in {model, None}:
                raise ValueError(f"{model} hardware does not match supplied hardware")

        return model

    @classmethod
    def from_description(cls, desc: PipelineImportDescription):
        return desc.pipeline.model_copy(update={"name": desc.name})


class PipelineSet:
    def __init__(self, pipelines: list[Pipeline] = []):
        self._pipelines = {}

        for P in pipelines:
            self.add(P)

    @classmethod
    def from_descriptions(cls, pipeline_descriptions):
        default = next(
            (Pdesc.name for Pdesc in pipeline_descriptions if Pdesc.default), None
        )
        pipelinesset = cls(
            [Pipeline.from_description(Pdesc) for Pdesc in pipeline_descriptions]
        )
        if default:
            pipelinesset.set_default(default)

        return pipelinesset

    def set_default(self, pipeline: Pipeline | str):
        if isinstance(pipeline, Pipeline):
            name = pipeline.name
            if not name in self._pipelines:
                raise Exception(f"Add pipeline using add_pipeline before setting default")
        else:
            name = pipeline
            if not name in self._pipelines:
                raise Exception(f"Cannot set default pipeline to unknown pipeline {name}")

        self._default_pipeline = name

    @property
    def default(self) -> str:
        """Returns the name of the current default pipeline"""
        return self._default_pipeline

    def add(self, pipeline: Pipeline, default=False):
        """Adds a pipeline for subsequent use for compilation and execution

        :param pipeline: A pipeline instance to add, indexed by pipeline.name
        :type pipeline: Pipeline
        :param default: Set the added pipeline as the default, defaults to False
        :type default: bool, optional
        """
        self._pipelines[pipeline.name] = pipeline
        if default:
            self.set_default(pipeline)

    def remove(self, pipeline: Pipeline | str):
        """Remove a pipeline

        :param pipeline: The name of a pipeline or a pipeline instance to remove
        :type pipeline: Pipeline | str
        """
        name = pipeline.name if isinstance(pipeline, Pipeline) else pipeline

        if not name in self._pipelines:
            raise Exception(f"Pipeline {pipeline.name} not found")

        if isinstance(pipeline, Pipeline):
            if not pipeline is self._pipelines[name]:
                raise Exception(
                    f"Pipeline {pipeline.name} is not the same as stored pipeline with the same name"
                )

        if self._default_pipeline == name:
            self._default_pipeline = None

        del self._pipelines[name]

    def get(self, pipeline: Pipeline | str):
        """Gets a stored pipeline by name (str) or passes through a pipeline instance

        :param pipeline: A pipeline instance or the string name of a stored pipeline
        :type pipeline: Pipeline | str
        """

        if isinstance(pipeline, str):
            if pipeline == "default":
                if self._default_pipeline is None:
                    raise Exception(f"No Default Pipeline Set")
                pipeline = self._default_pipeline
            elif not pipeline in self._pipelines:
                raise Exception(f"Pipeline {pipeline} not found")

            pipeline = self._pipelines[pipeline]

        return pipeline

    def list(self) -> list[str]:
        """Returns a list of available pipeline names"""
        return list(self._pipelines.keys())

    def __repr__(self):
        return str(self.list())
