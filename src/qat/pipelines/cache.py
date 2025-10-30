# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.base import BaseBackend
from qat.engines import NativeEngine
from qat.frontend import BaseFrontend
from qat.middleend.base import BaseMiddleend
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.target_data import TargetData
from qat.pipelines.base import AbstractPipeline
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.base import BaseRuntime


class CompilePipelineCache(CompilePipeline):
    """Accesses a cached full :class:`Pipeline`, allowing it to be recognised and treated as
    only a :class:`CompilePipeline`."""

    def __init__(self, pipeline: AbstractPipeline):
        if not pipeline.is_subtype_of(Pipeline):
            raise ValueError(
                f"Pipeline {pipeline.name} must be a full Pipeline to use the "
                "CompilePipelineCache wrapper."
            )
        self._pipeline = pipeline

    @property
    def name(self) -> str:
        return self._pipeline.name

    @property
    def model(self) -> QuantumHardwareModel | PhysicalHardwareModel:
        return self._pipeline.model

    @property
    def target_data(self) -> TargetData:
        return self._pipeline.target_data

    @property
    def frontend(self) -> BaseFrontend:
        return self._pipeline.frontend

    @property
    def middleend(self) -> BaseMiddleend:
        return self._pipeline.middleend

    @property
    def backend(self) -> BaseBackend:
        return self._pipeline.backend

    def copy_with_name(self, name):
        raise NotImplementedError(
            "CompilePipelineCache does not support copy_with_name. Use the original "
            "pipeline instead."
        )


class ExecutePipelineCache(ExecutePipeline):
    """Accesses a cached full :class:`Pipeline`, allowing it to be recognised and treated as
    only an :class:`ExecutePipeline`."""

    def __init__(self, pipeline: AbstractPipeline):
        if not pipeline.is_subtype_of(Pipeline):
            raise ValueError(
                f"Pipeline {pipeline.name} must be a full Pipeline to use the "
                "ExecutePipelineCache wrapper."
            )
        self._pipeline = pipeline

    @property
    def name(self) -> str:
        return self._pipeline.name

    @property
    def model(self) -> QuantumHardwareModel | PhysicalHardwareModel:
        return self._pipeline.model

    @property
    def target_data(self) -> TargetData:
        return self._pipeline.target_data

    @property
    def runtime(self) -> BaseRuntime:
        return self._pipeline.runtime

    @property
    def engine(self) -> NativeEngine:
        return self._pipeline.engine

    def copy_with_name(self, name):
        raise NotImplementedError(
            "ExecutePipelineCache does not support copy_with_name. Use the original "
            "pipeline instead."
        )
