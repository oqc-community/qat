# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import inspect
from pathlib import Path

import piny
from pydantic import ImportString, ValidationInfo, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

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
from qat.extensions import QatExtension
from qat.purr.qatconfig import QatConfig
from qat.utils.piny import VeryStrictMatcher


class QatSessionConfig(QatConfig):
    """
    Configuration for QAT() sessions extending qat.purr.qatconfig configuration.

    >>> import os
    >>> os.environ["QAT_MAX_REPEATS_LIMIT"] = "654321"
    >>> QatSessionConfig() # doctest: +ELLIPSIS
    QatSessionConfig(MAX_REPEATS_LIMIT=654321, ...)
    >>> QatSessionConfig(MAX_REPEATS_LIMIT=123) # doctest: +ELLIPSIS
    QatSessionConfig(MAX_REPEATS_LIMIT=123, ...)
    >>> del os.environ["QAT_MAX_REPEATS_LIMIT"]
    >>> qatconfig = QatSessionConfig()
    >>> qatconfig.MAX_REPEATS_LIMIT = 16000
    >>> qatconfig # doctest: +ELLIPSIS
    QatSessionConfig(MAX_REPEATS_LIMIT=16000, ...)

    >>> QatSessionConfig(MAX_REPEATS_LIMIT=100.5) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    pydantic_core._pydantic_core.ValidationError
    ...
    Input should be a valid integer, got a number with a fractional part
    """

    model_config = SettingsConfigDict(
        env_prefix="QAT_",
        env_nested_delimiter="_",
        validate_assignment=True,
        yaml_file="qatconfig.yaml",
    )

    EXTENSIONS: list[ImportString] = []
    """ QAT Extensions to initialise on start-up"""

    HARDWARE: list[HardwareLoaderDescription] = []
    """ QAT Hardware Models to load on start-up """

    ENGINES: list[EngineDescription] = []
    """ QAT Engines to add on start-up """

    PIPELINES: (
        list[
            PipelineInstanceDescription
            | UpdateablePipelineDescription
            | PipelineFactoryDescription
            | PipelineClassDescription
        ]
        | None
    ) = None
    """ QAT Pipelines for compilation and execution to add on start-up, None adds default
    pipelines"""

    COMPILE: (
        list[
            PipelineInstanceDescription
            | UpdateablePipelineDescription
            | PipelineFactoryDescription
            | CompilePipelineDescription
        ]
        | None
    ) = None
    """ QAT Pipelines for compilation to add on start-up, None adds default pipelines"""

    EXECUTE: (
        list[
            PipelineInstanceDescription
            | UpdateablePipelineDescription
            | PipelineFactoryDescription
            | ExecutePipelineDescription
        ]
        | None
    ) = None
    """ QAT Pipelines for execution to add on start-up, None adds default pipelines"""

    @field_validator("EXTENSIONS")
    def load_extensions(values):
        for value in values:
            if type(value) is type and issubclass(value, QatExtension):
                value.load()
            else:
                name = inspect.getmodule(value).__name__
                raise ValueError(
                    f"extension '{name}' must be a valid QatExtension class, type is "
                    f"actually {type(value)}"
                )
        return values

    @model_validator(mode="after")
    def one_default(cls, values: dict) -> dict:
        """Ensures that exactly one PIPELINE is marked as default, or alternatively, one of
        COMPILE and EXECUTE pipelines is marked as default."""
        compile_pipelines = values.COMPILE
        execute_pipelines = values.EXECUTE
        full_pipelines = values.PIPELINES

        if not (compile_pipelines or execute_pipelines):
            if full_pipelines and len(full_pipelines) == 1:
                full_pipelines[0].default = True
                return values

        num_full_defaults = (
            sum(desc.default for desc in full_pipelines)
            if full_pipelines is not None
            else 0
        )
        if num_full_defaults > 1:
            raise ValueError(
                "Expected at most one default pipeline in PIPELINES, found "
                f"{num_full_defaults}."
            )

        for pipelines in [compile_pipelines, execute_pipelines]:
            if not pipelines and not full_pipelines:
                continue

            num_pipelines = len(pipelines) if pipelines is not None else 0
            num_pipeline_defaults = (
                sum(desc.default for desc in pipelines) if num_pipelines != 0 else 0
            )

            if num_pipelines == 1 and not full_pipelines:
                pipelines[0].default = True
                continue

            if (total_defaults := num_pipeline_defaults + num_full_defaults) != 1:
                raise ValueError(
                    f"Expected exactly one default COMPILE and EXECUTE pipelines, found "
                    f"multiple (including PIPELINES): {total_defaults}."
                )

        return values

    @field_validator("HARDWARE", "ENGINES")
    @classmethod
    def no_duplicate_names(cls, val, info: ValidationInfo):
        """Checks that there are no duplicate names in the list of descriptions."""
        if val is None or len(val) == 0:
            return val

        names = set()
        for desc in val:
            if desc.name in names:
                raise ValueError(f"Duplicate name {desc.name} found in {info.field_name}")
            names.add(desc.name)
        return val

    @field_validator("COMPILE", "EXECUTE")
    @classmethod
    def no_duplicate_pipeline_names(cls, val, info: ValidationInfo):
        """For each of the PIPELINES, COMPILE, and EXECUTE fields, this checks that there
        are no duplicate names in the list of descriptions. Also checks there are no shared
        names between PIPELINES and COMPILE/EXECUTE."""
        from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline

        pipeline_ty = CompilePipeline if info.field_name == "COMPILE" else ExecutePipeline

        pipelines = [] if val is None else val
        full_pipelines = info.data.get("PIPELINES", [])
        full_pipelines = [] if full_pipelines is None else full_pipelines
        full_pipelines = [
            desc for desc in full_pipelines if desc.is_subtype_of(pipeline_ty)
        ]

        names = set()
        for desc in full_pipelines + pipelines:
            if desc.name in names:
                raise ValueError(
                    f"Duplicate name {desc.name} found in {info.field_name} pipelines."
                )
            names.add(desc.name)
        return val

    @field_validator("PIPELINES", "ENGINES", "COMPILE", "EXECUTE")
    @classmethod
    def matching_hardware_loaders(cls, pipelines, info: ValidationInfo):
        if pipelines is None:
            return pipelines
        if len(pipelines) == 0:
            return pipelines

        hardware_loader_names = {hld.name for hld in info.data.get("HARDWARE", [])}
        for Pdesc in pipelines:
            expected_loader = getattr(Pdesc, "hardware_loader", None)
            if expected_loader is not None and expected_loader not in hardware_loader_names:
                raise ValueError(f"Hardware Loader {expected_loader} not defined")

        return pipelines

    @field_validator("PIPELINES", "EXECUTE")
    @classmethod
    def matching_engines(cls, pipelines, info: ValidationInfo):
        if pipelines is None:
            return pipelines
        if len(pipelines) == 0:
            return pipelines

        engine_names = {desc.name for desc in info.data.get("ENGINES", [])}
        engine_loaders = {
            desc.name: desc.hardware_loader for desc in info.data.get("ENGINES", [])
        }

        for Pdesc in pipelines:
            engine = getattr(Pdesc, "engine", None)
            if engine is None:
                continue

            if engine not in engine_names:
                raise ValueError(
                    f"Pipeline {Pdesc.name} requires engine {engine}, but it is not "
                    "defined in ENGINES"
                )

            expected_loader = getattr(Pdesc, "hardware_loader", None)
            engine_loader = engine_loaders[engine]
            if (
                engine_loader is not None
                and expected_loader is not None
                and engine_loader != expected_loader
            ):
                raise ValueError(
                    f"Pipeline {Pdesc.name} has hardware_loader {expected_loader}, but "
                    f"the engine {engine} has a different hardware_loader {engine_loader}."
                )

        return pipelines

    @classmethod
    def from_yaml(cls, path: str | Path):
        blob = piny.YamlLoader(path=str(path), matcher=VeryStrictMatcher).load()
        return cls(**blob)
