import inspect
from pathlib import Path

import piny
from pydantic import ImportString, ValidationInfo, field_validator
from pydantic_settings import SettingsConfigDict

from qat.core.config.descriptions import (
    HardwareLoaderDescription,
    PipelineClassDescription,
    PipelineFactoryDescription,
    PipelineInstanceDescription,
    UpdateablePipelineDescription,
)
from qat.extensions import QatExtension
from qat.purr.qatconfig import QatConfig
from qat.purr.utils.logger import get_default_logger
from qat.utils.piny import VeryStrictMatcher

log = get_default_logger()


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

    PIPELINES: (
        list[
            PipelineInstanceDescription
            | UpdateablePipelineDescription
            | PipelineFactoryDescription
            | PipelineClassDescription
        ]
        | None
    ) = None
    """ QAT Pipelines to add on start-up, None adds default pipelines"""

    @field_validator("EXTENSIONS")
    def load_extensions(values):
        for value in values:
            if type(value) is type and issubclass(value, QatExtension):
                value.load()
            else:
                name = inspect.getmodule(value).__name__
                raise ValueError(
                    f"extension '{name}' must be a valid QatExtension class, type is actually {type(value)}"
                )
        return values

    @field_validator("PIPELINES")
    @classmethod
    def one_default(cls, v):
        if v is None:
            return v
        elif len(v) == 0:
            return v
        elif len(v) == 1:
            v[0].default = True
            return v
        num_defaults = sum(pipe.default for pipe in v)
        if not num_defaults == 1:
            raise ValueError(
                f"Exactly one pipeline must have default: true (found {num_defaults})"
            )
        return v

    @field_validator("PIPELINES")
    @classmethod
    def matching_hardware_loaders(cls, pipelines, info: ValidationInfo):
        if pipelines is None:
            return pipelines
        if len(pipelines) == 0:
            return pipelines

        hardware_loader_names = {hld.name for hld in info.data.get("HARDWARE", [])}
        for Pdesc in pipelines:
            expected_loader = getattr(Pdesc, "hardware_loader", None)
            if expected_loader and expected_loader not in hardware_loader_names:
                raise ValueError(f"Hardware Loader {expected_loader} not defined")

        return pipelines

    @classmethod
    def from_yaml(cls, path: str | Path):
        blob = piny.YamlLoader(path=str(path), matcher=VeryStrictMatcher).load()
        return cls(**blob)
