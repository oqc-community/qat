from pydantic import BaseModel, ImportString, BeforeValidator, field_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
    PydanticBaseSettingsSource,
)
from typing import Annotated, List, Literal, Type, Tuple, Optional

import yaml
from enum import Enum


class HardwareTypeEnum(str, Enum):
    rtcs = "rtcs"
    echo = "echo"


class HardwareDescription(BaseModel):
    qubit_count: Optional[int] = None  # This there should be a class for each type
    hardware_type: HardwareTypeEnum


class PipelineDescription(BaseModel):
    name: str
    compile: ImportString
    execute: ImportString
    hardware: HardwareDescription
    default: bool = False


class PipelinesConfig(BaseSettings):
    model_config = SettingsConfigDict(yaml_file="pipeline.config.yaml")
    pipelines: List[PipelineDescription] = []

    @field_validator("pipelines")
    @classmethod
    def one_default(cls, v):
        if len(v) == 0:
            return v
        num_defaults = sum(pipe.default for pipe in v)
        if not num_defaults == 1:
            raise ValueError(
                f"Exactly one pipeline must have default: true (found {num_defaults})"
            )
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)
