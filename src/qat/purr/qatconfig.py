# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import inspect
import warnings
from typing import Literal, Optional, Tuple, Type

import yaml
from compiler_config.config import CompilerConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    ValidationInfo,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from qiskit_aer import AerSimulator

from qat.core.config import (
    HardwareLoaderDescription,
    PipelineBuilderDescription,
    PipelineInstanceDescription,
)
from qat.extensions import QatExtension
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class QiskitSimulationConfig(BaseModel):
    """
    The default settings for the Qiskit Simulator, including overridden MPS settings.
    """

    model_config = ConfigDict(validate_assignment=True)
    allowed_methods: type = Literal[AerSimulator().available_methods()]
    METHOD: allowed_methods = "automatic"
    """The simulation method to use."""
    FALLBACK_SEQUENCE: list[allowed_methods] = ["automatic", "matrix_product_state"]
    """If the simulation fails, specify a fallback sequence of methods to call."""
    OPTIONS: dict = {
        "matrix_product_state_max_bond_dimension": 128,
        "matrix_product_state_truncation_threshold": 1e-12,
    }
    """
    Specify the options for a chosen AerSimulator backend. See
    https://docs.quantum.ibm.com/api/qiskit/0.37/qiskit.providers.aer.AerSimulator
    for options you can provide.
    """
    ENABLE_METADATA: bool = False
    """Returns the AerSimulator metadata if enabled."""


class QatSimulationConfig(BaseModel):
    """
    The default settings for QATs simulation backends.
    """

    model_config = ConfigDict(validate_assignment=True)
    QISKIT: QiskitSimulationConfig = QiskitSimulationConfig()


class InstructionValidationConfig(BaseModel):
    """
    The default settings for validation of instructions.
    """

    model_config = ConfigDict(validate_assignment=True)
    NO_MID_CIRCUIT_MEASUREMENT: bool = True
    MAX_INSTRUCTION_LENGTH: bool = True
    ACQUIRE_CHANNEL: bool = True
    PULSE_DURATION_LIMITS: bool = True

    @property
    def DISABLED(self):
        return not (
            self.NO_MID_CIRCUIT_MEASUREMENT
            | self.MAX_INSTRUCTION_LENGTH
            | self.ACQUIRE_CHANNEL
            | self.PULSE_DURATION_LIMITS
        )

    def disable(self):
        self.NO_MID_CIRCUIT_MEASUREMENT = False
        self.MAX_INSTRUCTION_LENGTH = False
        self.ACQUIRE_CHANNEL = False
        self.PULSE_DURATION_LIMITS = False

    @field_validator("PULSE_DURATION_LIMITS")
    def check_disable_pulse_duration_limits(cls, PULSE_DURATION_LIMITS):
        if not PULSE_DURATION_LIMITS:
            log.warning(
                "Disabled check for pulse duration limits, which should ideally only be used for calibration purposes."
            )
        return PULSE_DURATION_LIMITS


class QatConfig(BaseSettings):
    """
    Full settings for a single job. Allows environment variables to be overridden by direct assignment.

    >>> import os
    >>> os.environ["QAT_MAX_REPEATS_LIMIT"] = "654321"
    >>> QatConfig() # doctest: +ELLIPSIS
    QatConfig(MAX_REPEATS_LIMIT=654321, ...)
    >>> QatConfig(MAX_REPEATS_LIMIT=123) # doctest: +ELLIPSIS
    QatConfig(MAX_REPEATS_LIMIT=123, ...)
    >>> del os.environ["QAT_MAX_REPEATS_LIMIT"]
    >>> qatconfig = QatConfig()
    >>> qatconfig.MAX_REPEATS_LIMIT = 16000
    >>> qatconfig # doctest: +ELLIPSIS
    QatConfig(MAX_REPEATS_LIMIT=16000, ...)

    >>> QatConfig(MAX_REPEATS_LIMIT=100.5) # doctest: +IGNORE_EXCEPTION_DETAIL
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
    MAX_REPEATS_LIMIT: Optional[int] = Field(gt=0, default=100_000)
    """Max number of repeats / shots to be performed in a single job."""

    INSTRUCTION_VALIDATION: InstructionValidationConfig = InstructionValidationConfig()
    """Options for Instruction validation before execution."""

    @property
    def DISABLE_PULSE_DURATION_LIMITS(self):
        """Flag to disable the lower and upper pulse duration limits.
        Only needs to be set to True for calibration purposes."""
        warnings.warn(
            "'QatConfig().DISABLE_PULSE_DURATION_LIMITS' is being deprecated, please use "
            "'QatConfig().INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return not self.INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS

    @DISABLE_PULSE_DURATION_LIMITS.setter
    def DISABLE_PULSE_DURATION_LIMITS(self, val: bool):
        warnings.warn(
            "'QatConfig().DISABLE_PULSE_DURATION_LIMITS' is being deprecated, please use "
            "'QatConfig().INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS = not val

    SIMULATION: QatSimulationConfig = QatSimulationConfig()
    """Options for QATs simulation backends."""

    EXTENSIONS: list[ImportString] = []
    """ QAT Extensions to initialise on start-up"""

    HARDWARE: list[HardwareLoaderDescription] = []
    """ QAT Hardware Models to load on start-up """

    PIPELINES: list[PipelineInstanceDescription | PipelineBuilderDescription] | None = None
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
        if len(v) == 0:
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

        hardware_loader_names = {hld.name for hld in info.data["HARDWARE"]}
        for Pdesc in pipelines:
            expected_loader = getattr(Pdesc, "hardware_loader", None)
            if expected_loader and not expected_loader in hardware_loader_names:
                raise ValueError(f"Hardware Loader {expected_loader} not defined")

        return pipelines

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    def validate(self, compiler_config: CompilerConfig):
        """_summary_

        Args:
            compiler_config (CompilerConfig): _description_
        """
        if compiler_config.repeats and (compiler_config.repeats > self.MAX_REPEATS_LIMIT):
            raise ValueError(
                f"Number of shots {compiler_config.repeats} exceeds the maximum amount of {self.MAX_REPEATS_LIMIT}."
            )

    @staticmethod
    def from_yaml(filename: str):
        with open(filename) as f:
            blob = yaml.safe_load(f)
        return QatConfig(**blob)


qatconfig = QatConfig()
