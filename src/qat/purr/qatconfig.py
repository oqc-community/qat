# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from typing import Literal, Type

from compiler_config.config import CompilerConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from qiskit_aer import AerSimulator

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class QiskitSimulationConfig(BaseModel):
    """
    The default settings for the Qiskit Simulator, including overridden MPS settings.
    """

    model_config = ConfigDict(validate_assignment=True)
    _allowed_methods: type = Literal[AerSimulator().available_methods()]
    METHOD: _allowed_methods = "automatic"
    """The simulation method to use."""
    FALLBACK_SEQUENCE: list[_allowed_methods] = ["automatic", "matrix_product_state"]
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
    """

    model_config = SettingsConfigDict(
        env_prefix="QAT_",
        env_nested_delimiter="_",
        validate_assignment=True,
    )
    MAX_REPEATS_LIMIT: int | None = Field(gt=0, default=100_000)
    """Max number of repeats / shots to be performed in a single job."""

    INSTRUCTION_VALIDATION: InstructionValidationConfig = InstructionValidationConfig()
    """Options for Instruction validation before execution."""

    SIMULATION: QatSimulationConfig = QatSimulationConfig()
    """Options for QATs simulation backends."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
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
