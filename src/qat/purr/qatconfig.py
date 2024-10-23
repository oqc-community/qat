from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from qiskit_aer import AerSimulator

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


class QatConfig(BaseSettings):
    """
    Full settings for a single job. Allows environment variables to be overridden by direct assignment.

    >>> import os
    >>> os.environ["QAT_MAX_REPEATS_LIMIT"] = "654321"
    >>> QatConfig()
    QatConfig(MAX_REPEATS_LIMIT=654321, DISABLE_PULSE_DURATION_LIMITS=False)
    >>> QatConfig(MAX_REPEATS_LIMIT=123)
    QatConfig(MAX_REPEATS_LIMIT=123, DISABLE_PULSE_DURATION_LIMITS=False)

    >>> qatconfig = QatConfig()
    >>> qatconfig.MAX_REPEATS_LIMIT = 16000
    >>> qatconfig
    QatConfig(MAX_REPEATS_LIMIT=16000, DISABLE_PULSE_DURATION_LIMITS=False)

    >>> QatConfig(MAX_REPEATS_LIMIT=100.5) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    pydantic_core._pydantic_core.ValidationError
    ...
    Input should be a valid integer, got a number with a fractional part
    """

    model_config = SettingsConfigDict(
        env_prefix="QAT_", env_nested_delimiter="_", validate_assignment=True
    )
    MAX_REPEATS_LIMIT: Optional[int] = Field(gt=0, default=100_000)
    """Max number of repeats / shots to be performed in a single job."""
    DISABLE_PULSE_DURATION_LIMITS: bool = False
    """Flag to disable the lower and upper pulse duration limits. 
    Only needs to be set to True for calibration purposes."""

    SIMULATION: QatSimulationConfig = QatSimulationConfig()
    """Options for QATs simulation backends."""

    @field_validator("DISABLE_PULSE_DURATION_LIMITS")
    def check_disable_pulse_duration_limits(cls, DISABLE_PULSE_DURATION_LIMITS):
        if DISABLE_PULSE_DURATION_LIMITS:
            log.warning(
                "Disabled check for pulse duration limits, which should ideally only be used for calibration purposes."
            )
        return DISABLE_PULSE_DURATION_LIMITS


qatconfig = QatConfig()

if __name__ == "__main__":
    import doctest

    doctest.testmod()
