from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class QatConfig(BaseSettings, validate_assignment=True):
    """
    Full settings for a single job. All values are defaulted on initialization.
    """

    model_config = SettingsConfigDict(env_prefix="QAT_")
    MAX_REPEATS_LIMIT: int = 100_000
    """Max number of repeats / shots to be performed in a single job."""

    @field_validator("MAX_REPEATS_LIMIT", mode="before")
    def check_max_repeats_limit(cls, value):
        if value and not isinstance(value, int):
            raise TypeError(
                f"MAX_REPEATS_LIMIT must be an int, got {type(value).__name__}."
            )
        return value


qatconfig = QatConfig()
