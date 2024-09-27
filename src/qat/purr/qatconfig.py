from typing import Union

from compiler_config.config import CompilerConfig
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QatConfig(BaseSettings, validate_assignment=True):
    """
    Full settings for a single job. Allows environment variables to be overridden by direct assignment.

    >>> import os
    >>> os.environ["QAT_MAX_REPEATS_LIMIT"] = "654321"
    >>> QatConfig()
    QatConfig(MAX_REPEATS_LIMIT=654321)
    >>> QatConfig(MAX_REPEATS_LIMIT=123)
    QatConfig(MAX_REPEATS_LIMIT=123)

    >>> qatconfig = QatConfig()
    >>> qatconfig.MAX_REPEATS_LIMIT = 16000
    >>> qatconfig
    QatConfig(MAX_REPEATS_LIMIT=16000)

    >>> QatConfig(MAX_REPEATS_LIMIT=100.5) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    pydantic_core._pydantic_core.ValidationError
    ...
    Input should be a valid integer, got a number with a fractional part
    """

    model_config = SettingsConfigDict(env_prefix="QAT_")
    MAX_REPEATS_LIMIT: Union[None, int] = Field(gt=0, default=100_000)
    """Max number of repeats / shots to be performed in a single job."""

    def validate(self, compiler_config: CompilerConfig):
        """_summary_

        Args:
            compiler_config (CompilerConfig): _description_
        """
        if compiler_config.repeats and (compiler_config.repeats > self.MAX_REPEATS_LIMIT):
            raise ValueError(
                f"Number of shots {compiler_config.repeats} exceeds the maximum amount of {self.MAX_REPEATS_LIMIT}."
            )


qatconfig = QatConfig()
