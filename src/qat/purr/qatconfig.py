from typing import Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class QatConfig(BaseSettings, validate_assignment=True):
    """
    Full settings for a single job. All values are defaulted on initialization.
    """

    model_config = SettingsConfigDict(env_prefix="QAT_")
    MAX_REPEATS_LIMIT: Union[None, int] = 100_000
    """Max number of repeats / shots to be performed in a single job."""


qatconfig = QatConfig()
