from pydantic_settings import BaseSettings, SettingsConfigDict


class QatConfig(BaseSettings):
    """
    Full settings for a single job. All values are defaulted on initialization.
    """

    model_config = SettingsConfigDict(env_prefix="QAT_")
    MAX_REPEATS_LIMIT: int = 100_000
    """Max number of repeats / shots to be performed in a single job."""


qatconfig = QatConfig()
