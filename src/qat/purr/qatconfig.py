from pydantic_settings import BaseSettings


class QatConfig(BaseSettings):
    MAX_REPEATS_LIMIT: int = 100_000
    """Max number of repeats / shots to be performed in a single job."""


qatconfig = QatConfig()
