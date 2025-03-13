from pathlib import Path

from qat.model.loaders.legacy.base import BaseLegacyModelLoader
from qat.purr.compiler.hardware_models import (
    QuantumHardwareModel as LegacyQuantumHardwareModel,
)


class FileModelLoader(BaseLegacyModelLoader):
    def __init__(self, path: Path | str):
        self.path: Path = Path(path)

    def load(self) -> LegacyQuantumHardwareModel:
        return LegacyQuantumHardwareModel.load_calibration_from_file(str(self.path))
