from qat.model.loaders.legacy.base import BaseLegacyModelLoader
from qat.purr.backends.live import build_lucy_hardware
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class LucyModelLoader(BaseLegacyModelLoader):
    def load(self) -> QuantumHardwareModel:
        return build_lucy_hardware(QuantumHardwareModel())
