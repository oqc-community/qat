from qat.model.loaders.legacy.base import BaseLegacyModelLoader
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.hardware_models import HardwareModel as LegacyHardwareModel


class RTCSModelLoader(BaseLegacyModelLoader):
    def __init__(self, rotating_frame: bool = True):
        self.rotating_frame = rotating_frame

    def load(self) -> LegacyHardwareModel:
        return get_default_RTCS_hardware(rotating_frame=self.rotating_frame)
