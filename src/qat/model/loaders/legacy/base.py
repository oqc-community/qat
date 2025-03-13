import abc

from qat.model.loaders.base import BaseModelLoader
from qat.purr.compiler.hardware_models import HardwareModel as LegacyHardwareModel


class BaseLegacyModelLoader(BaseModelLoader):
    @abc.abstractmethod
    def load(self) -> LegacyHardwareModel:
        pass
