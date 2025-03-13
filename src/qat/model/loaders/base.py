import abc

from qat.model.hardware_model import LogicalHardwareModel, PhysicalHardwareModel
from qat.purr.compiler.hardware_models import HardwareModel as LegacyHardwareModel


class BaseModelLoader(abc.ABC):
    """ModelLoaders load HardwareModels from a source configured on initialisation."""

    def __init__(self): ...

    @abc.abstractmethod
    def load(self) -> LegacyHardwareModel | LogicalHardwareModel:
        """Load and return the Hardware Model.

        :return: A loaded Hardware Model
        :rtype: LegacyHardwareModel | LogicalHardwareModel
        """
        pass


class BaseLogicalModelLoader(BaseModelLoader):
    @abc.abstractmethod
    def load(self) -> LogicalHardwareModel:
        pass


class BasePhysicalModelLoader(BaseModelLoader):
    @abc.abstractmethod
    def load(self) -> PhysicalHardwareModel:
        pass
