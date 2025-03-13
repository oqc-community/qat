from qat.model.convert_legacy import convert_legacy_echo_hw_to_pydantic
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.loaders.base import BasePhysicalModelLoader
from qat.purr.backends.echo import Connectivity

from .legacy.echo import EchoModelLoader as LegacyEchoModelLoader


class EchoModelLoader(BasePhysicalModelLoader):
    def __init__(
        self,
        qubit_count: int = 4,
        connectivity: Connectivity | list[(int, int)] | None = None,
    ):
        self._legacy = LegacyEchoModelLoader(
            qubit_count=qubit_count, connectivity=connectivity
        )

    def load(self) -> PhysicalHardwareModel:
        legacy_model = self._legacy.load()
        return convert_legacy_echo_hw_to_pydantic(legacy_model)
