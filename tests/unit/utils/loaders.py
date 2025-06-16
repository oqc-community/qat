from qat.model.loaders.legacy.echo import EchoModelLoader
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class BrokenLoader(EchoModelLoader):
    """ModelLoaders load HardwareModels from a source configured on initialisation."""

    def __init__(self, on_init: bool = False, on_load: bool = False, *args, **kwargs):
        self.on_init = on_init
        self.on_load = on_load

        if self.on_init:
            raise ValueError("This loader is broken intentionally on init.")

        super().__init__(*args, **kwargs)

    def load(self) -> QuantumHardwareModel:
        """Load and return the Hardware Model.

        :return: A loaded Hardware Model
        :rtype: LegacyHardwareModel | LogicalHardwareModel
        """
        if self.on_load:
            raise ValueError("This loader is broken intentionally on load.")
        return super().load()
