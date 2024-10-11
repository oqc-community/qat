from typing import Dict

from qat.purr.backends.live_devices import ControlHardware, Instrument
from qat.purr.compiler.experimental.execution import QuantumExecutionEngine


class LiveDeviceEngine(QuantumExecutionEngine):
    """
    Backend that hooks up the logical hardware model to our QPU's, currently hardcoded to particular fridges.
    This will only work when run on a machine physically connected to a QPU.
    """

    startup_engine: bool = True
    control_hardware: ControlHardware | None = None
    instruments: Dict[str, Instrument] | None = None

    def __init__(self, **data):
        super.__init__(**data)
        if self.startup_engine:
            self.startup()

    def startup(self):
        pass

    def shutdown(self):
        pass

    def execute(self):
        pass
