import abc

from pydantic import BaseModel

from qat.purr.compiler.experimental.hardware_models import QuantumHardwareModel


class InstructionExecutionEngine(BaseModel, abc.ABC):
    model: QuantumHardwareModel
    startup_engine: bool = True

    def __init__(self, **data):
        super.__init__(**data)
        if self.startup_engine:
            self.startup()

    def startup(self):
        """Starts up the underlying hardware or does nothing if already started."""
        return True

    def run_calibrations(self, qubits_to_calibrate=None): ...

    def execute(self, instructions: List[Instruction]) -> Dict[str, Any]: ...
