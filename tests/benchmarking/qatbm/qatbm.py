from typing import Union

from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class QatCollection:
    def __init__(self, arg: Union[QuantumHardwareModel, QuantumExecutionEngine]):
        """
        Given a QuantumHardwareModel or QuantumExecutionEngine, create the relevant builder
        and execution engine. Also stores the qatfile and instruction timeline.
        """
        if isinstance(arg, QuantumHardwareModel):
            self.model = arg
            self.engine: QuantumExecutionEngine = arg.create_engine()
        else:
            self.engine = arg
            self.model = arg.model
        self.builder: InstructionBuilder = self.model.create_builder()
        self.qatfile = None
        self.timeline = None
        self.results = None
