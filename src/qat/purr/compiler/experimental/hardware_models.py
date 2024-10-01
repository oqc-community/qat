from typing import TYPE_CHECKING, Dict, List

from pydantic import BaseModel, Field

from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.experimental.devices import (
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    QuantumDevice,
    QubitCoupling,
)
from qat.purr.compiler.instructions import AcquireMode

if TYPE_CHECKING:
    from qat.purr.compiler.execution import InstructionExecutionEngine
    from qat.purr.compiler.experimental.error_mitigation import ErrorMitigation
    from qat.purr.compiler.runtime import QuantumRuntime


class HardwareModel(BaseModel):
    """
    Base class for all hardware models. Every model should return the builder class that
    should be used to build circuits/pulses for its particular back-end.
    """

    repeat_limit: int | None = None
    "The maximum number of shots / repeats for the hardware model."

    def create_builder(self) -> InstructionBuilder: ...

    def create_engine(self) -> InstructionExecutionEngine: ...

    def create_runtime(self, engine: InstructionExecutionEngine = None) -> QuantumRuntime:
        if engine is None:
            engine = self.create_engine()
        return QuantumRuntime(engine)


class QuantumHardwareModel(HardwareModel):
    """
    Object modelling our superconducting hardware. Holds up-to-date information about a
    current piece of hardware, whether simulated or physical machine.
    """

    default_acquire_mode: AcquireMode = AcquireMode.RAW
    default_repeat_count: int = 1000
    default_repetition_period: float = 100e-6

    quantum_devices: Dict[str, QuantumDevice] = Field(allow_mutation=False, default=dict())
    pulse_channels: Dict[str, PulseChannel] = Field(allow_mutation=False, default=dict())
    physical_channels: Dict[str, PhysicalChannel] = Field(
        allow_mutation=False, default=dict()
    )
    basebands: Dict[str, PhysicalBaseband] = Field(allow_mutation=False, default=dict())
    qubit_direction_couplings: List[QubitCoupling] = Field(allow_mutation=False, default=[])
    error_mitigation: ErrorMitigation | None = None
