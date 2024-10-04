from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field

from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.experimental.devices import (
    ChannelType,
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    QuantumDevice,
    Qubit,
    QubitCoupling,
    Resonator,
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

    model_config = ConfigDict(validate_assignment=True)

    default_acquire_mode: AcquireMode = AcquireMode.RAW
    default_repeat_count: int = 1000
    default_repetition_period: float = 100e-6

    quantum_devices: Dict[str, QuantumDevice] = Field(allow_mutation=False, default=dict())
    pulse_channels: Dict[str, PulseChannel] = Field(allow_mutation=False, default=dict())
    physical_channels: Dict[str, PhysicalChannel] = Field(
        allow_mutation=False, default=dict()
    )
    physical_basebands: Dict[str, PhysicalBaseband] = Field(
        allow_mutation=False, default=dict()
    )
    qubit_direction_couplings: List[QubitCoupling] = Field(allow_mutation=False, default=[])
    error_mitigation: ErrorMitigation | None = None

    def add_hardware_component(
        self,
        name: str,
        component: Union[
            Dict[str, QuantumDevice],
            Dict[str, PulseChannel],
            Dict[str, PhysicalChannel],
            Dict[str, PhysicalBaseband],
        ],
    ):
        if name not in self.__model_fields__.keys() or not isinstance(self[name], Dict):
            raise ValueError(
                f"Trying to add to {name}, which is not a type of hardware component."
            )

        elif component.id in self[name]:
            raise KeyError(
                f"Hardware component name with id '{component.id}' already exists."
            )

        return self.model_copy(update={name: self[name].append(component)})


class QuantumHardwareModelBuilder:
    def __init__(self):
        self.hardware_model = QuantumHardwareModel()

    def add_physical_baseband(self, **kwargs):
        self.hardware_model.add_hardware_component(
            "physical_basebands", PhysicalBaseband(kwargs)
        )

    def add_physical_channel(self, **kwargs):
        self.hardware_model.add_hardware_component(
            "physical_channels", PhysicalChannel(kwargs)
        )

    def add_qubit(self, frequency: float, **kwargs):
        pc_drive = PulseChannel(channel_type=ChannelType.drive, frequency=frequency)
        pc_drive_id = ChannelType.drive.name

        qubit = Qubit(kwargs, pulse_channels={pc_drive_id: pc_drive})
        self.hardware_model.add_hardware_component("quantum_devices", qubit)

    def add_resonator(self, frequency: float, **kwargs):
        pc_measure = PulseChannel(channel_type=ChannelType.measure, frequency=frequency)
        pc_measure_id = ChannelType.measure.name

        pc_acquire = PulseChannel(channel_type=ChannelType.acquire, frequency=frequency)
        pc_acquire_id = ChannelType.acquire.name

        resonator = Resonator(
            kwargs, pulse_channels={pc_measure_id: pc_measure, pc_acquire_id: pc_acquire}
        )
        self.hardware_model.add_hardware_component("quantum_devices", resonator)
