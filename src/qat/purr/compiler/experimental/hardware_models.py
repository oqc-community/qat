from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    Base class for all hardware models. Every model should return the builder class
    that should be used to build circuits/pulses for its particular back-end.
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
    Object modelling our superconducting hardware. Holds up-to-date information
    about a current piece of hardware, whether simulated or physical machine.
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

    @field_validator
    def check_default_repeat_count(self):
        if self.repeat_limit and self.default_repeat_count > self.repeat_limit:
            raise ValueError(
                f"Default repeat count {self.default_repeat_count} cannot be larger than the repeat limit {self.repeat_limit}."
            )

    @property
    def number_of_qubits(self):
        return len(self.qubit_devices)

    @property
    def qubit_devices(self):
        return [qd for qd in self.quantum_devices if isinstance(qd, Qubit)]

    @property
    def resonator_devices(self):
        return [qd for qd in self.quantum_devices if isinstance(qd, Resonator)]

    def get_qubit_with_index(self, i: int):
        for qd in self.quantum_devices.values():
            if isinstance(qd, Qubit) and qd.index == i:
                return qd

        raise KeyError(f"Qubit with index {i} does not exist.")

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

    @property
    def hardware_model(self):
        return self.hardware_model

    @hardware_model.setter
    def hardware_model(self, other):
        raise AttributeError(
            f"Cannot set hardware_model attribute to {other}."
            "Please use the builder to build the hardware model."
        )

    def add_physical_baseband(self, **kwargs):
        self.hardware_model.add_hardware_component(
            "physical_basebands", PhysicalBaseband(**kwargs)
        )

    def add_physical_channel(self, **kwargs):
        self.hardware_model.add_hardware_component(
            "physical_channels", PhysicalChannel(**kwargs)
        )

    def add_qubit(self, drive_frequency: float, **kwargs):
        pc_drive = PulseChannel(channel_type=ChannelType.drive, frequency=drive_frequency)
        pc_drive_id = ChannelType.drive.name

        qubit = Qubit(kwargs, pulse_channels={pc_drive_id: pc_drive})
        self.hardware_model.add_hardware_component("quantum_devices", qubit)

    def add_resonator(self, meas_acq_frequency: float, **kwargs):
        pc_measure = PulseChannel(
            channel_type=ChannelType.measure, frequency=meas_acq_frequency
        )
        pc_acquire = PulseChannel(
            channel_type=ChannelType.acquire, frequency=meas_acq_frequency
        )

        resonator = Resonator(
            pulse_channels={
                ChannelType.measure.name: pc_measure,
                ChannelType.acquire.name: pc_acquire,
            },
            **kwargs,
        )
        self.hardware_model.add_hardware_component("quantum_devices", resonator)

    def add_cross_resonance_pulse_channels(
        self,
        cross_res_frequency: float,
        cross_res_canc_frequency: float,
        cross_res_scale: float = 50.0,
        cross_res_canc_scale: float = 0.0,
        connectivity: List[Tuple[int, int]] = None,
    ):

        if connectivity is None:
            # Create a ring architecture where each qubit i is connected to qubits i-1 and i+1.
            n_qubits = self.hardware_model.number_of_qubits
            connectivity = [(i, i % n_qubits + 1) for i in range(1, n_qubits + 1)]

        def couple_qubits(qubit1, qubit2):
            self.hardware_model.get_qubit_with_index(qubit1).add_pulse_channel(
                PulseChannel(
                    channel_type=ChannelType.cross_resonance,
                    auxiliary_devices=[qubit2],
                    frequency=cross_res_frequency,
                    scale=cross_res_scale,
                )
            )
            self.hardware_model.get_qubit_with_index(qubit1).add_pulse_channel(
                PulseChannel(
                    channel_type=ChannelType.cross_resonance_cancellation,
                    auxiliary_devices=[qubit2],
                    frequency=cross_res_canc_frequency,
                    scale=cross_res_canc_scale,
                )
            )

            self.hardware_model.get_qubit_with_index(qubit1).add_coupled_qubit(qubit2)

        qubits_by_index = {
            qubit.index: qubit for qubit in self.hardware_model.qubit_devices
        }
        for index1, index2 in connectivity:
            qubit1 = qubits_by_index[index1]
            qubit2 = qubits_by_index[index2]

            couple_qubits(qubit1, qubit2)
            couple_qubits(qubit2, qubit1)
