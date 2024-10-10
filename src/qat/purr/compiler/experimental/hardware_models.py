from __future__ import annotations

from typing import Dict, List, Tuple, Union

from pydantic import ConfigDict, Field, model_validator

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
from qat.purr.compiler.experimental.error_mitigation import ErrorMitigation
from qat.purr.compiler.instructions import AcquireMode
from qat.purr.utils.pydantic import WarnOnExtraFieldsModel


class QuantumHardwareModel(WarnOnExtraFieldsModel):
    """
    Base class for modelling our superconducting hardware. Holds up-to-date information
    about a current piece of hardware, whether simulated or physical machine.

    Attributes:
        repeat_limit: The maximum number of shots / repeats for the hardware model.
    """

    model_config = ConfigDict(validate_assignment=True)

    default_acquire_mode: AcquireMode = AcquireMode.RAW
    default_repeat_count: int = 1000
    default_repetition_period: float = 100e-6
    repeat_limit: int | None = None
    min_pulse_length: float = Field(ge=0.0, default=1e-09)  # default value in seconds
    max_pulse_length: float = Field(gt=0.0, default=1e-03)  # default value in seconds

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

    @model_validator(mode="after")
    def check_default_repeat_count(self):
        if self.repeat_limit and self.default_repeat_count > self.repeat_limit:
            raise ValueError(
                f"Default repeat count {self.default_repeat_count} cannot be larger than the repeat limit {self.repeat_limit}."
            )
        return self

    @model_validator(mode="after")
    def check_pulse_limits(self):
        if self.min_pulse_length > self.max_pulse_length:
            raise ValueError(f"Min pulse length cannot be larger than max pulse length.")

    @property
    def number_of_qubits(self):
        return len(self.qubit_devices)

    @property
    def qubit_devices(self):
        return [qd for qd in self.quantum_devices if isinstance(qd, Qubit)]

    @property
    def number_of_resonators(self):
        return len(self.resonator_devices)

    @property
    def resonator_devices(self):
        return [qd for qd in self.quantum_devices if isinstance(qd, Resonator)]

    def get_qubit_with_index(self, i: int):
        for qd in self.qubit_devices:
            if qd.index == i:
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
        if not hasattr(self, name) or not isinstance(
            components := getattr(self, name), Dict
        ):
            raise ValueError(
                f"Trying to add to {name}, which is not a type of hardware component."
            )
        elif component.id in components:
            raise KeyError(
                f"Hardware component name with id '{component.id}' already exists."
            )

        components.update({component.id: component})

        return self.model_copy(update={name: components})


class QuantumHardwareModelBuilder:
    def __init__(self):
        self.model = QuantumHardwareModel()

    def add_physical_baseband(self, **kwargs):
        self.model = self.model.add_hardware_component(
            "physical_basebands", PhysicalBaseband(**kwargs)
        )

    def add_physical_channel(self, **kwargs):
        self.model = self.model.add_hardware_component(
            "physical_channels", PhysicalChannel(**kwargs)
        )

    def add_pulse_channel(self, **kwargs):
        self.model = self.model.add_hardware_component(
            "pulse_channels", PulseChannel(**kwargs)
        )

    def add_qubit(self, frequency, **kwargs):
        if "pulse_channels" not in kwargs:
            pc_drive = PulseChannel(
                channel_type=ChannelType.drive,
                frequency=frequency,
                physical_channel=kwargs["physical_channel"],
            )
            pc_drive_id = ChannelType.drive.name
            kwargs["pulse_channels"] = {pc_drive_id: pc_drive}

        qubit = Qubit(**kwargs)
        self.model = self.model.add_hardware_component("quantum_devices", qubit)

    def add_resonator(self, frequency, **kwargs):
        if "pulse_channels" not in kwargs:
            pc_measure = PulseChannel(
                channel_type=ChannelType.measure,
                frequency=frequency,
                physical_channel=kwargs["physical_channel"],
            )
            pc_acquire = PulseChannel(
                channel_type=ChannelType.acquire,
                frequency=frequency,
                physical_channel=kwargs["physical_channel"],
            )
            kwargs["pulse_channels"] = {
                ChannelType.measure.name: pc_measure,
                ChannelType.acquire.name: pc_acquire,
            }

        resonator = Resonator(**kwargs)
        self.model = self.model.add_hardware_component("quantum_devices", resonator)

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
            n_qubits = self.model.number_of_qubits
            connectivity = [(i, i % n_qubits + 1) for i in range(1, n_qubits + 1)]

        def couple_qubits(qubit1, qubit2):
            self.model = self.model.get_qubit_with_index(qubit1).add_hardware_component(
                "pulse_channels",
                PulseChannel(
                    channel_type=ChannelType.cross_resonance,
                    auxiliary_devices=[qubit2],
                    frequency=cross_res_frequency,
                    scale=cross_res_scale,
                ),
            )
            self.model = self.model.get_qubit_with_index(qubit1).add_hardware_component(
                "pulse_channels",
                PulseChannel(
                    channel_type=ChannelType.cross_resonance_cancellation,
                    auxiliary_devices=[qubit2],
                    frequency=cross_res_canc_frequency,
                    scale=cross_res_canc_scale,
                ),
            )

            self.model.get_qubit_with_index(qubit1).add_coupled_qubit(qubit2)

        qubits_by_index = {qubit.index: qubit for qubit in self.model.qubit_devices}
        for index1, index2 in connectivity:
            qubit1 = qubits_by_index[index1]
            qubit2 = qubits_by_index[index2]

            couple_qubits(qubit1, qubit2)
            couple_qubits(qubit2, qubit1)
