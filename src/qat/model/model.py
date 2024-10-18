from __future__ import annotations

from typing import Dict, List, Tuple, Union

from pydantic import ConfigDict, Field, model_validator

from qat.model.devices import (
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
        return self

    @property
    def number_of_qubits(self):
        return len(self.qubits)

    @property
    def qubits(self):
        return [
            self.quantum_devices[qd]
            for qd in self.quantum_devices
            if isinstance(self.quantum_devices[qd], Qubit)
        ]

    @property
    def number_of_resonators(self):
        return len(self.resonators)

    @property
    def resonators(self):
        return [
            self.quantum_devices[qd]
            for qd in self.quantum_devices
            if isinstance(self.quantum_devices[qd], Resonator)
        ]

    def get_qubit_with_index(self, i: int):
        for qd in self.qubits:
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

    def __str__(self):
        return f"{type(self).__name__}({self.qubits}, {self.resonators})"


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
                id=ChannelType.drive.name,
                channel_type=ChannelType.drive,
                frequency=frequency,
                physical_channel=kwargs["physical_channel"],
            )
            kwargs["pulse_channels"] = {pc_drive.id: pc_drive}

        qubit = Qubit(**kwargs)
        self.model = self.model.add_hardware_component("quantum_devices", qubit)

    def add_resonator(self, frequency, **kwargs):
        if "pulse_channels" not in kwargs:
            pc_measure = PulseChannel(
                id=ChannelType.measure.name,
                channel_type=ChannelType.measure,
                frequency=frequency,
                physical_channel=kwargs["physical_channel"],
            )
            pc_acquire = PulseChannel(
                id=ChannelType.acquire.name,
                channel_type=ChannelType.acquire,
                frequency=frequency,
                physical_channel=kwargs["physical_channel"],
            )
            kwargs["pulse_channels"] = {
                pc_measure.id: pc_measure,
                pc_acquire.id: pc_acquire,
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
            connectivity = [(i, i % (n_qubits - 1) + 1) for i in range(0, n_qubits)]

        def couple_qubits(qubit1, qubit2):
            cross_res_pulse_ch = PulseChannel(
                id=ChannelType.cross_resonance.name,
                channel_type=ChannelType.cross_resonance,
                auxiliary_devices=[qubit2],
                frequency=cross_res_frequency,
                physical_channel=qubit1.physical_channel,
                scale=cross_res_scale,
            )
            cross_res_canc_pulse_ch = PulseChannel(
                id=ChannelType.cross_resonance_cancellation.name,
                channel_type=ChannelType.cross_resonance_cancellation,
                auxiliary_devices=[qubit2],
                frequency=cross_res_canc_frequency,
                physical_channel=qubit1.physical_channel,
                scale=cross_res_canc_scale,
            )
            self.model.quantum_devices[qubit1.id].pulse_channels.update(
                {
                    cross_res_pulse_ch.id: cross_res_pulse_ch,
                    cross_res_canc_pulse_ch.id: cross_res_canc_pulse_ch,
                }
            )
            self.model.quantum_devices[qubit1.id].add_coupled_qubit(qubit2)

        qubits_by_index = {qubit.index: qubit for qubit in self.model.qubits}
        for index1, index2 in connectivity:
            qubit1 = qubits_by_index[index1]
            qubit2 = qubits_by_index[index2]

            couple_qubits(qubit1, qubit2)
            couple_qubits(qubit2, qubit1)


def build_hardware_model(qubit_count: int = 4, connectivity: list = None):
    builder = QuantumHardwareModelBuilder()

    channel_idx = 1
    for qubit_index in range(qubit_count):
        builder.add_physical_baseband(id=f"LO{channel_idx}", frequency=5.5e9)
        builder.add_physical_baseband(id=f"LO{channel_idx+1}", frequency=8.5e9)

        builder.add_physical_channel(
            id=f"CH{channel_idx}",
            sample_time=1.0e-9,
            baseband=builder.model.physical_basebands[f"LO{channel_idx}"],
            block_size=1,
        )
        builder.add_physical_channel(
            id=f"CH{channel_idx+1}",
            sample_time=1.0e-9,
            baseband=builder.model.physical_basebands[f"LO{channel_idx+1}"],
            acquire_allowed=True,
            block_size=1,
        )

        builder.add_resonator(
            id=f"R{qubit_index}",
            frequency=8.5e9,
            physical_channel=builder.model.physical_channels[f"CH{channel_idx+1}"],
        )
        builder.add_qubit(
            id=f"Q{qubit_index}",
            index=qubit_index,
            frequency=5.5e9,
            physical_channel=builder.model.physical_channels[f"CH{channel_idx}"],
            measure_device=builder.model.quantum_devices[f"R{qubit_index}"],
        )

        channel_idx += 2

    builder.add_cross_resonance_pulse_channels(
        connectivity=connectivity,
        cross_res_frequency=5.5e9,
        cross_res_canc_frequency=5.5e9,
        cross_res_scale=50.0,
        cross_res_canc_scale=0.0,
    )

    return builder.model


hw_model = build_hardware_model(qubit_count=4)
