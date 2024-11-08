from typing import Annotated, get_args, get_origin

from qat.model.component import Component
from qat.model.device import (
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    Resonator,
)
from qat.model.hardware_model import QuantumHardwareModel
from qat.purr.compiler.devices import ChannelType

from qat.model.serialisation import ComponentDict  # isort: skip


class QuantumHardwareModelBuilder:
    def __init__(self):
        self._current_model = QuantumHardwareModel()

    @property
    def model(self):
        return self._current_model

    def _update_component(self, component: Component):
        component_type = type(component)

        for field_name, field_type in self._current_model.__annotations__.items():
            if (
                self._current_model.model_fields[field_name].metadata
                and get_typevar_from_annotated(field_type) == component_type
            ):
                field_components = getattr(self._current_model, field_name)
                field_components.update({component.to_component_id(): component})

                self._current_model = self._current_model.model_copy(
                    update={field_name: field_components}
                )
                return component

        raise Exception(f"Unknown hardware component type {component_type}.")

    def _update_components(self, *components: Component):
        for component in components:
            self._update_component(component)

    def add_physical_baseband(self, frequency, **kwargs):
        return self._update_component(PhysicalBaseband(frequency=frequency, **kwargs))

    def add_physical_channel(self, baseband, sample_time, **kwargs):
        return self._update_component(
            PhysicalChannel(baseband=baseband, sample_time=sample_time, **kwargs)
        )

    def add_pulse_channel(self, frequency, physical_channel, **kwargs):
        return self._update_component(
            PulseChannel(frequency=frequency, physical_channel=physical_channel, **kwargs)
        )

    def add_qubit(
        self,
        index,
        frequency,
        physical_channel,
        measure_device,
        pulse_channels=None,
        **kwargs,
    ):
        if pulse_channels is None:
            pc_drive = PulseChannel(
                frequency=frequency,
                physical_channel=physical_channel,
                channel_type=ChannelType.drive,
            )
            pulse_channels = {pc_drive.to_component_id(): pc_drive}
            self._update_component(pc_drive)

        return self._update_component(
            Qubit(
                index=index,
                pulse_channels=pulse_channels,
                physical_channel=physical_channel,
                measure_device=measure_device,
                **kwargs,
            )
        )

    def add_resonator(self, frequency, physical_channel, pulse_channels=None, **kwargs):
        if pulse_channels is None:
            pc_measure = PulseChannel(
                id=ChannelType.measure.name,
                channel_type=ChannelType.measure,
                frequency=frequency,
                physical_channel=physical_channel,
            )
            self._update_component(pc_measure)

            pc_acquire = PulseChannel(
                id=ChannelType.acquire.name,
                channel_type=ChannelType.acquire,
                frequency=frequency,
                physical_channel=physical_channel,
            )
            self._update_component(pc_acquire)

            pulse_channels = {
                pc_measure.to_component_id(): pc_measure,
                pc_acquire.to_component_id(): pc_acquire,
            }

        resonator = Resonator(
            pulse_channels=pulse_channels, physical_channel=physical_channel, **kwargs
        )
        return self._update_component(resonator)

    def _couple_qubits(
        self,
        qubit1: Qubit,
        qubit2: Qubit,
        cross_res_frequency: float,
        cross_res_canc_frequency: float,
        cross_res_scale: float = 1.0,
        cross_res_canc_scale: float = 1.0,
    ):
        cross_res_pulse_ch = PulseChannel(
            frequency=cross_res_frequency,
            scale=cross_res_scale,
            channel_type=ChannelType.cross_resonance,
            auxiliary_qubits=[qubit2],
            physical_channel=qubit1.physical_channel,
        )
        cross_res_canc_pulse_ch = PulseChannel(
            frequency=cross_res_canc_frequency,
            scale=cross_res_canc_scale,
            channel_type=ChannelType.cross_resonance_cancellation,
            auxiliary_qubits=[qubit2],
            physical_channel=qubit1.physical_channel,
        )

        self._update_components(cross_res_pulse_ch, cross_res_canc_pulse_ch)

        qubit1 = qubit1._update_pulse_channels(cross_res_pulse_ch, cross_res_canc_pulse_ch)
        qubit1.coupled_qubits.append(qubit2)
        self._update_component(qubit1)

    def add_connectivity(self, connectivity: list[tuple[int, int]] = None):

        if connectivity is None:
            # Create a ring architecture where each qubit i is connected to qubits i-1 and i+1.
            n_qubits = self.model.number_of_qubits
            connectivity = [(i, i + 1) for i in range(0, n_qubits - 1)]

        qubits_by_index = {qubit.index: qubit for qubit in self.model.qubits.values()}
        for index1, index2 in connectivity:
            qubit1 = qubits_by_index[index1]
            qubit2 = qubits_by_index[index2]

            self._couple_qubits(
                qubit1,
                qubit2,
                cross_res_frequency=5.5e09,
                cross_res_canc_frequency=5.5e9,
                cross_res_scale=50.0,
                cross_res_canc_scale=0.0,
            )
            self._couple_qubits(
                qubit2,
                qubit1,
                cross_res_frequency=5.5e09,
                cross_res_canc_frequency=5.5e9,
                cross_res_scale=50.0,
                cross_res_canc_scale=0.0,
            )


def get_typevar_from_annotated(type_name: str):
    """Function to retrieve the type assigned to `T` from a string representation of Annotated types."""
    annotated_type = eval(type_name)

    if get_origin(annotated_type) is Annotated:
        inner_type = get_args(annotated_type)[0]
        container_type = get_origin(inner_type)

        if container_type is dict:
            return get_args(inner_type)[1]

        elif container_type is list:
            return get_args(inner_type)[0]

        elif container_type is None:
            return inner_type

    raise ValueError("Provided Annotated type does not have the required structure.")
