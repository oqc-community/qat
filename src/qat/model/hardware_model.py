from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from qat.model.device_models import (
    PhysicalBasebandData,
    PhysicalChannelData,
    PulseChannelData,
    QubitData,
    ResonatorData,
)
from qat.model.devices import (
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    Resonator,
)
from qat.utils.pydantic import WarnOnExtraFieldsModel


class QuantumHardwareDataParser(WarnOnExtraFieldsModel):
    qubits_data: List[QubitData] = Field(allow_mutation=False, default=dict())
    resonators_data: List[ResonatorData] = Field(allow_mutation=False, default=dict())

    pulse_channels_data: List[PulseChannelData] = Field(
        allow_mutation=False, default=dict()
    )
    physical_channels_data: List[PhysicalChannelData] = Field(
        allow_mutation=False, default=dict()
    )
    physical_basebands_data: List[PhysicalBasebandData] = Field(
        allow_mutation=False, default=dict()
    )

    def parse(self):
        physical_basebands = {}
        for bb_data in self.physical_basebands_data:
            physical_basebands[bb_data.id] = PhysicalBaseband(data=bb_data)

        physical_channels = {}
        for phys_ch_data in self.physical_channels_data:
            physical_channels[phys_ch_data.id] = PhysicalChannel(
                data=phys_ch_data, baseband=physical_basebands[phys_ch_data.baseband_id]
            )

        pulse_channels = {}
        for pulse_ch_data in self.pulse_channels_data:
            pulse_channels[pulse_ch_data.id] = PulseChannel(
                data=pulse_ch_data,
                physical_channel=physical_channels[pulse_ch_data.physical_channel_id],
            )
            # TODO: set auxiliary qubits

        resonators = {}
        for r_data in self.resonators_data:
            r_pulse_channels = {
                pulse_ch_id: pulse_channels[pulse_ch_id]
                for pulse_ch_id in r_data.pulse_channel_ids
            }
            resonators[r_data.id] = Resonator(
                data=r_data,
                physical_channel=physical_channels[r_data.physical_channel_id],
                pulse_channels=r_pulse_channels,
            )

        qubits = {}
        for q_data in self.qubits_data:
            q_pulse_channels = {
                pulse_ch_id: pulse_channels[pulse_ch_id]
                for pulse_ch_id in q_data.pulse_channel_ids
            }
            qubits[q_data.id] = Qubit(
                data=q_data,
                physical_channel=physical_channels[q_data.physical_channel_id],
                pulse_channels=q_pulse_channels,
            )

        return QuantumHardwareModel(
            qubits=qubits,
            resonators=resonators,
            pulse_channels=pulse_channels,
            physical_channels=physical_channels,
            physical_basebands=physical_basebands,
        )


class QuantumHardwareModel(WarnOnExtraFieldsModel):
    """
    Base class for calibrating our QPU hardware.

    Attributes:
        qubits:
    """

    qubits: Dict[str, Qubit] = Field(allow_mutation=False, default=dict())
    resonators: Dict[str, Resonator] = Field(allow_mutation=False, default=dict())

    pulse_channels: Dict[str, PulseChannel] = Field(allow_mutation=False, default=dict())
    physical_channels: Dict[str, PhysicalChannel] = Field(
        allow_mutation=False, default=dict()
    )
    physical_basebands: Dict[str, PhysicalBaseband] = Field(
        allow_mutation=False, default=dict()
    )

    @property
    def quantum_devices(self):
        qds = self.qubits.update(self.resonators)
        return qds

    @property
    def number_of_qubits(self):
        return len(self.qubits)

    @property
    def number_of_resonators(self):
        return len(self.resonators)

    def get_qubit_with_index(self, i: int):
        for qubit in self.qubits:
            if qubit.index == i:
                return qubit

        raise KeyError(f"Qubit with index {i} does not exist.")

    """
    def add_qubit(self, qubit: QubitData):
        qubits = getattr(self, name), Dict
        self.qubits.update({component.id.full_id: component})
    
    def add_hardware_component(
        self,
        name: str,
        component: QuantumComponent,
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

        components.update({component.id.id: component})

        return self.model_copy(update={name: components})
        
    """
