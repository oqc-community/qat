from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from qat.model.device_models import (
    PhysicalBasebandData,
    PhysicalChannelData,
    PulseChannelData,
    QuantumComponentData,
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
from qat.purr.compiler.devices import ChannelType
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
            physical_basebands[bb_data.full_id] = PhysicalBaseband(data=bb_data)

        physical_channels = {}
        for phys_ch_data in self.physical_channels_data:
            physical_channels[phys_ch_data.full_id] = PhysicalChannel(
                data=phys_ch_data,
                baseband=physical_basebands[phys_ch_data.baseband_id.uuid],
            )

        pulse_channels = {}
        for pulse_ch_data in self.pulse_channels_data:
            pulse_channels[pulse_ch_data.full_id] = PulseChannel(
                data=pulse_ch_data,
                physical_channel=physical_channels[pulse_ch_data.physical_channel_id.uuid],
            )
            # TODO: set auxiliary qubits

        resonators = {}
        for r_data in self.resonators_data:
            r_pulse_channels = {
                pulse_ch_id.uuid: pulse_channels[pulse_ch_id.uuid]
                for pulse_ch_id in r_data.pulse_channel_ids
            }
            resonators[r_data.full_id] = Resonator(
                data=r_data,
                physical_channel=physical_channels[r_data.physical_channel_id.uuid],
                pulse_channels=r_pulse_channels,
            )

        qubits = {}
        for q_data in self.qubits_data:
            q_pulse_channels = {
                pulse_ch_id.uuid: pulse_channels[pulse_ch_id.uuid]
                for pulse_ch_id in q_data.pulse_channel_ids
            }
            qubits[q_data.full_id] = Qubit(
                data=q_data,
                physical_channel=physical_channels[q_data.physical_channel_id.uuid],
                pulse_channels=q_pulse_channels,
                measure_device=resonators[q_data.measure_device_id.uuid],
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

    def model_dump(self):
        parser = QuantumHardwareDataParser(
            qubits_data=[qubit.data for qubit in self.qubits.values()],
            resonators_data=[resonator.data for resonator in self.resonators.values()],
            pulse_channels_data=[
                pulse_channel.data for pulse_channel in self.pulse_channels.values()
            ],
            physical_channels_data=[
                physical_channel.data
                for physical_channel in self.physical_channels.values()
            ],
            physical_basebands_data=[
                physical_baseband.data
                for physical_baseband in self.physical_basebands.values()
            ],
        )
        return parser.model_dump()

    @staticmethod
    def model_load(data: Dict[str, QuantumComponentData]):
        parser = QuantumHardwareDataParser.model_validate(data)
        return parser.parse()


bb = PhysicalBaseband(data=PhysicalBasebandData(frequency=5e09))
phys_ch = PhysicalChannel(
    data=PhysicalChannelData(baseband_id=bb.id, sample_time=1e-09), baseband=bb
)
pulse_ch1 = PulseChannel(
    data=PulseChannelData(
        frequency=1e09, channel_type=ChannelType.drive, physical_channel_id=phys_ch.id
    ),
    physical_channel=phys_ch,
)
pulse_ch2 = PulseChannel(
    data=PulseChannelData(
        frequency=5e09,
        channel_type=ChannelType.second_state,
        physical_channel_id=phys_ch.id,
    ),
    physical_channel=phys_ch,
)
resonator = Resonator(
    data=ResonatorData(
        pulse_channel_ids=[pulse_ch1.id, pulse_ch2.id], physical_channel_id=phys_ch.id
    ),
    physical_channel=phys_ch,
    pulse_channels={pulse_ch1.full_id: pulse_ch1, pulse_ch2.full_id: pulse_ch2},
)
qubit = Qubit(
    data=QubitData(
        pulse_channel_ids=[pulse_ch1.id, pulse_ch2.id],
        physical_channel_id=phys_ch.id,
        index=0,
        measure_device_id=resonator.id,
    ),
    physical_channel=phys_ch,
    pulse_channels={pulse_ch1.full_id: pulse_ch1, pulse_ch2.full_id: pulse_ch2},
    measure_device=resonator,
)

hwm_parser = QuantumHardwareDataParser(
    qubits_data=[qubit.data],
    resonators_data=[resonator.data],
    pulse_channels_data=[pulse_ch1.data, pulse_ch2.data],
    physical_channels_data=[phys_ch.data],
    physical_basebands_data=[bb.data],
)

# hwm_parser.parse()
