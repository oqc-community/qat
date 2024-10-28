from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import model_serializer, model_validator

from qat.model.device_models import (
    PhysicalBasebandData,
    PhysicalChannelData,
    PulseChannelData,
    QuantumComponentData,
    QuantumDeviceData,
    QubitData,
)
from qat.purr.compiler.devices import ChannelType
from qat.purr.compiler.instructions import QuantumInstruction
from qat.utils.pydantic import WarnOnExtraFieldsModel


class QuantumComponent(WarnOnExtraFieldsModel):
    data: QuantumComponentData

    @property
    def id(self):
        return self.data.id

    @property
    def full_id(self):
        return self.data.full_id

    @model_serializer()
    def dump(self):
        return self.data.model_dump()


class PhysicalBaseband(QuantumComponent):
    data: PhysicalBasebandData


class PhysicalChannel(QuantumComponent):
    data: PhysicalChannelData

    baseband: PhysicalBaseband

    model_validator(mode="before")

    def validate_baseband(cls, values):
        baseband_id = values["baseband"].id
        baseband_data_id = values["data"]["baseband"].id
        if baseband_id != baseband_data_id:
            raise ValueError(
                f"Mismatch for unique baseband ids, got {baseband_id} and {baseband_data_id} instead."
            )


class PulseChannel(QuantumComponent):
    data: PulseChannelData

    physical_channel: PhysicalChannel
    auxiliary_qubits: Optional[List[Qubit]] = []

    @property
    def channel_type(self):
        return self.data.channel_type


class QuantumDevice(QuantumComponent):
    data: QuantumDeviceData

    pulse_channels: Dict[str, PulseChannel]
    physical_channel: PhysicalChannel
    measure_device: Optional[Resonator] = None

    def get_pulse_channel(
        self,
        channel_type: ChannelType,
        auxiliary_qubits: List[QubitData] = None,
    ) -> PulseChannel:
        for pulse_channel in self.pulse_channels.values():
            if pulse_channel.channel_type == channel_type:
                if auxiliary_qubits:
                    if auxiliary_qubits == pulse_channel.auxiliary_qubits:
                        return pulse_channel
                    else:
                        raise KeyError(
                            f"Pulse channel with channel type '{channel_type}' and auxiliary qubits '{auxiliary_qubits}' not found on device '{self.id}'."
                        )

                else:
                    return pulse_channel

        raise KeyError(
            f"Pulse channel with channel type '{channel_type}' not found on device '{self.id}'."
        )


class Resonator(QuantumDevice):
    measure_device: None = None

    def get_measure_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.measure)

    def get_acquire_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.acquire)


class Qubit(QuantumDevice):
    measure_device: Resonator

    def get_acquire_channel(self) -> PulseChannel:
        return self.measure_device.get_acquire_channel()

    def get_cross_resonance_channel(self, linked_qubits: List[Qubit]) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.cross_resonance, linked_qubits)

    def get_cross_resonance_cancellation_channel(
        self, linked_qubits: List[Qubit]
    ) -> PulseChannel:
        return self.get_pulse_channel(
            ChannelType.cross_resonance_cancellation, linked_qubits
        )

    def get_drive_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.drive)

    def get_freq_shift_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.freq_shift)

    def get_measure_channel(self) -> PulseChannel:
        return self.measure_device.get_measure_channel()

    def get_second_state_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.second_state)

    def get_all_channels(self) -> List[PulseChannel]:
        """
        Returns all channels associated with this qubit, including resonator channel
        and other qubit devices that act as if they are on this object.
        """
        return [
            *self.data.pulse_channels.values(),
            self.get_measure_channel(),
            self.get_acquire_channel(),
        ]

    def X(self) -> QuantumInstruction:
        pass

    def Y(self) -> QuantumInstruction:
        pass

    def Z(self) -> QuantumInstruction:
        pass
