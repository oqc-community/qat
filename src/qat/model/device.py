from __future__ import annotations

from qat.model.component import Component
from qat.model.serialisation import Ref, RefDict, RefList


class PhysicalBaseband(Component):
    pass


class PhysicalChannel(Component):
    baseband: Ref[PhysicalBaseband]


class PulseChannel(Component):
    some_val: int = -100
    physical_channel: Ref[PhysicalChannel]
    auxiliary_qubits: RefList[Qubit] = []


class QuantumDevice(Component):
    pulse_channels: RefDict[PulseChannel]
    physical_channel: Ref[PhysicalChannel]
    measure_device: Ref[Resonator]


class Resonator(QuantumDevice):
    measure_device: None = None


class Qubit(QuantumDevice):
    measure_device: Ref[Resonator]
