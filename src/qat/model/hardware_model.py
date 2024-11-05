from __future__ import annotations

from pydantic import Field

from qat.model.autopopulate import AutoPopulate
from qat.model.device import (
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    Resonator,
)
from qat.model.serialisation import ComponentDict


class QuantumHardwareModel(AutoPopulate):
    """
    Base class for calibrating our QPU hardware.

    Attributes:
        qubits:
    """

    physical_basebands: ComponentDict[PhysicalBaseband] = Field(frozen=True, default=dict())
    physical_channels: ComponentDict[PhysicalChannel] = Field(frozen=True, default=dict())
    pulse_channels: ComponentDict[PulseChannel] = Field(frozen=True, default=dict())
    qubits: ComponentDict[Qubit] = Field(frozen=True, default=dict())
    resonators: ComponentDict[Resonator] = Field(frozen=True, default=dict())
