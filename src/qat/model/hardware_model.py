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
from qat.model.ref import IdDict


class QuantumHardwareModel(AutoPopulate):
    """
    Base class for calibrating our QPU hardware.

    Attributes:
        qubits:
    """

    physical_basebands: IdDict[PhysicalBaseband] = Field(
        allow_mutation=False, default=dict()
    )
    physical_channels: IdDict[PhysicalChannel] = Field(allow_mutation=False, default=dict())
    pulse_channels: IdDict[PulseChannel] = Field(allow_mutation=False, default=dict())
    qubits: IdDict[Qubit] = Field(allow_mutation=False, default=dict())
    resonators: IdDict[Resonator] = Field(allow_mutation=False, default=dict())
