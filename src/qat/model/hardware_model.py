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

    physical_basebands: ComponentDict[PhysicalBaseband] = Field(
        allow_mutation=False, default=dict()
    )
    physical_channels: ComponentDict[PhysicalChannel] = Field(
        allow_mutation=False, default=dict()
    )
    pulse_channels: ComponentDict[PulseChannel] = Field(
        allow_mutation=False, default=dict()
    )
    qubits: ComponentDict[Qubit] = Field(allow_mutation=False, default=dict())
    resonators: ComponentDict[Resonator] = Field(allow_mutation=False, default=dict())
