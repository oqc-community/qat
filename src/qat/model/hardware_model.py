from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from pydantic_extra_types.semantic_version import SemanticVersion
from semver import Version

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

    version: ClassVar[SemanticVersion] = Version(0, 0, 1)
    physical_basebands: ComponentDict[PhysicalBaseband] = Field(frozen=True, default=dict())
    physical_channels: ComponentDict[PhysicalChannel] = Field(frozen=True, default=dict())
    pulse_channels: ComponentDict[PulseChannel] = Field(frozen=True, default=dict())
    qubits: ComponentDict[Qubit] = Field(frozen=True, default=dict())
    resonators: ComponentDict[Resonator] = Field(frozen=True, default=dict())
