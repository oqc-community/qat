from __future__ import annotations

from pydantic import Field, field_validator
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

VERSION = Version(0, 0, 1)


class QuantumHardwareModel(AutoPopulate):
    """
    Base class for calibrating our QPU hardware.

    Attributes:
        version: The version of this hardware model.
        physical_basebands: The physical basebands that model the LOs.
        physical_channels: The physical channels that carry the pulses.
        pulse_channels: The pulse channels on a particular device.
        qubits: The superconducting qubits on the chip.
        resonators: The resonators on the chip.
    """

    version: SemanticVersion = Field(frozen=True, repr=False, default=VERSION)
    physical_basebands: ComponentDict[PhysicalBaseband] = Field(frozen=True, default=dict())
    physical_channels: ComponentDict[PhysicalChannel] = Field(frozen=True, default=dict())
    pulse_channels: ComponentDict[PulseChannel] = Field(frozen=True, default=dict())
    qubits: ComponentDict[Qubit] = Field(frozen=True, default=dict())
    resonators: ComponentDict[Resonator] = Field(frozen=True, default=dict())

    @field_validator("version")
    def version_compatibility(version):
        assert (
            version.major == VERSION.major
        ), f"Direct instantiation requires major version compatibility (expected {VERSION.major}.Y.Z, found {version})"
        assert (
            VERSION >= version
        ), f"Latest supported hardware model version {VERSION}, found {version}"
        return VERSION
