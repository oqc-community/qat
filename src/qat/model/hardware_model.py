from __future__ import annotations

from copy import deepcopy
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_extra_types.semantic_version import SemanticVersion
from semver import Version

from qat.model.device import Qubit
from qat.model.hardware_base import QubitId
from qat.utils.pydantic import WarnOnExtraFieldsModel

VERSION = Version(0, 0, 1)


class LogicalHardwareModel(WarnOnExtraFieldsModel):
    """Models a hardware with a given connectivity.

    :param version: Semantic version of the hardware model.
    :param logical_connectivity: Connectivity of the qubits in the hardware model.
    """

    version: SemanticVersion = Field(frozen=True, repr=False, default=VERSION)
    logical_connectivity: dict[QubitId, set[QubitId]]

    @field_validator("version")
    def version_compatibility(version: Version):
        assert (
            version.major == VERSION.major
        ), f"Direct instantiation requires major version compatibility (expected {VERSION.major}.Y.Z, found {version})"
        assert (
            VERSION >= version
        ), f"Latest supported hardware model version {VERSION}, found {version}"
        return VERSION

    def __eq__(self, other: LogicalHardwareModel) -> bool:
        if type(self) != type(other):
            return False

        if self.model_fields != other.model_fields:
            return False

        if self.version != other.version:
            return False

        if self.logical_connectivity != other.logical_connectivity:
            return False

        return True

    def __ne__(self, other: LogicalHardwareModel) -> bool:
        return not self.__eq__(other)


class PhysicalHardwareModel(LogicalHardwareModel):
    """Class for calibrating our QPU hardware.

    :param qubits: The superconducting qubits on the chip.
    :param physical_connectivity: The connectivities of the physical qubits on the QPU.
    :param logical_connectivity: The connectivities of the qubits used for compilation,
                    which is equal to `physical_connectivity` or a subset thereof.
    """

    qubits: dict[QubitId, Qubit]
    physical_connectivity: dict[QubitId, set[QubitId]] = Field(frozen=True)
    logical_connectivity: Optional[dict[QubitId, set[QubitId]]] = Field(default=None)

    @model_validator(mode="before")
    def validate_connectivity(cls, data):
        physical_connectivity = data["physical_connectivity"]
        logical_connectivity = data.get("logical_connectivity", None)

        if not logical_connectivity:
            logical_connectivity = deepcopy(data["physical_connectivity"])
            data["logical_connectivity"] = logical_connectivity

        for qubit_index in logical_connectivity:
            if not logical_connectivity[qubit_index] <= physical_connectivity[qubit_index]:
                raise ValueError(
                    "Logical connectivity must be a subgraph of the physical connectivity."
                )

        return data

    def __eq__(self, other: PhysicalHardwareModel) -> bool:
        base_eq = super().__eq__(other)

        s_qubits = list(self.qubits.values())
        o_qubits = list(other.qubits.values())
        if len(s_qubits) != len(o_qubits):
            return False

        for s, o in zip(s_qubits, o_qubits):
            if s != o:
                return False

        return base_eq

    @property
    def calibrated(self) -> bool:
        for qubit in self.qubits.values():
            if not qubit.is_calibrated:
                return False
        return True

    @property
    def number_of_qubits(self) -> int:
        return len(self.qubits)

    def qubit_with_index(self, index: int | QubitId) -> Qubit:
        return self.qubits[index]
