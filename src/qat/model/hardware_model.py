from __future__ import annotations

from copy import deepcopy
from typing import Optional

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_extra_types.semantic_version import SemanticVersion
from semver import Version

from qat.model.device import Qubit
from qat.model.hardware_base import CalibratableUnitInterval, FrozenDict, FrozenSet, QubitId
from qat.utils.pydantic import WarnOnExtraFieldsModel

VERSION = Version(0, 0, 1)


class LogicalHardwareModel(WarnOnExtraFieldsModel):
    """Models a hardware with a given connectivity.

    :param version: Semantic version of the hardware model.
    :param logical_connectivity: Connectivity of the qubits in the hardware model.
    """

    version: SemanticVersion = Field(frozen=True, repr=False, default=VERSION)
    logical_connectivity: FrozenDict[QubitId, FrozenSet[QubitId]]

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
    :param physical_connectivity_quality: Quality of the connections between the qubits.
    :param logical_connectivity: The connectivities of the qubits used for compilation,
                    which is equal to `physical_connectivity` or a subset thereof.
    """

    qubits: FrozenDict[QubitId, Qubit]
    physical_connectivity: FrozenDict[QubitId, FrozenSet[QubitId]] = Field(frozen=True)
    logical_connectivity: Optional[FrozenDict[QubitId, FrozenSet[QubitId]]] = Field(
        default=None
    )
    physical_connectivity_quality: FrozenDict[
        tuple[QubitId, QubitId], CalibratableUnitInterval
    ]

    @model_validator(mode="before")
    def default_logical_connectivity(cls, data):
        if not data.get("logical_connectivity", None):
            logical_connectivity = deepcopy(data["physical_connectivity"])
            data["logical_connectivity"] = logical_connectivity

        return data

    @model_validator(mode="before")
    def default_physical_connectivity_quality(cls, data):
        if not data.get("physical_connectivity_quality", None):
            physical_connectivity_quality = {}

            for q1_index, connected_qubits in data["physical_connectivity"].items():
                for q2_index in connected_qubits:
                    physical_connectivity_quality[(q1_index, q2_index)] = 1.0

            data["physical_connectivity_quality"] = physical_connectivity_quality

        return data

    @model_validator(mode="after")
    def validate_connectivity(self):
        for qubit_index in self.logical_connectivity:
            if (
                not self.logical_connectivity[qubit_index]
                <= self.physical_connectivity[qubit_index]
            ):
                raise ValueError(
                    "Logical connectivity must be a subgraph of the physical connectivity."
                )

        return self

    @model_validator(mode="after")
    def validate_connectivity_quality(self):
        for q1_index, q2_index in self.physical_connectivity_quality:
            if not self.physical_connectivity.get(q1_index, None):
                raise ValidationError(
                    f"The coupling ({q1_index}, {q2_index}) is not present in `physical_connectivity`."
                )

        for q1_index, connected_qubits in self.physical_connectivity.items():
            for q2_index in connected_qubits:
                if (q1_index, q2_index) not in self.physical_connectivity_quality:
                    raise ValidationError(
                        f"The coupling ({q1_index}, {q2_index}) is not present in `physical_connectivity_quality`."
                    )

        return self

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
    def is_calibrated(self) -> bool:
        for qubit in self.qubits.values():
            if not qubit.is_calibrated:
                return False
        return True

    @property
    def number_of_qubits(self) -> int:
        return len(self.qubits)

    def qubit_with_index(self, index: int | QubitId) -> Qubit:
        return self.qubits[index]
