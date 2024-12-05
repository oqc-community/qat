from __future__ import annotations

from copy import deepcopy
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_extra_types.semantic_version import SemanticVersion
from semver import Version

from qat.model.device import Qubit
from qat.model.hardware_base import CalibratableUnitInterval, FrozenDict, FrozenSet, QubitId
from qat.utils.pydantic import WarnOnExtraFieldsModel

VERSION = Version(0, 0, 1)


class LogicalHardwareModel(WarnOnExtraFieldsModel):
    """Models a hardware with a given connectivity.

    :param version: Semantic version of the hardware model.
    :param logical_connectivity: Connectivity of the qubits in the hardware model (directed graph).
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
    :param physical_connectivity: The connectivities of the physical qubits on the QPU (undirected graph).
    :param logical_connectivity: The connectivities (directed graph) of the qubits used for compilation, which can be a subgraph of `physical_connectivity`.
    :param logical_connectivity_quality: Quality of the connections between the qubits.
    """

    qubits: FrozenDict[QubitId, Qubit]
    physical_connectivity: FrozenDict[QubitId, FrozenSet[QubitId]] = Field(frozen=True)
    logical_connectivity: Optional[FrozenDict[QubitId, FrozenSet[QubitId]]] = Field(
        default=None
    )
    logical_connectivity_quality: FrozenDict[
        tuple[QubitId, QubitId], CalibratableUnitInterval
    ]

    @model_validator(mode="before")
    def default_logical_connectivity(cls, data):
        if not data.get("logical_connectivity", None):
            logical_connectivity = deepcopy(data["physical_connectivity"])
            data["logical_connectivity"] = logical_connectivity

        return data

    @model_validator(mode="before")
    def default_logical_connectivity_quality(cls, data):
        if not data.get("logical_connectivity_quality", None):
            logical_connectivity_quality = {}

            for q1_index, connected_qubits in data["logical_connectivity"].items():
                for q2_index in connected_qubits:
                    logical_connectivity_quality[(q1_index, q2_index)] = 1.0

            data["logical_connectivity_quality"] = logical_connectivity_quality

        return data

    @field_validator("physical_connectivity")
    def validate_physical_connectivity_symmetry(cls, physical_connectivity):
        for q, connected_qs in physical_connectivity.items():
            for connected_q in connected_qs:
                if q not in physical_connectivity[connected_q]:
                    raise ValueError(
                        f"The topology is not symmetric, qubit {q} not present in connected qubits of {connected_q}."
                    )
        return physical_connectivity

    @model_validator(mode="after")
    def validate_connectivity(self):

        # Check if all qubits exist in physical connectivity.
        if len(self.qubits) > 1:  # 1Q-systems do not have any connectivity
            assert (
                self.qubits.keys() == self.physical_connectivity.keys()
            ), f"Inconsistent qubit ids for {self.qubits} and {self.physical_connectivity}."

        # Check if logical connectivity is subset of physical connectivity.
        for qubit_index in self.logical_connectivity:
            if (
                not self.logical_connectivity[qubit_index]
                <= self.physical_connectivity[qubit_index]
            ):
                raise ValueError(
                    "Logical connectivity must be a subgraph of the physical connectivity."
                )

        # Check if qubit cross resonance (cancellation) pulse channels agree with logical connectivity.
        logical_connectivities = {
            (src, tgt) for (src, tgts) in self.logical_connectivity.items() for tgt in tgts
        }
        cross_resonance_edges = {
            (src, chan.auxiliary_qubit)
            for src, qubit in self.qubits.items()
            for chan in qubit.pulse_channels.cross_resonance_channels.values()
        }
        cross_cancellation_edges = {
            (src, chan.auxiliary_qubit)
            for src, qubit in self.qubits.items()
            for chan in qubit.pulse_channels.cross_resonance_cancellation_channels.values()
        }

        assert (
            cross_resonance_edges == cross_cancellation_edges
        ), "Cross resonance channels mismatch cross resonance cancellation channels."
        assert (
            logical_connectivities == cross_resonance_edges
        ), "Cross resonance channels mismatch logical connectivity."
        assert (
            logical_connectivities == cross_cancellation_edges
        ), "Cross resonance cancellation channels mismatch logical connectivity."

        return self

    @model_validator(mode="after")
    def validate_connectivity_quality(self):
        for q1_index, q2_index in self.logical_connectivity_quality:
            if not self.logical_connectivity.get(q1_index, None):
                raise IndexError(
                    f"The coupling ({q1_index}, {q2_index}) is not present in `logical_connectivity`."
                )

        for q1_index, connected_qubits in self.logical_connectivity.items():
            for q2_index in connected_qubits:
                if (q1_index, q2_index) not in self.logical_connectivity_quality:
                    raise IndexError(
                        f"The coupling ({q1_index}, {q2_index}) is not present in `logical_connectivity_quality`."
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
