from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_extra_types.semantic_version import SemanticVersion
from semver import Version

from qat.model.device import Qubit, QubitId
from qat.utils.pydantic import WarnOnExtraFieldsModel

VERSION = Version(0, 0, 1)


class LogicalHardwareModel(WarnOnExtraFieldsModel):
    version: SemanticVersion = Field(frozen=True, repr=False, default=VERSION)
    topology: dict[QubitId, set[QubitId]]

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

        return True

    def __ne__(self, other: LogicalHardwareModel) -> bool:
        return not self.__eq__(other)


class QuantumHardwareModel(LogicalHardwareModel):
    qubits: dict[QubitId, Qubit]
    constrained_topology: Optional[dict[QubitId, set[QubitId]]] = None

    @model_validator(mode="before")
    def validate_topology(cls, data):
        topology = data["topology"]

        try:
            constrained_topology = data["constrained_topology"]

            for qubit_index in topology:
                if not constrained_topology[qubit_index] <= topology[qubit_index]:
                    raise ValueError(
                        "Constrained topology must be a subgraph of the physical topology."
                    )
        except (KeyError, TypeError):
            data["constrained_topology"] = topology

        return data

    @field_validator("topology")
    def validate_topology_symmetry(cls, topology):
        for node, connected_nodes in topology.items():
            for connected_node in connected_nodes:
                if node not in topology[connected_node]:
                    raise ValueError(
                        f"The topology is not symmetric, node {node} not present in connected nodes of node {connected_node}."
                    )
        return topology

    @field_validator("constrained_topology")
    def validate_constrained_topology_symmetry(cls, constrained_topology):
        if constrained_topology:
            QuantumHardwareModel.validate_topology_symmetry(constrained_topology)
        return constrained_topology

    def __eq__(self, other: QuantumHardwareModel):
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
    def calibrated(self):
        for qubit in self.qubits.values():
            if not qubit.calibrated:
                return False
        return True

    @property
    def number_of_qubits(self):
        return len(self.qubits)

    def qubit_with_index(self, index: int | QubitId):
        return self.qubits[index]
