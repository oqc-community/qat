from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_extra_types.semantic_version import SemanticVersion
from semver import Version

from qat.model.device import Qubit, QubitId
from qat.utils.pydantic import WarnOnExtraFieldsModel

VERSION = Version(0, 0, 1)


class QuantumHardwareModel(WarnOnExtraFieldsModel):

    version: SemanticVersion = Field(frozen=True, repr=False, default=VERSION)
    qubits: dict[QubitId, Qubit]
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

    def __eq__(self, other: QuantumHardwareModel) -> bool:
        if type(self) != type(other):
            return False

        if self.model_fields != other.model_fields:
            return False

        if self.version != other.version:
            return False

        s_qubits = list(getattr(self, "qubits").values())
        o_qubits = list(getattr(other, "qubits").values())
        if len(s_qubits) != len(o_qubits):
            return False

        for s, o in zip(s_qubits, o_qubits):
            if s != o:
                return False

        return True

    def __ne__(self, other: QuantumHardwareModel) -> bool:
        return not self.__eq__(other)

    @property
    def calibrated(self):
        for qubit in self.qubits.values():
            if not qubit.calibrated:
                return False
        return True

    def qubit_with_index(self, index: int | QubitId):
        return self.qubits[index]
