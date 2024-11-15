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

    @property
    def calibrated(self):
        for qubit in self.qubits.values():
            if not qubit.calibrated:
                return False
        return True
