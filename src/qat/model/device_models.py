import uuid
from typing import Optional

from pydantic import Field, model_validator

from qat.utils.pydantic import WarnOnExtraFieldsModel


class DeviceIdMixin(WarnOnExtraFieldsModel):
    """
    Base class for any logical object which can act as a target of a quantum action
    - a Qubit or various channels for a simple example.

    Attributes:
        id: The string representation of the quantum component.
    """

    id: Optional[str] = ""
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), allow_mutation=False)

    @model_validator(mode="after")
    def set_id(self):
        if not self.id:
            self.id = self.uuid
        return self

    @model_validator(mode="before")
    def check_uuid_not_in_input(cls, values):
        # Prevent _uuid from being provided by the user.
        if "uuid" in values:
            raise ValueError(
                "'uuid' will be created automatically and is not allowed during instantiation."
            )
        return values

    @property
    def full_id(self):
        return self.uuid

    def __hash__(self):
        return hash(self.uuid)

    def __repr__(self):
        return f"{type(self).__name__}({self.id})"


class PhysicalBasebandId(DeviceIdMixin):
    pass


class PhysicalChannelId(DeviceIdMixin):
    pass


class PulseChannelId(DeviceIdMixin):
    pass


class QuantumDeviceId(DeviceIdMixin):
    pass


class ResonatorId(DeviceIdMixin):
    pass


class QubitId(DeviceIdMixin):
    pass
