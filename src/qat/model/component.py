from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, model_validator


def make_refdict(*items: list[Component]):
    return {i.to_component_id(): i for i in items}


def get_reftype(model, field):
    """gets the Ref(Dict/List) type of a field or returns None otherwise"""
    if not hasattr(model.model_fields[field], "metadata"):
        return None

    metadata = model.model_fields[field].metadata
    if len(metadata) == 0:
        return None

    ref_type = metadata[-1]
    if ref_type in {"Ref", "RefDict", "RefList"}:
        return ref_type
    else:
        return None


class ComponentId(BaseModel):
    """
    Base class for any logical object which can act as a target of a quantum action
    - a Qubit or various channels for a simple example.

    Attributes:
        id: The string representation of the quantum component.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), frozen=True)
    id_type: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        self.id_type = self.__class__.__name__

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other: ComponentId):
        return self.uuid == other.uuid

    def __ne__(self, other: ComponentId):
        return self.uuid != other.uuid

    def to_component_id(self):
        return ComponentId(uuid=self.uuid, id_type=self.id_type)

    def model_dump_id(self):
        return self.to_component_id().model_dump()

    def _deepequals(self, other) -> bool:
        if type(self) != type(other):
            return False

        if self.model_fields != other.model_fields:
            return False

        if self.uuid != other.uuid:
            return False

        for field_name in self.model_fields:
            if getattr(self, field_name) != getattr(other, field_name):
                return False

        return True


class Component(ComponentId):
    _ref_fields = {}

    @model_validator(mode="after")
    def set_ref_fields(self):
        for field_name in self.model_fields:
            reftype = get_reftype(self, field_name)
            if reftype:
                populated = self._is_populated(field_name)
                self._ref_fields[field_name] = {"type": reftype, "populated": populated}
        return self

    def _is_populated(self, field_name):
        field_value = getattr(self, field_name)
        match get_reftype(self, field_name):
            case "Ref":
                return isinstance(field_value, Component)
            case "RefList":
                return all(isinstance(v, Component) for v in field_value)
            case "RefDict":
                return all(isinstance(v, Component) for v in field_value.values())

    def populate(self, reference_targets):
        """Populate all references"""
        for field_name, v in self._ref_fields.items():
            if not v["populated"]:
                field_value = getattr(self, field_name)
                match v["type"]:
                    case "Ref":
                        new_value = reference_targets.get(field_value)
                    case "RefList":
                        new_value = [reference_targets.get(d) for d in field_value]
                    case "RefDict":
                        new_value = {
                            k: reference_targets.get(v) for (k, v) in field_value.items()
                        }
                setattr(self, field_name, new_value)
