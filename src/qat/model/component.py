from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import List

from pydantic import BaseModel, Field, model_validator


class ComponentId(BaseModel):
    """
    Base class for any logical object which can act as a target of a quantum action
    - a Qubit or various channels for a simple example.

    Attributes:
        id: The string representation of the quantum component.
    """

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), frozen=True)

    def __init__(self, **data):
        super().__init__(**data)

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other: ComponentId):
        return self.uuid == other.uuid

    def __ne__(self, other: ComponentId):
        return self.uuid != other.uuid

    def __repr__(self):
        return self.uuid

    def to_component_id(self):
        return self

    def model_dump_id(self):
        return self.to_component_id().model_dump()

    def _deepequals(self, other: ComponentId) -> bool:
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

    def __repr__(self):
        return f"{type(self).__name__}({self.uuid})"

    @property
    def full_id(self):
        return self.uuid

    def to_component_id(self):
        return ComponentId(uuid=self.uuid)

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

    @contextmanager
    def temporary_unfreeze(self, field_name: str):
        """Context manager to temporarily unfreeze a field and populate its references."""
        original_frozen = self.model_fields[field_name].frozen
        self.model_fields[field_name].frozen = False

        try:
            yield
        finally:
            self.model_fields[field_name].frozen = original_frozen

    def populate(self, reference_targets):
        """Populate all references."""
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

                with self.temporary_unfreeze(field_name):
                    setattr(self, field_name, new_value)


def get_reftype(model, field):
    """Gets the Ref(Dict/List) type of a field or returns None otherwise."""
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


def make_refdict(*items: List[Component]):
    return {i.to_component_id(): i for i in items}
