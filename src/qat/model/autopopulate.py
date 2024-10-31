from __future__ import annotations

import typing

from pydantic import BaseModel, ValidationError

from .component import Component, get_reftype


class AutoPopulate(BaseModel):
    _reference_targets: dict | None = None

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._set_refs()
        self.populate(self._reference_targets)

    def _set_refs(self):
        reference_targets = {}
        for field_name in self.model_fields:
            if get_reftype(self, field_name):
                raise ValidationError("AutoPopulate classes cannot have Ref fields")
            targets = {
                target.to_component_id(): target
                for target in self._get_components(field_name)
            }
            reference_targets.update(targets)

        self._reference_targets = reference_targets

    def _get_components(self, field) -> list[Component]:
        """Return all Components for a given field"""
        field_type = self.model_fields[field].annotation
        container = typing.get_origin(field_type)
        field_contents = getattr(self, field)

        if container is list:
            inner_cls = typing.get_args(field_type)[0]
            if issubclass(inner_cls, Component):
                return field_contents
        elif container is dict:
            inner_cls = typing.get_args(field_type)[1]
            if issubclass(inner_cls, Component):
                return list(field_contents.values())
        elif container is None:
            if issubclass(inner_cls, Component):
                return [field_contents]
        else:
            raise Exception(f"Unknown container {container}, {field_type}")
        return []

    def populate(self, reference_targets):
        """Populate all references"""
        for field_name in self.model_fields:
            for component in self._get_components(field_name):
                component.populate(reference_targets)

    def _deepequals(self, other) -> bool:
        if type(self) != type(other):
            return False

        if self.model_fields != other.model_fields:
            return False

        for field_name in self.model_fields:
            s_components = self._get_components(field_name)
            o_components = other._get_components(field_name)
            if len(s_components) != len(o_components):
                return False

            for s, o in zip(s_components, o_components):
                if not s._deepequals(o):
                    return False

        return True
