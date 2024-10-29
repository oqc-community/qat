import typing
from typing import Annotated, Dict, List

from pydantic import BaseModel, PlainSerializer

T = typing.TypeVar("T")
SerializeItemToRefs = PlainSerializer(lambda i: i.id, return_type=str, when_used="always")
refstr = str
Ref = Annotated[refstr | T, SerializeItemToRefs]

SerializeDictToRefs = PlainSerializer(
    lambda d: {k: v.id for (k, v) in d.items()}, return_type=dict, when_used="always"
)
RefDict = Annotated[Dict[refstr, refstr] | Dict[refstr, T], SerializeDictToRefs]

SerializeListToRefs = PlainSerializer(
    lambda d: [v.id for v in d], return_type=list, when_used="always"
)
RefList = Annotated[List[refstr] | List[T], SerializeListToRefs]


def _get_field_type_and_container(cls, field):
    field_type = cls.model_fields[field].annotation
    container = typing.get_origin(field_type)
    if container is typing.Union:
        field_type = typing.get_args(field_type)[1]
        container = typing.get_origin(field_type)
    field_cls = None
    if container is list:
        field_cls = typing.get_args(field_type)[0]
    elif container is dict:
        field_cls = typing.get_args(field_type)[1]
    elif container is None:
        field_cls = field_type
    else:
        raise Exception(f"Unknown container {container}, {field_type}")
    return field_cls, container


class HasId(BaseModel):
    id: str

    def populate(self, reference_targets):
        """Populate all references"""
        for field_name in self.model_fields:
            field_cls, container = _get_field_type_and_container(self, field_name)
            field_value = getattr(self, field_name)
            if issubclass(field_cls, HasId):
                if container is dict:
                    new_value = {
                        k: _get_or_create(field_cls, reference_targets, v)
                        for (k, v) in field_value.items()
                    }
                elif container is list:
                    new_value = [
                        _get_or_create(field_cls, reference_targets, d) for d in field_value
                    ]
                elif container is None:
                    new_value = _get_or_create(field_cls, reference_targets, field_value)

                setattr(self, field_name, new_value)


def _get_or_create(cls, context, data: str | dict | HasId):
    if issubclass(type(data), HasId):
        return data
    elif isinstance(data, str):
        return context.get(data)


def _get_reference_targets(model, field):
    field_cls, container = _get_field_type_and_container(model, field)
    if issubclass(field_cls, HasId):
        field_contents = getattr(model, field)
        if container is dict:
            ref_targets = field_contents.values()
        elif container is list:
            ref_targets = field_contents
        elif container is None:
            ref_targets = [field_contents]

        return {target.id: target for target in ref_targets}
    else:
        return {}


class AutoPopulate(BaseModel):
    _reference_targets: dict | None = None

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._reference_targets = self.get_refs()
        self.populate(self._reference_targets)

    def get_refs(self):
        reference_targets = {}

        for field_name in self.model_fields:
            reference_targets.update(_get_reference_targets(self, field_name))

        return reference_targets

    def populate(self, reference_targets):
        """Populate all references"""
        for field_name in self.model_fields:
            field_cls, container = _get_field_type_and_container(self, field_name)
            field_value = getattr(self, field_name)
            if issubclass(field_cls, HasId):
                if container is dict:
                    for item in field_value.values():
                        item.populate(reference_targets)
                elif container is list:
                    for item in field_value:
                        item.populate(reference_targets)
                elif container is None:
                    field_value.populate(reference_targets)
