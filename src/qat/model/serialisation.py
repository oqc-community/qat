from __future__ import annotations

import typing
from typing import Annotated, Dict, List

from pydantic import BeforeValidator, PlainSerializer

from qat.model.component import Component, ComponentId

T = typing.TypeVar("T", bound=Component)

SerializeItemToRefs = PlainSerializer(
    lambda o: o.model_dump_id(), return_type=dict, when_used="always"
)
Ref = Annotated[ComponentId | T, SerializeItemToRefs, "Ref"]

SerializeDictToRefs = PlainSerializer(
    lambda d: {k.uuid: v.model_dump_id() for (k, v) in d.items()},
    return_type=dict,
    when_used="always",
)


def deserialize_dict_to_refs(data):
    """Rehydrates Reference Dicts."""
    out = {}
    for k, v in data.items():
        if isinstance(v, (Component, ComponentId)):
            out[k] = v
        elif isinstance(v, dict):
            componentid = ComponentId(**v)
            assert k == componentid.uuid, "uuid key vs value mismatch on RefDict"
            out[componentid] = componentid
        else:
            assert False, f"Unexpected value type {type(v)}."
    return out


RefDict = Annotated[
    Dict[ComponentId, T] | Dict[ComponentId, ComponentId],
    SerializeDictToRefs,
    BeforeValidator(deserialize_dict_to_refs),
    "RefDict",
]

SerializeListToRefs = PlainSerializer(
    lambda d: [v.model_dump_id() for v in d], return_type=list, when_used="always"
)
RefList = Annotated[List[ComponentId] | List[T], SerializeListToRefs, "RefList"]


SerializeComponentDict = PlainSerializer(
    lambda d: {k.uuid: v for (k, v) in d.items()},
    return_type=dict,
    when_used="always",
)


def deserialize_component_dict(data):
    """Rehydrates ComponentDicts."""
    out = {}
    for k, v in data.items():
        if isinstance(k, ComponentId):
            out[k] = v
        elif isinstance(k, str):
            componentid = ComponentId(**v)
            assert k == componentid.uuid, "uuid key vs value mismatch on ComponentDict"
            out[componentid] = v
        else:
            assert False, f"Unexpected value type {type(k)}."
    return out


ComponentDict = Annotated[
    Dict[ComponentId, T],
    SerializeComponentDict,
    BeforeValidator(deserialize_component_dict),
    "ComponentDict",
]
