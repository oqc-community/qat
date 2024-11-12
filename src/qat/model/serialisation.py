from __future__ import annotations

import typing
from typing import Annotated, Any

from frozendict import frozendict
from pydantic import BeforeValidator, GetCoreSchemaHandler, PlainSerializer
from pydantic_core import core_schema

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
    dict[ComponentId, ComponentId] | dict[ComponentId, T],
    SerializeDictToRefs,
    BeforeValidator(deserialize_dict_to_refs),
    "RefDict",
]


class _PydanticFrozenDictAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def validate_from_dict(d: dict | frozendict) -> frozendict:
            return frozendict(d)

        frozendict_schema = core_schema.chain_schema(
            [
                handler.generate_schema(dict[*typing.get_args(source_type)]),
                core_schema.no_info_plain_validator_function(deserialize_dict_to_refs),
                core_schema.is_instance_schema(frozendict),
            ]
        )
        return core_schema.json_or_python_schema(
            json_schema=frozendict_schema,
            python_schema=frozendict_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(RefDict),
        )


FrozenRefDict = Annotated[frozendict[ComponentId, T], _PydanticFrozenDictAnnotation]

SerializeListToRefs = PlainSerializer(
    lambda d: [v.model_dump_id() for v in d], return_type=list, when_used="always"
)
RefList = Annotated[list[ComponentId] | list[T], SerializeListToRefs, "RefList"]


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
    dict[ComponentId, T],
    SerializeComponentDict,
    BeforeValidator(deserialize_component_dict),
    "ComponentDict",
]
