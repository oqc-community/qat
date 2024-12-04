from __future__ import annotations

from copy import deepcopy
from typing import Annotated, TypeVar, get_args

import numpy as np
from frozendict import frozendict
from pydantic import AfterValidator, RootModel
from pydantic_core import core_schema

# This base file is used to implement classes/methods common to all hardware.


def validate_non_negative(value: int):
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"Given value {value} must be an int and >=0.")
    return value


NonNegativeInt = Annotated[
    int,
    AfterValidator(validate_non_negative),
]

QubitId = NonNegativeInt


def validate_calibratable_positive_float(value: CalibratablePositiveFloat):
    if not np.isnan(value) and value < 0.0:
        raise ValueError(f"Given value {value} must be >=0.")
    return value


CalibratablePositiveFloat = Annotated[
    float,
    AfterValidator(validate_calibratable_positive_float),
]


def validate_calibratable_unit_interval(value: CalibratableUnitInterval):
    if not np.isnan(value):
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Given value {value} must be in the interval [0, 1].")
    return value


CalibratableUnitInterval = Annotated[
    float,
    AfterValidator(validate_calibratable_unit_interval),
]


K = TypeVar("GeneralKey")
V = TypeVar("GeneralValue")


def get_validator_from_annotated(annotated_type):
    metadata = get_args(annotated_type)
    for item in metadata:
        if isinstance(item, AfterValidator):
            return item.func
    return None


class PydSetBase(RootModel[set[V]]):
    root: set[V]

    def __iter__(self):
        return iter(self.root)

    def __eq__(self, other: PydSetBase):
        if isinstance(other, PydSetBase):
            return self.root.__eq__(other.root)
        elif isinstance(other, set):
            return self.root.__eq__(other)
        return False

    def __le__(self, other: PydSetBase):
        if isinstance(other, PydSetBase):
            return self.root.issubset(other.root)
        elif isinstance(other, set):
            return self.root.issubset(other)
        else:
            raise NotImplemented(f"Unsupported processing for {self} and {other}.")

    def __len__(self):
        return self.root.__len__()

    def __repr__(self):
        return self.root.__repr__()

    def __deepcopy__(self, memo):
        copied_root = deepcopy(self.root, memo)
        copied_instance = self.__class__(root=copied_root)
        memo[id(self)] = copied_instance
        return copied_instance


class FrozenSet(PydSetBase):
    root: frozenset[V]


class ValidatedSet(PydSetBase):

    def add(self, value: V):
        annotation = self.model_fields["root"].annotation
        f_validate = get_validator_from_annotated(get_args(annotation)[0])
        if f_validate:
            value = f_validate(value)
        self.root.add(value)

    def discard(self, value):
        self.root.discard(value)

    def pop(self):
        self.root.pop()

    def remove(self, value):
        self.root.remove(value)


class PydDictBase(RootModel[dict[K, V]]):
    root: dict[K, V]

    def get(self, key, default=None):
        return self.root.get(key, default)

    def keys(self):
        return self.root.keys()

    def items(self):
        return self.root.items()

    def values(self):
        return self.root.values()

    def __eq__(self, other: PydDictBase):
        if isinstance(other, PydDictBase):
            return self.root.__eq__(other.root)
        elif isinstance(other, dict):
            return self.root.__eq__(other)
        return False

    def __getitem__(self, key):
        return self.root.get(key, None)

    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return self.root.__len__()

    def __repr__(self):
        return self.root.__repr__()

    def __deepcopy__(self, memo):
        result = {}
        for k, v in self.root.items():
            result[deepcopy(k)] = deepcopy(v)

        result = self.__class__(result)
        memo[id(self)] = result
        return result


class _PydanticFrozenDictAnnotation:
    """
    Helper class since Pydantic `V2` does only offer support for `frozenset`, not `frozendict`.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        def validate_from_dict(d: dict | frozendict) -> frozendict:
            return frozendict(d)

        k, v = get_args(source_type)
        frozendict_schema = core_schema.chain_schema(
            [
                handler.generate_schema(dict[k, v]),
                core_schema.no_info_plain_validator_function(validate_from_dict),
                core_schema.is_instance_schema(frozendict),
            ]
        )
        return core_schema.json_or_python_schema(
            json_schema=frozendict_schema,
            python_schema=frozendict_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(dict),
        )


pyd_frozendict = Annotated[frozendict[K, V], _PydanticFrozenDictAnnotation]


class FrozenDict(PydDictBase):
    root: pyd_frozendict[K, V]


class ValidatedDict(PydDictBase):

    def update(self, data: dict[K, V]):
        for key, value in data.items():
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        annotation = self.model_fields["root"].annotation

        f_validate_key = get_validator_from_annotated(get_args(annotation)[0])
        if f_validate_key:
            key = f_validate_key(key)

        f_validate_value = get_validator_from_annotated(get_args(annotation)[1])
        if f_validate_value:
            value = f_validate_value(value)

        self.root[key] = value
