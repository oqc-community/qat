# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from __future__ import annotations

import re
from collections.abc import Iterable
from copy import deepcopy
from pydoc import locate
from typing import Annotated, Any, TypeVar, get_args

import numpy as np
from frozendict import frozendict
from numpydantic import NDArray, Shape
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    RootModel,
)
from pydantic._internal._model_construction import ModelMetaclass
from pydantic_core import core_schema

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class NoExtraFieldsModel(BaseModel):
    """
    A Pydantic `BaseModel` with the extra constraints:
        #. Assignment of fields after initialisation is checked again.
        #. Extra fields given to the model are not ignored (default behaviour in `BaseModel`),
          but raise an error now.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        extra="forbid",
        ser_json_inf_nan="constants",
    )

    def __str__(self):
        return self.__repr__()


class AllowExtraFieldsModel(BaseModel):
    """
    A Pydantic `BaseModel` with the extra constraints:
        #. Assignment of fields after initialisation is checked again.
        #. Extra fields given to the model are ignored (default behaviour in `BaseModel`).
    """

    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=False, extra="ignore"
    )

    def __str__(self):
        return self.__repr__()


# This base file is used to implement classes/methods common to all hardware.


def validate_non_negative(value: int):
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"Given value {value} must be an int and >=0.")
    return value


NonNegativeInt = Annotated[
    int,
    AfterValidator(validate_non_negative),
]


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


# A float in the unit interval [0, 1].
CalibratableUnitInterval = Annotated[
    float,
    AfterValidator(validate_calibratable_unit_interval),
]


def validate_calibratable_unit_interval_array(array: CalibratableUnitInterval2x2Array):
    if np.any(array > 1) or np.any(array < 0):
        raise ValueError(f"Given array elements must be in the interval [0, 1].")
    return array


CalibratableUnitInterval2x2Array = Annotated[
    NDArray[Shape["2, 2"], float], AfterValidator(validate_calibratable_unit_interval_array)
]


def validate_qubit_coupling(value: QubitCoupling):
    if isinstance(value, str):
        return tuple(map(int, re.findall(r"\d+", value)))
    elif isinstance(value, tuple):
        return value
    else:
        raise TypeError(
            "Invalid type for `QubitCoupling`. Please provide a `str` or `tuple`."
        )


# A qubit coupling represented as (q_i: int, q_j: int).
QubitCoupling = Annotated[
    tuple[NonNegativeInt, NonNegativeInt],
    BeforeValidator(validate_qubit_coupling),
]


def get_validator_from_annotated(annotated_type):
    metadata = get_args(annotated_type)
    for item in metadata:
        if isinstance(item, AfterValidator):
            return item.func
    return None


K = TypeVar("GeneralKey")
V = TypeVar("GeneralValue")


class PydListBase(RootModel[list[V]]):
    root: list[V] = list[V]()

    def __iter__(self):
        return iter(self.root)

    def __eq__(self, other: PydListBase | list):
        if isinstance(other, PydListBase):
            return self.root.__eq__(other.root)
        elif isinstance(other, list):
            return self.root.__eq__(other)
        return False

    def __getitem__(self, i: int):
        return self.root.__getitem__(i)

    def __len__(self):
        return self.root.__len__()

    def __repr__(self):
        return self.root.__repr__()

    def __str__(self):
        return self.root.__str__()

    def __deepcopy__(self, memo):
        copied_root = deepcopy(self.root, memo)
        copied_instance = self.__class__(root=copied_root)
        memo[id(self)] = copied_instance
        return copied_instance


class ValidatedList(PydListBase):
    """
    A list object that validates the input appended/extended after instantiation.
    This way, we are sure that the elements in a list are only of a certain type.
    (FYI: Pydantic containers only validate upon instantiation.)
    """

    def append(self, value: V):
        annotation = self.model_fields["root"].annotation
        value_type = get_args(annotation)[0]
        # Validate if type of value == `V`.
        if not isinstance(value, value_type):
            raise TypeError(
                f"Cannot add value {value} of type '{type(value)} to {annotation}'."
            )
        # Validate extra constraints on the value, provided via the annotation.
        f_validate = get_validator_from_annotated(value_type)
        if f_validate:
            value = f_validate(value)
        self.root.append(value)

    def extend(self, values: Iterable[V]):
        for value in values:
            self.append(value)


class PydSetBase(RootModel[set[V]]):
    root: set[V] = set[V]()

    def __iter__(self):
        return iter(self.root)

    def __eq__(self, other: PydSetBase):
        if isinstance(other, PydSetBase):
            return self.root.__eq__(other.root)
        elif isinstance(other, set):
            return self.root.__eq__(other)
        elif isinstance(other, (list, tuple)):
            return self.root.__eq__(set(other))
        return False

    def __le__(self, other: PydSetBase):
        if isinstance(other, PydSetBase):
            return self.root.issubset(other.root)
        elif isinstance(other, set):
            return self.root.issubset(other)
        else:
            raise NotImplemented(f"Unsupported processing for {self} and {other}.")

    def __sub__(self, other: PydSetBase):
        if isinstance(other, PydSetBase):
            return self.root.__sub__(other.root)
        elif isinstance(other, set):
            return self.root.__sub__(other)
        else:
            raise NotImplemented(f"Unsupported processing for {self} and {other}.")

    def __and__(self, other: PydSetBase):
        if isinstance(other, PydSetBase):
            return self.root.__and__(other.root)
        elif isinstance(other, set):
            return self.root.__and__(other)
        else:
            raise NotImplemented(f"Unsupported processing for {self} and {other}.")

    def __len__(self):
        return self.root.__len__()

    def __repr__(self):
        return self.root.__repr__()

    def __str__(self):
        return self.root.__str__()

    def __deepcopy__(self, memo):
        copied_root = deepcopy(self.root, memo)
        copied_instance = self.__class__(root=copied_root)
        memo[id(self)] = copied_instance
        return copied_instance


class FrozenSet(PydSetBase):
    """
    A Pydantic set that is immutable after instantiation.
    """

    root: frozenset[V]


class ValidatedSet(PydSetBase):
    """
    A set object that validates the input added after instantiation.
    This way, we are sure that the elements in a set are only of a certain type.
    (FYI: Pydantic containers only validate upon instantiation.)
    """

    def add(self, value: V):
        annotation = self.model_fields["root"].annotation

        annotated_value_type = get_args(annotation)[0]
        if len(args := get_args(annotated_value_type)) >= 1:
            # We assume that the arg type of the root is the first element.
            value_type = args[0]
        else:
            value_type = annotated_value_type
        # Validate if type of value == `V`.
        if not isinstance(value, value_type):
            raise TypeError(
                f"Cannot add value {value} of type '{type(value)} to {annotation}'."
            )
        # Validate extra constraints on the value, provided via the annotation.
        f_validate = get_validator_from_annotated(annotated_value_type)
        if f_validate:
            value = f_validate(value)
        self.root.add(value)

    def discard(self, value):
        self.root.discard(value)

    def pop(self):
        self.root.pop()

    def remove(self, value):
        self.root.remove(value)

    def update(self, *sets):
        for s in sets:
            for value in s:
                self.add(value)


class PydDictBase(RootModel[dict[K, V]]):
    root: dict[K, V] = dict[K, V]()

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

    def pop(self, key: Any, *args, **kwargs):
        return self.root.pop(key, *args, **kwargs)

    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return self.root.__len__()

    def __repr__(self):
        return self.root.__repr__()

    def __str__(self):
        return self.root.__str__()

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
    """
    A Pydantic dict that is immutable after instantiation.
    """

    root: pyd_frozendict[K, V]


class ValidatedDict(PydDictBase):
    """
    A dict object that validates the input added after instantiation.
    This way, we are sure that the elements in a dict are only of a certain type.
    (FYI: Pydantic containers only validate upon instantiation.)
    """

    def update(self, data: dict[K, V]):
        for key, value in data.items():
            self.__setitem__(key, value)

    def __setitem__(self, key: K, value: V):
        annotation = self.model_fields["root"].annotation

        annotated_key_type = get_args(annotation)[0]
        if len(args := get_args(annotated_key_type)) >= 1:
            # We assume that the arg type of the root is the first element.
            key_type = args[0]
        else:
            key_type = annotated_key_type

        annotated_value_type = get_args(annotation)[1]
        if len(args := get_args(annotated_value_type)) >= 1:
            # We assume that the arg type of the root is the first element.
            value_type = args[0]
        else:
            value_type = annotated_value_type

        # Validate if type of key == `K` and value == `V`.
        if not isinstance(key, key_type) or not isinstance(value, value_type):
            raise TypeError(
                f"Cannot add pair {key}:{value} of type '{type(key)}:{type(value)}' to {annotation}."
            )

        # Validate extra constraints on the key and value, provided via the annotation.
        f_validate_key = get_validator_from_annotated(annotated_key_type)
        if f_validate_key:
            key = f_validate_key(key)

        f_validate_value = get_validator_from_annotated(annotated_value_type)
        if f_validate_value:
            value = f_validate_value(value)

        self.root[key] = value


QubitId = NonNegativeInt


def validate_waveform_type(value: BaseModel):
    if isinstance(value, str):
        return locate("qat.ir.waveforms." + value)
    elif isinstance(value, ModelMetaclass):
        return value
    else:
        raise TypeError(
            "Invalid type for `WaveformType`. Please provide a `str` or `BaseModel`."
        )


WaveformType = Annotated[
    TypeVar("Waveform"),
    PlainSerializer(lambda wf: wf.__name__, when_used="json-unless-none"),
    BeforeValidator(validate_waveform_type),
]


def find_all_subclasses(cls: Type) -> list[Type]:
    """
    Recursively finds nested subclasses of a class.
    """
    subclasses = cls.__subclasses__()
    for subclass in subclasses:
        subclasses.extend(find_all_subclasses(subclass))
    return subclasses
