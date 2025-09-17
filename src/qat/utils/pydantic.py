# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from __future__ import annotations

import numbers
import re
from collections.abc import Iterable
from copy import deepcopy
from functools import cached_property
from pydoc import locate
from typing import Annotated, Any, Type, TypeVar, Union, get_args, get_origin

import numpy as np
from frozendict import frozendict
from numpydantic import NDArray, Shape
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    PrivateAttr,
    RootModel,
    computed_field,
    model_validator,
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


class NoExtraFieldsFrozenModel(NoExtraFieldsModel):
    """
    A Pydantic `BaseModel` with the extra constraints:
        #. Assignment of fields after initialisation is checked again.
        #. Extra fields given to the model are not ignored (default behaviour in `BaseModel`),
          but raise an error now.
        # All fields are frozen upon instantiation.
    """

    model_config = ConfigDict(frozen=True)


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


class RehydratableModel(BaseModel):
    @computed_field
    @cached_property
    def object_type(self) -> str:
        """
        Returns the type of the object, which is the class name.
        """
        return self.__class__.__module__ + "." + self.__class__.__name__

    @classmethod
    def _rehydrate_object(cls, data):
        type_str = data.get("object_type", None)
        if type_str is None:
            cls_type = cls
        else:
            # Locate the class using the type string.
            cls_type = locate(type_str)
        if cls_type is None:
            raise ValueError(
                f"Could not locate class for type string '{type_str}'. Ensure it is a valid class path."
            )
        # Create an instance of the located class with the provided data.
        if not issubclass(cls_type, cls):
            raise TypeError(
                f"Located class '{cls_type.__name__}' is not a subclass of '{cls.__name__}'."
            )
        return cls_type(**data)


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
        raise ValueError("Given array elements must be in the interval [0, 1].")
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

VALIDATORS = Union[BeforeValidator, AfterValidator, PlainValidator]


def validate_value(value: V, value_type: Type[V]):
    # Get the validator function if it's an annotated type.
    f_validate = None
    if get_origin(value_type) is Annotated:
        value_type, *metadata = get_args(value_type)
        for item in metadata:
            if isinstance(item, VALIDATORS):
                f_validate = item.func

    if get_origin(value_type) is Union:
        allowed_types = get_args(value_type)
    else:
        allowed_types = (value_type,)

    # Validate if type of value == `V`.
    if not isinstance(value, allowed_types):
        raise TypeError(
            f"Cannot add value {value} of type '{type(value)} to container of type {value_type}'."
        )

    # Validate extra constraints on the value, provided via the annotation.
    if f_validate:
        value = f_validate(value)


K = TypeVar("GeneralKey")
V = TypeVar("GeneralValue")


class PydListBase(RootModel[list[V]]):
    root: list[V] = Field(default_factory=list)
    _value_type: type = PrivateAttr()

    @model_validator(mode="after")
    def container_type(self):
        """
        Validate the type of elements in the list.
        """
        value_type = get_args(self.__class__.model_fields["root"].annotation)[0]
        self._value_type = value_type
        return self

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

    @model_validator(mode="after")
    def validate_root(self):
        for value in self.root:
            validate_value(value, self._value_type)
        return self

    def append(self, value: V):
        validate_value(value, self._value_type)
        self.root.append(value)

    def extend(self, values: Iterable[V]):
        for value in values:
            self.append(value)

    def remove(self, value: V):
        self.root.remove(value)


def _validate_set(value: float | int | str | Iterable | None):
    if isinstance(value, (float, int, str)):
        value = {value}
    elif isinstance(value, (list, ValidatedList, tuple)):
        value = set(value)
    return value


class PydSetBase(RootModel[set[V]]):
    root: set[V] = Field(default_factory=set)
    _value_type: type = PrivateAttr()

    @model_validator(mode="after")
    def container_type(self):
        """
        Validate the types of keys and values in the dictionary.
        """
        value_type = get_args(self.__class__.model_fields["root"].annotation)[0]
        self._value_type = value_type
        return self

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
            raise NotImplementedError(f"Unsupported processing for {self} and {other}.")

    def __sub__(self, other: PydSetBase):
        if isinstance(other, PydSetBase):
            return self.root.__sub__(other.root)
        elif isinstance(other, set):
            return self.root.__sub__(other)
        else:
            raise NotImplementedError(f"Unsupported processing for {self} and {other}.")

    def __and__(self, other: PydSetBase):
        if isinstance(other, PydSetBase):
            return self.root.__and__(other.root)
        elif isinstance(other, set):
            return self.root.__and__(other)
        else:
            raise NotImplementedError(f"Unsupported processing for {self} and {other}.")

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

    root: frozenset[V] = Field(default_factory=frozenset)


class ValidatedSet(PydSetBase):
    """
    A set object that validates the input added after instantiation.
    This way, we are sure that the elements in a set are only of a certain type.
    (FYI: Pydantic containers only validate upon instantiation.)
    """

    @model_validator(mode="after")
    def validate_root(self):
        for value in self.root:
            validate_value(value, self._value_type)
        return self

    def add(self, value: V):
        validate_value(value, self._value_type)
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
    root: dict[K, V] = Field(default_factory=dict)
    _key_type: type = PrivateAttr()
    _value_type: type = PrivateAttr()

    @model_validator(mode="after")
    def container_types(self):
        """
        Validate the types of keys and values in the dictionary.
        """
        key_type, value_type = get_args(self.__class__.model_fields["root"].annotation)
        self._key_type = key_type
        self._value_type = value_type
        return self

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

    root: pyd_frozendict[K, V] = Field(default_factory=frozendict)


class ValidatedDict(PydDictBase):
    """
    A dict object that validates the input added after instantiation.
    This way, we are sure that the elements in a dict are only of a certain type.
    (FYI: Pydantic containers only validate upon instantiation.)
    """

    @model_validator(mode="after")
    def validate_root(self):
        for key, value in self.root.items():
            validate_value(key, self._key_type)
            validate_value(value, self._value_type)
        return self

    def update(self, data: dict[K, V]):
        for key, value in data.items():
            self.__setitem__(key, value)

    def __setitem__(self, key: K, value: V):
        validate_value(key, self._key_type)
        validate_value(value, self._value_type)
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


# Efficient serializing of numeric numpy arrays
def _list_serializer(lst):
    """Lists of complex numbers can be expensive to serialise: by serializing type
    information and its value as a hex, we can have more performant serialization."""
    if isinstance(lst, PydArray):
        return {
            "dtype": lst.value.dtype.name,
            "shape": lst.value.shape,
            "value": lst.value.tobytes().hex(),
        }
    else:
        return {"dtype": lst.dtype.name, "shape": lst.shape, "value": lst.tobytes().hex()}


def _list_validator(lst, ty: type):
    """Reverts the hex value and type information into a numpy array."""
    if isinstance(lst, dict):
        dtype = lst.get("dtype", None)
        dtype = np.dtype(dtype) if dtype is not None else None
        if dtype is not None and not np.issubdtype(dtype, ty):
            return lst

        arr = PydArray(value=np.frombuffer(bytearray.fromhex(lst["value"]), dtype=dtype))
        arr.value = arr.value.reshape(lst["shape"])
        return arr

    if isinstance(lst, list | np.ndarray):
        # catches wrong types to allow e.g. FloatNDArray | ComplexNDArray
        try:
            return PydArray(value=np.asarray(lst, dtype=ty))
        except TypeError:
            return lst
    return lst


def get_annotated_array(ty: type):
    """Creates an annotated type for numeric lists with Pydantic serializers and validators
    for efficient serialization."""
    return Annotated[
        PydArray,
        BeforeValidator(lambda x: _list_validator(x, ty)),
        PlainSerializer(_list_serializer),
    ]


class PydArray(NoExtraFieldsModel, np.lib.mixins.NDArrayOperatorsMixin):
    """
    Subclass a NumPy mixin that auto-adds Python operator methods
    (like +, -, *, etc.) in terms of NumPy ufuncs. This class
    still controls behavior via `__array_ufunc__` below.
    """

    value: NDArray[Shape["*, ..."], int | float | complex | bool]  # noqa: F722
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __init__(self, *args, **kwargs):
        if args:
            if len(args) != 1:
                raise TypeError(
                    f"{type(self).__name__} accepts at most 1 positional argument ('value'), got {len(args)}."
                )

            if "value" in kwargs:
                raise TypeError("Pass either a positional value or 'value=', not both")

            kwargs["value"] = args[0]

        kwargs["value"] = (
            np.asarray(kwargs["value"])
            if (isinstance(lst := kwargs.get("value", []), list))
            else lst
        )
        super().__init__(**kwargs)

    def __array__(self, dtype=None):
        arr = self.value
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)

    @staticmethod
    def _wrap(x):
        """Wrap ndarrays as PydArray; leave scalars/others as-is."""
        return PydArray(x) if isinstance(x, np.ndarray) else x

    def __array_function__(self, func, types, args, kwargs):
        # Only handle if any argument is of PydArray type
        if not any(issubclass(t, PydArray) for t in types):
            return NotImplemented

        # Unwrap all PydArray to ndarrays
        def unwrap(a):
            return a.value if isinstance(a, PydArray) else a

        uargs = tuple(unwrap(a) for a in args)
        ukwargs = {k: unwrap(v) for k, v in (kwargs or {}).items()}

        out = func(*uargs, **ukwargs)
        if isinstance(out, tuple):
            return tuple(self._wrap(x) for x in out)
        return self._wrap(out)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Implements special methods for almost all of Python's built-in operators.

        Implementation inspired by
        https://numpy.org/doc/2.2/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html
        """
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Use PydArray instead of type(self)
            # for isinstance to allow subclasses that don't
            # override __array_ufunc__ to handle `PydArray` objects.
            if not isinstance(x, (np.ndarray, numbers.Number, PydArray)):
                return NotImplemented
        # Defer to the implementation of the ufunc
        # on unwrapped values.
        inputs = tuple(x.value if isinstance(x, PydArray) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(x.value if isinstance(x, PydArray) else x for x in out)

        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # ufunc.at performs in-place updates and returns None by design
            return None
        else:
            # one return value (default)
            return self._wrap(result)

    # Custom equality and inequality operators to improve compatibility of
    # Pydantic with empty arrays.
    def __eq__(self, other):
        if isinstance(other, PydArray):
            return np.array_equal(self.value, other.value)
        elif isinstance(other, np.ndarray):
            return np.array_equal(self.value, other)

        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # Other niceties from the ndarray interface we'd like to support for PydArrays.
    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying ndarray.
        This will handle .shape, .size, .reshape, etc. automatically.
        """
        attr = getattr(self.value, name)
        # If the attribute is callable (e.g. reshape), wrap its return value
        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                return self._wrap(result)

            return wrapper
        return attr

    def __getitem__(self, index):
        return self._wrap(self.value[index])

    def __setitem__(self, index, value):
        self.value[index] = value.value if isinstance(value, PydArray) else value

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        for x in self.value:
            yield self._wrap(x)

    def __repr__(self):
        return "%s(%r, dtype=%r)" % (type(self).__name__, self.value, self.dtype)


IntNDArray = get_annotated_array(int)
FloatNDArray = get_annotated_array(float)
ComplexNDArray = get_annotated_array(complex)
