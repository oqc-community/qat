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

VALIDATORS = BeforeValidator | AfterValidator | PlainValidator


K = TypeVar("GeneralKey")
V = TypeVar("GeneralValue")


class PydValidatedBase(RootModel):
    """
    Base class for validated Pydantic containers.
    """

    @model_validator(mode="after")
    def validation_setup(self):
        """
        Setup validation for the container.
        """
        annotation_args = get_args(self.__class__.model_fields["root"].annotation)
        value_type = annotation_args[0]
        annotation_type, allowed_types, validation_funcs = self._determine_validation_info(
            value_type
        )
        self._value_type = annotation_type
        self._value_types = allowed_types
        self._value_validators = validation_funcs

        return self

    @staticmethod
    def _determine_validation_info(annotation_type):
        validation_funcs = []
        if get_origin(annotation_type) is Annotated:
            annotation_type, *metadata = get_args(annotation_type)
            for item in metadata:
                if isinstance(item, VALIDATORS):
                    validation_funcs.append(item.func)
        if get_origin(annotation_type) is Union:
            allowed_types = get_args(annotation_type)
        else:
            allowed_types = (annotation_type,)
        return annotation_type, allowed_types, validation_funcs

    def validate_value(self, value: V):
        if not isinstance(value, self._value_types):
            raise TypeError(
                f"Cannot add value {value} of type '{type(value)}' to container of type {self._value_type}."
            )
        for validator in self._value_validators:
            value = validator(value)
        return value


class PydListBase(RootModel[list[V]]):
    root: list[V] = Field(default_factory=list)

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


class ValidatedList(PydListBase, PydValidatedBase):
    """
    A list object that validates the input appended/extended after instantiation.
    This way, we are sure that the elements in a list are only of a certain type.
    Pydantic containers only validate upon instantiation, not when modifying the
    container.
    """

    def append(self, value: V):
        self.validate_value(value)
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


class ValidatedSet(PydSetBase, PydValidatedBase):
    """
    A set object that validates the input added after instantiation.
    This way, we are sure that the elements in a set are only of a certain type.
    Pydantic containers only validate upon instantiation, not when modifying the
    container.
    """

    def add(self, value: V):
        self.validate_value(value)
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


class ValidatedDict(PydDictBase, PydValidatedBase):
    """
    A dict object that validates the input added after instantiation.
    This way, we are sure that the elements in a dict are only of a certain type.
    Pydantic containers only validate upon instantiation, not when modifying the
    container.
    """

    @model_validator(mode="after")
    def validation_setup(self):
        """
        Validate the types of keys and values in the dictionary.
        """
        annotation_args = get_args(self.__class__.model_fields["root"].annotation)
        key_type = annotation_args[0]
        annotation_type, allowed_types, validation_funcs = self._determine_validation_info(
            key_type
        )
        self._key_type = annotation_type
        self._key_types = allowed_types
        self._key_validators = validation_funcs
        value_type = annotation_args[1]
        annotation_type, allowed_types, validation_funcs = self._determine_validation_info(
            value_type
        )
        self._value_type = annotation_type
        self._value_types = allowed_types
        self._value_validators = validation_funcs
        return self

    def validate_key(self, key: K):
        if not isinstance(key, self._key_types):
            raise TypeError(
                f"Cannot add key {key} of type '{type(key)}' to container of type {self._key_type}."
            )
        for validator in self._key_validators:
            key = validator(key)
        return key

    def update(self, data: dict[K, V]):
        for key, value in data.items():
            self.__setitem__(key, value)

    def __setitem__(self, key: K, value: V):
        self.validate_key(key)
        self.validate_value(value)
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
def _validate_value(
    value: str | list | np.ndarray,
    implied_type: np.dtype,
    required_type: np.dtype,
):
    """
    If value is a string: Reverts the hex value and type information into a numpy array.
    If value is a list or a numpy array: Make a numpy array and validate its type against ty
    """

    if isinstance(value, str):
        value = np.frombuffer(bytearray.fromhex(value), dtype=implied_type)
    elif isinstance(value, (list, np.ndarray)):
        value = np.asarray(value, dtype=implied_type)
    else:
        raise ValueError(
            f"Expected value to be {str | list | np.ndarray}, got {type(value)}"
        )

    if np.can_cast(value.dtype, required_type):
        return value.astype(required_type)

    try:
        if np.all((cast_value := value.astype(required_type)) == value):
            return cast_value
    except Exception as e:
        raise ValueError(f"""Cannot cast {value.dtype} to {required_type}\n{str(e)}""")


def _serializer(obj, ty: type):
    """Lists of complex numbers can be expensive to serialise: by serializing type
    information and its value as a hex, we can have more performant serialization."""

    dtype = np.dtype(ty)
    if isinstance(obj, PydArray):
        obj.value = obj.value.astype(dtype)
        return {
            "dtype": obj.value.dtype.name,
            "shape": obj.value.shape,
            "value": obj.value.tobytes().hex(),
        }
    elif isinstance(obj, np.ndarray):
        obj = obj.astype(dtype)
        return {"dtype": obj.dtype.name, "shape": obj.shape, "value": obj.tobytes().hex()}
    else:
        raise ValueError(f"Expected obj to be {PydArray | np.ndarray}, got {type(obj)}")


def _validator(payload, ty: type):
    """
    Plain validator function for annotated numpy array types.
    The payload is assumed to be consumed as a string, a numpy array, or a dictionary
    and a PydArray is created from it.
    """

    if isinstance(payload, PydArray):
        return payload
    elif isinstance(payload, np.ndarray):
        payload = _validate_value(
            payload, implied_type=payload.dtype, required_type=np.dtype(ty)
        )
        payload = {"dtype": payload.dtype, "shape": payload.shape, "value": payload}
    elif isinstance(payload, (str, list)):
        payload = _validate_value(
            payload, implied_type=np.dtype(ty), required_type=np.dtype(ty)
        )
        payload = {"dtype": payload.dtype, "shape": payload.shape, "value": payload}
    elif isinstance(payload, dict):
        dtype = payload.get("dtype", None)
        dtype = np.dtype(dtype) if dtype is not None else None
        shape = payload.get("shape", (0,))
        shape = tuple(shape)
        value = payload.get("value", [])
        value = _validate_value(value, implied_type=dtype, required_type=np.dtype(ty))
        payload = {"dtype": dtype, "shape": shape, "value": value}
    else:
        raise ValueError(
            f"Expected payload to be {str | list | np.ndarray | dict} got {type(payload)}"
        )

    value = payload["value"]
    shape = payload["shape"]
    return value.reshape(shape)


def annotate_pyd_array(ty: type):
    """Creates an annotated type for numeric lists with Pydantic serializers and validators
    for efficient serialization."""
    return Annotated[
        PydArray,  # TODO - Encode custom data type COMPILER-769
        PlainValidator(lambda x: _validator(x, ty)),
        PlainSerializer(lambda x: _serializer(x, ty)),
    ]


class PydArray(NoExtraFieldsModel, np.lib.mixins.NDArrayOperatorsMixin):
    """
    A data class wrapper to handle the information needed to completely describe a numpy array:
        + Value is the numpy array itself
        + Shape is the shape of the numpy array
        + dtype is the data (implied) type of the numpy array

    Through annotations, this allows creation of metadata classes on top of PydArray that describe
    how to (de)serialise a (blob) PydArray object according to some (required) type. See _validator()
    and _serializer().

    Enlisting the array data as (nested) list(s) is not optimal, and this class renders (de)serialisation
    fast.

    This class also mixes in NDArrayOperatorsMixin, which defines Python arithmetic operators in terms of
    NumPy ufuncs. This does NOT guarantee full interoperability with NumPy but allows the option to extend
    this support in the future. For now, the sole purpose of this class is ONLY to be a data class wrapper
    and mediator for (de)serialisation purposes.
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

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        dtype = dtype or self.dtype
        return self.value.astype(dtype=dtype, copy=copy)

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


IntNDArray = annotate_pyd_array(int)
FloatNDArray = annotate_pyd_array(float)
ComplexNDArray = annotate_pyd_array(complex)

CalibratableUnitInterval2x2Array = Annotated[
    PydArray,
    AfterValidator(validate_calibratable_unit_interval_array),
    PlainSerializer(lambda x: _serializer(x, float)),
]
