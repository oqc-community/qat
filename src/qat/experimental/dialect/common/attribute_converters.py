# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Scalar converters for experimental dialect attributes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypeAlias, TypeVar, overload

from xdsl.dialects.builtin import (
    ArrayAttr,
    BoolAttr,
    Float64Type,
    FloatAttr,
    IntAttr,
    IntegerAttr,
    NoneAttr,
    StringAttr,
    i1,
)
from xdsl.ir import Attribute

OptionalFloat: TypeAlias = FloatAttr[Float64Type] | NoneAttr
OptionalInt: TypeAlias = IntAttr | NoneAttr
OptionalBool: TypeAlias = BoolAttr | NoneAttr
OptionalString: TypeAlias = StringAttr | NoneAttr

_F64 = Float64Type()
_AttributeT = TypeVar("_AttributeT", bound=Attribute)


@overload
def as_float(
    value: FloatAttr[Float64Type] | int | float,
    required: Literal[True],
) -> FloatAttr[Float64Type]: ...


@overload
def as_float(
    value: FloatAttr[Float64Type] | int | float | NoneAttr | None,
    required: Literal[False] = False,
) -> OptionalFloat: ...


def as_float(
    value: FloatAttr[Float64Type] | int | float | NoneAttr | None,
    required: bool = False,
) -> OptionalFloat:
    """Return an f64 attribute.

    :param value: Number, f64 attribute or absent value.
    :param required: Reject absent values.
    :returns: The converted attribute.
    :raises TypeError: For an invalid or absent required value.
    """

    if value is None or isinstance(value, NoneAttr):
        if required:
            raise TypeError("Float attribute value is required")
        return NoneAttr()
    if isinstance(value, FloatAttr):
        if value.type != _F64:
            raise TypeError(f"Expected f64 FloatAttr, got {value.type}")
        return value
    if isinstance(value, Attribute):
        raise TypeError(
            f"Expected f64 FloatAttr or numeric value, got {type(value).__name__}"
        )
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Expected numeric value, got {type(value).__name__}")
    return FloatAttr(float(value), _F64)


@overload
def as_int(value: IntAttr | int, required: Literal[True]) -> IntAttr: ...


@overload
def as_int(
    value: IntAttr | int | NoneAttr | None,
    required: Literal[False] = False,
) -> OptionalInt: ...


def as_int(
    value: IntAttr | int | NoneAttr | None,
    required: bool = False,
) -> OptionalInt:
    """Return an integer attribute.

    :param value: Integer, integer attribute or absent value.
    :param required: Reject absent values.
    :returns: The converted attribute.
    :raises TypeError: For an invalid or absent required value.
    """

    if value is None or isinstance(value, NoneAttr):
        if required:
            raise TypeError("Integer attribute value is required")
        return NoneAttr()
    if isinstance(value, IntAttr):
        return value
    if isinstance(value, Attribute):
        raise TypeError(f"Expected IntAttr or integer value, got {type(value).__name__}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected integer value, got {type(value).__name__}")
    return IntAttr(value)


@overload
def as_bool(value: BoolAttr | bool, required: Literal[True]) -> BoolAttr: ...


@overload
def as_bool(
    value: BoolAttr | bool | NoneAttr | None,
    required: Literal[False] = False,
) -> OptionalBool: ...


def as_bool(
    value: BoolAttr | bool | NoneAttr | None,
    required: bool = False,
) -> OptionalBool:
    """Return an i1 boolean attribute.

    :param value: Boolean, i1 attribute or absent value.
    :param required: Reject absent values.
    :returns: The converted attribute.
    :raises TypeError: For an invalid or absent required value.
    """

    if value is None or isinstance(value, NoneAttr):
        if required:
            raise TypeError("Boolean attribute value is required")
        return NoneAttr()
    if isinstance(value, IntegerAttr):
        if value.type != i1:
            raise TypeError(f"Expected i1 IntegerAttr, got {value.type}")
        return value
    if isinstance(value, Attribute):
        raise TypeError(
            f"Expected i1 IntegerAttr or boolean value, got {type(value).__name__}"
        )
    if not isinstance(value, bool):
        raise TypeError(f"Expected boolean value, got {type(value).__name__}")
    return BoolAttr(value, i1)


@overload
def as_string(value: StringAttr | str, required: Literal[True]) -> StringAttr: ...


@overload
def as_string(
    value: StringAttr | str | NoneAttr | None,
    required: Literal[False] = False,
) -> OptionalString: ...


def as_string(
    value: StringAttr | str | NoneAttr | None,
    required: bool = False,
) -> OptionalString:
    """Return a string attribute.

    :param value: String, string attribute or absent value.
    :param required: Reject absent values.
    :returns: The converted attribute.
    :raises TypeError: For an invalid or absent required value.
    """

    if value is None or isinstance(value, NoneAttr):
        if required:
            raise TypeError("String attribute value is required")
        return NoneAttr()
    if isinstance(value, StringAttr):
        return value
    if isinstance(value, Attribute):
        raise TypeError(f"Expected StringAttr or string value, got {type(value).__name__}")
    if not isinstance(value, str):
        raise TypeError(f"Expected string value, got {type(value).__name__}")
    return StringAttr(value)


def as_optional(value: _AttributeT | None) -> _AttributeT | NoneAttr:
    """Return an attribute or represent its absence explicitly."""

    return NoneAttr() if value is None else value


def as_array(
    value: ArrayAttr[_AttributeT] | Iterable[_AttributeT],
) -> ArrayAttr[_AttributeT]:
    """Return an array attribute from an existing array or attribute iterable."""

    return value if isinstance(value, ArrayAttr) else ArrayAttr(value)


def as_int_array(
    value: ArrayAttr[IntAttr] | Iterable[IntAttr | int],
) -> ArrayAttr[IntAttr]:
    """Return an integer array attribute from attributes or integer values."""

    if isinstance(value, ArrayAttr):
        return value
    return ArrayAttr(item if isinstance(item, IntAttr) else IntAttr(item) for item in value)
