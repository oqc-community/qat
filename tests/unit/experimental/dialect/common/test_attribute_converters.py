# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects.builtin import (
    ArrayAttr,
    BoolAttr,
    Float32Type,
    Float64Type,
    FloatAttr,
    IntAttr,
    NoneAttr,
    StringAttr,
    i1,
    i32,
)

from qat.experimental.dialect.common.attribute_converters import (
    as_array,
    as_bool,
    as_float,
    as_int,
    as_int_array,
    as_optional,
    as_string,
)


@pytest.mark.parametrize("value", [None, NoneAttr()])
@pytest.mark.parametrize("converter", [as_float, as_int, as_bool, as_string])
def test_optional_scalar_converters_preserve_absence(converter, value):
    assert isinstance(converter(value), NoneAttr)


@pytest.mark.parametrize("value", [None, NoneAttr()])
@pytest.mark.parametrize(
    ("converter", "kind"),
    [
        (as_float, "Float"),
        (as_int, "Integer"),
        (as_bool, "Boolean"),
        (as_string, "String"),
    ],
)
def test_required_scalar_converters_reject_absence(converter, kind, value):
    with pytest.raises(TypeError) as error:
        converter(value, True)
    assert str(error.value) == f"{kind} attribute value is required"


def test_scalar_converters_construct_expected_attributes():
    converted_float = as_float(2)
    converted_int = as_int(2)
    converted_bool = as_bool(True)
    converted_string = as_string("value")

    assert isinstance(converted_float, FloatAttr)
    assert converted_float.type == Float64Type()
    assert converted_float.value.data == 2.0
    assert isinstance(converted_int, IntAttr)
    assert converted_int.data == 2
    assert converted_bool == BoolAttr(True, i1)
    assert converted_string == StringAttr("value")


@pytest.mark.parametrize(
    ("converter", "value"),
    [
        (as_float, FloatAttr(1.0, Float64Type())),
        (as_int, IntAttr(1)),
        (as_bool, BoolAttr(True, i1)),
        (as_string, StringAttr("value")),
    ],
)
@pytest.mark.parametrize("required", [False, True])
def test_scalar_converters_preserve_expected_attributes(converter, value, required):
    assert converter(value, required) is value


@pytest.mark.parametrize(
    ("converter", "value", "expected"),
    [
        (
            as_float,
            StringAttr("1.0"),
            "Expected f64 FloatAttr or numeric value, got StringAttr",
        ),
        (as_float, FloatAttr(1.0, Float32Type()), "Expected f64 FloatAttr, got f32"),
        (as_int, StringAttr("1"), "Expected IntAttr or integer value, got StringAttr"),
        (
            as_bool,
            StringAttr("true"),
            "Expected i1 IntegerAttr or boolean value, got StringAttr",
        ),
        (as_bool, BoolAttr(True, i32), "Expected i1 IntegerAttr, got i32"),
        (as_string, IntAttr(1), "Expected StringAttr or string value, got IntAttr"),
    ],
)
def test_scalar_converters_reject_wrong_attribute_types(converter, value, expected):
    with pytest.raises(TypeError) as error:
        converter(value)
    assert str(error.value) == expected


@pytest.mark.parametrize(
    ("converter", "value", "expected"),
    [
        (as_float, True, "Expected numeric value, got bool"),
        (as_int, True, "Expected integer value, got bool"),
        (as_int, 1.5, "Expected integer value, got float"),
        (as_bool, 1, "Expected boolean value, got int"),
        (as_string, 1, "Expected string value, got int"),
    ],
)
def test_scalar_converters_reject_wrong_python_types(converter, value, expected):
    with pytest.raises(TypeError) as error:
        converter(value)
    assert str(error.value) == expected


@pytest.mark.parametrize("value", [IntAttr(1), NoneAttr()])
def test_optional_converter_preserves_attributes(value):
    assert as_optional(value) is value


def test_optional_converter_converts_none():
    assert isinstance(as_optional(None), NoneAttr)


def test_array_converter_preserves_array_attributes():
    value = ArrayAttr([StringAttr("value")])

    assert as_array(value) is value


def test_array_converter_constructs_array_attributes():
    assert as_array([StringAttr("first"), StringAttr("second")]) == ArrayAttr(
        [StringAttr("first"), StringAttr("second")]
    )


def test_int_array_converter_preserves_array_attributes():
    value = ArrayAttr([IntAttr(1)])

    assert as_int_array(value) is value


def test_int_array_converter_constructs_mixed_integer_arrays():
    assert as_int_array([IntAttr(1), 2]) == ArrayAttr([IntAttr(1), IntAttr(2)])
