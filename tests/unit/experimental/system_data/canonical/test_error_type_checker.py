# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from tests.unit.experimental.system_data.canonical.error_type_checker import (
    is_valid_error_type,
)


@pytest.mark.parametrize(
    "error_type",
    [
        # Built-in exception names — resolved via builtins
        "NotImplementedError",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "Exception",
        "AttributeError",
        "KeyError",
        "OSError",
        # Fully-qualified user-defined exception (importable dotted path)
        "qat.runtime.exceptions.ExecutionError",
    ],
)
def test_is_valid_error_type_accepts_valid_exceptions(error_type):
    """is_valid_error_type returns True for built-in exceptions and importable dotted
    paths."""
    assert is_valid_error_type(error_type) is True


@pytest.mark.parametrize(
    "error_type",
    [
        # Empty string
        "",
        # Numeric strings
        "3",
        # Simple unqualified names that are NOT built-in exceptions
        "ExecutionError",
        "MyCustomError",
        "foo",
        # Non-exception built-ins
        "int",
        "str",
        "print",
        # Non-importable dotted paths
        "nonexistent.module.SomeError",
        # Malformed dotted paths
        "foo..bar",
        ".foo",
        "foo.",
    ],
)
def test_is_valid_error_type_rejects_non_exceptions(error_type):
    """is_valid_error_type returns False for non-exceptions and unresolvable names."""
    assert is_valid_error_type(error_type) is False
