# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from tests.unit.experimental.system_data.materialisers.operations.type_expr_checker import (
    is_valid_type_expr,
)


@pytest.mark.parametrize(
    "expr",
    [
        # Bare names
        "int",
        "float",
        "str",
        "bool",
        "complex",
        # Custom domain names (not Python built-ins)
        "qubit_id",
        "mode_id",
        # Dotted attribute
        "typing.Optional",
        # PEP 604 unions
        "float | int",
        "int | str | bool",
        # Generics
        "list[int]",
        "list[qubit_id]",
        "dict[str, int]",
        "list[mode_id | qubit_id]",
        "dict[str, list[int]]",
    ],
)
def test_is_valid_type_expr_accepts_valid_expressions(expr):
    """is_valid_type_expr returns True for syntactically valid type-annotation strings."""
    assert is_valid_type_expr(expr) is True


@pytest.mark.parametrize(
    "expr",
    [
        # Numeric and string literals
        "3",
        "3.14",
        "'hello'",
        # Built-in constants that are NOT Names in Python 3.8+
        "True",
        "False",
        "None",
        # None-unions are also rejected for the same reason
        "int | None",
        "str | None",
        # Arithmetic and other non-union BinOps
        "1 + 2",
        "x + y",
        "x - y",
        "x * y",
        # Function calls
        "foo()",
        "int(x)",
        # List/set/dict displays
        "[int]",
        "{str}",
        # Empty string — SyntaxError
        "",
    ],
)
def test_is_valid_type_expr_rejects_invalid_expressions(expr):
    """is_valid_type_expr returns False for non-type expressions and literals."""
    assert is_valid_type_expr(expr) is False
