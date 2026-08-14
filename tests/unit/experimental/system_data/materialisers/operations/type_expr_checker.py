# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Utility for validating :attr:`OperationParameterData.type_expr` strings.

A valid type expression consists only of:

- :class:`ast.Name` — a bare identifier such as ``float`` or ``qubit_id``.
- :class:`ast.Attribute` — a dotted name such as ``typing.Optional``.
- :class:`ast.Subscript` — a generic type such as ``list[int]`` or
  ``dict[str, int]``.
- :class:`ast.BinOp` with :class:`ast.BitOr` — a PEP 604 union such as
  ``float | int``.
- :class:`ast.Tuple` — a comma-separated list of type expressions used as the
  subscript slice of a generic (e.g. the ``str, int`` inside ``dict[str, int]``).

All other nodes — numeric or string literals, ``None``, boolean constants
(``True`` / ``False``), arithmetic, function calls, list displays, etc. — are
rejected. Optionality is expressed via :attr:`OperationParameterData.optional`
rather than ``| None`` unions in the type expression.
"""

from __future__ import annotations

import ast


def _is_type_expr_node(node: ast.expr) -> bool:
    """Return ``True`` if *node* is a valid type-annotation AST node."""
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _is_type_expr_node(node.value)
    if isinstance(node, ast.Subscript):
        return _is_type_expr_node(node.value) and _is_type_expr_node(node.slice)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_type_expr_node(node.left) and _is_type_expr_node(node.right)
    if isinstance(node, ast.Tuple):
        return all(_is_type_expr_node(e) for e in node.elts)
    return False


def is_valid_type_expr(expr: str) -> bool:
    """Return ``True`` if *expr* is a syntactically valid Python type-annotation string.

    :param expr: The :attr:`~qat.experimental.system_data.canonical.schema.OperationParameterData.type_expr`
        string to validate.
    :returns: ``True`` when the string parses as a Python expression whose AST
        contains only nodes permitted in type annotations; ``False`` otherwise.
    """
    try:
        return _is_type_expr_node(ast.parse(expr, mode="eval").body)
    except SyntaxError:
        return False
