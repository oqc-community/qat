# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Utility for validating ErrorOperationStepData.error_type strings.

A valid error type is either:

1. A built-in Python exception name (resolved via the builtins module and
   verified to be a BaseException subclass), e.g. "NotImplementedError".
2. A fully-qualified dotted path to a user-defined exception class that is
   importable and verifiably a BaseException subclass, e.g.
   "qat.runtime.exceptions.ExecutionError".

This deliberately avoids eval/exec for dynamic name lookup.
Fully-qualified dotted paths are resolved via importlib/getattr.
Simple unqualified names that are not built-in exceptions must be given as
a fully-qualified dotted path to be verifiable.
"""

from __future__ import annotations

import builtins
import importlib


def is_valid_error_type(error_type: str) -> bool:
    """Return True if error_type names a Python exception class.

    Two paths are tried in order:

    1. **Built-in check**: if error_type contains no dot, look it up in
       builtins and confirm it is a BaseException subclass.
    2. **Import check**: if error_type is a dotted path, split on the last
       dot, import the module, and confirm the attribute is a BaseException
       subclass.

    :param error_type: The error_type string to validate.
    :returns: True when the string resolves to an exception class; False
        otherwise (unknown names, import failures, non-exception types, or
        malformed strings).
    """
    if not error_type:
        return False

    if "." not in error_type:
        obj = getattr(builtins, error_type, None)
        return isinstance(obj, type) and issubclass(obj, BaseException)

    module_path, _, class_name = error_type.rpartition(".")
    module_parts = module_path.split(".")
    if (
        not module_path
        or not class_name.isidentifier()
        or any(not part.isidentifier() for part in module_parts)
    ):
        return False
    try:
        module = importlib.import_module(module_path)
        obj = getattr(module, class_name, None)
        return isinstance(obj, type) and issubclass(obj, BaseException)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return False
