# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import warnings


class ExecutionError(Exception):
    """Base class for exceptions for execution of programs in QAT."""


def __getattr__(name):
    if name == "ExecutionException":
        warnings.warn(
            "ExecutionException is deprecated; use ExecutionError",
            DeprecationWarning,
            stacklevel=2,
        )
        return ExecutionError
    raise AttributeError(name)
