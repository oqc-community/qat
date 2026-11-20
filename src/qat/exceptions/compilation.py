# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Exceptions raised while compiling programs."""


class CompilationError(ValueError):
    """Base class for errors raised while compiling a program in QAT.

    It subclasses :class:`ValueError` so that code which catches :class:`ValueError` from
    compilation still catches it.
    """


class UnsupportedProgramError(CompilationError):
    """Raised when a program is valid, but the target cannot run it.

    For example, when a timing in the program exceeds a limit of the target's hardware.
    """
