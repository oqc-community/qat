# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Mixin for Q1 assembly argument value types."""

from __future__ import annotations


class WithArgument:
    """Mixin for types that can render themselves as a Q1 assembly argument string.

    Apply this mixin to any argument-value type that should participate in
    :meth:`~qat.experimental.dialect.q1.ir.abstract_ops.Q1AsmOperation.print_arg`.
    Concrete classes must implement :meth:`print_arg` to return the assembly text
    for their value.

    Classes that use this mixin: :class:`~qat.experimental.dialect.q1.ir.attrs.LabelAttr`,
    :class:`~qat.experimental.dialect.q1.ir.imm_desc.Q1Imm` (and all immediate subclasses),
    :class:`~qat.experimental.dialect.q1.ir.reg_desc.Q1RegisterType` (and all register
    subclasses).
    """

    def print_arg(self) -> str:
        """Return the textual form of this value as a Q1 assembly argument.

        :returns: Textual representation of this value for assembly emission.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement print_arg()")
