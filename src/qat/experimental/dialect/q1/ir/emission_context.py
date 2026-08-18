# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Emission context for Q1 assembly generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass()
class EmissionContext:
    """Controls which optional annotations are included during Q1 assembly emission.

    Pass a customised instance to
    :meth:`~qat.experimental.dialect.q1.ir.abstract_ops.Q1AsmOperation.assembly_line`
    or :func:`~qat.experimental.dialect.q1.target.emit_program` to select which
    annotations appear in the output.  Adding a new emission flag requires only a new
    field here; no other files need to change.

    :param emit_debug_info: When ``True``, debug-info inline comments
        derived from
        :class:`~qat.experimental.dialect.q1.ir.attrs.DebugInfoAttr` are appended to
        each instruction.  Defaults to ``False`` for comment-free output suitable for
        production upload.
    """

    emit_debug_info: bool = False
