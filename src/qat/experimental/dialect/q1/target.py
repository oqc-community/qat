# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from dataclasses import dataclass, field
from typing import IO

from xdsl.backend.assembly_printer import AssemblyPrinter
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Region
from xdsl.utils.target import Target

from qat.experimental.dialect.q1.ir.abstract_ops import Q1AsmOperation
from qat.experimental.dialect.q1.ir.emission_context import EmissionContext


def emit_program(region: Region, output: IO[str], *, emit_debug_info: bool = False):
    """Emits Q1 assembly for all ops in a region.

    :param region: Region containing Q1 operations.
    :param output: Text stream receiving the printed assembly.
    :param emit_debug_info: When ``True``, debug-info inline comments are included in
        the output.  Defaults to ``False`` for comment-free assembly suitable for
        production upload.
    """

    ctx = EmissionContext(emit_debug_info=emit_debug_info)
    printer = AssemblyPrinter(stream=output)
    for op in region.walk():
        if not isinstance(op, Q1AsmOperation):
            raise TypeError(f"Expected Q1AsmOperation op, got {type(op).__name__}")
        line = op.assembly_line(ctx)
        if line is not None:
            printer.print_string(line + "\n")


@dataclass(frozen=True)
class Q1asmTarget(Target):
    name = "q1asm"
    emit_debug_info: bool = field(default=False)

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        """Emits a Q1 module to Q1 assembly.

        :param ctx: xDSL context for the emission target.
        :param module: Module containing Q1 dialect operations.
        :param output: Text stream receiving the printed assembly.
        """

        emit_program(module.body, output, emit_debug_info=self.emit_debug_info)
