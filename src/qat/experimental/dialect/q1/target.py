# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from dataclasses import dataclass
from typing import IO

from xdsl.backend.assembly_printer import AssemblyPrintable, AssemblyPrinter
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Region
from xdsl.utils.target import Target


def emit_program(region: Region, output: IO[str]) -> None:
    """Emits Q1 assembly for all ops in a region.

    :param region: Region containing Q1 operations.
    :param output: Text stream receiving the printed assembly.
    """

    printer = AssemblyPrinter(stream=output)
    for op in region.walk():
        if not isinstance(op, AssemblyPrintable):
            raise TypeError(f"Expected AssemblyPrintable op, got {type(op).__name__}")
        op.print_assembly(printer)


@dataclass(frozen=True)
class Q1asmTarget(Target):
    name = "q1asm"

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        """Emits a Q1 module to Q1 assembly.

        :param ctx: xDSL context for the emission target.
        :param module: Module containing Q1 dialect operations.
        :param output: Text stream receiving the printed assembly.
        """

        emit_program(module.body, output)
