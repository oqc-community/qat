# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from dataclasses import dataclass
from io import StringIO
from typing import IO

from xdsl.backend.assembly_printer import AssemblyPrinter
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.target import Target


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    """Prints a Q1 module as Q1 assembly.

    :param module: Module containing Q1 operations.
    :param output: Text stream receiving the printed assembly.
    """

    printer = AssemblyPrinter(stream=output)
    printer.print_module(module)


def q1_code(module: ModuleOp) -> str:
    """Returns the Q1 assembly text for a Q1 module.

    :param module: Module containing Q1 dialect operations.
    :returns: Rendered Q1 assembly for the module.
    """

    stream = StringIO()
    print_assembly(module, stream)
    return stream.getvalue()


@dataclass(frozen=True)
class Q1asmTarget(Target):
    name = "q1asm"

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        """Emits a Q1 module to Q1 assembly.

        :param ctx: xDSL context for the emission target.
        :param module: Module containing Q1 dialect operations.
        :param output: Text stream receiving the printed assembly.
        """

        print_assembly(module, output)
