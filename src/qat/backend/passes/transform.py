# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import numpy as np
from more_itertools import partition

from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import (
    EndRepeat,
    EndSweep,
    Repeat,
    Sweep,
)


class ScopeSanitisation(TransformPass):
    """Bubbles up all sweeps and repeats to the beginning of the list.
    Adds delimiter instructions to the repeats and sweeps signifying the end of their scopes.

    Intended for legacy existing builders and the relative order of instructions guarantees
    backwards compatibility.
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        tail, head = partition(
            lambda inst: isinstance(inst, (Sweep, Repeat)), ir.instructions
        )
        tail, head = list(tail), list(head)

        delimiters = [
            EndSweep() if isinstance(inst, Sweep) else EndRepeat() for inst in head
        ]

        ir.instructions = head + tail + delimiters[::-1]
        return ir


class DesugaringPass(TransformPass):
    """Transforms syntactic sugars and implicit effects. Not that it applies here, but a
    classical example of a sugar syntax is the ternary expression
    :code:`"cond ? val1 : val2"` which is nothing more than an if else in disguise.

    The goal of this pass is to desugar any syntactic and semantic sugars. One of these
    sugars is iteration constructs such as :class:`Sweep` and :class:`Repeat`.
    """

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        for inst in ir.instructions:
            if isinstance(inst, Sweep):
                count = len(next(iter(inst.variables.values())))
                iter_name = f"sweep_{hash(inst)}"
                inst.variables[iter_name] = np.linspace(1, count, count, dtype=int).tolist()
        return ir
