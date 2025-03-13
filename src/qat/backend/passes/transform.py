# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import itertools

import numpy as np

from qat.core.pass_base import TransformPass, get_hardware_model
from qat.core.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    EndRepeat,
    EndSweep,
    Repeat,
    Return,
    Sweep,
)
from qat.utils.algorithm import stable_partition


class ScopeSanitisation(TransformPass):
    """Bubbles up all sweeps and repeats to the beginning of the list.
    Adds delimiter instructions to the repeats and sweeps signifying the end of their scopes.

    Intended for legacy existing builders and the relative order of instructions guarantees
    backwards compatibility.
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        head, tail = stable_partition(
            ir.instructions, lambda inst: isinstance(inst, (Sweep, Repeat))
        )

        delimiters = [
            EndSweep() if isinstance(inst, Sweep) else EndRepeat() for inst in head
        ]

        ir.instructions = head + tail + delimiters[::-1]
        return ir


class RepeatSanitisation(TransformPass):
    """Fixes repeat instructions if any with default values from the HW model.
    Adds a repeat instructions if none is found."""

    def __init__(self, model: QuantumHardwareModel = None):
        self.model = model

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        model = self.model or get_hardware_model(args, kwargs)

        repeats = [inst for inst in ir.instructions if isinstance(inst, Repeat)]
        if repeats:
            for rep in repeats:
                if rep.repeat_count is None:
                    rep.repeat_count = model.default_repeat_count
                if rep.repetition_period is None:
                    rep.repetition_period = model.default_repetition_period
        else:
            ir.repeat(model.default_repeat_count, model.default_repetition_period)
        return ir


class ReturnSanitisation(TransformPass):
    """Squashes all :class:`Return` instructions into a single one. Adds a :class:`Return`
    with all acquisitions if none is found."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        returns = [inst for inst in ir.instructions if isinstance(inst, Return)]
        acquires = [inst for inst in ir.instructions if isinstance(inst, Acquire)]

        if returns:
            unique_variables = set(itertools.chain(*[ret.variables for ret in returns]))
            for ret in returns:
                ir.instructions.remove(ret)
        else:
            # If we don't have an explicit return, imply all results.
            unique_variables = set(acq.output_variable for acq in acquires)

        ir.returns(list(unique_variables))
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
