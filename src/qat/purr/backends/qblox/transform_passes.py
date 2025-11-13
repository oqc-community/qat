# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import itertools

import numpy as np
from more_itertools import partition

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
from qat.purr.core.pass_base import TransformPass
from qat.purr.core.result_base import ResultManager


class ScopeSanitisation(TransformPass):
    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        Bubbles up all sweeps and repeats to the beginning of the list.
        Adds delimiter instructions to the repeats and sweeps signifying the end of their scopes.

        Intended for legacy existing builders and the relative order of instructions guarantees backwards
        compatibility.
        """

        tail, head = partition(
            lambda inst: isinstance(inst, (Sweep, Repeat)), ir.instructions
        )
        tail, head = list(tail), list(head)

        delimiters = [
            EndSweep() if isinstance(inst, Sweep) else EndRepeat() for inst in head
        ]

        ir.instructions = head + tail + delimiters[::-1]


class RepeatSanitisation(TransformPass):
    def __init__(self, model: QuantumHardwareModel = None):
        self.model = model

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        Fixes repeat instructions if any with default values from the HW model
        Adds a repeat instructions if none is found
        """

        repeats = [inst for inst in ir.instructions if isinstance(inst, Repeat)]
        if repeats:
            for rep in repeats:
                if rep.repeat_count is None:
                    rep.repeat_count = self.model.default_repeat_count
                if rep.repetition_period is None:
                    rep.repetition_period = self.model.default_repetition_period
        else:
            ir.repeat(self.model.default_repeat_count, self.model.default_repetition_period)


class ReturnSanitisation(TransformPass):
    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
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


class DesugaringPass(TransformPass):
    """
    Transforms syntactic sugars and implicit effects. Not that it applies here, but a classical
    example of a sugar syntax is the ternary expression "cond ? val1 : val2" which is nothing more
    than an if else in disguise.

    The goal of this pass is to desugar any syntactic and semantic sugars. One of these sugars
    is iteration constructs such as Sweep and Repeat.
    """

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        for inst in ir.instructions:
            if isinstance(inst, Sweep):
                count = len(next(iter(inst.variables.values())))
                iter_name = f"{hash(inst)}"
                inst.variables[iter_name] = np.linspace(1, count, count, dtype=int).tolist()
