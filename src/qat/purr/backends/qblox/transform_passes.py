# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import itertools
from typing import List, Tuple

import numpy as np
from more_itertools import partition

from qat.purr.backends.qblox.analysis_passes import BindingResult, TriageResult
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    EndRepeat,
    EndSweep,
    Instruction,
    Repeat,
    Return,
    Sweep,
)
from qat.purr.core.metrics_base import MetricsManager
from qat.purr.core.pass_base import TransformPass
from qat.purr.core.result_base import PreservedResults, ResultManager


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
    """Transforms syntactic sugars and implicit effects.

    The goal of this pass is to desugar any syntactic and semantic sugars. One of these
    sugars is iteration constructs such as :class:`Sweep` and :class:`Repeat`.

    1- Adds an iterator variable for sweep loops. We state that sweep loops implicitly
       declare an iterator variable to control progress of the sweep. See :class:`BindingPass`
    """

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        for inst in ir.instructions:
            if isinstance(inst, Sweep):
                count = len(next(iter(inst.variables.values())))
                iter_name = f"{hash(inst)}"
                inst.variables[iter_name] = np.linspace(1, count, count, dtype=int).tolist()


class ScopePeeling(TransformPass):
    """
    A transform pass to discard scopes assuming they’ve been identified.

    This pass is particularly useful for the code generator as it helps peel away any unwanted scopes.
    In fact, some sweeps are not lowerable and the user may choose to annotate certain sweeps and opt
    for static injection of variables with a switerator-like emitter. These sweep loops will be handled
    by the emitter statically while allowing it to lower nested constructs efficiently.
    """

    def run(
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        scopes: List[Tuple[Instruction, Instruction]] = kwargs.get("scopes", [])

        if scopes is None or len(scopes) == 0:
            return PreservedResults.all()

        if any(not isinstance(scope, Tuple) or len(scope) != 2 for scope in scopes):
            raise ValueError(
                f"Invalid scopes argument. Expected a list of tuples of length 2, got {scopes}"
            )

        if any(s is None or e is None for (s, e) in scopes):
            raise ValueError(
                "Invalid scopes argument. Delimiter instructions must not be none."
            )

        # Results assumed to be correct at this level of analysis
        triage_result = res_mgr.lookup_by_type(TriageResult)
        binding_result = res_mgr.lookup_by_type(BindingResult)

        for target, instructions in triage_result.target_map.items():
            scoping_result = binding_result.scoping_results[target]
            for scope in scopes:
                if scope not in scoping_result.scope2symbols:
                    raise ValueError(
                        f"Could not find scope {scope}, the binding analysis result may have been corrupt"
                    )

        delimiters = set(itertools.chain(*scopes))

        def predicate(inst):
            return inst not in delimiters

        instructions = ir.splice()
        ir.instructions = list(filter(predicate, instructions))

        return PreservedResults.discard(triage_result, binding_result)
