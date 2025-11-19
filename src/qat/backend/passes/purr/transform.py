# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import itertools

import numpy as np

from qat.backend.passes.purr.analysis import BindingResult, TriageResult
from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import TransformPass
from qat.core.result_base import PreservedResults, ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import Instruction, Sweep


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
        return ir


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
        scopes: list[tuple[Instruction, Instruction]] = kwargs.get("scopes", [])

        if scopes is None or len(scopes) == 0:
            return PreservedResults.all()

        if any(not isinstance(scope, tuple) or len(scope) != 2 for scope in scopes):
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
