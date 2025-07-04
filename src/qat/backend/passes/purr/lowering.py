# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from itertools import chain

from qat.core.pass_base import LoweringPass
from qat.ir.lowered import PartitionedIR
from qat.middleend.passes.purr.transform import LoopCount
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    GreaterThan,
    InlineResultsProcessing,
    Instruction,
    Jump,
    Label,
    PostProcessing,
    QuantumInstruction,
    Repeat,
    ResultsProcessing,
    Return,
    Synchronize,
    Variable,
)


class PartitionByPulseChannel(LoweringPass):
    """Separates a list of instructions into their respective pulse channels.

    For targets that do not have native support for
    :class:`Synchronize <qat.purr.compiler.instructions.Synchronize>`, we can separate the
    instruction list into many lists of instructions, with each list only containing
    instructions that act on a single channel.

    This pass achieves the same as the :class:`TriagePass <qat.backend.passes.legacy.analysis.TriagePass>`,
    but instead of saving the results in the results manager, returns the partitioned
    instructions. The only reason I've changed the name is to not confuse it with the
    TriagePass :)
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs) -> PartitionedIR:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        """
        shared_instructions = list()
        variables = defaultdict(list)
        partitioned_ir = PartitionedIR()

        if hasattr(ir, "shots") and hasattr(ir, "compiled_shots"):
            partitioned_ir.shots = ir.shots
            partitioned_ir.compiled_shots = ir.compiled_shots

        partitioned_ir.repetition_period = getattr(ir, "repetition_period", None)

        partitioned_ir.passive_reset_time = getattr(ir, "passive_reset_time", None)

        def add_to_all(inst: Instruction) -> None:
            shared_instructions.append(inst)
            for target_list in partitioned_ir.target_map.values():
                target_list.append(inst)
            partitioned_ir.target_map.default_factory = shared_instructions.copy

        repeat_seen = False
        for inst in ir.instructions:
            handled = False
            if isinstance(inst, QuantumInstruction) and not isinstance(
                inst, PostProcessing
            ):
                if isinstance(inst, Synchronize):
                    raise ValueError(
                        f"The `Synchronize` instruction {inst} is not supported by the "
                        "PartitionByPulseChannelPass. Please lower it to `Delay` instructions "
                        "using the `LowerSyncsToDelays` pass"
                    )
                handled = True
                for qt in inst.quantum_targets:
                    partitioned_ir.target_map[qt].append(inst)

            if isinstance(inst, Return):
                partitioned_ir.returns.append(inst)

            elif isinstance(inst, Variable):
                variables[inst.name].append(inst)
                add_to_all(inst)

            elif isinstance(inst, Assign):
                var_refs = variables[inst.name]
                if len(var_refs) == 0:
                    variables[inst.name].append(Variable(inst.name))
                if var_refs[-1].var_type == LoopCount:
                    add_to_all(inst)
                if isinstance(inst.value, list | str):
                    # Only list assigns that are readout/result assignements
                    # TODO: Supper different assigns for result readout vs loop
                    # variables etc.
                    partitioned_ir.assigns.append(inst)

            elif isinstance(inst, Acquire):
                for t in inst.quantum_targets:
                    partitioned_ir.acquire_map[t].append(inst)

            elif isinstance(inst, PostProcessing):
                partitioned_ir.pp_map[inst.output_variable].append(inst)

            elif isinstance(inst, ResultsProcessing):
                partitioned_ir.rp_map[inst.variable] = inst

            elif isinstance(inst, Repeat):
                if repeat_seen:
                    raise ValueError("Multiple Repeat instructions found.")
                if partitioned_ir.shots is None:
                    partitioned_ir.shots = inst.repeat_count
                    partitioned_ir.compiled_shots = inst.repeat_count
                if partitioned_ir.repetition_period is None:
                    partitioned_ir.repetition_period = inst.repetition_period
                if partitioned_ir.passive_reset_time is None:
                    partitioned_ir.passive_reset_time = inst.passive_reset_time
                repeat_seen = True

            elif isinstance(inst, Jump):
                if isinstance(inst.condition, GreaterThan):
                    if repeat_seen:
                        raise ValueError("Multiple Repeat or Jump instructions found.")

                    if partitioned_ir.shots is None:
                        limit = inst.condition.left
                        partitioned_ir.shots = limit
                        partitioned_ir.compiled_shots = limit

                add_to_all(inst)

            elif isinstance(inst, Label):
                add_to_all(inst)
            elif not handled:
                raise TypeError(f"Unexpected Instruction type: {type(inst)}.")

        # Assume that raw acquisitions are experiment results.
        # TODO: separate as ResultsProcessingSanitisation pass. (COMPILER-412)
        acquires = list(chain(*partitioned_ir.acquire_map.values()))
        missing_results = {
            acq.output_variable
            for acq in acquires
            if acq.output_variable not in partitioned_ir.rp_map
        }
        for missing_var in missing_results:
            partitioned_ir.rp_map[missing_var] = ResultsProcessing(
                missing_var, InlineResultsProcessing.Experiment
            )

        return partitioned_ir
