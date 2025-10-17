# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from itertools import chain

from compiler_config.config import InlineResultsProcessing

from qat.core.pass_base import LoweringPass
from qat.ir.instruction_builder import InstructionBuilder
from qat.ir.instructions import (
    Assign,
    GreaterThan,
    Instruction,
    Jump,
    Label,
    LoopCount,
    QuantumInstruction,
    Repeat,
    ResultsProcessing,
    Return,
    Synchronize,
    Variable,
)
from qat.ir.lowered import PartitionedIR
from qat.ir.measure import Acquire, PostProcessing


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
        :param ir: The list of instructions stored in a :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        """
        shared_instructions = list()
        variables = defaultdict(list)
        partitioned_ir = PartitionedIR()

        if hasattr(ir, "shots") and hasattr(ir, "compiled_shots"):
            partitioned_ir.shots = ir.shots
            partitioned_ir.compiled_shots = ir.compiled_shots

        def add_to_all(inst: Instruction) -> None:
            shared_instructions.append(inst)
            for target_list in partitioned_ir.target_map.values():
                target_list.append(inst)
            partitioned_ir.target_map.default_factory = shared_instructions.copy

        partitioned_ir.repetition_period = getattr(ir, "repetition_period", None)

        partitioned_ir.passive_reset_time = getattr(ir, "passive_reset_time", None)

        repeat_seen = False
        for instr in ir:
            handled = False
            if isinstance(instr, QuantumInstruction):
                if isinstance(instr, Synchronize):
                    raise ValueError(
                        f"The `Synchronize` instruction {instr} is not supported by the "
                        "PartitionByPulseChannel pass. Please lower it to `Delay` instructions "
                        "using the `LowerSyncsToDelays` pass"
                    )
                handled = True
                for target in instr.targets:
                    partitioned_ir.target_map[target].append(instr)
            if isinstance(instr, Return):
                partitioned_ir.returns.append(instr)

            elif isinstance(instr, Variable):
                variables[instr.name].append(instr)
                add_to_all(instr)

            elif isinstance(instr, Assign):
                var_refs = variables[instr.name]
                if len(var_refs) == 0:
                    variables[instr.name].append(Variable(name=instr.name))
                if var_refs[-1].var_type == LoopCount:
                    add_to_all(instr)
                if isinstance(instr.value, list | str):
                    # Only list assigns that are readout/result assignements
                    # TODO: Supper different assigns for result readout vs loop
                    # variables etc.
                    partitioned_ir.assigns.append(instr)

            elif isinstance(instr, Acquire):
                for target in instr.targets:
                    partitioned_ir.acquire_map[target].append(instr)

            elif isinstance(instr, PostProcessing):
                partitioned_ir.pp_map[instr.output_variable].append(instr)

            elif isinstance(instr, ResultsProcessing):
                partitioned_ir.rp_map[instr.variable] = instr

            elif isinstance(instr, Repeat):
                if repeat_seen:
                    raise ValueError("Multiple Repeat instructions found.")
                if partitioned_ir.shots is None:
                    partitioned_ir.shots = instr.repeat_count
                    partitioned_ir.compiled_shots = instr.repeat_count
                repeat_seen = True

            elif isinstance(instr, Jump):
                if isinstance(instr.condition, GreaterThan):
                    if repeat_seen:
                        raise ValueError("Multiple `Repeat` or `Jump` instructions found.")
                    if partitioned_ir.shots is None:
                        limit = instr.condition.left
                        partitioned_ir.shots = limit
                        partitioned_ir.compiled_shots = limit
                add_to_all(instr)

            elif isinstance(instr, Label):
                add_to_all(instr)
            elif not handled:
                raise TypeError(f"Unexpected Instruction type: {type(instr)}.")

        for pulse_channel_id in partitioned_ir.target_map:
            partitioned_ir.pulse_channels[pulse_channel_id] = ir.get_pulse_channel(
                pulse_channel_id
            )

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
                variable=missing_var, results_processing=InlineResultsProcessing.Experiment
            )

        return partitioned_ir


PydPartitionByPulseChannel = PartitionByPulseChannel
