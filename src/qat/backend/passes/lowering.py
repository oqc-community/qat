# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain

from qat.core.pass_base import LoweringPass
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    InlineResultsProcessing,
    Instruction,
    PostProcessing,
    QuantumInstruction,
    Repeat,
    ResultsProcessing,
    Return,
    Synchronize,
)


@dataclass
class PartitionedIR:
    """Stores the results of the :class:`PartitionByPulseChannel`."""

    # TODO: When refactoring into Pydantic instructions, we should replace this object with
    # a Pydantic base model that is somewhere in the :mod:`ir <qat.ir>` package. Ideally, we
    # would unify the different IR levels by making a general "QatIR` flexible enough to
    # support them all, but that might be too optimistic. (COMPILER-382)

    target_map: dict[PulseChannel, list[Instruction]] = field(
        default_factory=lambda: defaultdict(list)
    )
    shots: Repeat | None = field(default_factory=lambda: None)
    returns: list[Return] = field(default_factory=list)
    assigns: list[Assign] = field(default_factory=list)
    acquire_map: dict[PulseChannel, list[Acquire]] = field(
        default_factory=lambda: defaultdict(list)
    )
    pp_map: dict[str, list[PostProcessing]] = field(
        default_factory=lambda: defaultdict(list)
    )
    rp_map: dict[str, ResultsProcessing] = field(default_factory=dict)


class PartitionByPulseChannel(LoweringPass):
    """Separates a list of instructions into their respective pulse channels.

    For targets that do not have native support for
    :class:`Synchronize <qat.purr.compiler.instructions.Synchronize>`, we can separate the
    instruction list into many lists of instructions, with each list only containing
    instructions that act on a single channel.

    This pass achieves the same as the :class:`TriagePass <qat.backend.passes.analysis.TriagePass>`,
    but instead of saving the results in the results manager, returns the partitioned
    instructions. The only reason I've changed the name is to not confuse it with the
    TriagePass :)
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs) -> PartitionedIR:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        """
        partitioned_ir = PartitionedIR()
        for inst in ir.instructions:
            if isinstance(inst, QuantumInstruction) and not isinstance(
                inst, PostProcessing
            ):
                if isinstance(inst, Synchronize):
                    raise ValueError(
                        f"The `Synchronize` instruction {inst} is not supported by the "
                        "PartitionByPulseChannelPass. Please lower it to `Delay` instructions "
                        "using the `LowerSyncsToDelays` pass"
                    )
                for qt in inst.quantum_targets:
                    partitioned_ir.target_map[qt].append(inst)

            if isinstance(inst, Return):
                partitioned_ir.returns.append(inst)

            elif isinstance(inst, Assign):
                partitioned_ir.assigns.append(inst)

            elif isinstance(inst, Acquire):
                for t in inst.quantum_targets:
                    partitioned_ir.acquire_map[t].append(inst)

            elif isinstance(inst, PostProcessing):
                partitioned_ir.pp_map[inst.output_variable].append(inst)

            elif isinstance(inst, ResultsProcessing):
                partitioned_ir.rp_map[inst.variable] = inst

            elif isinstance(inst, Repeat):
                if partitioned_ir.shots:
                    raise ValueError("Multiple Repeat instructions found.")
                partitioned_ir.shots = inst

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
