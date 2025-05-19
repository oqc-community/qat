# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from numbers import Number

import numpy as np
from compiler_config.config import MetricsType

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    CustomPulse,
    Delay,
    EndRepeat,
    EndSweep,
    Instruction,
    Jump,
    Label,
    PhaseReset,
    PhaseSet,
    Pulse,
    QuantumInstruction,
    Repeat,
    Sweep,
    Synchronize,
)
from qat.purr.utils.logger import get_default_logger
from qat.utils.algorithm import stable_partition

log = get_default_logger()


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


class LowerSyncsToDelays(TransformPass):
    """Lowers :class:`Synchronize` instructions to :class:`Delay` instructions with static
    times.

    Increments through the instruction list, keeping track of the cumulative duration.
    When :class:`Synchronize` instructions are encountered, it is replaced with
    :class:`Delay` instructions with timings calculated from the cumulative durations.

    .. warning::

        Any manipulations of the instruction set that will alter the timeline and occur
        after this pass could invalidate the intention of the :class:`Synchronize`
        instruction.
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs) -> InstructionBuilder:
        durations: dict[str, float] = defaultdict(float)
        new_instructions: list[Instruction] = []

        for inst in ir.instructions:
            if isinstance(inst, (Pulse, Acquire, Delay, CustomPulse)):
                # only increment the durations for meaningful instructions
                pulse_chan_id = inst.quantum_targets[0].partial_id()
                durations[pulse_chan_id] += inst.duration
                new_instructions.append(inst)
            elif isinstance(inst, Synchronize):
                # determine the durations for syncs
                targets = inst.quantum_targets
                current_durations = np.asarray(
                    [durations[target.partial_id()] for target in targets]
                )
                max_duration = np.max(current_durations)
                sync_durations = max_duration - current_durations
                delay_instrs = [
                    Delay(target, sync_durations[i])
                    for i, target in enumerate(targets)
                    if sync_durations[i] > 0.0
                ]
                new_instructions.extend(delay_instrs)
                durations.update({target.partial_id(): max_duration for target in targets})
            else:
                # every other instruction is just added to the list
                new_instructions.append(inst)

        ir.instructions = new_instructions
        return ir


class SquashDelaysOptimisation(TransformPass):
    """Looks for consecutive :class:`Delay` instructions on a pulse channel and squashes
    them into a single instruction.

    Because :class:`Synchronize` instructions across multiple pulse channels are used so
    frequently to ensure pulses play at the correct timings, it means we can have sequences
    of many delays. Reducing the number of delays will simplify timing analysis later in
    the compilation.

    :class:`Delay` instructions commute with phase related instructions, so the only
    instructions that separate delays in a meaningful way are: :class:`Pulse`:,
    :class:`CustomPulse` and :class:`Acquire` instructions. We also need to be careful to
    not squash delays that contain a variable time.
    """

    # There is a strong argument for doing this pass and PhaseOptimisation after the
    # PartitionByPulseChannel pass. Its much easier to track consectuive instructions of a type
    # in a single channel if the instructions are partitioned off. However, ideally these
    # types of optimisations will be applied to compiled gates, so let's just leave this
    # here for now.

    def run(
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        :param met_mgr: The metrics manager to store the number of instructions after
            optimisation.
        """
        delimiter_types = (
            Acquire,
            Assign,
            CustomPulse,
            Delay,
            Jump,
            Label,
            PhaseReset,
            PhaseSet,
            Pulse,
        )
        accumulated_delays: dict[PulseChannel, float] = defaultdict(float)
        instructions: list[Instruction] = []
        for inst in ir.instructions:
            if isinstance(inst, Delay) and isinstance(inst.time, Number):
                for target in inst.quantum_targets:
                    accumulated_delays[target] += inst.time
            elif isinstance(inst, delimiter_types):
                if isinstance(inst, QuantumInstruction):
                    targets = inst.quantum_targets
                else:
                    targets = accumulated_delays.keys()
                for target in targets:
                    if (time := accumulated_delays[target]) > 0.0:
                        instructions.append(Delay(target, time))
                        accumulated_delays[target] = 0.0
                instructions.append(inst)
            else:
                instructions.append(inst)

        for key, val in accumulated_delays.items():
            if val != 0.0:
                instructions.append(Delay(key, val))
        ir.instructions = instructions

        met_mgr.record_metric(MetricsType.OptimizedInstructionCount, len(instructions))
        return ir
