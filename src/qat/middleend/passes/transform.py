# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import itertools
from collections import defaultdict

import numpy as np
from compiler_config.config import MetricsType

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import InstructionBuilder
from qat.ir.instructions import PhaseReset, PhaseShift, Repeat, Return
from qat.ir.measure import Acquire, MeasureBlock, PostProcessing
from qat.ir.waveforms import Pulse
from qat.model.target_data import TargetData
from qat.purr.compiler.instructions import (
    AcquireMode,
    PostProcessType,
    ProcessAxis,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class PhaseOptimisation(TransformPass):
    """Iterates through the list of instructions and compresses contiguous
    :class:`PhaseShift` instructions.
    """

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

        accum_phaseshifts: dict[str, float] = defaultdict(float)
        optimized_instructions: list = []
        previous_instruction = None
        for instruction in ir:
            if isinstance(instruction, PhaseShift):
                accum_phaseshifts[instruction.target] += instruction.phase

            elif isinstance(instruction, Pulse):
                target = instruction.target
                if not np.isclose(accum_phaseshifts[target] % (2 * np.pi), 0.0):
                    optimized_instructions.append(
                        PhaseShift(targets=target, phase=accum_phaseshifts.pop(target))
                    )
                optimized_instructions.append(instruction)

            elif isinstance(instruction, PhaseReset):
                for target in instruction.targets:
                    accum_phaseshifts.pop(target, None)

                if (
                    isinstance(previous_instruction, PhaseReset)
                    and previous_instruction.target == instruction.target
                ):
                    continue

                optimized_instructions.append(instruction)

            else:
                optimized_instructions.append(instruction)

            previous_instruction = instruction

        ir.instructions = optimized_instructions
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, len(optimized_instructions)
        )
        return ir


class PostProcessingSanitisation(TransformPass):
    """Checks that the :class:`PostProcessing` instructions that follow an acquisition are
    suitable for the acquisition mode, and removes them if not.

    Extracted from :meth:`qat.purr.backends.live.LiveDeviceEngine.optimize`.
    """

    def run(
        self,
        ir: InstructionBuilder,
        _: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param met_mgr: The metrics manager to store the number of instructions after
            optimisation.
        """

        acquire_mode_output_var_map = {}
        discarded_pp = []
        for instr in ir:
            if isinstance(instr, Acquire):
                if instr.mode == AcquireMode.RAW:
                    raise ValueError(
                        "Invalid acquire mode. The target machine does not support "
                        "a RAW acquire mode."
                    )

                if instr.output_variable:
                    acquire_mode_output_var_map[instr.output_variable] = instr.mode

            elif isinstance(instr, PostProcessing):
                acq_mode = acquire_mode_output_var_map.get(instr.output_variable, None)

                if not acq_mode:
                    log.warning(
                        f"Post-processing output variable {instr.output_variable} is not associated with any acquire output variable."
                    )
                    discarded_pp.append(instr)

                else:
                    if not self._valid_pp(acq_mode, instr):
                        discarded_pp.append(instr)

        ir.instructions = [instr for instr in ir.instructions if instr not in discarded_pp]
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, ir.number_of_instructions
        )
        return ir

    def _valid_pp(self, acquire_mode: AcquireMode, pp: PostProcessing) -> bool:
        """
        Validate whether the post-processing instruction is valid with a given
        acquire mode.
        """

        if acquire_mode == AcquireMode.SCOPE:
            if (
                pp.process_type == PostProcessType.MEAN
                and ProcessAxis.SEQUENCE in pp.axes
                and len(pp.axes) <= 1
            ):
                return False

        elif acquire_mode == AcquireMode.INTEGRATOR:
            if (
                pp.process_type == PostProcessType.DOWN_CONVERT
                and ProcessAxis.TIME in pp.axes
                and len(pp.axes) <= 1
            ):
                return False
            if (
                pp.process_type == PostProcessType.MEAN
                and ProcessAxis.TIME in pp.axes
                and len(pp.axes) <= 1
            ):
                return False

        return True


class ReturnSanitisation(TransformPass):
    """Squashes all :class:`Return` instructions into a single one. Adds a :class:`Return`
    with all acquisitions if none is found."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        returns = [inst for inst in ir if isinstance(inst, Return)]
        measure_blocks = [inst for inst in ir if isinstance(inst, MeasureBlock)]

        if returns:
            unique_variables = set(itertools.chain(*[ret.variables for ret in returns]))
            for ret in returns:
                ir.instructions.remove(ret)
        else:
            # If we do not have an explicit return, imply all results.
            unique_variables = set(
                itertools.chain(*[mb.output_variables for mb in measure_blocks])
            )

        ir.returns(list(unique_variables))
        return ir


class BatchedShots(TransformPass):
    """Determines how shots should be grouped when the total number exceeds that maximum
    allowed.

    The target machine might have an allowed number of shots that can be executed by a
    single execution call. To execute a number of shots greater than this value, shots can
    be batched, with each batch executed by its own "execute" call on the target machine.
    For example, if the maximum number of shots for a target machine is 2000, but you
    required 4000 shots, then this could be done as [2000, 2000] shots.

    Now consider the more complex scenario where  4001 shots are required. Clearly this can
    be done in three batches. While it is tempting to do this in batches of [2000, 2000, 1],
    for some target machines, specification of the number of shots can only be achieved at
    compilation (as opposed to runtime). Batching as described above would result in us
    needing to compile two separate programs. Instead, it makes more sense to batch the
    shots as three lots of 1334 shots, which gives a total of 4002 shots. The extra two
    shots can just be discarded at run time.

    .. warning::

        This pass makes certain assumptions about the IR, including that there is only
        at most one :class:`Repeat` instruction that contributes to the number of readouts,
        and there is at most one readout per shot. It will change the number of shots in
        the :class:`Repeat` instruction to the decided batch size, and store the total
        required shots in the IR, which is less than ideal. It would be nice to save this
        as an analysis result, but an unfortante side effect of the decoupled nature of
        the middle- and back-end is that the results manager might not necessarily be
        passed along.
    """

    def __init__(self, target_data: TargetData):
        """
        :param target_data: Target-related information.
        """
        self.max_shots = target_data.max_shots

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        repeats = [inst for inst in ir if isinstance(inst, Repeat)]
        if len(repeats) == 0:
            return ir

        if len(repeats) > 1:
            log.warning(
                "Multiple repeat instructions found. Using only the first for batching of "
                "shots."
            )

        repeat = repeats[0]
        num_shots = repeat.repeat_count
        num_batches = int(np.ceil(num_shots / self.max_shots))
        shots_per_batch = int(np.ceil(num_shots / num_batches))

        if num_batches > 1:
            log.info(
                f"The number of shots {num_shots} exceeds the maximum allowed on the "
                f"target. Batching as {num_batches} batches of {shots_per_batch} shots."
            )

        repeat.repeat_count = shots_per_batch
        ir.shots = num_shots
        ir.compiled_shots = shots_per_batch
        return ir


PydPhaseOptimisation = PhaseOptimisation
PydPostProcessingSanitisation = PostProcessingSanitisation
PydReturnSanitisation = ReturnSanitisation
PydBatchedShots = BatchedShots
