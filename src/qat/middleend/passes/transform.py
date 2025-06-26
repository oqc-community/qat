# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import itertools
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import singledispatchmethod

import numpy as np
from compiler_config.config import MetricsType

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import InstructionBuilder
from qat.ir.instructions import (
    Assign,
    Delay,
    EndRepeat,
    GreaterThan,
    Instruction,
    Jump,
    Label,
    LoopCount,
    PhaseReset,
    PhaseShift,
    Plus,
    Repeat,
    Return,
    Variable,
)
from qat.ir.measure import Acquire, MeasureBlock, PostProcessing
from qat.ir.waveforms import Pulse, SquareWaveform
from qat.model.target_data import TargetData
from qat.purr.compiler.instructions import (
    AcquireMode,
    PostProcessType,
    ProcessAxis,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class RepeatTranslation(TransformPass):
    """Transform :class:`Repeat` instructions so that they are replaced with:
    :class:`Variable`, :class:`Assign`, and :class:`Label` instructions at the start,
    and :class:`Assign` and :class:`Jump` instructions at the end."""

    def __init__(self, target_data: TargetData):
        self.target_data = target_data

    def run(self, ir: InstructionBuilder, *args, **kwargs) -> InstructionBuilder:
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""
        handler = RepeatTranslationHandler()
        for inst in ir.instructions:
            handler.run(inst)

        handler.close_repeats()

        if getattr(ir, "passive_reset_time", None) is None:
            ir.passive_reset_time = self.target_data.QUBIT_DATA.passive_reset_time
        ir.instructions = handler.instructions
        return ir


class RepeatTranslationHandler:
    """
    Handler used by PydRepeatTranslation to manage the translation of Repeat and EndRepeat instructions.
    """

    def __init__(self):
        self.instructions: list[Instruction] = []
        self.label_data: OrderedDict[Label, dict[str, Repeat | Variable]] = OrderedDict()

    @singledispatchmethod
    def run(self, instruction):
        """
        Default handler for instructions that do not have a specific repeat translation.
        """
        raise NotImplementedError(f"No repeat translation for {type(instruction)}")

    @run.register(Instruction)
    def _(self, instruction: Instruction):
        self.instructions.append(instruction)

    @run.register(Repeat)
    def _(self, instruction: Repeat):
        label = Label.with_random_name()
        var = Variable(name=label.name + "_count", var_type=LoopCount)
        assign = Assign(name=var.name, value=0)

        self.instructions.extend([var, assign, label])
        self.label_data[label.name] = {"repeat": instruction, "var": var}

    @run.register(EndRepeat)
    def _(self, instruction: EndRepeat):
        if len(self.label_data) < 1:
            raise ValueError("EndRepeat found without associated Repeat.")
        label, data = self.label_data.popitem()
        self.instructions.extend(self._close_repeat(label, data))

    def close_repeats(self):
        for label, data in reversed(self.label_data.items()):
            self.instructions.extend(self._close_repeat(label, data))

    @staticmethod
    def _close_repeat(label: Label, data: dict):
        # TODO: Adjust so that closing is done before finishing instructions such as
        #   returns and postprocessing. COMPILER-452
        repeat = data["repeat"]
        var = data["var"]
        assign = Assign(name=var.name, value=Plus(left=var, right=1))
        jump = Jump(label=label, condition=GreaterThan(left=repeat.repeat_count, right=var))
        return [assign, jump]


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


class InstructionLengthSanitisation(TransformPass):
    """
    Checks if quantum instructions are too long and splits if necessary.
    """

    def __init__(self, target_data: TargetData):
        """
        :param duration_limit: The maximum allowed clock cycles per instruction.

        .. warning::

            The pass will assume that the durations of instructions are sanitised to the
            granularity of the pulse channels. If instructions that do not meet the criteria are
            provided, it might produce incorrect instructions (i.e., instructions that are shorter than
            the clock cycle). This can be enforced using the :class:`InstructionGranularitySanitisation <qat.middleend.passes.transform.InstructionGranularitySanitisation>`
            pass.
        """
        self.duration_limit = target_data.QUBIT_DATA.pulse_duration_max

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """

        new_instructions = []
        for instr in ir.instructions:
            if isinstance(instr, Delay) and instr.duration > self.duration_limit:
                new_instructions.extend(self._batch_delay(instr, self.duration_limit))

            elif (
                isinstance(instr, Pulse)
                and isinstance(instr.waveform, SquareWaveform)
                and instr.waveform.width > self.duration_limit
            ):
                new_instructions.extend(
                    self._batch_square_pulse(instr, self.duration_limit)
                )
            else:
                new_instructions.append(instr)
        ir.instructions = new_instructions
        return ir

    @staticmethod
    def _batch_delay(instruction: Delay, max_duration: float):
        n_instr = int(instruction.duration // max_duration)
        remainder = instruction.duration % max_duration

        batch_instr = []
        for _ in range(n_instr):
            batch_instr.append(Delay(target=instruction.targets, duration=max_duration))

        if remainder > 0.0:
            batch_instr.append(Delay(target=instruction.targets, duration=remainder))

        return batch_instr

    @staticmethod
    def _batch_square_pulse(instruction: Pulse, max_width: float):
        n_instr = int(instruction.waveform.width // max_width)
        remainder = instruction.waveform.width % max_width

        batch_instr = []
        pulse = deepcopy(instruction)
        pulse.update_duration(max_width)
        for _ in range(n_instr):
            batch_instr.append(deepcopy(pulse))

        if remainder > 0.0:
            pulse = deepcopy(pulse)
            pulse.update_duration(remainder)
            batch_instr.append(pulse)

        return batch_instr


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
PydRepeatTranslation = RepeatTranslation
PydInstructionLengthSanitisation = InstructionLengthSanitisation
