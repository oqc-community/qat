# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import itertools
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import singledispatchmethod

import numpy as np
from compiler_config.config import MetricsType
from more_itertools import partition

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
    PhaseSet,
    PhaseShift,
    Plus,
    QuantumInstruction,
    Repeat,
    Reset,
    Return,
    Synchronize,
    Variable,
)
from qat.ir.measure import Acquire, MeasureBlock, PostProcessing
from qat.ir.waveforms import Pulse, SquareWaveform
from qat.middleend.passes.analysis import ActivePulseChannelResults
from qat.model.device import FreqShiftPulseChannel, PulseChannel, Qubit
from qat.model.hardware_model import PhysicalHardwareModel
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

        handler = PhaseOptimisationHandler()
        for instruction in ir:
            handler.run(instruction)

        ir.instructions = handler.optimized_instructions
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, ir.number_of_instructions
        )
        return ir


class PhaseOptimisationHandler:
    def __init__(self):
        self.accum_phaseshifts: dict[str, float] = defaultdict(float)
        self.optimized_instructions: list = []
        self.previous_phase_instruction_is_phase_set = defaultdict(bool)

    @singledispatchmethod
    def run(self, instruction):
        self.optimized_instructions.append(instruction)

    @run.register(PhaseShift)
    def _(self, instruction: PhaseShift):
        self.accum_phaseshifts[instruction.target] += instruction.phase

    @run.register(PhaseReset)
    def _(self, instruction: PhaseReset):
        for target in instruction.targets:
            self.accum_phaseshifts[target] = 0.0
            self.previous_phase_instruction_is_phase_set[target] = True

    @run.register(PhaseSet)
    def _(self, instruction: PhaseSet):
        self.accum_phaseshifts[instruction.target] = instruction.phase
        self.previous_phase_instruction_is_phase_set[instruction.target] = True

    @run.register(Pulse)
    @run.register(Acquire)
    def _(self, instruction: Pulse | Acquire):
        target = instruction.target
        if not np.isclose(self.accum_phaseshifts[target] % (2 * np.pi), 0.0):
            phase = self.accum_phaseshifts.pop(target)
            if self.previous_phase_instruction_is_phase_set[target]:
                phase_op = PhaseSet(target=target, phase=phase)
            else:
                phase_op = PhaseShift(target=target, phase=phase)
            self.optimized_instructions.append(phase_op)

        self.optimized_instructions.append(instruction)
        self.previous_phase_instruction_is_phase_set[target] = False

    @run.register(Delay)
    @run.register(Synchronize)
    def _(self, instruction: Delay | Synchronize):
        for target in instruction.targets:
            if self.previous_phase_instruction_is_phase_set[target]:
                self.optimized_instructions.append(
                    PhaseSet(target=target, phase=self.accum_phaseshifts.pop(target))
                )
                self.previous_phase_instruction_is_phase_set[target] = False
        self.optimized_instructions.append(instruction)


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


class InactivePulseChannelSanitisation(TransformPass):
    """Removes instructions that act on inactive pulse channels.

    Many channels that aren't actually needed to execute the program contain instructions,
    mainly :class:`Synchronize` instructions and :class:`PhaseShift` instructions which
    can happen when either of the instructions are applied to qubits. To simplify analysis
    and optimisations in later passes, it makes sense to filter these out to reduce the
    overall instruction amount.

    .. note::

        This pass requires results from the
        :class:`ActivePulseChannelAnalysis <qat.middleend.passes.analysis.ActivePulseChannelAnalysis>`
        to be stored in the results manager.
    """

    def run(
        self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs
    ) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        active_channels = [
            target for target in res_mgr.lookup_by_type(ActivePulseChannelResults).targets
        ]
        instructions: list[Instruction] = []
        for inst in ir.instructions:
            if isinstance(inst, Synchronize):
                # inactive channels need stripping from syncs
                targets = [target for target in inst.targets if target in active_channels]
                if len(targets) > 1:
                    inst.targets = targets
                    instructions.append(inst)
            elif isinstance(inst, QuantumInstruction):
                # other instructions need their targets checking
                if inst.target in active_channels:
                    instructions.append(inst)
            else:
                instructions.append(inst)
        ir.instructions = instructions
        return ir


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


class ResetsToDelays(TransformPass):
    """
    Transforms :class:`Reset` operations to :class:`Delay`s.

    Note that the delays do not necessarily agree with the granularity of the underlying target machine.
    This can be enforced using the :class:`InstructionGranularitySanitisation` pass.
    """

    def __init__(self, model: PhysicalHardwareModel, target_data: TargetData):
        """
        :param model: The hardware model that holds calibrated information on the qubits on the QPU.
        :param target_data: Target-related information.
        """
        self.model = model
        self.passive_reset_time = target_data.QUBIT_DATA.passive_reset_time

    def run(
        self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs
    ) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """
        active_pulse_channels: ActivePulseChannelResults = res_mgr.lookup_by_type(
            ActivePulseChannelResults
        )

        new_instructions = []
        for instr in ir:
            if isinstance(instr, Reset):
                qubit = self.model.qubits[instr.qubit_target]
                for pulse_ch in active_pulse_channels.from_qubit(qubit):
                    new_instructions.append(
                        Delay(
                            target=pulse_ch.uuid,
                            duration=self.passive_reset_time,
                        )
                    )

            else:
                new_instructions.append(instr)

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
            Delay,
            Jump,
            Label,
            PhaseReset,
            PhaseSet,
            Pulse,
        )
        accumulated_delays: dict[str, float] = defaultdict(float)
        instructions: list[Instruction] = []
        for inst in ir:
            if isinstance(inst, Delay):
                accumulated_delays[inst.target] += inst.duration

            elif isinstance(inst, delimiter_types):
                if isinstance(inst, QuantumInstruction):
                    targets = {inst.target}
                else:
                    targets = accumulated_delays.keys()
                for target in targets:
                    if (duration := accumulated_delays[target]) > 0.0:
                        instructions.append(Delay(target=target, duration=duration))
                        accumulated_delays[target] = 0.0
                instructions.append(inst)
            else:
                instructions.append(inst)

        for target, duration in accumulated_delays.items():
            if duration != 0.0:
                instructions.append(Delay(target=target, duration=duration))
        ir.instructions = instructions

        met_mgr.record_metric(MetricsType.OptimizedInstructionCount, len(instructions))
        return ir


class ScopeSanitisation(TransformPass):
    """Bubbles up all sweeps and repeats to the beginning of the list and adds delimiter
    instructions to the repeats and sweeps signifying the end of their scopes.

    .. warning::

        This pass is intended for use with legacy builders that do not use
        :class:`EndRepeat` and :class:`EndSweep` instructions. It will add a delimiter
        instruction for each :class:`Repeat` and :class:`Sweep` found in the IR.
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        tail, head = partition(lambda inst: isinstance(inst, Repeat), ir.instructions)
        tail, head = list(tail), list(head)

        ir.instructions = head + tail
        return ir


class EndOfTaskResetSanitisation(TransformPass):
    """Checks for a reset on each active qubit at the end of a task, and adds Reset
    operations if not found.

    After each shot, it is expected that the qubit is returned to its ground state, ready
    for the next shot. This pass ensures this is the case by checking if the last "active"
    operation on an qubit is a :class:`Reset`, and if not, adds a :class:`Reset` to the end
    of the instruction list.
    """

    def __init__(self, model: PhysicalHardwareModel):
        """
        :param model: The hardware model that holds calibrated information on the qubits on the QPU.
        """
        self.model = model

    def run(
        self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs
    ) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        active_pulse_channels: ActivePulseChannelResults = res_mgr.lookup_by_type(
            ActivePulseChannelResults
        )
        qubit_map: dict[Qubit, None | bool] = {
            qubit: None for qubit in active_pulse_channels.qubits
        }

        for inst in reversed(ir.instructions):
            if not isinstance(inst, (Pulse, Acquire, Reset)):
                continue

            elif isinstance(inst, Reset):
                # If a qubit only sees a reset, its not "active", so ignore.
                qubit = self.model.qubits[inst.qubit_target]
                if qubit in qubit_map and qubit_map[qubit] is None:
                    qubit_map[qubit] = True
            else:
                qubit = active_pulse_channels.target_map[inst.target]
                if qubit_map[qubit] is None:
                    qubit_map[qubit] = False

            if all([valid_reset is not None for valid_reset in qubit_map.values()]):
                break

        for qubit, valid_reset in qubit_map.items():
            if not valid_reset:
                ir.add(Reset(qubit_target=self.model.index_of_qubit(qubit)))

        return ir


class FreqShiftSanitisation(TransformPass):
    """
    Looks for any active frequency shift pulse channels in the hardware model and adds
    square pulses for the duration.

    .. warning::

        This pass assumes the following, which is achieved via other passes:

        * :class:`Synchronize` instructions have already been lowered to :class:`Delay`
          instructions.
        * Durations are static.
    """

    def __init__(self, model: PhysicalHardwareModel):
        """:param model: The hardware model containing the frequency shift channels."""

        self.model = model
        self.active_freq_shift_pulse_channels = self.get_active_freq_shift_pulse_channels(
            model
        )

    def run(
        self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs
    ) -> InstructionBuilder:
        """
        :param ir: The QatIR as an instruction builder.
        :param res_mgr: The results manager.
        """

        freq_shift_channels = self.active_freq_shift_pulse_channels
        if len(freq_shift_channels) == 0:
            return ir
        ir = self.add_freq_shift_to_ir(ir, set(freq_shift_channels.keys()))

        if res_mgr.check_for_type(ActivePulseChannelResults):
            res: ActivePulseChannelResults = res_mgr.lookup_by_type(
                ActivePulseChannelResults
            )
            res.target_map.update(
                {
                    fs_pulse_ch.uuid: qubit
                    for fs_pulse_ch, qubit in freq_shift_channels.items()
                }
            )

        return ir

    @staticmethod
    def get_active_freq_shift_pulse_channels(model) -> dict[FreqShiftPulseChannel, Qubit]:
        """Returns all active frequency shift pulse channels found in the hardware model."""
        pulse_channels = dict()
        for qubit in model.qubits.values():
            freq_shift_pulse_ch = qubit.freq_shift_pulse_channel
            if freq_shift_pulse_ch.active:
                pulse_channels[freq_shift_pulse_ch] = qubit

        return pulse_channels

    @staticmethod
    def add_freq_shift_to_ir(
        ir: InstructionBuilder, freq_shift_channels: set[PulseChannel]
    ) -> InstructionBuilder:
        """Adds frequency shift instructions to the instruction builder."""

        durations = defaultdict(float)
        idx = None  # Index of first quantum instruction.
        new_instructions = []
        for i, inst in enumerate(ir):
            if isinstance(inst, QuantumInstruction):
                if idx is None:
                    idx = i

                if (duration := inst.duration) > 0:
                    durations[inst.target] += duration

            new_instructions.append(inst)

        # If no quantum instructions found, return the IR as is.
        if len(durations) == 0:
            return ir

        max_duration = max(durations.values())
        for pulse_ch in freq_shift_channels:
            pulse = Pulse(
                target=pulse_ch.uuid,
                waveform=SquareWaveform(
                    amp=pulse_ch.amp, width=max_duration, phase=pulse_ch.phase
                ),
            )
            new_instructions.insert(idx, pulse)

        ir.instructions = new_instructions
        return ir


class InitialPhaseResetSanitisation(TransformPass):
    """
    Checks if every active pulse channel has a phase reset in the beginning.

    .. warning::

        This pass implies that an `ActivePulseChannelAnalysis` is performed prior to this pass.
    """

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        active_targets = list(res_mgr.lookup_by_type(ActivePulseChannelResults).targets)

        if active_targets:
            index = next(
                filter(
                    lambda tup: True if isinstance(tup[1], QuantumInstruction) else None,
                    enumerate(ir.instructions),
                )
            )[0]
            head, tail = ir.instructions[:index], ir.instructions[index:]
            resets = [PhaseReset(target=target) for target in active_targets]
            ir.instructions = head + resets + tail

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
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        durations: dict[str, float] = defaultdict(float)
        new_instructions: list[Instruction] = []

        for inst in ir.instructions:
            self.process_instruction(inst, new_instructions, durations)

        ir.instructions = new_instructions
        return ir

    @singledispatchmethod
    def process_instruction(
        self, instruction, new_instructions: list[Instruction], durations: dict[str, float]
    ):
        raise NotImplementedError(
            f"No processing method for instruction type {type(instruction)}"
        )

    @process_instruction.register(Instruction)
    def _(
        self,
        instruction: Instruction,
        new_instructions: list[Instruction],
        durations: dict[str, float],
    ):
        """Default handler for instructions that do not have a specific processing
        method."""
        new_instructions.append(instruction)

    @process_instruction.register(QuantumInstruction)
    def _(
        self,
        instruction: QuantumInstruction,
        new_instructions: list[Instruction],
        durations: dict[str, float],
    ):
        """Processes QuantumInstructions by updating their target durations."""
        durations[instruction.target] += instruction.duration
        new_instructions.append(instruction)

    @process_instruction.register(Synchronize)
    def _(
        self,
        instruction: Synchronize,
        new_instructions: list[Instruction],
        durations: dict[str, float],
    ):
        """Process Synchronize instructions by converting them to Delay instructions."""
        targets = instruction.targets
        current_durations = np.asarray([durations[target] for target in targets])
        max_duration = np.max(current_durations)
        sync_durations = max_duration - current_durations
        delay_instrs = [
            Delay(target=target, duration=sync_durations[i])
            for i, target in enumerate(targets)
            if sync_durations[i] > 0.0
        ]
        new_instructions.extend(delay_instrs)
        durations.update({target: max_duration for target in targets})


PydPhaseOptimisation = PhaseOptimisation
PydPostProcessingSanitisation = PostProcessingSanitisation
PydReturnSanitisation = ReturnSanitisation
PydBatchedShots = BatchedShots
PydResetsToDelays = ResetsToDelays
PydSquashDelaysOptimisation = SquashDelaysOptimisation
PydRepeatTranslation = RepeatTranslation
PydInactivePulseChannelSanitisation = InactivePulseChannelSanitisation
PydInstructionLengthSanitisation = InstructionLengthSanitisation
PydScopeSanitisation = ScopeSanitisation
PydEndOfTaskResetSanitisation = EndOfTaskResetSanitisation
PydFreqShiftSanitisation = FreqShiftSanitisation
PydInitialPhaseResetSanitisation = InitialPhaseResetSanitisation
PydLowerSyncsToDelays = LowerSyncsToDelays
