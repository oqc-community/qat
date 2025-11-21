# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import itertools
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import singledispatchmethod

import numpy as np
from compiler_config.config import CompilerConfig, MetricsType
from more_itertools import partition
from numpy.typing import NDArray

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import InstructionBuilder, QuantumInstructionBuilder
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
from qat.ir.waveforms import Pulse, SampledWaveform, SquareWaveform, Waveform
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


class PopulateWaveformSampleTime(TransformPass):
    """Populates instructions within the IR with the sample time from their coupled
    physical channel.

    This is a symptom of the IR builder not having access to the target data when assembling
    the instructions; this will likely be addressed later.
    """

    def __init__(self, hardware_model: PhysicalHardwareModel, target_data: TargetData):
        """
        :param hardware_model: The hardware model that holds calibrated information on the
            qubits on the QPU.
        :param target_data: Target-related information.
        """
        self._channel_data = self._create_channel_data(hardware_model, target_data)

    @staticmethod
    def _create_channel_data(
        hardware_model: PhysicalHardwareModel, target_data
    ) -> dict[str, float]:
        """Maps physical channels onto their sample times."""
        channel_data = {}

        for qubit in hardware_model.qubits.values():
            channel_data[qubit.physical_channel.uuid] = target_data.QUBIT_DATA.sample_time
            channel_data[qubit.resonator.physical_channel.uuid] = (
                target_data.RESONATOR_DATA.sample_time
            )
        return channel_data

    def run(
        self, ir: QuantumInstructionBuilder, *args, **kwargs
    ) -> QuantumInstructionBuilder:
        for inst in ir.instructions:
            target = getattr(inst, "target", None)
            if target is None:
                continue
            physical_channel_id = ir.get_pulse_channel(target).physical_channel_id
            self._add_sample_time(inst, physical_channel_id)
        return ir

    @singledispatchmethod
    def _add_sample_time(self, inst: Instruction, target: str):
        pass

    @_add_sample_time.register(Pulse)
    def _(self, inst: Pulse, target: str):
        self._add_sample_time(inst.waveform, target)
        inst.duration = inst.waveform.duration

    @_add_sample_time.register(Acquire)
    def _(self, inst: Acquire, target: str):
        if inst.filter is not None:
            self._add_sample_time(inst.filter, target)

    @_add_sample_time.register(SampledWaveform)
    def _(self, inst: SampledWaveform, target: str):
        inst.sample_time = self._channel_data[target]


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

    @run.register(PhaseSet)
    def _(self, instruction: PhaseSet):
        self.accum_phaseshifts[instruction.target] = instruction.phase
        self.previous_phase_instruction_is_phase_set[instruction.target] = True

    @run.register(Pulse)
    @run.register(Acquire)
    def _(self, instruction: Pulse | Acquire):
        target = instruction.target
        emit = True
        if self.previous_phase_instruction_is_phase_set[target]:
            op_class = PhaseSet
        elif not np.isclose(self.accum_phaseshifts[target] % (2 * np.pi), 0.0):
            op_class = PhaseShift
        else:
            emit = False

        if emit:
            phase = self.accum_phaseshifts.pop(target)
            self.optimized_instructions.append(op_class(target=target, phase=phase))
            self.previous_phase_instruction_is_phase_set[target] = False

        self.optimized_instructions.append(instruction)

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
                pp.process_type == PostProcessType.MEAN
                and ProcessAxis.TIME in pp.axes
                and len(pp.axes) <= 1
            ):
                return False
        return True


class MeasurePhaseResetSanitisation(TransformPass):
    """
    Adds a phase reset before every measure pulse.
    """

    def __init__(self, hardware_model: PhysicalHardwareModel):
        """
        :param hardware_model: The hardware model that holds calibrated information on the qubits on the QPU.
        """
        self.model = hardware_model
        self.measure_pulse_channels = self._get_measure_pulse_channels(hardware_model)

    @staticmethod
    def _get_measure_pulse_channels(hardware_model: PhysicalHardwareModel):
        """
        Returns a list of pulse channels that are used for measure pulses.
        """
        measure_pulse_channels = {}
        for qubit_id, qubit in hardware_model.qubits.items():
            measure_pulse_channels[qubit_id] = qubit.measure_pulse_channel.uuid
        return measure_pulse_channels

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        new_instructions = []
        for instr in ir.instructions:
            if isinstance(instr, MeasureBlock):
                for qubit_target in instr.qubit_targets:
                    new_instructions.append(
                        PhaseReset(target=self.measure_pulse_channels[qubit_target])
                    )
            elif (
                isinstance(instr, Pulse)
                and instr.target in self.measure_pulse_channels.values()
            ):
                new_instructions.append(PhaseReset(target=instr.target))
            new_instructions.append(instr)

        ir.instructions = new_instructions
        return ir


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

        active_channels = set(res_mgr.lookup_by_type(ActivePulseChannelResults).targets)
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

        # TODO: To be fixed with a module passed down the pipeline. (COMPILER-794)
        ir._pulse_channels = {
            k: v for k, v in ir._pulse_channels.items() if k in active_channels
        }
        return ir


class InstructionGranularitySanitisation(TransformPass):
    """Rounds the durations of quantum instructions so they are multiples of the clock
    cycle.

    Only supports quantum instructions with static non-zero durations. Assumes that
    instructions with a non-zero duration only act on a single pulse channel. The
    santisiation is done for all instructions simultaneously using numpy for performance.

    For :class:`Pulse` instructions with :class:`SampledWaveform` waveforms, the durations
    are rounded up by padding the pulse with zero amplitude at the end. For other relevant
    instructions, we round down: this is for compatibility with calibration files that are
    calibrated using legacy code. However, in the future we might consider changing this
    to round up for consistency.

    .. warning::

        This pass has the potential to invalidate the timings for sequences of instructions
        that are time-sensitive. For example, if a pulse has an invalid time, it will round
        it up to the nearest integer multiple. Furthemore, it will assume that
        :class:`Acquire` instructions have no delay. This can be forced explicitly using the
        :class:`AcquireSanitisation` pass.
    """

    def __init__(self, target_data: TargetData):
        """:param target_data: Target-related information."""

        self.clock_cycle = target_data.clock_cycle

    def run(self, ir: InstructionBuilder, *args, **kwargs) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """

        quantum_instructions = []
        custom_pulses = []
        for inst in ir.instructions:
            if isinstance(inst, (Acquire, Delay)) or (
                isinstance(inst, Pulse) and not isinstance(inst.waveform, SampledWaveform)
            ):
                quantum_instructions.append(inst)
            elif isinstance(inst, Pulse) and isinstance(inst.waveform, SampledWaveform):
                custom_pulses.append(inst)

        self._sanitise_quantum_instructions(quantum_instructions)
        self._sanitise_custom_pulses(custom_pulses)

        return ir

    def _clock_cycle_multiples(
        self, instructions: list[Pulse | Acquire | Delay]
    ) -> NDArray[float]:
        """
        Extracts the number of clock cycles for each instruction
        """
        durations = np.asarray([inst.duration for inst in instructions])
        return durations / self.clock_cycle

    def _sanitise_quantum_instructions(self, instructions: list[Pulse | Acquire | Delay]):
        """Sanitises the durations quantum instructions with non-zero duration by rounding
        down to the nearest clock cycle."""

        multiples = self._clock_cycle_multiples(instructions)
        # 1e-10 for floating point errors
        rounded_multiples = np.floor(multiples + 1e-10).astype(int)
        durations_equal = np.isclose(multiples, rounded_multiples)

        invalid_instructions: set[str] = set()
        for idx in np.where(np.logical_not(durations_equal))[0]:
            inst = instructions[idx]
            invalid_instructions.add(str(inst))
            new_duration = rounded_multiples[idx] * self.clock_cycle
            if isinstance(inst, Pulse):
                inst.update_duration(new_duration)
            else:
                if isinstance(inst, Acquire) and inst.filter is not None:
                    self._sanitise_acquire_pulse(inst, new_duration)
                inst.duration = new_duration

        if len(invalid_instructions) >= 1:
            log.info(
                "The following instructions do not have durations that are integer "
                f"multiples of the clock cycle {self.clock_cycle}, and will be rounded "
                "down: " + ", ".join(set(invalid_instructions))
            )

    @staticmethod
    def _sanitise_acquire_pulse(instruction: Acquire, new_duration: float):
        """
        Acquire instructions with filters have a duration equal to the filter duration,
        so we need to update the filter as well.
        """
        if (
            isinstance(instruction.filter.waveform, SampledWaveform)
            and new_duration < instruction.duration
        ):
            # This is a temporary workaround due to the fact that the acquire duration gets rounded
            # down to the nearest clock cycle, which results in cutting off the sample.
            # TODO: Review for COMPILER-488 changes.
            n_samples = int(
                np.floor(new_duration / instruction.filter.waveform.sample_time)
            )
            instruction.filter.waveform.samples = instruction.filter.waveform.samples[
                :n_samples
            ]
            instruction.filter.duration = new_duration
        instruction.filter.update_duration(new_duration)

    def _sanitise_custom_pulses(self, instructions: list[Pulse]):
        """Sanitises the durations of :class:`SampledWaveform`s by padding the
        waveforms with zero amplitudes."""

        multiples = self._clock_cycle_multiples(instructions)
        # 1e-10 for floating point errors
        rounded_multiples = np.ceil(multiples - 1e-10).astype(int)
        durations_equal = np.isclose(multiples, rounded_multiples)

        invalid_instructions: set[str] = set()
        for idx in np.where(np.logical_not(durations_equal))[0]:
            inst = instructions[idx]
            invalid_instructions.add(str(inst))
            new_duration = rounded_multiples[idx] * self.clock_cycle
            inst.update_duration(new_duration)

        if len(invalid_instructions) > 1:
            log.info(
                "The following sampled waveform pulses do not have durations that are integer "
                f"multiples of the clock cycle {self.clock_cycle}, and will be rounded "
                "up by padding with zero amplitudes: "
                + ", ".join(set(invalid_instructions))
            )


class InstructionLengthSanitisation(TransformPass):
    """
    Checks if quantum instructions are too long and splits if necessary.
    """

    def __init__(self, model: PhysicalHardwareModel, target_data: TargetData):
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
        self.sample_times = self._create_sample_time_map(model, target_data)

    @staticmethod
    def _create_sample_time_map(
        model: PhysicalHardwareModel, target_data: TargetData
    ) -> dict[str, float]:
        """Maps physical channels onto their sample times."""
        channel_data = {}
        for qubit in model.qubits.values():
            channel_data[qubit.physical_channel.uuid] = target_data.QUBIT_DATA.sample_time
            channel_data[qubit.resonator.physical_channel.uuid] = (
                target_data.RESONATOR_DATA.sample_time
            )
        return channel_data

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

            elif (
                isinstance(instr, Pulse)
                and isinstance(instr.waveform, SampledWaveform)
                and instr.duration > self.duration_limit
            ):
                target = next(iter(instr.targets))
                physical_channel_id = ir.get_pulse_channel(target).physical_channel_id
                sample_time = self.sample_times[physical_channel_id]
                new_instructions.extend(
                    self._batch_custom_waveform(instr, self.duration_limit, sample_time)
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

    def _batch_custom_waveform(
        self, instruction: Pulse, max_duration: float, sample_time: float
    ):
        """Breaks up a custom pulse into multiple custom pulses with none exceeding the
        maximum duration."""

        n_instr = int(instruction.duration // max_duration)
        remainder = instruction.duration % max_duration
        max_samples = int(round(max_duration / sample_time))

        batch_instr = []
        for i in range(n_instr):
            waveform = SampledWaveform(
                samples=instruction.waveform.samples[
                    i * max_samples : (i + 1) * max_samples
                ],
                sample_time=instruction.waveform.sample_time,
            )
            pulse = Pulse(
                targets=instruction.target,
                waveform=waveform,
                ignore_channel_scale=instruction.ignore_channel_scale,
                duration=max_duration,
            )
            batch_instr.append(pulse)

        if remainder:
            num_samples = int(round(remainder / sample_time))
            waveform = SampledWaveform(
                samples=instruction.waveform.samples[-num_samples:],
                sample_time=instruction.waveform.sample_time,
            )
            pulse = Pulse(
                targets=instruction.target,
                waveform=waveform,
                ignore_channel_scale=instruction.ignore_channel_scale,
                duration=remainder,
            )
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


class RepeatSanitisation(TransformPass):
    """Adds repeat counts and repetition periods to :class:`Repeat` instructions. If none
    is found, a repeat instruction is added."""

    def __init__(self, target_data: TargetData):
        """
        :param target_data: Target-related information.
        """
        self.default_shots = target_data.default_shots

    def run(
        self,
        ir: InstructionBuilder,
        *args,
        compiler_config: CompilerConfig = None,
        **kwargs,
    ):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""
        num_shots = self.default_shots
        configured = False
        if compiler_config is not None and compiler_config.repeats is not None:
            configured = True
            num_shots = compiler_config.repeats

        for inst in ir.instructions:
            if isinstance(inst, Repeat):
                if inst.repeat_count is None:
                    inst.repeat_count = num_shots
                elif configured and inst.repeat_count != num_shots:
                    raise ValueError(
                        f"Repeat instruction 'repeat_count' [{inst.repeat_count}] not "
                        f"matching CompilerConfig value [{num_shots}]."
                    )
                return ir

        ir.repeat(num_shots)
        log.warning(
            "Could not find any repeat instructions. "
            f"One has been added with {num_shots} shots."
        )
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
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager,
        *args,
        compiler_config: CompilerConfig = None,
        **kwargs,
    ) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        :param compiler_config: The compiler configuration.
        """
        active_pulse_channels = res_mgr.lookup_by_type(ActivePulseChannelResults)

        reset_time = self._get_reset_time(compiler_config)

        new_instructions = []
        for instr in ir:
            if isinstance(instr, Reset):
                qubit = self.model.qubits[instr.qubit_target]
                for pulse_ch in active_pulse_channels.from_qubit(qubit):
                    new_instructions.append(
                        Delay(
                            target=pulse_ch.uuid,
                            duration=reset_time,
                        )
                    )

            else:
                new_instructions.append(instr)

        ir.instructions = new_instructions
        return ir

    def _get_reset_time(
        self,
        compiler_config: CompilerConfig,
    ) -> float:
        """Gets the reset time for a qubit from the compiler configuration, or falls back to
        the passive reset time from the target data."""

        if getattr(compiler_config, "passive_reset_time", None) is not None:
            return compiler_config.passive_reset_time

        if getattr(compiler_config, "repetition_period", None) is not None:
            log.warning(
                "The `repetition_period` in `CompilerConfig` is deprecated. "
                "Please use `passive_reset_time` instead. "
                f"Using the default `passive_reset_time` {self.passive_reset_time}."
            )

        return self.passive_reset_time


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

        active_pulse_channels = res_mgr.lookup_by_type(ActivePulseChannelResults)
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
                qubit = active_pulse_channels.pulse_channel_to_qubit_map[inst.target]
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
            res = res_mgr.lookup_by_type(ActivePulseChannelResults)
            for pulse_channel, qubit in freq_shift_channels.items():
                res.add_target(pulse_channel, qubit)

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
        active_targets = res_mgr.lookup_by_type(ActivePulseChannelResults).targets

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


class EvaluateWaveforms(TransformPass):
    """Evaluates the amplitudes of :class:`Waveform`s within :class:`Pulse` instructions,
    replacing them with a :class:`SampledWaveform` and accounting for the scale of the
    pulse channel.

    :class:`Waveform` dataclasses are defined by (often many) parameters. With the exception
    of specific shapes, they cannot be implemented directly on hardware. Instead, we
    evaluate the waveform at discrete times, and communicate these values to the target.
    This pass evaluates pulses early in the compilation pipeline.

    The resulting :class:`Pulse` instructions will have :code:`ignore_channel_scale=True`.
    """

    def __init__(
        self,
        model: PhysicalHardwareModel,
        target_data: TargetData,
        ignored_shapes: tuple[type] | None = None,
        acquire_ignored_shapes: tuple[type] | None = None,
    ):
        """
        :param model: The hardware model that holds calibrated information on the qubits,
            which is required to extract the scale factor of pulse channels.
        :param target_data: Target-related information, which is used to extract the sample
            time of the pulse channels.
        :param ignored_shapes: A list of pulse shapes that are not evaluated to a custom
            pulse, defaults to `[PulseShapeType.SQUARE]`.
        :param acquire_ignored_shapes: A list of pulse shapes that are not evaluated to a
            custom pulse for :class:`Acquire` filters, defaults to `[]`.
        """

        self.ignored_shapes = self._sanitise_ignored_shapes(ignored_shapes, SquareWaveform)
        self.acquire_ignored_shapes = self._sanitise_ignored_shapes(
            acquire_ignored_shapes, ()
        )
        self.sample_times = self._extract_pulse_channel_features(model, target_data)

    @staticmethod
    def _sanitise_ignored_shapes(
        ignored_shapes: tuple[type] | type, default: tuple[type] | type
    ) -> tuple[type]:
        """Sanitises the ignored shapes to ensure they are a list of types."""
        ignored_shapes = ignored_shapes if ignored_shapes is not None else default
        return ignored_shapes if isinstance(ignored_shapes, tuple) else (ignored_shapes,)

    @staticmethod
    def _extract_pulse_channel_features(
        model: PhysicalHardwareModel, target_data: TargetData
    ) -> dict[str, float]:
        """Extracts the  sample rate for all pulse channels as dicts, with physical channel
        IDs as keys."""

        sample_times = {}
        for qubit in model.qubits.values():
            sample_times[qubit.physical_channel.uuid] = target_data.QUBIT_DATA.sample_time
            sample_times[qubit.resonator.physical_channel.uuid] = (
                target_data.RESONATOR_DATA.sample_time
            )

        return sample_times

    def run(
        self, ir: QuantumInstructionBuilder, *args, **kwargs
    ) -> QuantumInstructionBuilder:
        """:param ir: The list of instructions as an instruction builder."""

        waveform_lookup = defaultdict(dict)
        instructions = []
        for inst in ir.instructions:
            new_inst = self.process_instruction(inst, ir, waveform_lookup=waveform_lookup)
            instructions.append(new_inst)
        ir.instructions = instructions
        return ir

    @singledispatchmethod
    def process_instruction(
        self, instruction: Instruction, ir: QuantumInstructionBuilder, **kwargs
    ):
        """Default handler for instructions that do not have a waveform evaluation."""
        return instruction

    @process_instruction.register(Acquire)
    def _(self, instruction: Acquire, ir: QuantumInstructionBuilder, **kwargs) -> Acquire:
        """Implements the waveform evaluation on the filter for an :class:`Acquire`
        instruction."""

        instruction.filter = self.process_instruction(
            instruction.filter, ir, ignored_shapes=self.acquire_ignored_shapes, **kwargs
        )
        return instruction

    @process_instruction.register(Pulse)
    def _(self, instruction: Pulse, ir: QuantumInstructionBuilder, **kwargs) -> Pulse:
        """Evaluates the waveform within the :class:`Pulse` and handles the channel scale
        factor if required."""

        pulse_channel = ir.get_pulse_channel(instruction.target)
        scale = pulse_channel.scale if not instruction.ignore_channel_scale else 1.0

        instruction.waveform = self.evaluate_waveform(
            instruction.waveform,
            ignored_shapes=kwargs.pop("ignored_shapes", self.ignored_shapes),
            target=instruction.target,
            physical_channel=pulse_channel.physical_channel_id,
            scale=scale,
            **kwargs,
        )
        instruction.ignore_channel_scale = True
        return instruction

    @singledispatchmethod
    def evaluate_waveform(self, waveform, *args, **kwargs):
        """Default handler for waveform evaluation that does not have a specific
        implementation."""
        return ValueError(
            f"No waveform evaluation method for waveform type {type(waveform)}."
        )

    @evaluate_waveform.register(Waveform)
    def _(
        self,
        waveform: Waveform,
        ignored_shapes: list[type],
        waveform_lookup: dict[str, dict[str, NDArray]],
        target: str,
        physical_channel: str,
        scale: complex | float = 1.0 + 0.0j,
    ) -> Waveform | SampledWaveform:
        """Implements the waveform evaluation for :class:`Waveform`s."""

        # We cannot bake the waveform type into singledispatch here as the ignored shapes
        # is dynamic..
        if isinstance(waveform, ignored_shapes):
            if np.isclose(scale, 1.0):
                return waveform
            else:
                # We reinstantiate a new waveform as a preacaution against accidently
                # accidently shared instances.
                attrs = waveform.model_dump()
                attrs["amp"] *= scale
                return type(waveform)(**attrs)

        samples = waveform_lookup[target].get(waveform, None)
        if samples is not None:
            new_waveform = SampledWaveform(samples=samples)
        else:
            # Evaluate the waveform at discrete times
            sample_time = self.sample_times[physical_channel]
            edge = waveform.duration / 2.0 - sample_time * 0.5
            num_samples = int(np.ceil(waveform.duration / sample_time - 1e-10))
            t = np.linspace(start=-edge, stop=edge, num=num_samples)
            new_waveform = waveform.sample(t)
            waveform_lookup[target][waveform] = new_waveform.samples

        if not np.isclose(scale, 1.0):
            new_waveform.samples = new_waveform.samples * scale
        return new_waveform

    @evaluate_waveform.register(SampledWaveform)
    def _(
        self, waveform: SampledWaveform, scale: complex | float = 1.0 + 0.0j, **kwargs
    ) -> SampledWaveform:
        """Multiplies the waveform by a scale factor if required."""

        if not np.isclose(scale, 1.0):
            waveform.samples = waveform.samples * scale
        return waveform


class SynchronizeTask(TransformPass):
    """Synchronizes all active pulse channels in a task.

    Adds a synchronize to the end of the instruction list for all active pulse channels,
    which is extracted from the :class:`ActivePulseChannelAnalysis` pass. This is useful to
    do before resetting the qubits, as it ensures no qubit is reset while the task is still
    "active", reducing the effects of cross-talk.
    """

    def run(
        self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs
    ) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        active_channels = res_mgr.lookup_by_type(ActivePulseChannelResults).targets
        if len(active_channels) > 1:
            ir.add(Synchronize(targets=active_channels))
        return ir


class AcquireSanitisation(TransformPass):
    """Sanitises the :class:`Acquire` instruction to separate the :class:`Delay` from
    :class:`Acquire` instructions.

    :class:`Acquire` instructions are defined by a "duration" for which they instruct the
    target to readout. They also contain a "delay" attribute, which instructions the
    acquisition to start after some given time. This pass separates acqusitions with a
    delay into two instructions for the first acquire that acts on the channel.
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """

        new_instructions: list[Instruction] = []

        for inst in ir:
            if isinstance(inst, Acquire):
                if inst.delay:
                    delay = Delay(target=inst.target, duration=inst.delay)
                    inst.delay = 0.0
                    new_instructions.extend([delay, inst])
                else:
                    new_instructions.append(inst)
            else:
                new_instructions.append(inst)
        ir.instructions = new_instructions
        return ir


PydPhaseOptimisation = PhaseOptimisation
PydPostProcessingSanitisation = PostProcessingSanitisation
PydReturnSanitisation = ReturnSanitisation
PydRepeatSanitisation = RepeatSanitisation
PydBatchedShots = BatchedShots
PydResetsToDelays = ResetsToDelays
PydSquashDelaysOptimisation = SquashDelaysOptimisation
PydRepeatTranslation = RepeatTranslation
PydMeasurePhaseResetSanitisation = MeasurePhaseResetSanitisation
PydInactivePulseChannelSanitisation = InactivePulseChannelSanitisation
PydInstructionGranularitySanitisation = InstructionGranularitySanitisation
PydInstructionLengthSanitisation = InstructionLengthSanitisation
PydScopeSanitisation = ScopeSanitisation
PydEndOfTaskResetSanitisation = EndOfTaskResetSanitisation
PydFreqShiftSanitisation = FreqShiftSanitisation
PydInitialPhaseResetSanitisation = InitialPhaseResetSanitisation
PydLowerSyncsToDelays = LowerSyncsToDelays
PydEvaluateWaveforms = EvaluateWaveforms
PydSynchronizeTask = SynchronizeTask
PydAcquireSanitisation = AcquireSanitisation
