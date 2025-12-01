# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import itertools
from collections import OrderedDict, defaultdict
from copy import copy, deepcopy
from numbers import Number

import numpy as np
from compiler_config.config import CompilerConfig, MetricsType
from more_itertools import partition

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.middleend.passes.purr.analysis import ActiveChannelResults
from qat.model.target_data import TargetData
from qat.purr.backends.qiskit_simulator import QiskitBuilder, QiskitBuilderWrapper
from qat.purr.backends.utilities import evaluate_shape
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, Qubit
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    Assign,
    CustomPulse,
    Delay,
    DeviceUpdate,
    EndRepeat,
    EndSweep,
    GreaterThan,
    Instruction,
    Jump,
    Label,
    MeasurePulse,
    PhaseReset,
    PhaseSet,
    PhaseShift,
    Plus,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
    PulseShapeType,
    QuantumInstruction,
    Repeat,
    Reset,
    Return,
    Sweep,
    Synchronize,
    Variable,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class IntegratorAcquireSanitisation(TransformPass):
    """Changes `AcquireMode.INTEGRATOR` acquisitions to `AcquireMode.RAW`.

    The legacy echo/RTCS engines expect the acquisition mode to be either `RAW` or `SCOPE`.
    While the actual execution can process `INTEGRATOR` by treating it as `RAW`, they are
    typically santitised the runtime using :meth:`EchoEngine.optimize()`. If not done in the
    new pipelines, it will conflict with :class:`PostProcessingSantisiation`, and return the
    wrong results. The new echo engine supports all acquisition modes, so this is not a
    problem here.
    """

    def run(
        self,
        ir: InstructionBuilder,
        *args,
        **kwargs,
    ):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""
        for inst in [instr for instr in ir.instructions if isinstance(instr, Acquire)]:
            if inst.mode == AcquireMode.INTEGRATOR:
                inst.mode = AcquireMode.RAW
        return ir


class QiskitInstructionsWrapper(TransformPass):
    """Wraps the Qiskit builder in a wrapper to match the pipelines API.

    A really silly pass needed to wrap the :class:`QiskitBuilder` in an object that allows
    `QiskitBuilderWrapper.instructions` to be called, allowing the builder to be used in the
    the :class:`LegacyRuntime`. This is needed because the qiskit engine has a different API
    to other `purr` engines, requiring the whole builder to be passed (as opposed to
    `builder.instructions`).
    """

    def run(self, ir: QiskitBuilder, *args, **kwargs) -> QiskitBuilderWrapper:
        """:param ir: The Qiskit instructions"""
        return QiskitBuilderWrapper(ir)


class LegacyPhaseOptimisation(TransformPass):
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

        accum_phaseshifts: dict[PulseChannel, PhaseShift] = {}
        optimized_instructions: list[Instruction] = []
        for instruction in ir.instructions:
            if isinstance(instruction, PhaseShift) and isinstance(
                instruction.phase, Number
            ):
                if accum_phaseshift := accum_phaseshifts.get(instruction.channel, None):
                    accum_phaseshift.phase += instruction.phase
                else:
                    accum_phaseshifts[instruction.channel] = PhaseShift(
                        instruction.channel, instruction.phase
                    )
            elif isinstance(instruction, (Pulse, CustomPulse)):
                quantum_targets = getattr(instruction, "quantum_targets", [])
                if not isinstance(quantum_targets, list):
                    quantum_targets = [quantum_targets]
                for quantum_target in quantum_targets:
                    if quantum_target in accum_phaseshifts:
                        optimized_instructions.append(accum_phaseshifts.pop(quantum_target))
                optimized_instructions.append(instruction)
            elif isinstance(instruction, PhaseReset):
                for channel in instruction.quantum_targets:
                    accum_phaseshifts.pop(channel, None)
                optimized_instructions.append(instruction)
            else:
                optimized_instructions.append(instruction)

        ir.instructions = optimized_instructions
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, len(optimized_instructions)
        )
        return ir


class LoopCount(int): ...


class RepeatTranslation(TransformPass):
    """Transform :class:`Repeat` instructions so that they are replaced with:
    :class:`Variable`, :class:`Assign`, and :class:`Label` instructions at the start,
    and :class:`Assign` and :class:`Jump` instructions at the end."""

    def __init__(self, target_data: TargetData):
        self.target_data = target_data

    def run(self, ir: InstructionBuilder, *args, **kwargs) -> InstructionBuilder:
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""
        instructions: list[Instruction] = []
        label_data: OrderedDict[Label, dict[str, Repeat | Variable]] = OrderedDict()
        repetition_period = None
        passive_reset_time = None

        def _close_repeat(label: Label, data: dict, index: int = None) -> None:
            # TODO: Adjust so that closing is done before finishing instructions such as
            # returns and postprocessing. COMPILER-452
            repeat = data["repeat"]
            var = data["var"]
            assign = Assign(var.name, Plus(var, 1))
            jump = Jump(label, GreaterThan(repeat.repeat_count, var))
            instructions.extend([assign, jump])

        for inst in ir.instructions:
            if isinstance(inst, Repeat):
                label = Label.with_random_name(ir.existing_names)
                var = Variable(label.name + "_count", LoopCount)
                assign = Assign(var.name, 0)
                label_data[label] = {"repeat": inst, "var": var}
                instructions.extend([var, assign, label])
                if repetition_period is None and inst.repetition_period is not None:
                    repetition_period = inst.repetition_period
                if passive_reset_time is None and inst.passive_reset_time is not None:
                    passive_reset_time = inst.passive_reset_time
            elif isinstance(inst, EndRepeat):
                if len(label_data) < 1:
                    raise ValueError("EndRepeat found without associated Repeat.")
                label, data = label_data.popitem()
                _close_repeat(label, data)
            else:
                instructions.append(inst)

        for label, data in reversed(label_data.items()):
            _close_repeat(label, data)

        if passive_reset_time is None and repetition_period is None:
            passive_reset_time = self.target_data.QUBIT_DATA.passive_reset_time
        ir.repetition_period = repetition_period
        ir.passive_reset_time = passive_reset_time

        ir.instructions = instructions
        return ir


class PhaseOptimisation(TransformPass):
    """Iterates through the list of instructions and compresses contiguous
    :class:`PhaseShift` instructions. This pass will change :class:`PhaseReset` to
    :class:`PhaseSet` instructions.
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

        accum_phaseshifts: dict[str, PhaseShift | PhaseSet] = dict()
        optimized_instructions: list[Instruction] = []
        for instruction in ir.instructions:
            if isinstance(instruction, (PhaseShift, PhaseSet)) and isinstance(
                instruction.phase, Number
            ):
                key = instruction.channel.partial_id()
                accum_phaseshifts[key] = self.merge_phase_instructions(
                    instruction.channel, accum_phaseshifts.get(key, None), instruction
                )
            elif isinstance(instruction, PhaseReset):
                for channel in instruction.quantum_targets:
                    key = channel.partial_id()
                    accum_phaseshifts[key] = self.merge_phase_instructions(
                        channel, accum_phaseshifts.get(key, None), instruction
                    )
            elif isinstance(
                instruction, (CustomPulse, Pulse, Acquire, PhaseSet, PhaseShift)
            ):
                quantum_targets = getattr(instruction, "quantum_targets", [])
                if not isinstance(quantum_targets, list):
                    quantum_targets = [quantum_targets]
                for quantum_target in quantum_targets:
                    if quantum_target.partial_id() in accum_phaseshifts:
                        new_instruction = accum_phaseshifts.pop(quantum_target.partial_id())
                        if not (
                            isinstance(new_instruction, PhaseShift)
                            and np.isclose(new_instruction.phase % (2 * np.pi), 0.0)
                        ):
                            optimized_instructions.append(new_instruction)
                optimized_instructions.append(instruction)
            elif isinstance(instruction, (Delay, Synchronize)):
                for channel in instruction.quantum_targets:
                    key = channel.partial_id()
                    if isinstance(accum_phaseshifts.get(key, None), PhaseSet):
                        optimized_instructions.append(accum_phaseshifts.pop(key))
                optimized_instructions.append(instruction)
            elif isinstance(instruction, (Jump, Label)):
                accum_phaseshifts = dict()
                optimized_instructions.append(instruction)
            else:
                optimized_instructions.append(instruction)

        ir.instructions = optimized_instructions
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, len(optimized_instructions)
        )
        return ir

    @staticmethod
    def merge_phase_instructions(
        target: PulseChannel,
        phase1: PhaseSet | PhaseReset | PhaseShift | None,
        phase2: PhaseSet | PhaseReset | PhaseShift,
    ):
        if isinstance(phase2, PhaseReset):
            return PhaseSet(target, 0.0)
        elif isinstance(phase2, PhaseSet):
            return PhaseSet(target, phase2.phase)
        elif phase1 is None:
            return phase2
        elif isinstance(phase1, PhaseReset):
            return PhaseSet(target, phase2.phase)
        elif isinstance(phase1, PhaseSet):
            return PhaseSet(target, phase1.phase + phase2.phase)
        else:
            return PhaseShift(target, phase1.phase + phase2.phase)


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

        # TODO -COMPILER-851 - Specify ctrl HW features in TargetData
        enable_hw_averaging = kwargs.get("enable_hw_averaging", False)

        pp_insts = [val for val in ir.instructions if isinstance(val, PostProcessing)]
        discarded = []
        for pp in pp_insts:
            if pp.acquire.mode == AcquireMode.SCOPE:
                if (
                    not enable_hw_averaging
                    and pp.process == PostProcessType.MEAN
                    and ProcessAxis.SEQUENCE in pp.axes
                    and len(pp.axes) <= 1
                ):
                    discarded.append(pp)

            elif pp.acquire.mode == AcquireMode.INTEGRATOR:
                if (
                    pp.process == PostProcessType.DOWN_CONVERT
                    and ProcessAxis.TIME in pp.axes
                    and len(pp.axes) <= 1
                ):
                    discarded.append(pp)
                if (
                    pp.process == PostProcessType.MEAN
                    and ProcessAxis.TIME in pp.axes
                    and len(pp.axes) <= 1
                ):
                    discarded.append(pp)
        ir.instructions = [val for val in ir.instructions if val not in discarded]
        met_mgr.record_metric(MetricsType.OptimizedInstructionCount, len(ir.instructions))
        return ir


class AcquireSanitisation(TransformPass):
    """Sanitises the :class:`Acquire` instruction: the first per pulse channel is split into
    an :class:`Acquire` and a :class:`Delay`, and other acquisitions have their delay
    removed.

    :class:`Acquire` instructions are defined by a "duration" for which they instruct the
    target to readout. They also contain a "delay" attribute, which instructions the
    acquisition to start after some given time. This pass separates acqusitions with a
    delay into two instructions for the first acquire that acts on the channel. For multiple
    acquisitions on a single channel, the delay is not needed for the following
    acquisitions, and is set to zero.
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """

        new_instructions: list[Instruction] = []

        for inst in ir.instructions:
            if isinstance(inst, Acquire):
                if inst.delay:
                    delay = Delay(inst.quantum_targets, inst.delay)
                    inst.delay = 0.0
                    new_instructions.extend([delay, inst])
                else:
                    new_instructions.append(inst)
            else:
                new_instructions.append(inst)
        ir.instructions = new_instructions
        return ir


class InstructionGranularitySanitisation(TransformPass):
    """Rounds the durations of quantum instructions so they are multiples of the clock
    cycle.

    Only supports quantum instructions with static non-zero durations. Assumes that
    instructions with a non-zero duration only act on a single pulse channel. The
    santisiation is done for all instructions simultaneously using numpy for performance.

    For :class:`CustomPulse` instructions, the durations are rounded up by padding the pulse
    with zero amplitude at the end. For other relevant instructions, we round down: this is
    for compatibility with calibration files that are calibrated using legacy code. However,
    in the future we might consider changing this to round up for consistency.

    .. warning::

        This pass has the potential to invalidate the timings for sequences of instructions
        that are time-sensitive. For example, if a pulse has an invalid time, it will round
        it up to the nearest integer multiple. Furthemore, it will assume that
        :class:`Acquire` instructions have no delay. This can be forced explicitly using the
        :class:`AcquireSanitisation` pass.
    """

    def __init__(self, model: QuantumHardwareModel, target_data: TargetData):
        """:param target_data: Target-related information."""

        self.clock_cycle = target_data.clock_cycle
        qubit_sample_times = {
            qubit.physical_channel: target_data.QUBIT_DATA.sample_time
            for qubit in model.qubits
        }
        res_sample_times = {
            resonator.physical_channel: target_data.RESONATOR_DATA.sample_time
            for resonator in model.resonators
        }
        self.sample_times = qubit_sample_times | res_sample_times

    def run(self, ir: InstructionBuilder, *args, **kwargs) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """

        self.sanitise_quantum_instructions(
            [inst for inst in ir.instructions if isinstance(inst, (Pulse, Acquire, Delay))]
        )
        self.sanitise_custom_pulses(
            [inst for inst in ir.instructions if isinstance(inst, CustomPulse)]
        )
        self.sanitise_acquire_filters(
            [inst for inst in ir.instructions if isinstance(inst, Acquire)]
        )
        return ir

    def sanitise_quantum_instructions(self, instructions: list[Pulse | Acquire | Delay]):
        """Sanitises the durations quantum instructions with non-zero duration by rounding
        down to the nearest clock cycle."""

        durations = np.asarray([inst.duration for inst in instructions])
        multiples = durations / self.clock_cycle
        rounded_multiples = np.floor(multiples + 1e-10).astype(int)
        # 1e-10 for floating point errors
        durations_equal = np.isclose(multiples, rounded_multiples)

        invalid_instructions: set[str] = set()
        for idx in np.where(np.logical_not(durations_equal))[0]:
            inst = instructions[idx]
            invalid_instructions.add(str(inst))
            new_duration = rounded_multiples[idx] * self.clock_cycle
            if isinstance(inst, Pulse):
                inst.width = new_duration
            else:
                inst.time = new_duration

        if len(invalid_instructions) >= 1:
            log.info(
                "The following instructions do not have durations that are integer "
                f"multiples of the clock cycle {self.clock_cycle}, and will be rounded "
                "down: " + ", ".join(set(invalid_instructions))
            )

    def sanitise_custom_pulses(self, instructions: list[CustomPulse]):
        """Sanitises the durations of :class:`CustomPulse` instructions by padding the
        pulses with zero amplitudes."""

        durations = np.asarray([inst.duration for inst in instructions])
        multiples = durations / self.clock_cycle
        # 1e-10 for floating point errors
        rounded_multiples = np.ceil(multiples - 1e-10).astype(int)
        durations_equal = np.isclose(multiples, rounded_multiples)

        invalid_instructions: set[str] = set()
        for idx in np.where(np.logical_not(durations_equal))[0]:
            inst = instructions[idx]
            invalid_instructions.add(str(inst))
            new_duration = rounded_multiples[idx] * self.clock_cycle

            padding = int(
                np.round(
                    (new_duration - durations[idx])
                    / self.sample_times[inst.quantum_targets[0].physical_channel],
                    0,
                )
            )

            if isinstance(inst.samples, list):
                inst.samples.extend([0.0 + 0.0j] * padding)
            elif isinstance(inst.samples, np.ndarray):
                inst.samples = np.append(
                    inst.samples, np.zeros(padding, dtype=inst.samples.dtype)
                )
            else:
                raise TypeError(
                    f"Unsupported type for samples in CustomPulse: {type(inst.samples)}"
                )

        if len(invalid_instructions) > 1:
            log.info(
                "The following custom pulses do not have durations that are integer "
                f"multiples of the clock cycle {self.clock_cycle}, and will be rounded "
                "up by padding with zero amplitudes: "
                + ", ".join(set(invalid_instructions))
            )

    def sanitise_acquire_filters(self, instructions: list[Acquire]):
        """Sanitises the durations of :class:`Acquire` filters by matching it to the
        duration of the :class:`Acquire`. For :class:`CustomPulse` filters, this strips
        away the samples that are not needed."""
        for instruction in instructions:
            if isinstance(instruction.filter, Pulse):
                new_filter = copy(instruction.filter)
                new_filter.width = instruction.duration
                instruction.filter = new_filter
            elif isinstance(instruction.filter, CustomPulse):
                num_samples = int(
                    np.round(
                        instruction.duration
                        / self.sample_times[instruction.quantum_targets[0].physical_channel]
                    )
                )
                if instruction.filter.duration > instruction.duration:
                    # if the filter is longer than the acquire, truncate it
                    samples = instruction.filter.samples[:num_samples]
                else:
                    # if the filter is shorter than the acquire, pad it with zeros
                    samples = np.zeros(num_samples, dtype=np.complex128)
                    samples[: len(instruction.filter.samples)] = instruction.filter.samples
                instruction.filter = CustomPulse(
                    instruction.filter.channel,
                    samples,
                    instruction.filter.ignore_channel_scale,
                )


class InitialPhaseResetSanitisation(TransformPass):
    """
    Checks if every active pulse channel has a phase reset in the beginning.

    .. warning::

        This pass implies that an `ActiveChannelAnalysis` is performed prior to this pass.
    """

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        active_targets = list(res_mgr.lookup_by_type(ActiveChannelResults).targets)

        if active_targets:
            index = next(
                filter(
                    lambda tup: True if isinstance(tup[1], QuantumInstruction) else None,
                    enumerate(ir._instructions),
                )
            )[0]
            ir.insert(PhaseReset(active_targets), index)

        return ir


class MeasurePhaseResetSanitisation(TransformPass):
    """
    Adds a phase reset before every measure pulse.
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        new_instructions = []
        for instr in ir.instructions:
            if isinstance(instr, MeasurePulse):
                new_instructions.append(PhaseReset(instr.quantum_targets))
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

        active_channels = [
            target.full_id()
            for target in res_mgr.lookup_by_type(ActiveChannelResults).targets
        ]
        instructions: list[Instruction] = []
        for inst in ir.instructions:
            if isinstance(inst, (PostProcessing, Pulse, CustomPulse, Acquire)):
                # instructions which define an active channel
                instructions.append(inst)
            elif isinstance(inst, Synchronize):
                # inactive channels need stripping from syncs
                inst.quantum_targets = [
                    target
                    for target in inst.quantum_targets
                    if target.full_id() in active_channels
                ]
                if len(inst.quantum_targets) > 0:
                    instructions.append(inst)
            elif isinstance(inst, QuantumInstruction):
                # other instructions need their targets checking
                target = next(iter(inst.quantum_targets))
                if target.full_id() in active_channels:
                    instructions.append(inst)
            else:
                instructions.append(inst)
        ir.instructions = instructions
        return ir


class InstructionLengthSanitisation(TransformPass):
    """
    Checks if quantum instructions are too long and splits if necessary.
    """

    def __init__(self, model: QuantumHardwareModel, target_data: TargetData):
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
        self.qubit_sample_time = target_data.QUBIT_DATA.sample_time
        self.resonator_sample_time = target_data.RESONATOR_DATA.sample_time
        self.is_resonator = self._create_resonator_map(model)

    @staticmethod
    def _create_resonator_map(model: QuantumHardwareModel) -> dict[str, bool]:
        """Creates a mapping between physical channel and pulse channel to specify if the
        channel is a resonator or not."""

        is_resonator = {}
        for qubit in model.qubits:
            is_resonator[qubit.physical_channel.id] = False
            is_resonator[qubit.measure_device.physical_channel.id] = True
        return is_resonator

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
                and instr.width > self.duration_limit
                and instr.shape == PulseShapeType.SQUARE
            ):
                new_instructions.extend(
                    self._batch_square_pulse(instr, self.duration_limit)
                )

            elif isinstance(instr, CustomPulse) and instr.duration > self.duration_limit:
                new_instructions.extend(
                    self._batch_custom_pulse(instr, self.duration_limit)
                )
            else:
                new_instructions.append(instr)

        ir.instructions = new_instructions
        return ir

    def _batch_delay(self, instruction: Delay, max_duration: float):
        n_instr = int(instruction.duration // max_duration)
        remainder = instruction.duration % max_duration

        batch_instr = []
        for _ in range(n_instr):
            batch_instr.append(Delay(instruction.quantum_targets, time=max_duration))

        if remainder:
            batch_instr.append(Delay(instruction.quantum_targets, time=remainder))

        return batch_instr

    def _batch_square_pulse(self, instruction: Pulse, max_width: float):
        n_instr = int(instruction.width // max_width)
        remainder = instruction.width % max_width

        batch_instr = []
        pulse = deepcopy(instruction)
        pulse.width = max_width
        for _ in range(n_instr):
            batch_instr.append(deepcopy(pulse))

        if remainder:
            pulse = deepcopy(pulse)
            pulse.width = remainder
            batch_instr.append(pulse)

        return batch_instr

    def _batch_custom_pulse(self, instruction: CustomPulse, max_duration: float):
        """Breaks up a custom pulse into multiple custom pulses with none exceeding the
        maximum duration."""

        n_instr = int(instruction.duration // max_duration)
        remainder = instruction.duration % max_duration
        sample_time = (
            self.resonator_sample_time
            if self.is_resonator[instruction.channel.physical_channel.id]
            else self.qubit_sample_time
        )
        max_samples = int(round(max_duration / sample_time))

        batch_instr = []
        for i in range(n_instr):
            pulse = CustomPulse(
                instruction.channel,
                instruction.samples[i * max_samples : (i + 1) * max_samples],
                instruction.ignore_channel_scale,
            )
            batch_instr.append(pulse)

        if remainder:
            num_samples = int(round(remainder / sample_time))
            pulse = CustomPulse(
                instruction.channel,
                instruction.samples[-num_samples:],
                instruction.ignore_channel_scale,
            )
            batch_instr.append(pulse)

        return batch_instr


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

        active_channels = res_mgr.lookup_by_type(ActiveChannelResults).targets
        if len(active_channels) > 0:
            ir.add(Synchronize(list(active_channels)))
        return ir


class EndOfTaskResetSanitisation(TransformPass):
    """Checks for a reset on each active qubit at the end of a task, and adds Reset
    operations if not found.

    After each shot, it is expected that the qubit is returned to its ground state, ready
    for the next shot. This pass ensures this is the case by checking if the last "active"
    operation on an qubit is a :class:`Reset`, and if not, adds a :class:`Reset` to the end
    of the instruction list. By default, the :class:`Reset` operation will have the drive
    channel of a qubit as its target. However, if the drive channel is not an active
    channel, a different active channel on the qubit will be used its place.

    :class:`Reset` instructions currently sit in a weird place. Their targets are drive
    channels to match the semantics of other quantum instructions (but are instantiated with
    a :class:Qubit). However, like measurements, resets are a qubit-level operation. You
    cannot reset the state of a pulse channel, the state is a property of the qubit! To
    avoid breaking changes, we'll just deal with that for now, but hope to do better in the
    refactored instructions...
    """

    def run(
        self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs
    ) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        active_pulse_channels = res_mgr.lookup_by_type(ActiveChannelResults)
        qubit_map: dict[Qubit, None | bool] = {
            qubit: None for qubit in active_pulse_channels.qubits
        }

        for inst in reversed(ir.instructions):
            if not isinstance(inst, (Pulse, CustomPulse, Acquire, Reset)):
                continue

            elif isinstance(inst, Reset):
                for target in inst.quantum_targets:
                    # if a qubit only sees a reset, its not "active", so ignore
                    qubit = active_pulse_channels.target_map[target]
                    if qubit in qubit_map and qubit_map[qubit] is None:
                        qubit_map[qubit] = True
            else:
                target = active_pulse_channels.target_map[next(iter(inst.quantum_targets))]
                if qubit_map[target] is None:
                    qubit_map[target] = False

            if all([val is not None for val in qubit_map.values()]):
                break

        for qubit, val in qubit_map.items():
            if not val:
                if (
                    target := qubit.get_drive_channel()
                ) not in active_pulse_channels.target_map:
                    # if the drive channel isn't an active channel, choose something else
                    target = active_pulse_channels.from_qubit(qubit)[0]
                ir.add(Reset(target))

        return ir


class ResetsToDelays(TransformPass):
    """
    Transforms :class:`Reset` operations to :class:`Delay`s.

    Note that the delays do not necessarily agree with the granularity of the underlying target machine.
    This can be enforced using the :class:`InstructionGranularitySanitisation`
    pass.
    """

    def __init__(self, target_data: TargetData):
        """
        :param target_data: Target-related information.
        """
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
        active_pulse_channels = res_mgr.lookup_by_type(ActiveChannelResults)

        reset_time = self._get_reset_time(compiler_config, ir)

        new_instructions = []
        for instr in ir.instructions:
            if isinstance(instr, Reset):
                qubit_targets = {
                    active_pulse_channels.target_map[target]
                    for target in instr.quantum_targets
                }

                for qubit in qubit_targets:
                    new_instructions.append(
                        Delay(
                            active_pulse_channels.from_qubit(qubit),
                            time=reset_time,
                        )
                    )

            else:
                new_instructions.append(instr)

        ir.instructions = new_instructions
        return ir

    def _get_reset_time(
        self,
        compiler_config: CompilerConfig,
        ir: InstructionBuilder = None,
    ) -> float:
        """Gets the reset time for a qubit from the compiler configuration, or falls back to
        the passive reset time from the target data."""

        if getattr(compiler_config, "passive_reset_time", None) is not None:
            return compiler_config.passive_reset_time

        if getattr(compiler_config, "repetition_period", None) is not None:
            log.warning(
                "The `repetition_period` in `CompilerConfig` will soon be deprecated. "
                "Please use `passive_reset_time` instead. "
            )
            return self._calculate_reset_time_for_repetition_period(compiler_config, ir)

        return self.passive_reset_time

    def _calculate_reset_time_for_repetition_period(
        self,
        compiler_config: CompilerConfig,
        ir: InstructionBuilder,
    ) -> float:
        """Calculates the reset time for a qubit based on the repetition period in the
        compiler configuration."""

        total_duration = self._get_total_duration(ir)
        reset_time = compiler_config.repetition_period - total_duration

        if reset_time < 0:
            log.warning(
                "The specified `repetition_period` is shorter than the total instruction duration. "
                "Setting the repetition period to equal the total instruction duration."
            )
            return 0.0
        return reset_time

    @staticmethod
    def _get_total_duration(ir: InstructionBuilder) -> float:
        durations: dict[str, float] = defaultdict(float)
        for inst in ir.instructions:
            if isinstance(inst, (Pulse, Acquire, Delay, CustomPulse)):
                # only increment the durations for meaningful instructions
                pulse_chan_id = inst.quantum_targets[0].partial_id()
                durations[pulse_chan_id] += inst.duration
        return max(durations.values())


class EvaluatePulses(TransformPass):
    """Evaluates the amplitudes of :class:`Pulse` instructions, replacing them with a
    :class:`CustomPulse` and accounting for the scale of the pulse channel.

    :class:`Pulse` instructions are defined by (often many) parameters. With the exception
    of specific shapes, they cannot be implemented directly on hardware. Instead, we
    evaluate the waveform at discrete times, and communicate these values to the target.
    This pass evaluates pulses early in the compilation pipeline.

    The :class:`CustomPulse` instructions will have :code:`ignore_channel_scale=True`.
    """

    def __init__(
        self,
        ignored_shapes: list[PulseShapeType] | None = None,
        acquire_ignored_shapes: list[PulseShapeType] | None = None,
        eval_function: callable = evaluate_shape,
    ):
        """
        :param ignored_shapes: A list of pulse shapes that are not evaluated to a custom
            pulse, defaults to `[PulseShapeType.SQUARE]`.
        :param acquire_ignored_shapes: A list of pulse shapes that are not evaluated to a
            custom pulse for :class:`Acquire` filters, defaults to `[]`.
        :param eval_function: Allows a pulse evaluation function to be injected, defaults
            to :meth:`evaluate_shape`.
        """

        ignored_shapes = (
            ignored_shapes if ignored_shapes is not None else [PulseShapeType.SQUARE]
        )
        self.ignored_shapes = (
            ignored_shapes if isinstance(ignored_shapes, list) else [ignored_shapes]
        )
        acquire_ignored_shapes = (
            acquire_ignored_shapes if acquire_ignored_shapes is not None else []
        )
        self.acquire_ignored_shapes = (
            acquire_ignored_shapes
            if isinstance(acquire_ignored_shapes, list)
            else [acquire_ignored_shapes]
        )
        self.evaluate = eval_function

    def run(self, ir: InstructionBuilder, *args, **kwargs) -> InstructionBuilder:
        """:param ir: The list of instructions as an instruction builder."""

        pulses: dict[str, CustomPulse] = dict()

        instructions = []
        for inst in ir.instructions:
            if isinstance(inst, (Pulse, CustomPulse)):
                inst = self.evaluate_waveform(inst, self.ignored_shapes, pulses)
            elif isinstance(inst, Acquire) and inst.filter:
                inst.filter = self.evaluate_waveform(
                    inst.filter, self.acquire_ignored_shapes, pulses
                )
            instructions.append(inst)
        ir.instructions = instructions
        return ir

    def evaluate_waveform(
        self,
        inst: Pulse | CustomPulse,
        ignored_shapes: list[PulseShapeType],
        pulse_lookup: dict[str, CustomPulse],
    ) -> Pulse | CustomPulse:
        """Evaluates the waveform for a :class:`Pulse` or :class:`CustomPulse`, accounting
        for the pulse channel scale."""

        if isinstance(inst, CustomPulse):
            # custom pulses need to be changed to account for scale
            if not inst.ignore_channel_scale:
                target: PulseChannel = inst.channel
                inst.samples = np.asarray(inst.samples) * target.scale
                inst.ignore_channel_scale = True
            return inst

        if isinstance(inst, Pulse):
            if inst.shape not in ignored_shapes:
                # check if the pulse has already been compiled!
                hash_ = self.hash_pulse(inst)
                if hash_ in pulse_lookup:
                    inst = pulse_lookup[hash_]
                    return inst

                # evaluate for non-ignored shapes
                edge = inst.duration / 2.0 - inst.channel.sample_time * 0.5
                samples = int(np.ceil(inst.duration / inst.channel.sample_time - 1e-10))
                t = np.linspace(start=-edge, stop=edge, num=samples)
                pulse_shape = self.evaluate(inst, t, 0.0)
                if not inst.ignore_channel_scale:
                    pulse_shape = pulse_shape * inst.channel.scale
                inst = CustomPulse(inst.channel, pulse_shape, True)
                pulse_lookup[hash] = inst

            elif not inst.ignore_channel_scale:
                # shapes that are ignored are still checked for scale!
                inst.amp *= inst.channel.scale
                inst.ignore_channel_scale = True

            return inst

        raise ValueError(
            f"Expected to see a Pulse or CustomPulse type, got {type(inst)} instead."
        )

    def hash_pulse(self, pulse: Pulse) -> str:
        """Hashs a pulse object."""
        return hash(
            (
                pulse.channel.partial_id,
                pulse.shape,
                pulse.width,
                pulse.amp,
                pulse.phase,
                pulse.drag,
                pulse.rise,
                pulse.amp_setup,
                pulse.scale_factor,
                pulse.zero_at_edges,
                pulse.beta,
                pulse.frequency,
                pulse.internal_phase,
                pulse.std_dev,
                pulse.square_width,
                pulse.ignore_channel_scale,
            )
        )


class ReturnSanitisation(TransformPass):
    """Squashes all :class:`Return` instructions into a single one. Adds a :class:`Return`
    with all acquisitions if none is found."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        returns = [inst for inst in ir.instructions if isinstance(inst, Return)]
        acquires = [inst for inst in ir.instructions if isinstance(inst, Acquire)]

        if returns:
            unique_variables = set(itertools.chain(*[ret.variables for ret in returns]))
            for ret in returns:
                ir._instructions.remove(ret)
        else:
            # If we don't have an explicit return, imply all results.
            unique_variables = set(acq.output_variable for acq in acquires)

        ir.returns(list(unique_variables))
        return ir


class RepeatSanitisation(TransformPass):
    """Adds repeat counts and repetition periods to :class:`Repeat` instructions. If none
    is found, a repeat instruction is added."""

    def __init__(self, model: QuantumHardwareModel, target_data: TargetData):
        """
        :param model: The hardware model contains the default repeat value, defaults to
            None.
        :param target_data: Target-related information.
        """
        self.model = model
        self.target_data = target_data

    def run(
        self,
        ir: InstructionBuilder,
        *args,
        compiler_config: CompilerConfig = None,
        **kwargs,
    ):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""
        num_shots = self.target_data.default_shots
        configured = False
        if compiler_config is not None and compiler_config.repeats is not None:
            configured = True
            num_shots = compiler_config.repeats

        repeats = [inst for inst in ir._instructions if isinstance(inst, Repeat)]
        if repeats:
            for rep in repeats:
                if rep.repeat_count is None:
                    rep.repeat_count = num_shots
                if rep.passive_reset_time is None and rep.repetition_period is None:
                    rep.passive_reset_time = self.target_data.QUBIT_DATA.passive_reset_time
            if not configured:
                num_shots = repeats[0].repeat_count
            if not all([rep.repeat_count == num_shots for rep in repeats]):
                raise ValueError(
                    "Inconsistent repeat_count information found. "
                    + f"Repeat instruction values: {[rep.repeat_count for rep in repeats]}"
                    + (f", CompilerConfig repeats: {num_shots}." if configured else ".")
                )
        else:
            ir.repeat(
                num_shots,
                passive_reset_time=self.target_data.QUBIT_DATA.passive_reset_time,
            )
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

    def __init__(self, model: QuantumHardwareModel):
        """:param model: The hardware model containing the frequency shift channels."""

        self.model = model

    def run(
        self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs
    ) -> InstructionBuilder:
        """
        :param ir: The QatIR as an instruction builder.
        :param res_mgr: The results manager.
        """

        freq_shift_channels = self.get_freq_shift_channels()
        if len(freq_shift_channels) == 0:
            return ir
        ir = self.add_freq_shift_to_ir(ir, set(freq_shift_channels.keys()))

        if res_mgr.check_for_type(ActiveChannelResults):
            res = res_mgr.lookup_by_type(ActiveChannelResults)
            res.target_map.update(freq_shift_channels)

        return ir

    def get_freq_shift_channels(self) -> dict[PulseChannel, Qubit]:
        """Returns all active frequency shift pulse channels found in the hardware model."""

        channels: dict[PulseChannel, Qubit] = dict()
        for qubit in self.model.qubits:
            try:
                freq_shift_pulse_ch = qubit.get_freq_shift_channel()
                if freq_shift_pulse_ch.active:
                    channels[freq_shift_pulse_ch] = qubit
            except KeyError:
                continue

        return channels

    @staticmethod
    def add_freq_shift_to_ir(
        ir: InstructionBuilder, freq_shift_channels: set[PulseChannel]
    ) -> InstructionBuilder:
        """Adds frequency shift instructions to the instruction builder."""

        durations = defaultdict(list)
        for inst in ir.instructions:
            if isinstance(inst, QuantumInstruction) and (duration := inst.duration) > 0:
                for target in inst.quantum_targets:
                    durations[target].append(duration)

        if len(durations) == 0:
            return ir

        max_duration = np.max([np.sum(duration) for duration in durations.values()])
        if max_duration == 0.0:
            return ir

        # Find the index if the first quantum instruction
        index = next(
            filter(
                lambda tup: True if isinstance(tup[1], QuantumInstruction) else None,
                enumerate(ir._instructions),
            )
        )[0]

        for channel in freq_shift_channels:
            pulse = Pulse(
                channel,
                shape=PulseShapeType.SQUARE,
                amp=channel.amp,
                phase=getattr(channel, "phase", 0),
                width=max_duration,
            )
            ir.insert(pulse, index)
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
            if isinstance(inst, (Pulse, Acquire, Delay, CustomPulse)):
                # only increment the durations for meaningful instructions
                pulse_chan_id = inst.quantum_targets[0].partial_id()
                durations[pulse_chan_id] += inst.duration
                new_instructions.append(inst)
            elif isinstance(inst, Synchronize):
                # determine the durations for syncs
                targets = inst.quantum_targets
                if (num_targets := len(targets)) <= 1:
                    log.info(
                        f"Synchronize instructions with {num_targets} does not have enough "
                        "targets to synchronise, it will be ignored."
                    )
                    continue
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

        repeats = [inst for inst in ir.instructions if isinstance(inst, Repeat)]
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

        tail, head = partition(
            lambda inst: isinstance(inst, (Sweep, Repeat)), ir.instructions
        )
        tail, head = list(tail), list(head)

        delimiters = [
            EndSweep() if isinstance(inst, Sweep) else EndRepeat() for inst in head
        ]

        ir.instructions = head + tail + delimiters[::-1]
        return ir


class DeviceUpdateSanitisation(TransformPass):
    """
    Duplicate DeviceUpdate instructions upsets the device injection mechanism, which causes corruption
    of the HW model.

    In fact, a DeviceInjector is currently 1-1 associated with a DeviceUpdate instruction. When multiple
    DeviceUpdate instructions (sequentially) inject the same "target", the first DeviceInjector assigns the
    (correct) value of the attribute (on the target) to the revert_value. At this point the HW model (or any
    other target kind) is dirty, and any subsequent DeviceInjector updater would surely assign
    the (wrong, usually a placeholder `Variable`) to its revert_value. This results in a corrupt HW model whereby
    reversion wouldn't have the desired effect.

    This pass is a (lazy) fix, which is to analyse when such cases happen and eliminate duplicate DeviceUpdate
    instructions that target THE SAME "attribute" on THE SAME "target" with THE SAME variable.
    """

    def run(
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        target_attr2value = defaultdict(list)
        for inst in ir.instructions:
            if isinstance(inst, DeviceUpdate):
                # target already validated to be a QuantumComponent during initialisation
                if not hasattr(inst.target, inst.attribute):
                    raise ValueError(
                        f"Attempting to assign {inst.value} to non existing attribute {inst.attribute}"
                    )

                if isinstance(inst.value, Variable):
                    target_attr2value[(inst.target, inst.attribute)].append(inst.value)

        for (target, attr), values in target_attr2value.items():
            if len(values) > 1:
                log.warning(
                    f"Multiple DeviceUpdate instructions attempting to update the same attribute '{attr}' on {target}"
                )

            unique_values = set(values)
            if len(unique_values) > 1:
                raise ValueError(
                    f"Cannot update the same attribute '{attr}' on {target} with distinct values {unique_values}"
                )

        new_instructions = []
        for inst in ir.instructions:
            if isinstance(inst, DeviceUpdate) and isinstance(inst.value, Variable):
                if next(iter(target_attr2value[(inst.target, inst.attribute)]), None):
                    target_attr2value[(inst.target, inst.attribute)].clear()
                    new_instructions.append(inst)
            else:
                new_instructions.append(inst)

        ir.instructions = new_instructions

        return ir
