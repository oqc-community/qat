# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from copy import deepcopy
from numbers import Number
from typing import List

import numpy as np
from compiler_config.config import MetricsType

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.ir.instructions import PhaseReset as PydPhaseReset
from qat.ir.instructions import PhaseShift as PydPhaseShift
from qat.ir.measure import Acquire as PydAcquire
from qat.ir.measure import PostProcessing as PydPostProcessing
from qat.ir.waveforms import Pulse as PydPulse
from qat.middleend.passes.analysis import ActiveChannelResults
from qat.purr.backends.utilities import evaluate_shape
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, Qubit
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CustomPulse,
    Delay,
    Instruction,
    MeasurePulse,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
    PulseShapeType,
    QuantumInstruction,
    Reset,
    Synchronize,
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

        previous_instruction = None
        accum_phaseshifts: dict[PulseChannel, float] = defaultdict(float)
        optimized_instructions: list[Instruction] = []
        for instruction in ir.instructions:
            if isinstance(instruction, PhaseShift) and isinstance(
                instruction.phase, Number
            ):
                accum_phaseshifts[instruction.channel] += instruction.phase
            elif isinstance(instruction, (Pulse, CustomPulse, PhaseShift)):
                quantum_targets = getattr(instruction, "quantum_targets", [])
                if not isinstance(quantum_targets, List):
                    quantum_targets = [quantum_targets]
                for quantum_target in quantum_targets:
                    if not np.isclose(accum_phaseshifts[quantum_target] % (2 * np.pi), 0.0):
                        optimized_instructions.append(
                            PhaseShift(
                                quantum_target, accum_phaseshifts.pop(quantum_target)
                            )
                        )
                optimized_instructions.append(instruction)

            elif isinstance(instruction, PhaseReset):
                for channel in instruction.quantum_targets:
                    accum_phaseshifts.pop(channel, None)

                if isinstance(previous_instruction, PhaseReset):
                    unseen_targets = list(
                        set(instruction.quantum_targets)
                        - set(previous_instruction.quantum_targets)
                    )
                    previous_instruction.quantum_targets.extend(unseen_targets)
                else:
                    optimized_instructions.append(instruction)

            else:
                optimized_instructions.append(instruction)

            previous_instruction = instruction

        ir.instructions = optimized_instructions
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, len(optimized_instructions)
        )
        return ir


class PydPhaseOptimisation(TransformPass):
    """Iterates through the list of instructions and compresses contiguous
    :class:`PhaseShift` instructions.
    """

    def run(
        self,
        ir: PydQuantumInstructionBuilder,
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
            if isinstance(instruction, PydPhaseShift):
                accum_phaseshifts[instruction.target] += instruction.phase

            elif isinstance(instruction, PydPulse):
                target = instruction.target
                if not np.isclose(accum_phaseshifts[target] % (2 * np.pi), 0.0):
                    optimized_instructions.append(
                        PydPhaseShift(targets=target, phase=accum_phaseshifts.pop(target))
                    )
                optimized_instructions.append(instruction)

            elif isinstance(instruction, PydPhaseReset):
                for target in instruction.targets:
                    accum_phaseshifts.pop(target, None)

                if isinstance(previous_instruction, PydPhaseReset):
                    unseen_targets = instruction.targets - previous_instruction.targets
                    previous_instruction.targets.update(unseen_targets)
                else:
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

        pp_insts = [val for val in ir.instructions if isinstance(val, PostProcessing)]
        discarded = []
        for pp in pp_insts:
            if pp.acquire.mode == AcquireMode.SCOPE:
                if (
                    pp.process == PostProcessType.MEAN
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


class PydPostProcessingSanitisation(TransformPass):
    """Checks that the :class:`PostProcessing` instructions that follow an acquisition are
    suitable for the acquisition mode, and removes them if not.

    Extracted from :meth:`qat.purr.backends.live.LiveDeviceEngine.optimize`.
    """

    def run(
        self,
        ir: PydQuantumInstructionBuilder,
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
            if isinstance(instr, PydAcquire):
                if instr.mode == AcquireMode.RAW:
                    raise ValueError(
                        "Invalid acquire mode. The target machine does not support "
                        "a RAW acquire mode."
                    )

                if instr.output_variable:
                    acquire_mode_output_var_map[instr.output_variable] = instr.mode

            elif isinstance(instr, PydPostProcessing):
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

    def _valid_pp(self, acquire_mode: AcquireMode, pp: PydPostProcessing) -> bool:
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
    """Rounds up the durations of quantum instructions so they are multiples of the clock
    cycle.

    Only supports quantum instructions with static non-zero durations. Assumes that
    instructions with a non-zero duration only act on a single pulse channel. The
    santisiation is done for all instructions simultaneously using numpy for performance.

    The Pydantic version of the pass will require the (pydantic equivalent) pass
    :class:`ActivePulseChannelAnalysis <qat.middleend.passes.analysis.ActivePulseChannelAnalysis>`
    to have run, with results saved to the results manager to extract pulse channel
    information.

    .. warning::

        This pass has the potential to invalidate the timings for sequences of instructions
        that are time-sensitive. For example, if a pulse has an invalid time, it will round
        it up to the nearest integer multiple. Furthemore, it will assume that
        :class:`Acquire` instructions have no delay. This can be forced explicitly using the
        :class:`AcquireSanitisation` pass.
    """

    # TODO: PydInstructionGranularitySanitisation: will require the PydActivePulseChannelAnalysis
    # to extract the pulse/physical channel information (COMPILER-394)
    # TODO: replace clock_cycle with target data (COMPILER-395)

    def __init__(self, clock_cycle: float = 8e-9):
        """:param clock_cycle: The clock cycle to round to."""

        self.clock_cycle = clock_cycle

    def run(self, ir: InstructionBuilder, *args, **kwargs) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """

        filter_instructions = [
            inst
            for inst in ir.instructions
            if isinstance(inst, (Pulse, Acquire, Delay, CustomPulse))
        ]

        durations = np.asarray([inst.duration for inst in filter_instructions])

        # We substract 1e-10 before rounding as floating point errors can lead to rounding
        # problems.
        multiples = durations / self.clock_cycle
        rounded_multiples = np.ceil(multiples - 1e-10).astype(int)
        durations_equal = np.isclose(multiples, rounded_multiples)

        invalid_instructions: set[str] = set()
        for idx in np.where(np.logical_not(durations_equal))[0]:
            inst = filter_instructions[idx]
            invalid_instructions.add(str(inst))
            new_duration = rounded_multiples[idx] * self.clock_cycle
            if isinstance(inst, CustomPulse):
                sample_time = inst.quantum_targets[0].sample_time
                padding = int(np.round((new_duration - durations[idx]) / sample_time, 0))
                inst.samples.extend([0.0 + 0.0j] * padding)
            elif isinstance(inst, Pulse):
                inst.width = new_duration
            else:
                inst.time = new_duration

        if len(invalid_instructions) > 1:
            log.info(
                "The following instructions do not have durations that are integer "
                f"multiples of the clock cycle {self.clock_cycle}, and will be rounded up: "
                + ", ".join(set(invalid_instructions))
            )

        return ir


class InitialPhaseResetSanitisation(TransformPass):
    """
    Checks if every active pulse channel has a phase reset in the beginning.

    .. warning::

        This pass implies that an `ActiveChannelAnalysis` is performed prior to this pass.
    """

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        active_targets = list(res_mgr.lookup_by_type(ActiveChannelResults).targets)

        if active_targets:
            ir.insert(PhaseReset(active_targets), 0)

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

    def __init__(self, duration_limit: float = 1e-03):
        """
        :param duration_limit: The maximum allowed clock cycles per instruction.

        .. warning::

            The pass will assume that the durations of instructions are sanitised to the
            granularity of the pulse channels. If instructions that do not meet the criteria are
            provided, it might produce incorrect instructions (i.e., instructions that are shorter than
            the clock cycle). This can be enforced using the :class:`InstructionGranularitySanitisation <qat.middleend.passes.transform.InstructionGranularitySanitisation>`
            pass.
        """

        # TODO: update to target data (COMPILER-395)
        if duration_limit == 0:
            raise ValueError("Instruction duration limit cannot be zero.")
        self.duration_limit = duration_limit

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """
        duration_limit = self.duration_limit

        new_instructions = []
        for instr in ir.instructions:
            if isinstance(instr, Delay) and instr.duration > duration_limit:
                new_instructions.extend(self._batch_delay(instr, duration_limit))

            elif (
                isinstance(instr, Pulse)
                and instr.width > duration_limit
                and instr.shape == PulseShapeType.SQUARE
            ):
                new_instructions.extend(self._batch_square_pulse(instr, duration_limit))

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
    of the instruction list.

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

        active_pulse_channels: ActiveChannelResults = res_mgr.lookup_by_type(
            ActiveChannelResults
        )
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
                    if qubit in qubit_map and qubit_map[qubit] == None:
                        qubit_map[qubit] = True
            else:
                target = active_pulse_channels.target_map[next(iter(inst.quantum_targets))]
                if qubit_map[target] == None:
                    qubit_map[target] = False

            if all([val != None for val in qubit_map.values()]):
                break

        for qubit, val in qubit_map.items():
            if not val:
                ir.add(Reset(qubit))
        return ir


class ResetsToDelays(TransformPass):
    """
    Transforms :class:`Reset` operations to :class:`Delay`s.

    Note that the delays do not necessarily agree with the granularity of the underlying target machine.
    This can be enforced using the :class:`InstructionGranularitySanitisation <qat.middleend.passes.transform.InstructionGranularitySanitisation>`
    pass.
    """

    def __init__(self, passive_reset_time: float):
        """
        :param passive_reset_time: The time added to the end of each shot to allow the
                                state of the qubits to reset.
        """
        self.passive_reset_time = passive_reset_time

    def run(
        self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs
    ) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """
        active_pulse_channels: ActiveChannelResults = res_mgr.lookup_by_type(
            ActiveChannelResults
        )

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
                            time=self.passive_reset_time,
                        )
                    )

            else:
                new_instructions.append(instr)

        ir.instructions = new_instructions
        return ir


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
