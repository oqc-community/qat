# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from numbers import Number
from typing import Dict, List

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
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CustomPulse,
    Delay,
    Instruction,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class PhaseOptimisation(TransformPass):
    """Iterates through the list of instructions and compresses contiguous
    :class:`PhaseShift` instructions.

    Extracted from :meth:`qat.purr.compiler.execution.QuantumExecutionEngine.optimize`.
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

        accum_phaseshifts: Dict[PulseChannel, PhaseShift] = {}
        optimized_instructions: List[Instruction] = []
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
                if not isinstance(quantum_targets, List):
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


class PydPhaseOptimisation(TransformPass):
    """Iterates through the list of instructions and compresses contiguous
    :class:`PhaseShift` instructions.

    Extracted from :meth:`qat.purr.compiler.execution.QuantumExecutionEngine.optimize`.
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

        accum_phaseshifts: dict[str, PydPhaseShift] = dict()
        optimized_instructions: list = []
        for instruction in ir:
            if isinstance(instruction, PydPhaseShift):
                if accum_phaseshift := accum_phaseshifts.get(instruction.target, None):
                    accum_phaseshift.phase += instruction.phase
                else:
                    accum_phaseshifts[instruction.target] = PydPhaseShift(
                        targets=instruction.target, phase=instruction.phase
                    )

            elif isinstance(instruction, PydPulse):
                if (target := instruction.target) in accum_phaseshifts:
                    optimized_instructions.append(accum_phaseshifts.pop(target))
                optimized_instructions.append(instruction)

            elif isinstance(instruction, PydPhaseReset):
                for target in instruction.targets:
                    accum_phaseshifts.pop(target, None)
                optimized_instructions.append(instruction)
            else:
                optimized_instructions.append(instruction)

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
        acquired_channels: set[PulseChannel] = set()

        for inst in ir.instructions:
            if isinstance(inst, Acquire):
                if inst.quantum_targets[0] in acquired_channels:
                    # The acquire has already been seen, so set the delay to zero.
                    inst.delay = 0.0
                acquired_channels.update(inst.quantum_targets)

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
    :class:`ActiveChannelAnalysis <qat.middleend.passes.analysis.ActiveChannelAnalysis>`
    to have run, with results saved to the results manager to extract pulse channel
    information.

    .. warning::

        This pass has the potential to invalidate the timings for sequences of instructions
        that are time-sensitive. For example, if a pulse has an invalid time, it will round
        it up to the nearest integer multiple. Furthemore, it will assume that
        :class:`Acquire` instructions have no delay. This can be forced explicitly using the
        :class:`AcquireSanitisation` pass.
    """

    # TODO: PydInstructionGranularitySanitisation: will require the PydActiveChannelAnalysis
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
