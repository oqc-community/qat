# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from numbers import Number
from typing import Dict, List

from compiler_config.config import MetricsType

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import PhaseReset as PydPhaseReset
from qat.ir.instructions import PhaseShift as PydPhaseShift
from qat.ir.waveforms import Pulse as PydPulse
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import (
    AcquireMode,
    CustomPulse,
    Instruction,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
)


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
        ir: QuantumInstructionBuilder,
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
