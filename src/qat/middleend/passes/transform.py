# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import itertools
from collections import defaultdict

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
from qat.ir.instructions import Return as PydReturn
from qat.ir.measure import Acquire as PydAcquire
from qat.ir.measure import MeasureBlock as PydMeasureBlock
from qat.ir.measure import PostProcessing as PydPostProcessing
from qat.ir.waveforms import Pulse as PydPulse
from qat.purr.compiler.instructions import (
    AcquireMode,
    PostProcessType,
    ProcessAxis,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


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

                if (
                    isinstance(previous_instruction, PydPhaseReset)
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


class PydReturnSanitisation(TransformPass):
    """Squashes all :class:`Return` instructions into a single one. Adds a :class:`Return`
    with all acquisitions if none is found."""

    def run(self, ir: PydQuantumInstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`QuantumInstructionBuilder`."""

        returns = [inst for inst in ir if isinstance(inst, PydReturn)]
        measure_blocks = [inst for inst in ir if isinstance(inst, PydMeasureBlock)]

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
