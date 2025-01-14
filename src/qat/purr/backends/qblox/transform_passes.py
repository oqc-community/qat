# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import itertools
from numbers import Number
from typing import Dict, List

import numpy as np
from compiler_config.config import MetricsType

from qat.purr.backends.qblox.algorithm import stable_partition
from qat.purr.backends.qblox.metrics_base import MetricsManager
from qat.purr.backends.qblox.pass_base import QatIR, TransformPass, get_hardware_model
from qat.purr.backends.qblox.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CustomPulse,
    EndRepeat,
    EndSweep,
    Instruction,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
    Repeat,
    Return,
    Sweep,
)


class PhaseOptimisation(TransformPass):
    """
    Extracted from QuantumExecutionEngine.optimize()
    """

    def run(
        self, ir: QatIR, res_mgr: ResultManager, met_mgr: MetricsManager, *args, **kwargs
    ):
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        accum_phaseshifts: Dict[PulseChannel, PhaseShift] = {}
        optimized_instructions: List[Instruction] = []
        for instruction in builder.instructions:
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

        builder.instructions = optimized_instructions
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, len(optimized_instructions)
        )


class PostProcessingOptimisation(TransformPass):
    """
    Extracted from LiveDeviceEngine.optimize()
    Better pass name/id ?
    """

    def run(
        self, ir: QatIR, res_mgr: ResultManager, met_mgr: MetricsManager, *args, **kwargs
    ):
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        pp_insts = [val for val in builder.instructions if isinstance(val, PostProcessing)]
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
        builder.instructions = [val for val in builder.instructions if val not in discarded]
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, len(builder.instructions)
        )


class ScopeSanitisation(TransformPass):

    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        """
        Bubbles up all sweeps and repeats to the beginning of the list.
        Adds delimiter instructions to the repeats and sweeps signifying the end of their scopes.

        Intended for legacy existing builders and the relative order of instructions guarantees backwards
        compatibility.
        """

        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        head, tail = stable_partition(
            builder.instructions, lambda inst: isinstance(inst, (Sweep, Repeat))
        )

        delimiters = [
            EndSweep() if isinstance(inst, Sweep) else EndRepeat() for inst in head
        ]

        builder.instructions = head + tail + delimiters[::-1]


class RepeatSanitisation(TransformPass):
    def __init__(self, model: QuantumHardwareModel = None):
        self.model = model

    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        """
        Fixes repeat instructions if any with default values from the HW model
        Adds a repeat instructions if none is found
        """

        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        model = self.model or get_hardware_model(args, kwargs)

        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        if repeats:
            for rep in repeats:
                if rep.repeat_count is None:
                    rep.repeat_count = model.default_repeat_count
                if rep.repetition_period is None:
                    rep.repetition_period = model.default_repetition_period
        else:
            builder.repeat(model.default_repeat_count, model.default_repetition_period)


class ReturnSanitisation(TransformPass):
    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        returns = [inst for inst in builder.instructions if isinstance(inst, Return)]
        acquires = [inst for inst in builder.instructions if isinstance(inst, Acquire)]

        if returns:
            unique_variables = set(itertools.chain(*[ret.variables for ret in returns]))
            for ret in returns:
                builder.instructions.remove(ret)
        else:
            # If we don't have an explicit return, imply all results.
            unique_variables = set(acq.output_variable for acq in acquires)

        builder.returns(list(unique_variables))


class DesugaringPass(TransformPass):
    """
    Transforms syntactic sugars and implicit effects. Not that it applies here, but a classical
    example of a sugar syntax is the ternary expression "cond ? val1 : val2" which is nothing more
    than an if else in disguise.

    The goal of this pass is to desugar any syntactic and semantic sugars. One of these sugars
    is iteration constructs such as Sweep and Repeat.
    """

    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        for inst in builder.instructions:
            if isinstance(inst, Sweep):
                count = len(next(iter(inst.variables.values())))
                iter_name = f"sweep_{hash(inst)}"
                inst.variables[iter_name] = np.linspace(1, count, count, dtype=int).tolist()
