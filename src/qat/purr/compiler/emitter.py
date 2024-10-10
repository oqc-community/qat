# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from __future__ import annotations

from typing import Any, List, Set

from qat.purr.compiler.instructions import (
    Acquire,
    PostProcessing,
    QuantumInstruction,
    Repeat,
    Return,
    Sweep,
)


class TimelineSegment:
    """Segment of the execution timeline."""

    def __init__(
        self,
        instruction: QuantumInstruction,
        dependencies: Set[str] = None,
        reliant: List[Any] = None,
    ):
        self.instruction = instruction
        self.scheduling_dependencies = dependencies or {
            comp.full_id() for comp in instruction.quantum_targets
        }
        self.reliant_instructions = reliant or []

    def __repr__(self):
        sep = ";" if any(self.reliant_instructions) else ""
        sched_deps = (
            f"[{','.join(self.scheduling_dependencies)}]"
            if any(self.scheduling_dependencies)
            else ""
        )
        return (
            f"{self.instruction.duration}: {sched_deps} {str(self.instruction)}"
            f"{sep}{';'.join(str(inst) for inst in self.reliant_instructions)}"
        )


class QatFile:
    """In-memory representation of our instruction file."""

    def __init__(self):
        self.timeline: List[TimelineSegment] = []
        self.meta_instructions = []

    def add(self, *args, **kwargs):
        self.timeline.append(TimelineSegment(*args, **kwargs))

    def add_meta(self, instruction):
        if isinstance(instruction, Return):
            existing_return = next(
                iter(meta for meta in self.meta_instructions if isinstance(meta, Return)),
                None,
            )
            if existing_return is not None:
                existing_return.variables.extend(instruction.variables)
            else:
                self.meta_instructions.append(instruction)
        else:
            self.meta_instructions.append(instruction)

    @property
    def instructions(self):
        return [seg.instruction for seg in self.timeline]

    def get_pp_for_variable(self, target_var):
        results = []
        for instruction in self.instructions:
            if (
                isinstance(instruction, PostProcessing)
                and instruction.output_variable == target_var
            ):
                results.append(instruction)

        return results

    @property
    def sweeps(self):
        return [sw for sw in self.meta_instructions if isinstance(sw, Sweep)]

    @property
    def repeat(self):
        return next(
            iter(sw for sw in self.meta_instructions if isinstance(sw, Repeat)), None
        )

    @property
    def return_(self):
        return next(
            iter(sw for sw in self.meta_instructions if isinstance(sw, Return)), None
        )


class InstructionEmitter:
    """
    Interface to the LLVM-driven QAT optimization and construction passes. For now we
    simulate what it might do in the future and ust output a Python object that
    simulates what our instruction set might look like.
    """

    def emit(self, instructions, hardware):
        qatf = QatFile()

        for inst in instructions:
            if isinstance(inst, PostProcessing):
                qatf.add(inst, [inst.acquire.channel.full_id()])
            elif isinstance(inst, QuantumInstruction):
                qatf.add(inst)
            else:
                qatf.add_meta(inst)

        # If we don't have an explicit return, imply all results.
        if not any(ret for ret in qatf.meta_instructions if isinstance(ret, Return)):
            # Only gather each variable once for the return.
            unique_variables = []
            for var in [
                acq.output_variable for acq in qatf.instructions if isinstance(acq, Acquire)
            ]:
                if var not in unique_variables:
                    unique_variables.append(var)
            qatf.add_meta(Return(unique_variables))

        # If we don't have a repeat, set it as 1.
        repeat_inst = next(
            (rep for rep in qatf.meta_instructions if isinstance(rep, Repeat)), None
        )
        if repeat_inst is None:
            qatf.add_meta(
                Repeat(hardware.default_repeat_count, hardware.default_repetition_period)
            )
        else:
            if repeat_inst.repeat_count is None:
                repeat_inst.repeat_count = hardware.default_repeat_count
            if repeat_inst.repetition_period is None:
                repeat_inst.repetition_period = hardware.default_repetition_period

        return qatf
