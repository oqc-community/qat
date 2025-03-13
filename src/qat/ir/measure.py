# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from pydantic import Field, ValidationInfo, field_validator, model_validator

from qat.ir.instructions import (
    Instruction,
    QuantumInstruction,
    QuantumInstructionBlock,
    Synchronize,
)
from qat.ir.waveforms import Pulse

# The following things from legacy instructions are unchanged, so just import for now.
from qat.purr.compiler.instructions import AcquireMode, PostProcessType, ProcessAxis
from qat.utils.pydantic import QubitId, ValidatedSet


class Acquire(QuantumInstruction):
    inst: Literal["Acquire"] = "Acquire"
    targets: ValidatedSet[str] = Field(max_length=1)
    suffix_incrementor: int = 0
    duration: float = 1e-6
    mode: AcquireMode = AcquireMode.INTEGRATOR
    delay: Optional[float] = 0.0
    filter: Optional[Pulse] = Field(default=None)
    output_variable: str | None = None

    @property
    def pulse_channel(self):
        return next(iter(self.targets))

    @property
    def target(self):
        return next(iter(self.targets))

    @field_validator("filter", mode="before")
    @classmethod
    def _validate_filter(cls, filter, info: ValidationInfo):
        duration = info.data["duration"]

        if filter:
            if filter.duration == 0:  # < 0 condition already tested in `Waveform`
                raise ValueError(f"Filter duration cannot be equal to zero.")

            if not np.isclose(filter.duration, duration, atol=1e-9):
                raise ValueError(
                    f"Filter duration '{filter.duration}' must be equal to Acquire "
                    f"duration '{duration}'."
                )

        return filter


class PostProcessing(Instruction):
    """
    States what post-processing should happen after data has been acquired. This can
    happen in the FPGA's or a software post-process.
    """

    inst: Literal["PostProcessing"] = "PostProcessing"
    output_variable: str | None = None
    process_type: PostProcessType
    axes: list[ProcessAxis] = []
    args: list[float | complex] = []
    result_needed: bool = False

    @classmethod
    def _from_legacy(cls, legacy_pp):
        # private as we dont want to support this in the long-term
        return cls(
            output_variable=legacy_pp.output_variable,
            process_type=legacy_pp.process.value,
            axes=legacy_pp.axes,
            args=legacy_pp.args,
            result_needed=legacy_pp.result_needed,
        )

    @field_validator("axes", mode="before")
    @classmethod
    def _axes_as_list(cls, axes=None):
        if axes:
            return axes if isinstance(axes, list) else [axes]
        return []


class MeasureBlock(QuantumInstructionBlock):
    """
    Encapsulates a measurement of a single (or multiple) qubit(s).
    It should only contain instructions that are associated with a
    measurement such as a measure pulse, an acquire or a synchronize.
    """

    inst: Literal["MeasureBlock"] = "MeasureBlock"
    qubit_targets: ValidatedSet[QubitId] = ValidatedSet()
    _valid_instructions: Literal[(Synchronize, Pulse, Acquire)] = (
        Synchronize,
        Pulse,
        Acquire,
    )

    def add(self, *instructions: QuantumInstruction):
        self._validate_instruction(*instructions)
        for instruction in instructions:
            if isinstance(instruction, Acquire):
                self._duration_per_target[instruction.target] += instruction.delay
            QuantumInstructionBlock.add(self, instruction)

    def _validate_instruction(self, *instructions: QuantumInstruction):
        for instruction in instructions:
            if not isinstance(instruction, self._valid_instructions):
                raise TypeError(
                    f"Instruction {instruction} not suitable for `MeasureBlock`. Instruction type should be in {self._valid_instructions}."
                )
            elif isinstance(instruction, Pulse) and instruction.type.value != "measure":
                raise TypeError(f"Pulse {instruction} is not a measure pulse.")

    @model_validator(mode="before")
    def validate_targets(cls, data, field_name="qubit_targets"):
        data = super().validate_targets(data, field_name="targets")
        data = super().validate_targets(data, field_name="qubit_targets")
        return data


acq_mode_process_axis = {
    ProcessAxis.SEQUENCE: AcquireMode.INTEGRATOR,
    ProcessAxis.TIME: AcquireMode.SCOPE,
    "sequence": AcquireMode.INTEGRATOR,
    "time": AcquireMode.SCOPE,
}
