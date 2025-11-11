# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import (
    BeforeValidator,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from qat.ir.instructions import (
    Delay,
    Instruction,
    QuantumInstruction,
    QuantumInstructionBlock,
    Synchronize,
)
from qat.ir.waveforms import Pulse

# The following things from legacy instructions are unchanged, so just import for now.
from qat.purr.compiler.instructions import AcquireMode, PostProcessType, ProcessAxis
from qat.utils.pydantic import QubitId, ValidatedList, ValidatedSet, _validate_set


class Acquire(QuantumInstruction):
    suffix_incrementor: int = 0
    duration: float = 1e-6
    mode: AcquireMode = AcquireMode.INTEGRATOR
    delay: float | None = 0.0
    filter: Pulse | None = Field(default=None)
    output_variable: str | None = None
    rotation: float | None = 0.0
    threshold: float | None = 0.0

    @property
    def pulse_channel(self):
        return next(iter(self.targets))

    @property
    def target(self):
        return next(iter(self.targets))

    @model_validator(mode="before")
    def _validate_filter(cls, data: dict):
        if isinstance(data, dict):
            duration = data.get("duration")
            filter = data.get("filter")

            if filter:
                filter_duration = (
                    filter.duration if isinstance(filter, Pulse) else filter["duration"]
                )
                # TODO: COMPILER-722 -- with this in place, it required to add a duration to both the pulse
                #  and the Acquire instruction, would be better to figure out how to share information in some cases.
                if filter_duration == 0:  # < 0 condition already tested in `Waveform`
                    raise ValueError("Filter duration cannot be equal to zero.")

                if not np.isclose(filter_duration, duration, atol=1e-9):
                    raise ValueError(
                        f"Filter duration '{filter_duration}' must be equal to Acquire "
                        f"duration '{duration}'."
                    )

        return data


class PostProcessing(Instruction):
    """
    States what post-processing should happen after data has been acquired. This can
    happen in the FPGA's or a software post-process.
    """

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

    @field_validator("args", mode="before")
    def _validate_args(cls, args=[]):
        """Ensures that the args are not numpy arrays or numpy numbers."""

        args = [args] if not isinstance(args, (list, np.ndarray)) else args
        return np.asarray(args).tolist()


VALID_MEASURE_INSTR = Synchronize | Acquire | Pulse | Delay


class MeasureBlock(QuantumInstructionBlock):
    """
    Encapsulates a measurement of a single (or multiple) qubit(s).
    It should only contain instructions that are associated with a
    measurement such as a measure pulse, an acquire or a synchronize.
    """

    instructions: ValidatedList[VALID_MEASURE_INSTR] = Field(
        default_factory=lambda: ValidatedList[VALID_MEASURE_INSTR]()
    )

    qubit_targets: Annotated[ValidatedSet[QubitId], BeforeValidator(_validate_set)] = Field(
        default_factory=lambda: ValidatedSet()
    )
    _output_variables: list[str] = PrivateAttr(default=[])

    def add(self, *instructions: QuantumInstruction):
        for instruction in instructions:
            if isinstance(instruction, Acquire) and (delay := instruction.delay) > 0.0:
                self._output_variables.append(instruction.output_variable)
                super().add(Delay(target=instruction.target, duration=delay))
                instruction.delay = 0.0
            super().add(instruction)
        return self

    @property
    def output_variables(self):
        return self._output_variables


acq_mode_process_axis = {
    ProcessAxis.SEQUENCE: AcquireMode.INTEGRATOR,
    ProcessAxis.TIME: AcquireMode.SCOPE,
    "sequence": AcquireMode.INTEGRATOR,
    "time": AcquireMode.SCOPE,
}
