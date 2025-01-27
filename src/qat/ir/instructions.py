# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, List, Literal, Optional

import numpy as np
from compiler_config.config import InlineResultsProcessing
from pydantic import Field, field_validator, model_validator

from qat.purr.utils.logger import get_default_logger
from qat.utils.pydantic import NoExtraFieldsModel, ValidatedList, ValidatedSet

log = get_default_logger()


### Instructions
class Instruction(NoExtraFieldsModel):
    inst: Literal["Instruction"] = "Instruction"

    def __iter__(self):
        yield self

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @property
    def number_of_instructions(self):
        return 1


class InstructionBlock(Instruction, Iterable):
    inst: Literal["InstructionBlock"] = "InstructionBlock"
    instructions: ValidatedList[Instruction] = ValidatedList[Instruction]([])

    def add(self, *instructions: Instruction, flatten: bool = False):
        for instruction in instructions:
            if flatten:
                for sub_instruction in instruction:
                    self.instructions.append(sub_instruction)
            else:
                self.instructions.append(instruction)

    def __iter__(self):
        for instruction in self.instructions:
            yield from instruction

    @property
    def number_of_instructions(self):
        n = 0
        for instr in self:
            n += instr.number_of_instructions
        return n


class Repeat(Instruction):
    """
    Global meta-instruction that applies to the entire list of instructions. Repeat
    value of the current operations, also known as shots.
    """

    inst: Literal["Repeat"] = "Repeat"
    repeat_count: Optional[int] = None
    repetition_period: Optional[float] = None


class Assign(Instruction):
    """
    Assigns the variable 'x' the value 'y'. This can be performed as a part of running
    on the QPU or by a post-processing pass.
    """

    # set as any for now... not sure what the restrictions would be here
    inst: Literal["Assign"] = "Assign"
    name: str
    value: Any

    @classmethod
    def _from_legacy(cls, legacy_assign):
        # private as we dont want to support this in the long-term
        return cls(name=legacy_assign.name, value=legacy_assign.value)


class Return(Instruction):
    """A statement defining what to return from a quantum execution."""

    inst: Literal["Return"] = "Return"
    variables: List[str] = []

    @field_validator("variables", mode="before")
    @classmethod
    def _variables_as_list(cls, variables):
        variables = (
            []
            if variables == None
            else ([variables] if not isinstance(variables, List) else variables)
        )
        return variables


class ResultsProcessing(Instruction):
    """
    A meta-instruction that stores how the results are processed.
    """

    inst: Literal["ResultsProcessing"] = "ResultsProcessing"
    variable: str
    results_processing: InlineResultsProcessing = InlineResultsProcessing.Raw

    @classmethod
    def _from_legacy(cls, legacy_rp):
        # private as we dont want to support this in the long-term
        return cls(
            variable=legacy_rp.variable, results_processing=legacy_rp.results_processing
        )


### Quantum Instructions
class QuantumInstruction(Instruction):
    """
    Any node that deals particularly with quantum operations. All quantum operations
    must have some sort of target on the quantum computer, such as a qubit, channel, or
    another form of component.
    """

    inst: Literal["QuantumInstruction"] = "QuantumInstruction"
    targets: ValidatedSet[str]
    duration: float = Field(ge=0, default=0)  # in seconds

    @model_validator(mode="before")
    def validate_targets(cls, data, field_name="targets"):
        targets = data.get(field_name, [])
        annotation = cls.model_fields[field_name].annotation

        if isinstance(targets, (ValidatedSet, set)):
            data[field_name] = annotation(targets)

        elif isinstance(targets, (list, tuple, np.ndarray)):
            if len(unique_targets := set(targets)) < len(targets):
                log.warning(
                    f"`QuantumInstruction` has duplicate targets {targets}. Duplicates have been removed."
                )
            data[field_name] = annotation(unique_targets)

        elif isinstance(targets, (str, int)):
            data[field_name] = annotation({targets})

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(duration={self.duration}, targets={self.targets})"
        )


class QuantumInstructionBlock(QuantumInstruction, InstructionBlock):
    inst: Literal["QuantumInstructionBlock"] = "QuantumInstructionBlock"
    instructions: ValidatedList[QuantumInstruction] = []
    targets: ValidatedSet[str] = ValidatedSet()
    _duration_per_target: dict[str, float] = defaultdict(lambda: 0.0, dict())

    def add(self, *instructions: QuantumInstruction):
        super().add(*instructions)
        for instruction in instructions:
            for target in instruction.targets:
                self.targets.add(target)
                self._duration_per_target[target] += instruction.duration

            self.duration = max(self._duration_per_target.values())


class PhaseShift(QuantumInstruction):
    """
    A PhaseShift instruction is used to change the phase of waveforms sent down
    the pulse channel.
    """

    inst: Literal["PhaseShift"] = "PhaseShift"
    targets: ValidatedSet[str] = Field(max_length=1)
    phase: float = 0.0

    @property
    def target(self):
        return next(iter(self.targets))


class FrequencyShift(QuantumInstruction):
    """Change the frequency of a pulse channel."""

    inst: Literal["FrequencyShift"] = "FrequencyShift"
    targets: ValidatedSet[str] = Field(max_length=1)
    frequency: float = 0.0

    @property
    def target(self):
        return next(iter(self.targets))


class Delay(QuantumInstruction):
    """Instructs a quantum target to do nothing for a fixed time."""

    inst: Literal["Delay"] = "Delay"


class Synchronize(QuantumInstruction):
    """
    Tells the QPU to wait for all the target channels to be free before continuing
    execution on any of them.
    """

    inst: Literal["Synchronize"] = "Synchronize"


class PhaseReset(QuantumInstruction):
    """
    Reset the phase shift of given pulse channels, or the pulse channels of given qubits.
    """

    inst: Literal["PhaseReset"] = "PhaseReset"


class Reset(QuantumInstruction):
    """Resets this qubit to its starting state."""

    inst: Literal["Reset"] = "Reset"
