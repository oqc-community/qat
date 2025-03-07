# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable
from pydoc import locate
from typing import Any, List, Literal, Optional, Union

import numpy as np
from compiler_config.config import InlineResultsProcessing
from pydantic import Field, field_serializer, field_validator, model_validator

# The following things from legacy instructions are unchanged, so just import for now.
from qat.purr.compiler.instructions import IndexAccessor as LegacyIndexAccessor
from qat.purr.compiler.instructions import Variable as LegacyVariable
from qat.purr.utils.logger import get_default_logger
from qat.utils.pydantic import NoExtraFieldsModel, QubitId, ValidatedList, ValidatedSet

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

    @property
    def head(self):
        return self

    @property
    def tail(self):
        return self


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
        return sum(1 for _ in self)

    @property
    def head(self):
        if self.number_of_instructions:
            return self.instructions[0]
        raise IndexError(
            f"Cannot access first element of an empty `{self.__class__.__name__}`."
        )

    @property
    def tail(self):
        if self.number_of_instructions:
            return self.instructions[-1]
        raise IndexError(
            f"Cannot access last element of an empty `{self.__class__.__name__}`."
        )


class Repeat(Instruction):
    """
    Global meta-instruction that applies to the entire list of instructions. Repeat
    value of the current operations, also known as shots.
    """

    inst: Literal["Repeat"] = "Repeat"
    repeat_count: Optional[int] = None
    repetition_period: Optional[float] = None


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


### Variables
class Variable(NoExtraFieldsModel):
    """
    States that this value is actually a variable that should be fetched instead.
    """

    name: str
    var_type: Optional[type] = None
    value: Any = None

    @staticmethod
    def with_random_name():
        return Variable(name=str(uuid.uuid4()))

    def __repr__(self):
        return self.name

    @field_serializer("var_type", when_used="json")
    def _serialise_type(self, var_type: type) -> str:
        # Types can't be (de)serialized, so we serialise as a string.
        return var_type.__name__

    @field_validator("var_type", mode="before")
    @classmethod
    def _validate_var_type(cls, var_type):
        # Types can't be (de)serialised, so we use a validator to manually
        # find the correct type.
        if isinstance(var_type, str):
            var_type = locate(var_type)
        return var_type

    @field_validator("value", mode="after")
    @classmethod
    def _validate_value_type(cls, value, val_info):
        var_type = val_info.data["var_type"]
        if var_type != None and not isinstance(value, var_type) and value != None:
            raise ValueError(f"Value provided has type {type(value)}: must be {var_type}")
        return value


class Assign(Instruction):
    """
    Assigns a value to a variable.

    This is used to assign some value (e.g. the results from an acquisition) to a variable.
    In the legacy instructions, `Assign` could be given a `Variable` that declares the value
    as another variable. It could also be given more complex structures, should as a list of
    `Variables`, or an `IndexAccessor`. For example, this could be used to assign each of the
    qubit acquisitions to a single register,

    ```
        c = [
            Qubit 1 measurements,
            Qubit 2 measurements,
            ...
        ]
    ```

    In general, declarations and allocations could be improved in future iterations, and
    functionality may change with improvements to the front end. For now, `Assign` has been
    adapted to support the required front end behaviour. The value is allowed to be a list
    (with recurssions supported) of strings that declare what Variable to point to, or a tuple
    for `IndexAccessor`.
    """

    # set as any for now... not sure what the restrictions would be here
    inst: Literal["Assign"] = "Assign"
    name: str
    value: Any

    @classmethod
    def _from_legacy(cls, legacy_assign):
        # private as we dont want to support this in the long-term

        def recursively_strip(value):
            if isinstance(value, list):
                return [recursively_strip(val) for val in value]
            elif isinstance(value, LegacyIndexAccessor):
                return (value.name, value.index)
            elif isinstance(value, LegacyVariable):
                return value.name
            return value

        return cls(name=legacy_assign.name, value=recursively_strip(legacy_assign.value))


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
    targets: ValidatedSet[str] = Field(max_length=0, default=ValidatedSet[str](set()))
    qubit_targets: Union[QubitId, set[QubitId]]

    @model_validator(mode="after")
    def validate_targets(self):
        if isinstance(self.qubit_targets, int):
            self.qubit_targets = set({self.qubit_targets})

        return self
