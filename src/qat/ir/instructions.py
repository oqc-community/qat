# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable
from functools import cached_property
from pydoc import locate
from typing import Annotated, Any, Literal

from compiler_config.config import InlineResultsProcessing
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)
from typing_extensions import TypeAliasType

from qat.purr.compiler.instructions import BinaryOperator as LegacyBinaryOperator
from qat.purr.compiler.instructions import IndexAccessor as LegacyIndexAccessor
from qat.purr.compiler.instructions import Variable as LegacyVariable
from qat.purr.utils.logger import get_default_logger
from qat.utils.pydantic import (
    AllowExtraFieldsModel,
    FrozenSet,
    NoExtraFieldsModel,
    QubitId,
    ValidatedList,
    ValidatedSet,
    _validate_set,
)

log = get_default_logger()


### Instructions

BASE_INSTR = "qat.ir.instructions.Instruction"


class Instruction(AllowExtraFieldsModel):
    @computed_field
    @cached_property
    def instr_type(self) -> str:
        """
        Returns the type of the instruction, which is the class name.
        """
        return self.__class__.__module__ + "." + self.__class__.__name__

    def __iter__(self):
        yield self

    def __reversed__(self):
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
    instructions: ValidatedList[Instruction] = []

    def add(self, *instructions: Instruction, flatten: bool = False):
        for instruction in instructions:
            if flatten:
                for sub_instruction in instruction:
                    self.instructions.append(sub_instruction)
            else:
                self.instructions.append(instruction)
        return self

    def __iter__(self):
        """Iterator over the flattened instructions in the block."""
        for instruction in self.instructions:
            yield from instruction

    def __reversed__(self):
        """Reversed iterator over the flattened instructions in the block."""
        for instruction in reversed(self.instructions):
            yield from reversed(instruction)

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

    def flatten(self):
        flattened_instructions = []
        for instr in self:
            flattened_instructions.append(instr)
        self.instructions = flattened_instructions
        return self

    @field_serializer("instructions")
    def _serialize_instructions(self, instructions: list[Instruction]):
        return [inst.model_dump() for inst in self]

    @field_validator("instructions", mode="before")
    @classmethod
    def _rehydrate_instructions(cls, instructions):
        if isinstance(instructions, Instruction):
            return instructions
        if not isinstance(instructions, (list, ValidatedList)):
            raise TypeError(
                f"Expected `instructions` to be a list of `Instruction` instances or dictionaries, got {type(instructions)}."
            )

        rehydrated = []
        # Cache for located types to avoid repeated lookups
        type_cache: dict[str, type[Instruction]] = {}

        for instr in instructions:
            if isinstance(instr, dict):
                instr_type_str = instr.get("instr_type", BASE_INSTR)

                if instr_type_str not in type_cache:
                    type_cache[instr_type_str] = locate(instr_type_str)
                instr_type = type_cache[instr_type_str]
                rehydrated.append(instr_type(**instr))
            elif isinstance(instr, Instruction):
                rehydrated.append(instr)
            else:
                raise TypeError(
                    f"Instruction must be an Instruction instance or dict, got {type(instr)}."
                )

        return rehydrated


class Repeat(Instruction):
    """
    Global meta-instruction that applies to the entire list of instructions. Repeat
    value of the current operations, also known as shots.
    """

    repeat_count: int | None = None


class EndRepeat(Instruction): ...


class Return(Instruction):
    """A statement defining what to return from a quantum execution."""

    variables: list[str] = []

    @field_validator("variables", mode="before")
    @classmethod
    def _variables_as_list(cls, variables):
        variables = (
            []
            if variables is None
            else ([variables] if not isinstance(variables, list) else variables)
        )
        return variables


class ResultsProcessing(Instruction):
    """
    A meta-instruction that stores how the results are processed.
    """

    variable: str
    results_processing: InlineResultsProcessing = InlineResultsProcessing.Raw

    @classmethod
    def _from_legacy(cls, legacy_rp):
        # private as we dont want to support this in the long-term
        return cls(
            variable=legacy_rp.variable, results_processing=legacy_rp.results_processing
        )


### Variables
class Variable(Instruction):
    """
    States that this value is actually a variable that should be fetched instead.
    """

    name: str
    var_type: type | None = None
    value: Any = None

    @staticmethod
    def with_random_name():
        return Variable(name=str(uuid.uuid4()))

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, type={self.var_type}, value={self.value})"

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
        if var_type is not None and not isinstance(value, var_type) and value is not None:
            raise ValueError(f"Value provided has type {type(value)}: must be {var_type}")
        return value


class Label(Instruction):
    """
    Label to apply to a line of code. Used as anchors for other instructions like jumps.
    """

    name: str

    @staticmethod
    def with_random_name():
        return Label(name=f"label_{uuid.uuid4()}")

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class Jump(Instruction):
    """
    Classic jump instruction, should be linked to label with an optional condition.
    """

    label: Label | str
    condition: BinaryOperator | None = None

    @field_validator("label", mode="after")
    @classmethod
    def _validate_label(cls, label: str | Label):
        if isinstance(label, str):
            label = Label(name=label)
        return label

    @property
    def target(self):
        return self.label.name

    def __repr__(self):
        if self.condition is not None:
            return f"Jump if {str(self.condition)} -> {str(self.target)}"
        else:
            return f"Jump -> {str(self.target)}"


class LoopCount(int): ...


### Binary Operators


class BinaryOperator(NoExtraFieldsModel):
    """Binary operator, such as ``x == y``, ``x != y`` etc."""

    left: int | float | Variable
    right: int | float | Variable


class Equals(BinaryOperator):
    def __repr__(self):
        return f"{str(self.left)} == {str(self.right)}"


class NotEquals(BinaryOperator):
    def __repr__(self):
        return f"{str(self.left)} != {str(self.right)}"


class GreaterThan(BinaryOperator):
    def __repr__(self):
        return f"{str(self.left)} > {str(self.right)}"


class GreaterOrEqualThan(BinaryOperator):
    def __repr__(self):
        return f"{str(self.left)} >= {str(self.right)}"


class LessThan(BinaryOperator):
    def __repr__(self):
        return f"{str(self.left)} < {str(self.right)}"


class LessOrEqualThan(BinaryOperator):
    def __repr__(self):
        return f"{str(self.left)} <= {str(self.right)}"


class Plus(BinaryOperator):
    # TODO: Improve operators in Pydantic version
    def __repr__(self):
        return f"{str(self.left)} + {str(self.right)}"


AssignTypes = int | float | str | Variable | BinaryOperator
RecursiveAssignTypes = TypeAliasType(
    "ResursiveAssignTypes", "AssignTypes | list[RecursiveAssignTypes]"
)


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

    name: str
    value: RecursiveAssignTypes

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
            elif isinstance(value, LegacyBinaryOperator):
                return str(value).replace("variable ", "")
            return value

        return cls(name=legacy_assign.name, value=recursively_strip(legacy_assign.value))

    def __repr__(self):
        return f"Assign {self.name} = {str(self.value)}"


### Quantum Instructions
class QuantumInstruction(Instruction):
    """
    Any node that deals particularly with quantum operations. All quantum operations
    must have some sort of target on the quantum computer, such as a qubit, channel, or
    another form of component.
    """

    targets: Annotated[FrozenSet[str], BeforeValidator(_validate_set)] | None = Field(
        min_length=1, max_length=1, alias="target"
    )
    duration: float = Field(ge=0, default=0)  # in seconds

    model_config = ConfigDict(validate_by_name=True, populate_by_name=True)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(duration={self.duration}, targets={self.targets})"
        )

    @property
    def target(self):
        return next(iter(self.targets))


class QuantumInstructionBlock(QuantumInstruction, InstructionBlock):
    instructions: ValidatedList[QuantumInstruction] = []
    targets: Annotated[ValidatedSet[str], BeforeValidator(_validate_set)] = Field(
        default_factory=lambda: ValidatedSet[str]()
    )
    _duration_per_target: dict[str, float] = defaultdict(lambda: 0.0, dict())

    def add(self, *instructions: QuantumInstruction):
        super().add(*instructions)
        for instruction in instructions:
            for target in instruction.targets:
                self.targets.add(target)
                self._duration_per_target[target] += instruction.duration

            self.duration = max(self._duration_per_target.values())

        return self


class PhaseShift(QuantumInstruction):
    """
    A PhaseShift instruction is used to change the phase of waveforms sent down
    the pulse channel.
    """

    phase: float = 0.0


class FrequencySet(QuantumInstruction):
    """Set the frequency of a pulse channel."""

    frequency: float = 0.0


class FrequencyShift(QuantumInstruction):
    """Change the frequency of a pulse channel."""

    frequency: float = 0.0


class Delay(QuantumInstruction):
    """Instructs a quantum target to do nothing for a fixed time."""


class Synchronize(QuantumInstruction):
    """
    Tells the QPU to wait for all the target channels to be free before continuing
    execution on any of them.
    """

    duration: Literal[0] = 0

    targets: Annotated[FrozenSet[str], BeforeValidator(_validate_set)] = Field(min_length=2)


class PhaseSet(QuantumInstruction):
    """Sets the phase for a pulse channel.

    This sets the absolute phase of an NCO, unlike the :class:`PhaseShift`, which changes
    the phase relative to the current phase.
    """

    phase: float = 0.0


class PhaseReset(PhaseSet):
    phase: Literal[0.0] = 0.0


class Reset(QuantumInstruction):
    """Resets this qubit to its starting state."""

    targets: Annotated[FrozenSet[str], BeforeValidator(_validate_set)] = Field(
        max_length=0, default=FrozenSet[str](set())
    )
    qubit_targets: Annotated[FrozenSet[QubitId], BeforeValidator(_validate_set)] = Field(
        min_length=1, max_length=1, alias="qubit_target"
    )

    @property
    def target(self):
        return None

    @property
    def qubit_target(self):
        return next(iter(self.qubit_targets))
