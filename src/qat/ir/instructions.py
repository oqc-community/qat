from __future__ import annotations

from copy import deepcopy
from pydoc import locate
from typing import Any, Dict, List, Literal, Optional, Union

from compiler_config.config import InlineResultsProcessing
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
)

from qat.purr.compiler.devices import PulseChannel, QuantumComponent, Qubit

# The following things from legacy instructions are unchanged, so just import for now.
from qat.purr.compiler.instructions import build_generated_name


### Sweeps
class SweepOperation(BaseModel):
    """Common parent for all things that need differentiating during a sweep."""

    pass


class SweepValue(SweepOperation):
    name: str
    value: Any


### Variables
class Variable(BaseModel):
    """
    States that this value is actually a variable that should be fetched instead.
    """

    name: str
    var_type: Optional[type] = None
    value: Any = None
    model_config = ConfigDict(validate_assignment=True)

    @staticmethod
    def with_random_name(existing_names=None, var_type=None, value=None):
        return Variable(
            name=build_generated_name(existing_names), var_type=var_type, value=value
        )

    def __repr__(self):
        return self.name

    @field_serializer("var_type", when_used="json")
    def _serialize_type(self, var_type: type) -> str:
        # Types can't be (de)serialized, so we serialize as a string.
        return var_type.__name__

    @field_validator("var_type", mode="before")
    @classmethod
    def _validate_var_type(cls, var_type):
        # Types can't be (de)serialized, so we use a validator to manually
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


class IndexAccessor(Variable):
    """Used to access an array index on a particular variable."""

    index: int

    def __repr__(self):
        return f"{self.name}[{self.index}]"


### Instructions
class Instruction(BaseModel):
    inst: Literal["Instruction"] = "Instruction"


class Repeat(Instruction):
    """
    Global meta-instruction that applies to the entire list of instructions. Repeat
    value of the current operations, also known as shots.
    """

    inst: Literal["Repeat"] = "Repeat"
    repeat_count: Optional[int] = None
    repetition_period: Optional[float] = None

    def __repr__(self):
        return f"repeat {self.repeat_count},{self.repetition_period}"


class QuantumMetadata(Instruction):
    inst: Literal["QuantumMetadata"] = "QuantumMetadata"


class Assign(Instruction):
    """
    Assigns the variable 'x' the value 'y'. This can be performed as a part of running
    on the QPU or by a post-processing pass.
    """

    # set as any for now... not sure what the restrictions would be here
    inst: Literal["Assign"] = "Assign"
    name: str
    value: Any

    def __repr__(self):
        return f"{self.name} = {str(self.value)}"

    @field_validator("value", mode="before")
    @classmethod
    def _list_to_vars(cls, val):
        # Variables are serialised as dicts: we need to defined how to deserialize them.
        if isinstance(val, list):
            lst = []
            for itm in val:
                if isinstance(itm, dict):
                    lst.append(Variable(**itm))
                else:
                    lst.append(itm)
            return lst
        return val


class Return(Instruction):
    """A statement defining what to return from a quantum execution."""

    inst: Literal["Return"] = "Return"
    variables: List[str] = []

    def __repr__(self):
        return f"return {','.join(self.variables)}"

    @field_validator("variables", mode="before")
    @classmethod
    def _variables_as_list(cls, variables):
        variables = (
            []
            if variables == None
            else ([variables] if not isinstance(variables, List) else variables)
        )
        return variables


class Sweep(Instruction):
    """
    This is a global meta-instruction that performs a sweep over values, effectively
    performing a loop over the instructions replacing a variable with a specific value
    per time.

    Nested sweeps are run in the order they're added and are performed after repeats. So
    a 1000 repeat with a sweep of four values, followed by a sweep with two values will
    run a total of 8000 iterations.
    """

    inst: Literal["Sweep"] = "Sweep"
    operations: List[SweepValue] = []
    variables: Dict[str, List[Any]] = None

    @property
    def length(self):
        return next(iter([len(value) for value in self.variables.values()]), 0)

    def __repr__(self):
        args = ",".join(key + "=" + str(value) for key, value in self.variables.items())
        return f"sweep {args}"

    @field_validator("operations", mode="before")
    @classmethod
    def _operations_as_list(cls, operations):
        if operations is None:
            operations = []
        elif not isinstance(operations, List):
            operations = [operations]
        return operations

    @field_validator("variables", mode="before")
    @classmethod
    def _create_variable_dict(cls, variables, info: ValidationInfo):
        if not variables:
            variables = {op.name: op.value for op in info.data["operations"]}

        # Get the length of the variables, which we will then assume is the sweep
        # length.
        sweep_lengths = [len(value) for value in variables.values()]
        if len(set(sweep_lengths)) > 1:
            raise ValueError("Sweep variables have inconsistent lengths.")
        return variables


class EndSweep(Instruction):
    """
    Basic scoping. Marks the end of the most recent sweep
    """

    inst: Literal["EndSweep"] = "EndSweep"

    def __repr__(self):
        return f"end_sweep"


class Label(Instruction):
    """
    Label to apply to a line of code. Used as anchors for other instructions like jumps.
    """

    inst: Literal["Label"] = "Label"
    name: str

    @staticmethod
    def with_random_name(existing_names=None):
        """Build a label with a randomly generated name."""
        return Label(name=build_generated_name(existing_names))

    def __repr__(self):
        return f"{self.name}:"


class Jump(Instruction):
    """
    Classic jump instruction, should be linked to label with an optional condition.
    The target can be a label or a string that points to a label.
    """

    inst: Literal["Jump"] = "Jump"
    target: str
    condition: Any = None

    def __repr__(self):
        if self.condition is not None:
            return f"if {str(self.condition)} -> {str(self.target)}"
        else:
            return f"-> {str(self.target)}"

    @field_validator("target", mode="before")
    @classmethod
    def _target_from_label(cls, target):
        if isinstance(target, Label):
            return target.name
        return target


class ResultsProcessing(Instruction):
    """
    A meta-instruction that stores how the results are processed.
    """

    inst: Literal["ResultsProcessing"] = "ResultsProcessing"
    variable: str
    results_processing: InlineResultsProcessing

    def __repr__(self):
        return f"{self.variable}: {str(self.results_processing.name)}"


### Quantum Instructions
class QuantumInstruction(Instruction):
    """
    Any node that deals particularly with quantum operations. All quantum operations
    must have some sort of target on the quantum computer, such as a qubit, channel, or
    another form of component.
    """

    inst: Literal["QuantumInstruction"] = "QuantumInstruction"
    targets: Union[set[str], str]

    def __init__(self, targets, **kwargs):
        # overwrite the init to accept quantum targets as a position argument
        return super().__init__(targets=targets, **kwargs)

    @property
    def duration(self):
        return 0.0

    @field_validator("targets", mode="before")
    @classmethod
    def _components_to_ids(cls, targets):
        """
        Fetches the IDs from quantum components.
        """
        if isinstance(targets, (tuple, list)) and len(targets) == 1:
            targets = targets[0]

        if isinstance(targets, (set, list, tuple)):
            targets = [
                itm.full_id() if isinstance(itm, QuantumComponent) else itm
                for itm in targets
            ]
        else:
            targets = (
                targets.full_id() if isinstance(targets, QuantumComponent) else targets
            )
        return targets


class PhaseShift(QuantumInstruction):
    """
    A PhaseShift instruction is used to change the phase of waveforms sent down
    the pulse channel.
    """

    inst: Literal["PhaseShift"] = "PhaseShift"
    targets: str
    phase: Union[float, Variable] = 0.0

    @property
    def channel(self):
        return self.targets

    def __repr__(self):
        return f"phaseshift {self.channel},{self.phase}"

    @field_validator("targets", mode="before")
    @classmethod
    def _is_pulse_channel(cls, target):
        if isinstance(target, list) and len(target) == 1:
            target = target[0]

        if not isinstance(target, (str, PulseChannel)):
            raise ValueError(
                f"channel has type {type(target).__name__}: it must have type PulseChannel."
            )
        return target.full_id() if isinstance(target, PulseChannel) else target


class FrequencyShift(QuantumInstruction):
    """Change the frequency of a pulse channel."""

    inst: Literal["FrequencyShift"] = "FrequencyShift"
    targets: str
    frequency: Union[float, Variable] = 0.0

    @property
    def channel(self):
        return self.targets

    def __repr__(self):
        return f"frequencyshift {self.channel},{self.frequency}"

    @field_validator("targets", mode="before")
    @classmethod
    def _is_pulse_channel(cls, target):
        if isinstance(target, list) and len(target) == 1:
            target = target[0]

        if not isinstance(target, (str, PulseChannel)):
            raise ValueError(
                f"channel has type {type(target).__name__}: it must have type PulseChannel"
            )
        return target.full_id() if isinstance(target, PulseChannel) else target


class Id(QuantumInstruction):
    """Simply a no-op, called an Identity gate."""

    inst: Literal["Id"] = "Id"

    def __repr__(self):
        return "id"


class Delay(QuantumInstruction):
    """Instructs a quantum target to do nothing for a fixed time."""

    inst: Literal["Delay"] = "Delay"
    time: float = Field(ge=0.0, default=0.0)

    @property
    def duration(self):
        return self.time

    def __repr__(self):
        return f"delay {str(self.time)}"


class GroupInstruction(QuantumInstruction):
    """
    Some instructions are just a collection of multiple targets with a label
    attached (e.g. synchronize) that tells us what to do with them. This group
    of instructions share the same properties, so we just define them here.
    """

    inst: Literal["GroupInstruction"] = "GroupInstruction"

    @field_validator("targets", mode="before")
    @classmethod
    def _pulse_channels_to_strs(cls, targets):
        targets = [targets] if not isinstance(targets, list) else targets
        unique_targets = set()
        for target in (
            chan
            for val in targets
            for chan in (val.pulse_channels.values() if isinstance(val, Qubit) else [val])
        ):
            if isinstance(target, str):
                unique_targets.add(target)
            elif isinstance(target, PulseChannel):
                unique_targets.add(target.full_id())
            else:
                raise ValueError(
                    f"Attempted to add a non PulseChannel ({target}) to the instruction."
                )
        return set(sorted(unique_targets))

    def add_channels(
        self,
        sync_channels: Union[Qubit, PulseChannel, List[Union[Qubit, PulseChannel]]],
    ):
        new_targets = self._pulse_channels_to_strs(sync_channels)
        self.targets.update(new_targets)
        return self

    def __add__(self, other):
        new_sync = deepcopy(self)
        new_sync += other
        return new_sync

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            self.targets.update(other.targets)
        elif isinstance(other, (QuantumComponent, list)):
            self.add_channels(other)
        else:
            raise ValueError(
                f"Object {other} must a Component, or a list of Components or a "
                + f"{self.__class__.__name__}."
            )
        return self


class Synchronize(GroupInstruction):
    """
    Tells the QPU to wait for all the related channels to be free before continuing
    execution on any of them.
    """

    inst: Literal["Synchronize"] = "Synchronize"

    def __repr__(self):
        return f"sync {','.join(self.targets)}"


class PhaseReset(GroupInstruction):
    """
    Reset the phase shift of given pulse channels, or the pulse channels of given qubits.
    """

    inst: Literal["PhaseReset"] = "PhaseReset"

    def __repr__(self):
        return f"phase reset {','.join(self.targets)}"


class Reset(QuantumInstruction):
    """Resets this qubit to its starting state."""

    inst: Literal["Reset"] = "Reset"

    def __repr__(self):
        return f"reset {','.join(self.targets)}"

    @field_validator("targets", mode="before")
    @classmethod
    def _components_to_ids(cls, targets):
        if not isinstance(targets, (set, list, tuple)):
            targets = [targets]

        invalid_reset_targets = [
            str(val) for val in targets if not isinstance(val, (str, Qubit, PulseChannel))
        ]
        if any(invalid_reset_targets):
            raise ValueError(
                "Tried to reset on non-qubit/pulse channel "
                f"{', '.join(invalid_reset_targets)}."
            )
        targets = [
            (
                val.get_drive_channel().full_id()
                if isinstance(val, Qubit)
                else (val.full_id() if isinstance(val, PulseChannel) else val)
            )
            for val in targets
        ]
        return targets


class DeviceUpdate(QuantumInstruction):
    """
    Dynamically assigns a value to a particular symbol or hardware attribute during
    execution.

    .. note:: It's still unknown how this will be represented in the instructions themselves, but that'll come later.
    For now we perform programatic modification and a before/after state.
    """

    inst: Literal["DeviceUpdate"] = "DeviceUpdate"
    attribute: str
    value: Any

    def __repr__(self):
        return f"{self.target}.{self.attribute} = {str(self.value)}"
