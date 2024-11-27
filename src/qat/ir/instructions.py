from __future__ import annotations

from copy import deepcopy
from pydoc import locate
from typing import Any, Dict, List, Literal, Optional, Set, Union

import numpy as np
from compiler_config.config import InlineResultsProcessing
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
)
from typing_extensions import Annotated

from qat.purr.compiler.devices import PulseChannel, PulseShapeType, QuantumComponent, Qubit

# The following things from legacy instructions are unchanged, so just import for now.
from qat.purr.compiler.instructions import AcquireMode, PostProcessType, ProcessAxis
from qat.purr.utils.logger import get_default_logger


### Sweeps
class SweepOperation(BaseModel):
    """Common parent for all things that need differentiating during a sweep."""

    pass


class SweepValue(SweepOperation):
    name: str
    value: Any


### Variables
def build_generated_name(existing_names=None, prefix=None):
    if existing_names is None:
        existing_names = set()

    if prefix is None:
        prefix = ""

    if any(prefix) and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    variable_name = f"{prefix}generated_name_{np.random.randint(np.iinfo(np.int32).max)}"
    while variable_name in existing_names:
        variable_name = (
            f"{prefix}generated_name_{np.random.randint(np.iinfo(np.int32).max)}"
        )

    existing_names.add(variable_name)
    return variable_name


class Variable(BaseModel):
    """
    States that this value is actually a variable that should be fetched instead.
    """

    name: str
    # TODO: we might consider limiting what types a variable can take...
    var_type: Optional[type] = None
    value: Any = None
    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, name, var_type=None, value=None, **kwargs):
        super().__init__(name=name, var_type=var_type, value=value)

    @staticmethod
    def with_random_name(existing_names=None, var_type=None, value=None):
        return Variable(Variable.generate_name(existing_names), var_type, value)

    @staticmethod
    def generate_name(existing_names=None):
        return build_generated_name(existing_names)

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
        if var_type != None and not isinstance(value, var_type):
            raise ValueError(f"Value provided has type {type(value)}: must be {var_type}")
        return value


class IndexAccessor(Variable):
    """Used to access an array index on a particular variable."""

    index: int

    def __init__(self, name, index, **kwargs):
        super().__init__(name)
        self.index = index

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
    repeat_count: Any
    repetition_period: Any = None  # TODO: probably can only be a float, but not sure...

    def __init__(self, repeat_count: int = None, repetition_period=None, **kwargs):
        super().__init__(repeat_count=repeat_count, repetition_period=repetition_period)

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

    def __init__(self, name, value, **kwargs):
        super().__init__(name=name, value=value)

    def __repr__(self):
        return f"{self.name} = {str(self.value)}"

    @field_validator("value", mode="before")
    @classmethod
    def _list_to_vars(cls, val):
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
    variables: Optional[List[str]]

    def __init__(self, variables: List[str] = None, **kwargs):
        if variables is None:
            variables = []

        if not isinstance(variables, List):
            variables = [variables]

        super().__init__(variables=variables)

    def __repr__(self):
        return f"return {','.join(self.variables)}"

    @field_validator("variables")
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
    operations: List[SweepValue]
    variables: Dict[str, List[Any]]

    def __init__(
        self,
        operations: Union[SweepValue, List[SweepValue]] = None,
        variables: Dict[str, List[Any]] = None,
        **kwargs,
    ):
        if operations is None:
            operations = []
        elif not isinstance(operations, List):
            operations = [operations]

        # Get the length of the variables, which we will then assume is the sweep
        # length.
        if not variables:
            variables = {op.name: op.value for op in operations}
        sweep_lengths = [len(value) for value in variables.values()]
        if len(set(sweep_lengths)) > 1:
            raise ValueError("Sweep variables have inconsistent lengths.")

        super().__init__(operations=operations, variables=variables)

    @property
    def length(self):
        return next(iter([len(value) for value in self.variables.values()]), 0)

    def __repr__(self):
        args = ",".join(key + "=" + str(value) for key, value in self.variables.items())
        return f"sweep {args}"


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

    def __init__(self, name, **kwargs):
        """If you need a name, use generate_name and pass in existing values."""
        super().__init__(name=name)

    @staticmethod
    def with_random_name(existing_names=None):
        return Label(Label.generate_name(existing_names))

    @staticmethod
    def generate_name(existing_names=None):
        return build_generated_name(existing_names)

    def __repr__(self):
        return f"{self.name}:"


class Jump(Instruction):
    """
    Classic jump instruction, should be linked to label with an optional condition.
    """

    inst: Literal["Jump"] = "Jump"
    target: str
    condition: Any

    def __init__(self, target: Union[str, Label], condition=None, **kwargs):
        return super().__init__(
            target=target.name if isinstance(target, Label) else target,
            condition=condition,
        )

    def __repr__(self):
        if self.condition is not None:
            return f"if {str(self.condition)} -> {str(self.target)}"
        else:
            return f"-> {str(self.target)}"


class ResultsProcessing(Instruction):
    """
    A meta-instruction that stores how the results are processed.
    """

    inst: Literal["ResultsProcessing"] = "ResultsProcessing"
    variable: str
    res_processing: InlineResultsProcessing

    def __init__(self, variable: str, res_processing: InlineResultsProcessing, **kwargs):
        super().__init__(variable=variable, res_processing=res_processing)

    def __repr__(self):
        return f"{self.variable}: {str(self.res_processing.name)}"


### Quantum Instructions
class QuantumInstruction(Instruction):
    """
    Any node that deals particularly with quantum operations. All quantum operations
    must have some sort of target on the quantum computer, such as a qubit, channel, or
    another form of component.
    """

    inst: Literal["QuantumInstruction"] = "QuantumInstruction"
    quantum_targets: Union[set[str], str]

    @property
    def duration(self):
        return 0.0

    @field_validator("quantum_targets", mode="before")
    @classmethod
    def _components_to_ids(cls, targets):
        """
        Fetches the IDs from quantum components.
        """
        if isinstance(targets, (set, list, tuple)):
            targets = set(
                [
                    itm.full_id() if isinstance(itm, QuantumComponent) else itm
                    for itm in targets
                ]
            )
            targets = [val for val in targets][0] if len(targets) == 1 else targets
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
    quantum_targets: str
    phase: Union[float, Variable] = 0.0

    def __init__(self, quantum_targets: PulseChannel, phase: float, **kwargs):
        super().__init__(quantum_targets=quantum_targets, phase=phase)

    @property
    def channel(self):
        return self.quantum_targets

    def __repr__(self):
        return f"phaseshift {self.channel},{self.phase}"

    @field_validator("quantum_targets", mode="before")
    @classmethod
    def _is_pulse_channel(cls, target):
        if not isinstance(target, (str, PulseChannel)):
            raise ValueError(
                f"channel has type {str(type(target))}: it must have type PulseChannel"
            )
        return target.full_id() if isinstance(target, PulseChannel) else target


class FrequencyShift(QuantumInstruction):
    """Change the frequency of a pulse channel."""

    inst: Literal["FrequencyShift"] = "FrequencyShift"
    quantum_targets: str
    frequency: Union[float, Variable] = 0.0

    def __init__(self, quantum_targets: PulseChannel, frequency: float, **kwargs):
        super().__init__(quantum_targets=quantum_targets, frequency=frequency)

    @property
    def channel(self):
        return self.quantum_targets

    def __repr__(self):
        return f"frequencyshift {self.channel},{self.frequency}"

    @field_validator("quantum_targets", mode="before")
    @classmethod
    def _is_pulse_channel(cls, target):
        if not isinstance(target, (str, PulseChannel)):
            raise ValueError(
                f"channel has type {str(type(target))}: it must have type PulseChannel"
            )
        return target.full_id() if isinstance(target, PulseChannel) else target


class Id(QuantumInstruction):
    """Simply a no-op, called an Identity gate."""

    # TODO: do we really need this?
    inst: Literal["Id"] = "Id"

    def __repr__(self):
        return "id"


class Delay(QuantumInstruction):
    """Instructs a quantum target to do nothing for a fixed time."""

    inst: Literal["Delay"] = "Delay"
    time: float = Field(ge=0.0, default=0.0)

    def __init__(self, quantum_targets: QuantumComponent, time: float, **kwargs):
        super().__init__(quantum_targets=quantum_targets, time=time)

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

    def __init__(
        self,
        quantum_targets: Union[Qubit, PulseChannel, List[Union[Qubit, PulseChannel]]],
        **kwargs,
    ):
        super().__init__(quantum_targets=quantum_targets)

    @field_validator("quantum_targets", mode="before")
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
        self.quantum_targets.update(new_targets)
        return self

    def __add__(self, other):
        new_sync = deepcopy(self)
        new_sync += other
        return new_sync

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            self.quantum_targets.update(other.quantum_targets)
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
        return f"sync {','.join(self.quantum_targets)}"


class PhaseReset(GroupInstruction):
    """
    Reset the phase shift of given pulse channels, or the pulse channels of given qubits.
    """

    inst: Literal["PhaseReset"] = "PhaseReset"

    def __repr__(self):
        return f"phase reset {','.join(self.quantum_targets)}"


class Waveform(QuantumInstruction):
    # TODO: how to deal with the this not storing the actual pulse channel?
    inst: Literal["Waveform"] = "Waveform"

    @property
    def channel(self):
        return (
            self.quantum_targets
            if isinstance(self.quantum_targets, str)
            else self.quantum_targets[0]
        )


class CustomPulse(Waveform):
    """
    Send a waveform with a custom shape down this particular channel.
    """

    # changed sample to numpy array: not sure of the consequences of this.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    inst: Literal["CustomPulse"] = "CustomPulse"
    samples: np.ndarray
    ignore_channel_scale: bool
    sample_time: float

    def __init__(
        self,
        quantum_targets: PulseChannel,
        samples: np.ndarray,
        ignore_channel_scale: bool = False,
        sample_time=None,
        **kwargs,
    ):
        # TODO: should we be storing properties of the pulse channel in the instruction?
        # It is only used for calcualting the duration (which was previously done externally).
        if not sample_time:
            chan = quantum_targets
            if isinstance(chan, list):
                chan = chan[0]
            if not isinstance(chan, PulseChannel):
                raise (
                    ValueError("Sample time cannot be determined without a pulse channel.")
                )
            sample_time = chan.sample_time

        super().__init__(
            quantum_targets=quantum_targets,
            samples=samples,
            ignore_channel_scale=ignore_channel_scale,
            sample_time=sample_time,
        )

    @property
    def duration(self):
        return len(self.samples) * self.sample_time

    def __repr__(self):
        return f"custom pulse {self.channel},{self.duration}"

    @field_validator("samples", mode="before")
    @classmethod
    def _samples_as_np_array(cls, val):
        """
        Used in deserialization.

        Numpyas arrays of complex numbers are serialized as list of strings.
        """
        if not isinstance(val, np.ndarray):
            return np.array([complex(v) for v in val])
        return val

    @field_serializer("samples", when_used="json")
    def _np_array_to_list(self, samples):
        return samples.tolist()


class Pulse(Waveform):
    """
    Send a pulse down this particular channel.
    """

    # TODO: The amount of variables for different pulses here is overwhelming,
    # and won't do us any favours for serialization. How we deal with different
    # pulses is an open question, but a first step could be to standarize notation
    # between different pulse shapes.
    inst: Literal["Pulse"] = "Pulse"
    shape: PulseShapeType
    width: Union[Variable, float] = 0.0
    amp: Union[Variable, float] = 1.0
    phase: Union[Variable, float] = 0.0
    drag: Union[Variable, float] = 0.0
    rise: Union[Variable, float] = 0.0
    amp_setup: Union[Variable, float] = 0.0
    scale_factor: Union[Variable, float] = 1.0
    zero_at_edges: bool = False
    beta: Union[Variable, float] = 0.0
    frequency: Union[Variable, float] = 0.0
    internal_phase: Union[Variable, float] = 0.0
    std_dev: Union[Variable, float] = 0.0
    square_width: Union[Variable, float] = 0.0
    ignore_channel_scale: bool = False

    def __init__(
        self,
        quantum_targets: PulseChannel,
        shape: PulseShapeType,
        width: float,
        **kwargs,
    ):
        super().__init__(
            quantum_targets=quantum_targets, shape=shape, width=width, **kwargs
        )

    @property
    def duration(self):
        return self.width

    def __repr__(self):
        return (
            f"pulse {self.channel},{self.shape.value},{self.amp},"
            f"{self.phase},{self.width},{self.drag},{self.rise}"
        )


class MeasurePulse(Pulse):
    inst: Literal["MeasurePulse"] = "MeasurePulse"


class DrivePulse(Pulse):
    inst: Literal["DrivePulse"] = "DrivePulse"


class SecondStatePulse(Pulse):
    inst: Literal["SecondStatePulse"] = "SecondStatePulse"


class CrossResonancePulse(Pulse):
    inst: Literal["CrossResonancePulse"] = "CrossResonancePulse"


class CrossResonanceCancelPulse(Pulse):
    inst: Literal["CrossResonanceCancelPulse"] = "CrossResonanceCancelPulse"


# previously also inheritted QuantumComponent: I do not know why...
# might need adding back later
class Acquire(QuantumInstruction):
    inst: Literal["Acquire"] = "Acquire"
    suffix_incrementor: int = 0
    time: Union[float, Variable]
    mode: AcquireMode
    output_variable: str = None
    delay: Optional[float] = None
    # was previously union of pulse and custom pulse: can this be extended to waveform?
    filter: Optional[Waveform] = None

    def __init__(
        self,
        quantum_targets: PulseChannel,
        time: Union[float, Variable] = None,
        mode: AcquireMode = None,
        output_variable: Optional[str] = None,
        existing_names: Set[str] = None,
        delay: Optional[float] = None,
        filter: Optional[Waveform] = None,
        suffix_incrementor: int = 0,
        **kwargs,
    ):
        super().__init__(
            quantum_targets=quantum_targets,
            time=time or 1.0e-6,
            mode=mode or AcquireMode.RAW,
            delay=delay,
            output_variable=output_variable
            or self.generate_name(existing_names, quantum_targets.full_id()),
            filter=filter,
            suffix_incrementor=suffix_incrementor,
        )

    def generate_name(self, existing_names=None, channel=None):
        return build_generated_name(existing_names, channel if channel else self.channel)

    @property
    def duration(self):
        return self.time

    @property
    def channel(self):
        return self.quantum_targets

    def __repr__(self):
        out_var = f"->{self.output_variable}" if self.output_variable else ""
        mode = f",{self.mode.value}" if self.mode is not None else ""
        return f"acquire {self.channel},{self.time}{mode}{out_var}"

    @field_validator("filter", mode="before")
    @classmethod
    def _validate_filter(cls, filter, info: ValidationInfo):
        if filter == None:
            return filter

        time = info.data["time"]
        if isinstance(time, Variable):
            get_default_logger.warning(f"Filter cannot be applied when time is variable.")
            return None

        if not isinstance(filter, Waveform):
            raise ValueError("Filter must be a Waveform.")

        if filter.duration < 0:
            raise ValueError(
                f"Filter duration {filter.duration} cannot be less than or equal to zero."
            )

        if not np.isclose(filter.duration, time, atol=1e-9):
            raise ValueError(
                f"Filter duration '{filter.duration}' must be equal to Acquire "
                f"duration '{time}'."
            )

        return filter


# TODO: Used to be a quantum instruction, determine if this needs to be
class PostProcessing(Instruction):
    """
    States what post-processing should happen after data has been acquired. This can
    happen in the FPGA's or a software post-process.
    """

    inst: Literal["PostProcessing"] = "PostProcessing"
    acquire: Acquire
    process: PostProcessType
    axes: List[ProcessAxis] = []
    args: List[Any] = []
    result_needed: bool = False

    def __init__(
        self,
        acquire: Acquire,
        process,
        axes=[],
        args=[],
        results_needed: bool = False,
        **kwargs,
    ):
        # TODO: Should find out why ags can't reference model objects in builder serialization.
        #   But deep copying them, as the values shouldn't vary, should be fine.

        super().__init__(
            acquire=acquire,
            process=process,
            args=args or [],
            axes=[] if axes == None else axes if isinstance(axes, List) else [axes],
            results_needed=results_needed,
        )

    @property
    def output_variable(self):
        return self.acquire.output_variable

    def __repr__(self):
        axis = ",".join([axi.value for axi in self.axes])
        args = f",{','.join(str(arg) for arg in self.args)}" if len(self.args) > 0 else ","
        output_var = f"->{self.output_variable}" if self.output_variable is not None else ""
        return (
            f"{self.process.value} {self.acquire.output_variable}{args}{axis}{output_var}"
        )


class Reset(QuantumInstruction):
    """Resets this qubit to its starting state."""

    inst: Literal["Reset"] = "Reset"

    def __init__(self, quantum_targets, **kwargs):
        return super().__init__(quantum_targets=quantum_targets)

    def __repr__(self):
        return f"reset {','.join(self.quantum_targets)}"

    @field_validator("quantum_targets", mode="before")
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

    def __init__(self, target: QuantumComponent, attribute: str, value, **kwargs):
        super().__init__(target, attribute=attribute, value=value)

    def __repr__(self):
        return f"{self.target}.{self.attribute} = {str(self.value)}"


### Instruction blocks
def _all_instructions(start_types=[Instruction]):
    """
    Creates a tuple of all possible instructions, used in seralization.
    """
    instructions = set()
    check = start_types
    while check:
        parent = check.pop()
        for child in parent.__subclasses__():
            if child not in instructions:
                check.append(child)
                instructions.add(child)

    return tuple(instructions)


Inst = Annotated[
    Union[_all_instructions()],
    Field(discriminator="inst"),
]
InstList = List[Inst]


class InstructionBlock(BaseModel):
    """
    An Instruction grouping type. Allows working with blocks of Instructions as a unit.
    """

    inst: Literal["InstructionBlock"] = "InstructionBlock"

    def _validate_types(self, items, valid_types, label="targets"):
        if items is None:
            items = []
        elif not isinstance(items, List):
            items = [items]

        invalid_items = [item for item in items if not isinstance(item, valid_types)]
        if any(invalid_items):
            invalid_items_str = ",".join([str(item) for item in invalid_items])
            raise ValueError(f"Invalid {label} for {type(self)}: {invalid_items_str}")

        return items


class MeasureData(BaseModel):
    mode: AcquireMode
    output_variable: Optional[str] = None
    measure: MeasurePulse
    acquire: Acquire
    duration: float
    # TODO: think of a better solution than storing here
    targets: List[str] = None


class MeasureBlock(InstructionBlock):
    """Groups multiple qubit measurements together."""

    inst: Literal["MeasureBlock"] = "MeasureBlock"
    target_dict: Dict[str, MeasureData] = {}
    existing_names: Optional[Set[str]] = set()
    duration: float = 0.0

    @property
    def instructions(self):
        instructions = [Synchronize(self.pulse_channel_targets)]
        for _, val in self.target_dict.items():
            instructions.extend([val.measure, val.acquire])
        instructions.append(Synchronize(self.pulse_channel_targets))
        return instructions

    @property
    def quantum_targets(self):
        return list(self.target_dict.keys())

    @property
    def pulse_channel_targets(self):
        targets = []
        for target in self.target_dict.values():
            targets.extend(target.targets)
        return targets

    @staticmethod
    def create_block(
        qubit: Union[Qubit, List[Qubit]],
        mode: AcquireMode,
        output_variables: str = None,
        existing_names=set(),
    ):
        # set as a seperate static method as overwriting __init__ would
        # make this unserialisable...
        measure_block = MeasureBlock(existing_names=existing_names)
        measure_block.add_measurements(qubit, mode, output_variables)
        return measure_block

    def add_measurements(
        self,
        targets: Union[Qubit, List[Qubit]],
        mode: AcquireMode,
        output_variables: Union[str, List[str]] = None,
        existing_names: Set[str] = None,
    ):
        targets = self._validate_types(targets, (Qubit))
        if (
            len((duplicates := [t for t in targets if t.full_id() in self.quantum_targets]))
            > 0
        ):
            raise ValueError(
                "Target can only be measured once in a 'MeasureBlock'. "
                f"Duplicates: {duplicates}"
            )
        if not isinstance(output_variables, list):
            output_variables = [] if output_variables is None else [output_variables]
        if (num_out_vars := len(output_variables)) == 0:
            output_variables = [None] * len(targets)
        elif num_out_vars != len(targets):
            raise ValueError(
                f"Unsupported number of `output_variables`: {num_out_vars}, "
                f"must be `None` or match numer of targets: {len(targets)}."
            )
        for target, output_variable in zip(targets, output_variables):
            meas, acq = self._generate_measure_acquire(
                target, mode, output_variable, existing_names
            )
            duration = max(meas.duration, acq.delay + acq.duration)
            self.target_dict[target.full_id()] = MeasureData(
                mode=mode,
                output_variable=output_variable,
                measure=meas,
                acquire=acq,
                duration=duration,
                targets=[pc.full_id() for pc in target.pulse_channels.values()],
            )
            self.duration = max(self.duration, duration)

    def _generate_measure_acquire(self, qubit, mode, output_variable, existing_names):
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()
        existing_names = existing_names or self.existing_names
        weights = (
            qubit.measure_acquire.get("weights", None)
            if qubit.measure_acquire.get("use_weights", False)
            else None
        )

        measure_instruction = MeasurePulse(measure_channel, **qubit.pulse_measure)
        acquire_instruction = Acquire(
            acquire_channel,
            (
                qubit.pulse_measure["width"]
                if qubit.measure_acquire["sync"]
                else qubit.measure_acquire["width"]
            ),
            mode,
            output_variable,
            existing_names,
            qubit.measure_acquire["delay"],
            weights,
        )

        return [
            measure_instruction,
            acquire_instruction,
        ]

    def get_acquires(self, targets: Union[Qubit, List[Qubit]]):
        if not isinstance(targets, list):
            targets = [targets]
        targets = [t.full_id() if isinstance(t, Qubit) else t for t in targets]
        return [self.target_dict[qt].acquire for qt in targets]

    def __repr__(self):
        target_strings = []
        for q, d in self.target_dict.items():
            out_var = f"->{d.output_variable}" if d.output_variable is not None else ""
            mode = f":{d.mode.value}" if d.mode is not None else ""
            target_strings.append(f"{q}{mode}{out_var}")
        return f"Measure {', '.join(target_strings)}"
