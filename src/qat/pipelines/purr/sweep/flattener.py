# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from itertools import product
from math import prod

from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.instructions import DeviceUpdate, Instruction, Sweep, Variable
from qat.purr.utils.logger import get_default_logger

logger = get_default_logger()


@dataclass
class VariableAccessor:
    """Contains the information needed to access a variable instance in an instruction
    builder."""

    instruction_index: int
    attribute: str
    variable_name: str

    @classmethod
    def create_accessors(
        cls, instruction: Instruction, index: int, names: list[str]
    ) -> list["VariableAccessor"]:
        """Creates a list of VariableAccessors for the given instruction, looking for
        variables with the given names."""

        accessors = []
        for key, val in instruction.__dict__.items():
            if key.startswith("__"):
                continue
            if isinstance(val, Variable) and val.name in names:
                accessors.append(cls(index, key, val.name))
        return accessors


class DeviceAssignSet:
    """Contains a set of device assignments to be applied to a builder."""

    def __init__(self, assigns: list[DeviceUpdate]):
        self.assigns = assigns

    @contextmanager
    def apply(self):
        """Temporarily applies the device assignments to the targets.

        .. warning::

            This will mutate the components of the hardware model, potentially invalidating
            any analysis already done on instantiation of pipelines. Futhermore, since this
            is used in conjuntion with sweeping, it is not thread safe. This is only
            intended to be used in the SweepPipeline, and should not be used elsewhere.
            It is recommended that this is used with a hardware model, or hardware model
            loader that is not shared with any other pipeline.
        """

        original_values = [self._get_attribute(assign) for assign in self.assigns]

        try:
            for assign in self.assigns:
                self._set_attribute(assign, assign.value)
            yield
        finally:
            for assign, original in zip(self.assigns, original_values):
                self._set_attribute(assign, original)

    def _get_attribute(self, assign: DeviceUpdate):
        if isinstance(assign.target, dict):
            return assign.target[assign.attribute]
        else:
            return getattr(assign.target, assign.attribute)

    def _set_attribute(self, assign: DeviceUpdate, value):
        if isinstance(assign.target, dict):
            assign.target[assign.attribute] = value
        else:
            setattr(assign.target, assign.attribute, value)

    def __len__(self):
        return len(self.assigns)


@dataclass
class SweepInstance:
    """Contains a single instance of a sweep, with the variable names and values."""

    variables: dict[str, complex | float]
    builder: QuantumInstructionBuilder
    device_assigns: DeviceAssignSet

    @property
    def has_device_assigns(self) -> bool:
        return len(self.device_assigns) > 0


class SweepFlattener:
    """Extracts sweeps and device assigns from a builder, and acts as a factory for creating
    a set of new builder that is free of sweeps and device assigns.

    This can be used as an iterator to return a series of builders with the sweep values
    injected into the relevant instructions. This works by first extracting sweep
    instructions, and then locating all variables within instructions that match the sweep
    variable names. It only searches a single depth (e.g. within the top-level dictionary
    describing the instruction). This may be adjusted in the future if we need to perform
    a deeper search.
    """

    def __init__(self, builder: QuantumInstructionBuilder):
        """:param builder: The instruction builder containing sweeps."""
        self.sweeps: list[Sweep] = []
        self.sweep_names: list[str] = []  # list to preserve order
        self.sweep_sizes: list[int] = []
        self.device_assigns: list[DeviceUpdate] = []
        self.accessors: list[VariableAccessor] = []

        self._extract_sweeps(builder)
        self.total_combinations = prod(self.sweep_sizes)
        self._validate_device_assigns_variables(self.device_assigns)
        self._locate_variables(self.sweep_names)

    def create_flattened_builders(self) -> list[SweepInstance]:
        """Creates a list of builders with all combinations of sweep values injected.

        Returns a list of instances of the builder, containing the variable values, the
        builder with the injected values, and the device assignments to be applied before
        compiling the builders (if any).
        """

        builders = []
        for indices in product(*(range(size) for size in self.sweep_sizes)):
            variable_values = {}
            for i, index in enumerate(indices):
                sweep = self.sweeps[i]
                for key, val in sweep.variables.items():
                    variable_values[key] = val[index]
            new_builder = self._copy_and_inject_builder(variable_values)
            new_device_assigns = self._copy_and_inject_device_assigns(variable_values)
            builders.append(
                SweepInstance(
                    variable_values, new_builder, DeviceAssignSet(new_device_assigns)
                )
            )
        return builders

    def _create_new_builder(
        self, builder: QuantumInstructionBuilder, instructions: list[Instruction]
    ) -> QuantumInstructionBuilder:
        """Used to instantiate a new builder from an existing builder, defaulting to a
        QuantumInstructionBuilder if no hardware model is set within a builder."""
        new_builder = (
            builder.model.create_builder()
            if builder.model is not None
            else QuantumInstructionBuilder(hardware_model=None)
        )
        new_builder._instructions = instructions
        return new_builder

    def _extract_sweeps(self, builder: QuantumInstructionBuilder = None):
        """Extracts the sweeps from the instruction builder, removing them from the builder.

        A convinent side effect is that this also flattens any nested blocks in the builder,
        making it easier to inject variables.
        """

        instructions = []
        for instr in builder.instructions:
            if isinstance(instr, Sweep):
                self.sweeps.append(instr)
                self.sweep_names.extend(list(instr.variables.keys()))
                self.sweep_sizes.append(len(next(iter(instr.variables.values()))))
            elif isinstance(instr, DeviceUpdate):
                self.device_assigns.append(instr)
            else:
                instructions.append(instr)
        self._builder = self._create_new_builder(builder, instructions)

    def _locate_variables(self, names: list[str]):
        """Locate all variables in the instruction builder with the given names, returning
        a list of VariableAccesors that can be used to locate sweep variable instances.
        """

        self.accessors = []
        for i, instr in enumerate(self._builder._instructions):
            self.accessors.extend(VariableAccessor.create_accessors(instr, i, names))

    def _validate_device_assigns_variables(self, device_assigns: list[DeviceUpdate]):
        """Checks that any variables in device assignments corerspon"""
        for assign in device_assigns:
            if (
                isinstance(assign.value, Variable)
                and assign.value.name not in self.sweep_names
            ):
                raise ValueError(
                    f"Variable '{assign.value.name}' in device assignment is a variable "
                    "that is not allocated to any sweep."
                )

    def _copy_and_inject_builder(
        self, values: dict[str, complex | float]
    ) -> QuantumInstructionBuilder:
        """Injects the given values into the instructions using the injectors, and returns
        a copy of the builder with the modified instructions."""

        instructions_copy = [copy(instr) for instr in self._builder._instructions]
        builder = self._create_new_builder(self._builder, instructions_copy)

        for accessor in self.accessors:
            instr = builder._instructions[accessor.instruction_index]
            setattr(instr, accessor.attribute, values[accessor.variable_name])

        return builder

    def _copy_and_inject_device_assigns(
        self, values: dict[str, complex | float]
    ) -> list[DeviceUpdate]:
        """Injects the given values into the device assignments, returning a list of
        modified device assignments."""

        device_assigns = []
        for assign in self.device_assigns:
            new_assign = copy(assign)
            if isinstance((value := new_assign.value), Variable):
                new_assign.value = values[value.name]
            device_assigns.append(new_assign)
        return device_assigns
