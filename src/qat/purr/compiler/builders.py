# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import itertools
import math
import warnings
from collections import defaultdict
from enum import Enum, auto
from typing import List, Set, Union

import jsonpickle
import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.purr.compiler.devices import (
    ChannelType,
    CyclicRefPickler,
    CyclicRefUnpickler,
    PulseChannel,
    PulseChannelView,
    Qubit,
    Resonator,
)
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    Assign,
    CrossResonancePulse,
    Delay,
    DeviceUpdate,
    FrequencyShift,
    Instruction,
    Jump,
    Label,
    MeasureBlock,
    MeasurePulse,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
    QuantumInstruction,
    QuantumInstructionBlock,
    Repeat,
    Reset,
    ResultsProcessing,
    Return,
    Sweep,
    Synchronize,
)
from qat.purr.utils.logger import get_default_logger

warnings.simplefilter("always", DeprecationWarning)
log = get_default_logger()


class Axis(Enum):
    X = auto()
    Y = auto()
    Z = auto()


class InstructionBuilder:
    """
    Base instruction builder class that leaves unimplemented the methods that vary on a
    hardware-by-hardware basis.
    """

    def __init__(
        self,
        hardware_model,
        instructions: List[Union["InstructionBuilder", Instruction]] = None,
    ):
        super().__init__()
        self._instructions = []
        self.existing_names = set()
        self._entanglement_map = {qubit: {qubit} for qubit in hardware_model.qubits}
        self.model = hardware_model
        self.add(instructions)

    @property
    def instructions(self):
        flat_list = []
        for inst in self._instructions:
            if isinstance(inst, QuantumInstructionBlock):
                flat_list.extend(inst.instructions)
            else:
                flat_list.append(inst)
        return flat_list

    @instructions.setter
    def instructions(self, value):
        self._instructions = value

    @staticmethod
    def deserialize(blob) -> "InstructionBuilder":
        builder = jsonpickle.decode(blob, context=CyclicRefUnpickler())
        if not isinstance(builder, InstructionBuilder):
            raise ValueError("Attempt to deserialize has failed.")

        return builder

    def serialize(self):
        """
        Currently only serializes the instructions, not the supporting objects of the builder itself.
        This could be supported pretty easily, but not required right now.
        """
        return jsonpickle.encode(self, indent=4, context=CyclicRefPickler())

    def splice(self):
        """Clears the builder and returns its current instructions."""
        final_instructions = self.instructions
        self.clear()
        return final_instructions

    def clear(self):
        """Resets builder internal state for building a new set of instructions."""
        self._instructions = []
        self.existing_names = set()
        self._entanglement_map = defaultdict(set)

    def get_child_builder(self, inherit=False):
        builder = InstructionBuilder(self.model)
        builder._entanglement_map = self._entanglement_map
        if inherit:
            builder.existing_names = set(self.existing_names)
        return builder

    def _fix_clashing_label_names(
        self, invalid_label_names: Set[str], existing_names: Set[str]
    ):
        """
        Fixes up auto-generated label names if there are clashes. invalid_label_names is
        a set of names to be re-generated, existing_names is the full set of existing
        names (union of all builders names' who are being merged).
        """
        regenerated_names = dict()
        for inst in self._instructions:
            if not isinstance(inst, Label) or inst.name not in invalid_label_names:
                continue

            new_name = regenerated_names.setdefault(inst.name, None)
            if new_name is None:
                regenerated_names[inst.name] = new_name = Label.generate_name(
                    existing_names
                )

            inst.name = new_name

    def merge_builder(self, other_builder: "InstructionBuilder", index: int = None) -> int:
        """
        Merge this builder into the current instance. Checks for label name clashes and
        resolves them if any are found.

        :param index: index of self at which to insert the instructions of `other_builder`.
        :type index: int
        :returns: the next index to use for additional inserts.
        :rtype: int
        """
        name_clashes = other_builder.existing_names.intersection(self.existing_names)
        self.existing_names.update(other_builder.existing_names)
        if any(name_clashes):
            log.warning(
                "When merging builders these labels had name clashes: "
                f"{', '.join(name_clashes)}. Regenerating auto-assigned variable names."
            )
            other_builder._fix_clashing_label_names(name_clashes, self.existing_names)

        index = index or len(self._instructions)
        self.insert(other_builder._instructions, index)
        return index + len(other_builder._instructions)

    def add(
        self,
        components: Union[
            "InstructionBuilder",
            Instruction,
            List[Union["InstructionBuilder", Instruction]],
        ],
    ):
        """
        Adds an instruction to this builder. All methods should use this instead of
        accessing the instructions list directly as it deals with nested builders and
        merging.
        """
        return self.insert(components, len(self._instructions))

    def insert(
        self,
        components: Union[
            "InstructionBuilder",
            Instruction,
            List[Union["InstructionBuilder", Instruction]],
        ],
        index: int,
    ):
        """
        Inserts one or more instruction(s) into this builder, starting at index. All methods
        should use this instead of accessing the instructions list directly as it deals with
        nested builders and merging.
        """
        if components is None:
            return self

        if not isinstance(components, List):
            components = [components]

        for component in components:
            if isinstance(component, InstructionBuilder):
                index = self.merge_builder(component)
            else:
                # Naive entanglement checker for syncronization.
                if isinstance(component, CrossResonancePulse):
                    ent_qubits = self._get_entangled_qubits(component)
                    for qubit in ent_qubits:
                        self._entanglement_map[qubit].update(ent_qubits)
                    for qubit in self.model.qubits:
                        # entanglement is transitive, if A<>B and B<>C then C<>A
                        tmp = set()
                        for entangled in self._entanglement_map[qubit]:
                            tmp.update(self._entanglement_map[entangled])
                        self._entanglement_map[qubit].update(tmp)
                self._instructions.insert(index, component)
                index += 1
        return self

    def _get_entangled_qubits(self, inst):
        """
        Gets qubit ID's in relation to quantum entanglement for the current instruction.
        Important to note that it will route up or out to a qubit from things like resonators and pulse channels.
        """
        if not isinstance(inst, CrossResonancePulse):
            # Crossress pulses are the only mechanism by which entanglement is generated.
            return set()
        results = set()
        pulse_channel_view: PulseChannelView
        for pulse_channel_view in inst.quantum_targets:
            for target_qubit in pulse_channel_view.auxilary_devices:
                results.update(target_qubit)
            physical_channel = pulse_channel_view.pulse_channel.physical_channel
            for control_qubit in self.model.qubits:
                if control_qubit.physical_channel == physical_channel:
                    results.update(control_qubit)
        return results

    def results_processing(self, variable: str, res_format: InlineResultsProcessing):
        return self.add(ResultsProcessing(variable, res_format))

    def measure_single_shot_z(
        self, target: Qubit, axis: ProcessAxis = None, output_variable: str = None
    ):
        return self.measure(target, axis, output_variable)

    def measure_single_shot_signal(
        self, target: Qubit, axis: ProcessAxis = None, output_variable: str = None
    ):
        return self.measure(target, axis, output_variable)

    def measure_mean_z(
        self, target: Qubit, axis: ProcessAxis = None, output_variable: str = None
    ):
        return self.measure(target, axis, output_variable)

    def measure_mean_signal(self, target: Qubit, output_variable: str = None):
        return self.measure(target, output_variable=output_variable)

    def measure(
        self, target: Qubit, axis: ProcessAxis = None, output_variable: str = None
    ) -> "InstructionBuilder":
        raise NotImplementedError("Not available on this hardware model.")

    def X(self, target: Union[Qubit, PulseChannel], radii=None):
        raise NotImplementedError("Not available on this hardware model.")

    def Y(self, target: Union[Qubit, PulseChannel], radii=None):
        raise NotImplementedError("Not available on this hardware model.")

    def Z(self, target: Union[Qubit, PulseChannel], radii=None):
        raise NotImplementedError("Not available on this hardware model.")

    def U(self, target: Union[Qubit, PulseChannel], theta, phi, lamb):
        return self.Z(target, lamb).Y(target, theta).Z(target, phi)

    def swap(self, target: Qubit, destination: Qubit):
        raise NotImplementedError("Not available on this hardware model.")

    def had(self, qubit: Qubit):
        self.Y(qubit, math.pi / 2)
        return self.Z(qubit)

    def post_processing(
        self, acq: Acquire, process, axes=None, target: Qubit = None, args=None
    ):
        raise NotImplementedError("Not available on this hardware model.")

    def sweep(self, variables_and_values):
        raise NotImplementedError("Not available on this hardware model.")

    def pulse(self, *args, **kwargs):
        raise NotImplementedError("Not available on this hardware model.")

    def acquire(self, *args, **kwargs):
        raise NotImplementedError("Not available on this hardware model.")

    def delay(self, target: Union[Qubit, PulseChannel], time: float):
        raise NotImplementedError("Not available on this hardware model.")

    def synchronize(self, targets: Union[Qubit, List[Qubit]]):
        raise NotImplementedError("Not available on this hardware model.")

    def phase_shift(self, target: PulseChannel, phase):
        raise NotImplementedError("Not available on this hardware model.")

    def SX(self, target):
        return self.X(target, np.pi / 2)

    def SXdg(self, target):
        return self.X(target, -(np.pi / 2))

    def S(self, target):
        return self.Z(target, np.pi / 2)

    def Sdg(self, target):
        return self.Z(target, -(np.pi / 2))

    def T(self, target):
        return self.Z(target, np.pi / 4)

    def Tdg(self, target):
        return self.Z(target, -(np.pi / 4))

    def controlled(
        self, controllers: Union[Qubit, List[Qubit]], builder: "InstructionBuilder"
    ):
        raise NotImplementedError("Not available on this hardware model.")

    def cX(self, controllers: Union[Qubit, List[Qubit]], target: Qubit, radii=None):
        builder = self.get_child_builder()
        return self.controlled(controllers, builder.X(target, radii))

    def cY(self, controllers: Union[Qubit, List[Qubit]], target: Qubit, radii=None):
        builder = self.get_child_builder()
        return self.controlled(controllers, builder.Y(target, radii))

    def cZ(self, controllers: Union[Qubit, List[Qubit]], target: Qubit, radii=None):
        builder = self.get_child_builder()
        return self.controlled(controllers, builder.Z(target, radii))

    def cnot(self, control: Union[Qubit, List[Qubit]], target_qubit: Qubit):
        return self.cX(control, target_qubit, np.pi)

    def ccnot(self, controllers: List[Qubit], target_qubit: Qubit):
        raise NotImplementedError("Not available on this hardware model.")

    def cswap(self, controllers: Union[Qubit, List[Qubit]], target, destination):
        builder = self.get_child_builder()
        return self.controlled(controllers, builder.swap(target, destination))

    def ECR(self, control: Qubit, target: Qubit):
        raise NotImplementedError("Not available on this hardware model.")

    def jump(self, label: Union[str, Label], condition=None):
        return self.add(Jump(label, condition))

    def repeat(self, repeat_value: int, repetition_period=None):
        return self.add(Repeat(repeat_value, repetition_period))

    def assign(self, name, value):
        return self.add(Assign(name, value))

    def returns(self, variables=None):
        """Add return statement."""
        return self.add(Return(variables))

    def reset(self, qubits):
        self.add(Reset(qubits))
        return self.add(PhaseReset(qubits))

    def device_assign(self, target, attribute, value):
        """
        Special node that allows manipulation of device attributes during execution.
        """
        return self.add(DeviceUpdate(target, attribute, value))

    def create_label(self, name=None):
        """
        Creates and returns a label. Generates a non-clashing name if none is provided.
        """
        if name is None:
            name = Label.generate_name(self.existing_names)
        elif name in self.existing_names:
            new_name = Label.generate_name(self.existing_names)
            log.warning(f"Label name {name} already exists. Replacing with {new_name}.")

        return Label(name)


class FluidBuilderWrapper(tuple):
    """
    Wrapper to allow builders to return a tuple of values while also allowing fluid API
    consumption if those values are not required. Think of it like optional return
    values that don't require unpacking to discard.

    Examples of the two ways you should be able to call a builder when using this class.

    .. code-block:: python

        builder = ...
            .builder_method()
            .wrapped_value_returned()
            .builder_method()

            builder, value = ...
                .builder_method()
                .wrapped_value_returned()

            builder.builder_method(value)
                .builder_method()

    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            raise ValueError("Need at least one value to wrap.")

        instance = super().__new__(cls, args)
        both_contain = [
            val
            for val in set(dir(instance[0])) & set(dir(instance))
            if not val.startswith("__")
        ]
        if any(both_contain):
            raise ValueError(
                "Object being wrapped has the same attributes as tuple: "
                f"{', '.join(both_contain)}. This will cause shadowing and highly "
                "unlikely to be what is intended."
            )

        return instance

    def __getattr__(self, item):
        return object.__getattribute__(self[0], item)

    def __setattr__(self, key, value):
        object.__setattr__(self[0], key, value)


class QuantumInstructionBuilder(InstructionBuilder):
    def get_child_builder(self, inherit=False):
        builder = QuantumInstructionBuilder(self.model)
        builder._entanglement_map = self._entanglement_map
        if inherit:
            builder.existing_names = set(self.existing_names)
        return builder

    def measure_single_shot_z(
        self, target: Qubit, axis: str = None, output_variable: str = None
    ):
        _, acquire = self.measure(target, axis, output_variable)
        self.post_processing(
            acquire, PostProcessType.DOWN_CONVERT, ProcessAxis.TIME, target
        )
        self.post_processing(acquire, PostProcessType.MEAN, ProcessAxis.TIME, target)
        return self.post_processing(
            acquire, PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL, qubit=target
        )

    def measure_single_shot_signal(
        self, target: Qubit, axis: str = None, output_variable: str = None
    ):
        _, acquire = self.measure(target, axis, output_variable)
        self.post_processing(
            acquire, PostProcessType.DOWN_CONVERT, ProcessAxis.TIME, target
        )
        return self.post_processing(acquire, PostProcessType.MEAN, ProcessAxis.TIME, target)

    def measure_mean_z(self, target: Qubit, axis: str = None, output_variable: str = None):
        _, acquire = self.measure(target, axis, output_variable)
        self.post_processing(
            acquire, PostProcessType.DOWN_CONVERT, ProcessAxis.TIME, target
        )
        self.post_processing(acquire, PostProcessType.MEAN, ProcessAxis.TIME, target)
        self.post_processing(acquire, PostProcessType.MEAN, ProcessAxis.SEQUENCE, target)
        return self.post_processing(
            acquire, PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL, qubit=target
        )

    def measure_mean_signal(self, target: Qubit, output_variable: str = None):
        _, acquire = self.measure(target, ProcessAxis.SEQUENCE, output_variable)
        self.post_processing(
            acquire, PostProcessType.DOWN_CONVERT, ProcessAxis.TIME, target
        )
        self.post_processing(acquire, PostProcessType.MEAN, ProcessAxis.TIME, target)
        return self.post_processing(
            acquire, PostProcessType.MEAN, ProcessAxis.SEQUENCE, target
        )

    def measure_scope_mode(self, target: Qubit, output_variable: str = None):
        # Note: currently, the _execute_measure in base_quantum adds an acquire, so the
        # acquire in post_processing will not be taken into consideration since there
        # already exists one
        _, acquire = self.measure(target, ProcessAxis.TIME, output_variable)
        self.post_processing(
            acquire, PostProcessType.DOWN_CONVERT, ProcessAxis.TIME, target
        )
        return self.post_processing(
            acquire, PostProcessType.MEAN, ProcessAxis.SEQUENCE, target
        )

    def measure_single_shot_binned(
        self,
        target: Qubit,
        axis: Union[str, List[str]] = None,
        output_variable: str = None,
    ):
        _, acquire = self.measure(
            target, axis if axis is not None else ProcessAxis.SEQUENCE, output_variable
        )
        self.post_processing(
            acquire, PostProcessType.DOWN_CONVERT, ProcessAxis.TIME, target
        )
        self.post_processing(acquire, PostProcessType.MEAN, ProcessAxis.TIME, target)
        self.post_processing(
            acquire, PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL, qubit=target
        )
        return self.post_processing(acquire, PostProcessType.DISCRIMINATE, qubit=target)

    def _get_entangled_qubits(self, inst):
        """
        Gets qubit ID's in relation to quantum entanglement for the current instruction.
        Important to note that it will route up or out to a qubit from things like
        resonators and pulse channels.
        """
        if not isinstance(inst, Pulse):
            return []

        results = []
        for target in inst.quantum_targets:
            if isinstance(target, PulseChannel):
                devices = self.model.get_devices_from_pulse_channel(target.full_id())
                for device in devices:
                    if isinstance(device, Resonator):
                        for qubit in self.model.qubits:
                            if qubit.measure_device.id == device.id:
                                results.append(qubit)
                    else:
                        results.append(device)
                        results.extend(device.get_auxiliary_devices(target))

            else:
                results.append(target)

        return results

    def _generate_legacy_measure_block(
        self,
        qubit: Qubit,
        mode: AcquireMode,
        entangled_qubits: List[Qubit],
        output_variable: str = None,
    ):
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()
        weights = (
            qubit.measure_acquire.get("weights", None)
            if qubit.measure_acquire.get("use_weights", False)
            else None
        )
        # Naive entanglement checker to assure all entangled qubits are sync before a
        # measurement is done on any of them.

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
            self.existing_names,
            qubit.measure_acquire["delay"],
            weights,
        )

        return [
            Synchronize(entangled_qubits),
            measure_instruction,
            acquire_instruction,
            Synchronize(qubit),
            PhaseReset(entangled_qubits),
        ], acquire_instruction

    def _generate_measure_block(
        self,
        qubit: Qubit,
        mode: AcquireMode,
        entangled_qubits: List[Qubit],
        output_variable: str = None,
        **kwargs,
    ):
        measure_block = MeasureBlock(
            qubit, mode, output_variable, entangled_qubits, self.existing_names
        )
        acquire_instruction = measure_block.get_acquires(qubit)[0]

        return measure_block, acquire_instruction

    def _find_previous_measurement_block(
        self,
        mblock_types: List[Instruction] = [Acquire, MeasurePulse],
        optional_block_types: List[Instruction] = [Synchronize, PhaseReset],
    ):
        # List of node types that a measurement can be made up of.
        mblock_types_cycle = itertools.cycle(mblock_types)

        # Look at the instructions immediately before this measure, and try to find a
        # set of instructions that look exactly like a measure selection. It's a little
        # bit loose with matching, but if it sees a measure followed by an acquire,
        # surrounded by syncs and post-processing, it'll accept that block as a
        # 'measurement'.
        current_type = next(mblock_types_cycle)
        previous_measure_block = []
        instructions = reversed(self._instructions)

        for inst in instructions:
            if not isinstance(
                inst, QuantumInstruction | QuantumInstructionBlock
            ) or isinstance(inst, PostProcessing):
                continue

            if isinstance(inst, MeasureBlock) and len(previous_measure_block) == 0:
                return inst

            if not isinstance(inst, (*optional_block_types, current_type)):
                current_type = next(mblock_types_cycle)
                if not isinstance(inst, (*optional_block_types, current_type)):
                    break

            previous_measure_block.insert(0, inst)

        pre_syncs = [val for val in previous_measure_block if isinstance(val, Synchronize)]
        pre_phase_resets = [
            val for val in previous_measure_block if isinstance(val, PhaseReset)
        ]
        full_measure_block = set([val.__class__ for val in previous_measure_block]) == set(
            mblock_types + optional_block_types
        )
        if full_measure_block and len(pre_syncs) >= 2 and len(pre_phase_resets) >= 1:
            return previous_measure_block
        return None

    def _join_legacy_measure_blocks(self, previous_measure_block, new_measure_block):
        # If we detect a full measure block before us, merge it together if we're
        # entangled and/or can do it validly.
        pre_syncs = [val for val in previous_measure_block if isinstance(val, Synchronize)]
        new_syncs = [val for val in new_measure_block if isinstance(val, Synchronize)]
        pre_phase_resets = [
            val for val in previous_measure_block if isinstance(val, PhaseReset)
        ]
        new_phase_resets = [val for val in new_measure_block if isinstance(val, PhaseReset)]
        full_measure_block = set([val.__class__ for val in previous_measure_block]) == set(
            [val.__class__ for val in new_measure_block]
        )
        if full_measure_block and len(pre_syncs) >= 2 and len(pre_phase_resets) >= 1:
            # Merge the first and last sync with the new.
            pre_syncs[0] += new_syncs[0]
            pre_syncs[-1] += new_syncs[-1]

            # reset all qubits
            pre_phase_resets[-1] += new_phase_resets[-1]

            # Add in our current changes.
            self.insert(
                [
                    val
                    for val in new_measure_block
                    if isinstance(val, (MeasurePulse, Acquire))
                ],
                index=self._instructions.index(pre_syncs[-1]),
            )
        else:
            self.add(new_measure_block)

    def _append_measure_block(
        self,
        previous_measure_block: MeasureBlock,
        qubit: Qubit,
        mode: AcquireMode,
        entangled_qubits: List[Qubit],
        output_variable: str = None,
        **kwargs,
    ):
        previous_measure_block.add_measurements(
            qubit, mode, output_variable, entangled_qubits, self.existing_names
        )
        acquire_instruction = previous_measure_block.get_acquires(qubit)[0]

        return previous_measure_block, acquire_instruction

    def measure(
        self,
        qubit: Qubit,
        axis: ProcessAxis = None,
        output_variable: str = None,
        **kwargs,
    ) -> "QuantumInstructionBuilder":
        """
        Adds a measure instruction. Important note: this only adds the instruction, not
        any post-processing instructions as well. If you're wanting to perform generic
        common operations use the more specific measurement types, as they add all the
        additional information.
        """
        if axis is None:
            axis = ProcessAxis.SEQUENCE

        if axis == ProcessAxis.SEQUENCE:
            mode = AcquireMode.INTEGRATOR
        elif axis == ProcessAxis.TIME:
            mode = AcquireMode.SCOPE
        else:
            raise ValueError(f"Wrong measure axis '{str(axis)}'!")

        entangled_qubits = list(self._entanglement_map.get(qubit, []))
        previous_measure_block = self._find_previous_measurement_block()

        if isinstance(previous_measure_block, list):
            warnings.warn(
                "Use of legacy measurement block recognition is deprecated. "
                "Please use the 'MeasureBlock' type instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            new_measure_block, acquire_instruction = self._generate_legacy_measure_block(
                qubit, mode, entangled_qubits, output_variable
            )
            self._join_legacy_measure_blocks(previous_measure_block, new_measure_block)
        elif (
            isinstance(previous_measure_block, MeasureBlock)
            and qubit not in previous_measure_block.quantum_targets
        ):
            _, acquire_instruction = self._append_measure_block(
                previous_measure_block,
                qubit,
                mode,
                entangled_qubits,
                output_variable,
                **kwargs,
            )
        else:
            new_measure_block, acquire_instruction = self._generate_measure_block(
                qubit, mode, entangled_qubits, output_variable, **kwargs
            )
            self.add(new_measure_block)

        return FluidBuilderWrapper(self, acquire_instruction)

    def X(self, target: Union[Qubit, PulseChannel], radii=None):
        qubit, channel = self.model._resolve_qb_pulse_channel(target)
        return self.add(
            self.model.get_gate_X(qubit, math.pi if radii is None else radii, channel)
        )

    def Y(self, target: Union[Qubit, PulseChannel], radii=None):
        qubit, channel = self.model._resolve_qb_pulse_channel(target)
        return self.add(
            self.model.get_gate_Y(qubit, math.pi if radii is None else radii, channel)
        )

    def Z(self, target: Union[Qubit, PulseChannel], radii=None):
        qubit, channel = self.model._resolve_qb_pulse_channel(target)
        return self.add(
            self.model.get_gate_Z(qubit, math.pi if radii is None else radii, channel)
        )

    def U(self, target: Union[Qubit, PulseChannel], theta, phi, lamb):
        qubit, channel = self.model._resolve_qb_pulse_channel(target)
        return self.add(self.model.get_gate_U(qubit, theta, phi, lamb, channel))

    def post_processing(
        self, acq: Acquire, process, axes=None, qubit: Qubit = None, args=None
    ):
        # Default the mean z map args if none supplied.
        if args is None or not any(args):
            if process == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL:
                if not isinstance(qubit, Qubit):
                    raise ValueError(
                        f"Need qubit to infer {process} arguments. "
                        "Pass in either args or a qubit."
                    )

                args = qubit.mean_z_map_args
            if process == PostProcessType.DISCRIMINATE:
                if not isinstance(qubit, Qubit):
                    raise ValueError(
                        f"Need qubit to infer {process} arguments. "
                        "Pass in either args or a qubit."
                    )

                args = [qubit.discriminator]
            elif process == PostProcessType.DOWN_CONVERT:
                phys = acq.channel.physical_channel
                resonator = next(
                    dev
                    for dev in self.model.get_devices_from_physical_channel(phys.id)
                    if isinstance(dev, Resonator)
                )
                pulse = resonator.get_measure_channel()
                base = phys.baseband
                if pulse.fixed_if:
                    args = [base.if_frequency, phys.sample_time]
                else:
                    args = [pulse.frequency - base.frequency, phys.sample_time]

        return self.add(PostProcessing(acq, process, axes, args))

    def sweep(self, variables_and_values):
        return self.add(Sweep(variables_and_values))

    def pulse(self, *args, **kwargs):
        return self.add(Pulse(*args, **kwargs))

    def acquire(self, channel: "PulseChannel", *args, delay=None, **kwargs):
        if delay is None:
            devices = self.model.get_devices_from_pulse_channel(channel)
            qubits = [i for i in devices if isinstance(i, Qubit)]
            if len(qubits) > 1:
                raise ValueError(
                    "Wrong channel type given to acquire, please give a channel with a single qubit!"
                )
            delay = qubits[0].measure_acquire["delay"]
        return self.add(Acquire(channel, *args, delay=delay, **kwargs))

    def delay(self, target: Union[Qubit, PulseChannel], time: float):
        _, channel = self.model._resolve_qb_pulse_channel(target)
        return self.add(Delay(channel, time))

    def synchronize(self, targets: Union[Qubit, List[Qubit]]):
        if not isinstance(targets, List):
            targets = [targets]

        channels = []
        for target in targets:
            if isinstance(target, PulseChannel):
                channels.append(target)
            elif isinstance(target, Qubit):
                channels.append(target.get_acquire_channel())
                channels.append(target.get_measure_channel())
                channels.extend(target.pulse_channels.values())

        return self.add(Synchronize(channels))

    def phase_shift(self, target: PulseChannel, phase):
        if phase == 0:
            return self

        _, channel = self.model._resolve_qb_pulse_channel(target)
        return self.add(PhaseShift(channel, phase))

    def frequency_shift(self, target: PulseChannel, frequency):
        if frequency == 0:
            return self

        _, channel = self.model._resolve_qb_pulse_channel(target)
        return self.add(FrequencyShift(channel, frequency))

    def cnot(self, controlled_qubit: Qubit, target_qubit: Qubit):
        if isinstance(controlled_qubit, List):
            if len(controlled_qubit) > 1:
                raise ValueError("CNOT requires one control qubit.")
            else:
                controlled_qubit = controlled_qubit[0]

        self.ECR(controlled_qubit, target_qubit)
        self.X(controlled_qubit)
        self.Z(controlled_qubit, -np.pi / 2)
        return self.X(target_qubit, -np.pi / 2)

    def ECR(self, control: Qubit, target: Qubit):
        if not isinstance(control, Qubit) or not isinstance(target, Qubit):
            raise ValueError("The quantum targets of the ECR node must be qubits!")
        pulse_channels = [
            control.get_pulse_channel(ChannelType.drive),
            control.get_pulse_channel(ChannelType.cross_resonance, [target]),
            control.get_pulse_channel(ChannelType.cross_resonance_cancellation, [target]),
            target.get_pulse_channel(ChannelType.drive),
        ]

        sync_instr = Synchronize(pulse_channels)
        results = [sync_instr]
        results.extend(self.model.get_gate_ZX(control, np.pi / 4.0, target))
        results.append(sync_instr)
        results.extend(self.model.get_gate_X(control, np.pi))
        results.append(sync_instr)
        results.extend(self.model.get_gate_ZX(control, -np.pi / 4.0, target))
        results.append(sync_instr)

        return self.add(results)
