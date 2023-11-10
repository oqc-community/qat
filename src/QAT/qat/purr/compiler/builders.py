# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import itertools
import math
from typing import List, Set, Union

import numpy
import numpy as np
from qat.purr.compiler.config import InlineResultsProcessing
from qat.purr.compiler.devices import ChannelType, PulseChannel, Qubit, Resonator
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    Assign,
    Delay,
    DeviceUpdate,
    Instruction,
    Jump,
    Label,
    MeasurePulse,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
    QuantumInstruction,
    Repeat,
    Reset,
    ResultsProcessing,
    Return,
    Sweep,
    Synchronize,
)
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.serializer import json_dumps, json_loads

log = get_default_logger()


class InstructionBuilder:
    """
    Base instruction builder class that leaves unimplemented the methods that vary on a
    hardware-by-hardware basis.
    """
    def __init__(
        self,
        hardware_model,
        instructions: List[Union["InstructionBuilder", Instruction]] = None
    ):
        super().__init__()
        self._instructions = []
        self.existing_names = set()
        self.model = hardware_model
        self.add(instructions)

    @property
    def instructions(self):
        return list(self._instructions)

    @staticmethod
    def deserialize(blob, model) -> "QuantumInstructionBuilder":
        # TODO: At this point everything is a Quantum builder variation, base class losing meaning.
        builder = QuantumInstructionBuilder(model)
        builder._instructions = json_loads(blob, model=model)
        return builder

    def serialize(self):
        """
        Currently only serializes the instructions, not the supporting objects of the builder itself.
        This could be supported pretty easily, but not required right now.
        """
        return json_dumps(self._instructions)

    def splice(self):
        """ Clears the builder and returns its current instructions. """
        final_instructions = self.instructions
        self.clear()
        return final_instructions

    def clear(self):
        """ Resets builder internal state for building a new set of instructions."""
        self._instructions = []
        self.existing_names = set()

    def get_child_builder(self):
        return InstructionBuilder(self.model)

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
                regenerated_names[inst.name
                                 ] = new_name = Label.generate_name(existing_names)

            inst.name = new_name

    def merge_builder(self, other_builder: "InstructionBuilder"):
        """
        Merge this builder into the current instance. Checks for label name clashes and
        resolves them if any are found.
        """
        name_clashes = other_builder.existing_names.intersection(self.existing_names)
        self.existing_names.update(other_builder.existing_names)
        if any(name_clashes):
            log.warning(
                "When merging builders these labels had name clashes: "
                f"{', '.join(name_clashes)}. Regenerating auto-assigned variable names."
            )
            other_builder._fix_clashing_label_names(name_clashes, self.existing_names)

        self.add(other_builder._instructions)

    def add(
        self,
        components: Union["InstructionBuilder",
                          Instruction,
                          List[Union["InstructionBuilder", Instruction]]]
    ):
        """
        Adds an instruction to this builder. All methods should use this instead of
        accessing the instructions list directly as it deals with nested builders and
        merging.
        """
        if components is None:
            return self

        if not isinstance(components, List):
            components = [components]

        inst_list = []
        for component in components:
            if isinstance(component, InstructionBuilder):
                self.merge_builder(component)
            else:
                inst_list.append(component)

        self._instructions.extend(inst_list)
        return self

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
        return self.X(target, numpy.pi / 2)

    def SXdg(self, target):
        return self.X(target, -(numpy.pi / 2))

    def S(self, target):
        return self.Z(target, numpy.pi / 2)

    def Sdg(self, target):
        return self.Z(target, -(numpy.pi / 2))

    def T(self, target):
        return self.Z(target, numpy.pi / 4)

    def Tdg(self, target):
        return self.Z(target, -(numpy.pi / 4))

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
        """ Add return statement. """
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
            val for val in set(dir(instance[0])) & set(dir(instance))
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
    def get_child_builder(self):
        return QuantumInstructionBuilder(self.model)

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
        output_variable: str = None
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

    def measure(
        self, qubit: Qubit, axis: ProcessAxis = None, output_variable: str = None
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

        # Naive entanglement checker to assure all entangled qubits are sync before a
        # measurement is done on any of them.
        entangled_qubits = set()
        for inst in self._instructions:
            if isinstance(inst, QuantumInstruction):
                qubits = self._get_entangled_qubits(inst)
                if qubit in qubits:
                    entangled_qubits.update(qubits)

        entangled_qubits = list(entangled_qubits)

        # List of node types that a measurement can be made up of.
        mblock_types = [Acquire, MeasurePulse]
        mblock_types_cycle = itertools.cycle(mblock_types)
        optional_block_types = [PostProcessing, Synchronize, PhaseReset]

        # Look at the instructions immediately before this measure, and try to find a
        # set of instructions that look exactly like a measure selection. It's a little
        # bit loose with matching, but if it sees a measure followed by an acquire,
        # surrounded by syncs and post-processing, it'll accept that block as a
        # 'measurement'.
        current_type = next(mblock_types_cycle)
        previous_measure_block = []
        for inst in reversed(self._instructions):
            # We skip classic instructions since they have no relevance.
            if not isinstance(inst, QuantumInstruction):
                continue

            if not isinstance(inst, (*optional_block_types, current_type)):
                current_type = next(mblock_types_cycle)
                if not isinstance(inst, (*optional_block_types, current_type)):
                    break

            previous_measure_block.append(inst)

        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()
        acquire_instruction = Acquire(
            acquire_channel,
            qubit.pulse_measure['width']
            if qubit.measure_acquire['sync'] else qubit.measure_acquire['width'],
            mode,
            output_variable,
            self.existing_names,
            qubit.measure_acquire['delay']
        )

        # If we detect a full measure block before us, merge it together if we're
        # entangled and/or can do it validly.
        syncs = [val for val in previous_measure_block if isinstance(val, Synchronize)]
        phase_resets = [
            val for val in previous_measure_block if isinstance(val, PhaseReset)
        ]
        full_measure_block = set([val.__class__ for val in previous_measure_block])\
                             == set(mblock_types + optional_block_types)
        if full_measure_block and len(syncs) >= 2 and len(phase_resets) >= 1:
            # Find the pre-measure sync in the preceeding measure block and merge our
            # values into it.
            syncs[-1].add_channels(entangled_qubits)

            # Find the post-measure sync, merge with ours, then remove it.
            final_syncronize = syncs[0] + qubit
            self._instructions.remove(syncs[0])
            # reset all qubits
            final_phase_reset = phase_resets[0] + entangled_qubits
            self._instructions.remove(phase_resets[0])

            # Add in our current changes.
            self.add([
                MeasurePulse(measure_channel, **qubit.pulse_measure),
                acquire_instruction,
                final_syncronize,
                final_phase_reset
            ])

            # Move all post-processing until after our newly-shifted syncronize block.
            # Order matters here.
            for pp in [
                val for val in reversed(previous_measure_block)
                if isinstance(val, PostProcessing)
            ]:
                self._instructions.remove(pp)
                self.add(pp)
        else:
            self.add([
                Synchronize(entangled_qubits),
                MeasurePulse(measure_channel, **qubit.pulse_measure),
                acquire_instruction,
                Synchronize(qubit),
                PhaseReset(entangled_qubits)
            ])

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
                    dev for dev in self.model.get_devices_from_physical_channel(phys.id)
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
                raise ValueError("Wrong channel type given to acquire, please give a channel with a single qubit!")
            delay = qubits[0].measure_acquire['delay']
        return self.add(Acquire(channel, *args, delay, **kwargs))

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
            target.get_pulse_channel(ChannelType.drive)
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
