# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from uuid import uuid4

import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.ir.instructions import (
    Assign,
    BinaryOperator,
    Delay,
    FrequencyShift,
    Instruction,
    InstructionBlock,
    Jump,
    Label,
    PhaseReset,
    PhaseShift,
    Repeat,
    Reset,
    ResultsProcessing,
    Return,
    Synchronize,
)
from qat.ir.measure import (
    Acquire,
    AcquireMode,
    MeasureBlock,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    acq_mode_process_axis,
)
from qat.ir.pulse_channel import PulseChannel
from qat.ir.waveforms import Pulse, SampledWaveform
from qat.model.device import Component, PhysicalChannel, Qubit
from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.utils.pydantic import QubitId, ValidatedList

log = get_default_logger()


class InstructionBuilder(ABC):
    """Abstract class for assembling quantum programs.

    Provides a number of methods to deal with common quantum semantics, such as qubits,
    gates, measurements, and control flow. The details of how quantum operations are
    implemented are left to child classes.
    """

    def __init__(
        self,
        hardware_model: PhysicalHardwareModel,
        instructions: list[Instruction] = [],
        _qubit_index_by_uuid: dict[str, int] | None = None,
        _qubits_ordered_by_index: list[Qubit] | None = None,
    ):
        self.hw = hardware_model
        self._qubit_index_by_uuid = (
            _qubit_index_by_uuid
            if _qubit_index_by_uuid is not None
            else {qubit.uuid: idx for (idx, qubit) in hardware_model.qubits.items()}
        )
        self._qubits_ordered_by_index = (
            _qubits_ordered_by_index
            if _qubits_ordered_by_index is not None
            else [self.hw.qubits[idx] for idx in sorted(self.hw.qubits.keys())]
        )
        self._ir = InstructionBlock(instructions=instructions)
        self.compiled_shots: int | None = None
        self.shots: int | None = None

    @property
    def qubits(self) -> list[Qubit]:
        """Returns the list of qubits, sorted by index."""
        return self._qubits_ordered_by_index

    def get_physical_qubit(self, index: int) -> Qubit:
        """Returns the qubit assigned with the given physical index.

        :param index: The index of the qubit to return.
        :return: The qubit with the given index.
        """
        return self.hw.qubit_with_index(index)

    def get_logical_qubit(self, index: int) -> Qubit:
        """Returns the qubit assigned with the given logical index.

        :param index: The logical index of the qubit to return.
        :return: The qubit with the given logical index.
        """

        if index >= len(self.qubits):
            raise IndexError(f"Qubit with logical index {index} does not exist.")
        return self.qubits[index]

    @abstractmethod
    def X(
        self, target: Qubit, theta: float = np.pi, pulse_channel: PulseChannel = None
    ): ...

    @abstractmethod
    def Y(
        self, target: Qubit, theta: float = np.pi, pulse_channel: PulseChannel = None
    ): ...

    @abstractmethod
    def Z(
        self, target: Qubit, theta: float = np.pi, pulse_channel: PulseChannel = None
    ): ...

    @abstractmethod
    def U(
        self,
        target: Qubit,
        theta: float,
        phi: float,
        lamb: float,
        pulse_channel: PulseChannel = None,
    ): ...

    @abstractmethod
    def swap(self, target: Qubit, destination: Qubit): ...

    def had(self, target: Qubit):
        return self.Z(target).Y(target, theta=np.pi / 2.0)

    def SX(self, target: Qubit):
        return self.X(target, theta=np.pi / 2.0)

    def SXdg(self, target: Qubit):
        return self.X(target, theta=-(np.pi / 2.0))

    def S(self, target: Qubit):
        return self.Z(target, theta=np.pi / 2.0)

    def Sdg(self, target: Qubit):
        return self.Z(target, theta=-(np.pi / 2))

    def T(self, target: Qubit):
        return self.Z(target, theta=np.pi / 4.0)

    def Tdg(self, target: Qubit):
        return self.Z(target, -(np.pi / 4.0))

    @abstractmethod
    def controlled(self, controllers: Qubit | list[Qubit], builder: InstructionBuilder): ...

    def cX(self, controllers: Qubit | list[Qubit], target: Qubit, theta=np.pi):
        return self.controlled(controllers, self.X(target, theta=theta))

    def cY(self, controllers: Qubit | list[Qubit], target: Qubit, theta=np.pi):
        return self.controlled(controllers, self.Y(target, theta=theta))

    def cZ(self, controllers: Qubit | list[Qubit], target: Qubit, theta=np.pi):
        return self.controlled(controllers, self.Z(target, theta=theta))

    def cnot(self, control: Qubit | list[Qubit], target: Qubit):
        return self.cX(control, target, theta=np.pi)

    @abstractmethod
    def ccnot(self, controllers: list[Qubit], target: Qubit): ...

    def cswap(self, controllers: Qubit | list[Qubit], target: Qubit, destination: Qubit):
        return self.controlled(controllers, self.swap(target, destination))

    @abstractmethod
    def ECR(self, control: Qubit, target: Qubit): ...

    @abstractmethod
    def measure_single_shot_z(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ): ...

    @abstractmethod
    def reset(self, targets: Qubit | list[Qubit], **kwargs): ...

    def repeat(self, repeat_count: int):
        return self.add(Repeat(repeat_count=repeat_count))

    def returns(self, variables: list[str] = None):
        """Add return statement."""
        variables = variables if variables is not None else []
        return self.add(Return(variables=variables))

    def assign(self, name: str, value):
        return self.add(Assign(name=name, value=value))

    def jump(self, label: str | Label, condition: BinaryOperator | None = None):
        return self.add(Jump(label=label, condition=condition))

    def results_processing(self, variable: str, res_format: InlineResultsProcessing):
        return self.add(ResultsProcessing(variable=variable, results_processing=res_format))

    def add(self, *instructions: Instruction, flatten: bool = False):
        """
        Add one or more instruction(s) into this builder. All methods should use this
        instead of accessing the instructions tree directly as it deals with composite instructions.
        """
        self._ir.add(*instructions, flatten=flatten)
        return self

    def __add__(self, other: QuantumInstructionBuilder):
        comp_builder = self.__class__(
            hardware_model=self.hw, instructions=self._ir.instructions.model_copy()
        )
        if isinstance(other, InstructionBuilder):
            comp_builder.add(*other._ir.instructions)
        else:
            raise TypeError(
                "Only another `{self.__class__.__name__}` can be added to this builder."
            )
        return comp_builder

    @staticmethod
    def constrain(angle: float):
        """
        Constrain the rotation angle to avoid redundant rotations around the Bloch sphere.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @classmethod
    def _check_identity_operation(cls, f):
        """
        Wrapper method to constrain the rotation angle and to determine whether to avoid redundant rotations around the Bloch sphere.
        """

        def wrapper(self, target, theta=np.pi, *args, **kwargs):
            theta = self.constrain(theta)
            return self if np.isclose(theta, 0) else f(self, target, theta, *args, **kwargs)

        return wrapper

    @property
    def instructions(self):
        return self._ir.instructions

    @instructions.setter
    def instructions(self, instructions: ValidatedList[Instruction]):
        self._ir.instructions = instructions

    @property
    def number_of_instructions(self):
        return self._ir.number_of_instructions

    def __iter__(self):
        return self._ir.__iter__()

    def __reversed__(self):
        return self._ir.__reversed__()

    def flatten(self):
        """
        Flatten the instruction builder by removing nested structures like InstructionBlocks.
        """
        self._ir.flatten()
        return self


class QuantumInstructionBuilder(InstructionBuilder):
    """A pulse-level instruction builder, that provides implementations of quantum gates,
    and an API for pulse-level instructions."""

    def __init__(self, *args, **kwargs):
        pulse_channel_map = kwargs.pop("_pulse_channel_map", None)
        super().__init__(*args, **kwargs)
        if pulse_channel_map is None:
            pulse_channel_map = self._build_pulse_channel_mapping(self.hw)
        self._pulse_channels: dict[str, PulseChannel] = pulse_channel_map

    def pretty_print(self):
        output_str = ""
        for instr in self:
            targets = getattr(instr, "targets", None)
            if targets:
                format_target = ""
                for target in targets:
                    format_target = f"{instr.__class__.__name__} -> {self.hw._ids_to_pulse_channels[target]} "
                    device = self.hw._pulse_channel_ids_to_device[target]
                    if isinstance(device, Qubit):
                        format_target += f" @Q{self.hw._qubits_to_qubit_ids[device]}"
                    else:
                        qubit = self.hw._resonators_to_qubits[device]
                        format_target += f" @R{self.hw._qubits_to_qubit_ids[qubit]}"
                output_str += format_target

            else:
                output_str += str(instr)

            output_str += "\n"

        return output_str.rstrip()

    @InstructionBuilder._check_identity_operation
    def X(self, target: Qubit, theta: float = np.pi, pulse_channel: PulseChannel = None):
        """
        Adds a gate that drives the qubit with a rotation angle `theta` to the builder.

        :param target: The qubit to be rotated.
        :param theta: The applied rotation angle.
        :param pulse_channel: The pulse channel the pulses get sent to.
        """
        if np.isclose(np.abs(theta), np.pi / 2.0):
            xpi_pulse = self._hw_X_pi_2(target, pulse_channel=pulse_channel)
            angle = 0 if theta > 0 else np.pi
            return self._apply_z_transform_on_operation(
                target, xpi_pulse, angle, pulse_channel=pulse_channel
            )

        elif (
            np.isclose(np.abs(theta), np.pi)
            and getattr(target, "direct_x_pi", False)
            and getattr(target.drive_pulse_channel, "pulse_x_pi", None) is not None
        ):
            xpi_pulse = self._hw_X_pi(target, pulse_channel=pulse_channel)
            angle = 0 if theta > 0 else np.pi
            return self._apply_z_transform_on_operation(
                target, xpi_pulse, angle, pulse_channel=pulse_channel
            )

        return self.U(
            target,
            theta=theta,
            phi=-np.pi / 2.0,
            lamb=np.pi / 2.0,
            pulse_channel=pulse_channel,
        )

    @InstructionBuilder._check_identity_operation
    def Y(self, target: Qubit, theta: float = np.pi, pulse_channel: PulseChannel = None):
        """
        Adds a gate that drives the qubit with a rotation angle `theta` to the builder.

        :param target: The qubit to be rotated.
        :param theta: The applied rotation angle.
        :param pulse_channel: The pulse channel the pulses get sent to.
        """
        if np.isclose(np.abs(theta), np.pi / 2.0):
            xpi_pulse = self._hw_X_pi_2(target, pulse_channel=pulse_channel)
            angle = np.pi / 2 if theta > 0 else -np.pi / 2
            return self._apply_z_transform_on_operation(
                target, xpi_pulse, angle, pulse_channel=pulse_channel
            )

        elif (
            np.isclose(np.abs(theta), np.pi)
            and getattr(target, "direct_x_pi", False)
            and getattr(target.drive_pulse_channel, "pulse_x_pi", None) is not None
        ):
            xpi_pulse = self._hw_X_pi(target, pulse_channel=pulse_channel)
            angle = np.pi / 2 if theta > 0 else -np.pi / 2
            return self._apply_z_transform_on_operation(
                target, xpi_pulse, angle, pulse_channel=pulse_channel
            )

        return self.U(target, theta=theta, phi=0.0, lamb=0.0, pulse_channel=pulse_channel)

    @InstructionBuilder._check_identity_operation
    def Z(self, target: Qubit, theta: float = np.pi, pulse_channel: PulseChannel = None):
        """
        Adds a virtual gate that rotates the reference frame of the qubit with a phase `theta`
        to the builder.

        :param target: The qubit to be rotated.
        :param theta: The applied rotation angle.
        """
        if pulse_channel:
            log.warning(
                "Pulse channel in Z-gate will be ignored in the `QuantumInstructionBuilder`."
            )

        return self.add(*self._hw_Z(target=target, theta=theta))

    def U(
        self,
        target: Qubit,
        theta: float,
        phi: float,
        lamb: float,
        pulse_channel: PulseChannel = None,
    ):
        """
        Adds an arbitrary rotation around the Bloch sphere with 3 Euler angles
        to the builder (see https://doi.org/10.48550/arXiv.1707.03429).

        :param target: The qubit to be rotated.
        :param theta: Rotation angle.
        :param phi: Rotation angle.
        :param lamb: Rotation angle.
        :param pulse_channel: The pulse channel the pulses get sent to.
        """
        theta = self.constrain(theta)
        return self.add(
            *self._hw_Z(target, theta=lamb + np.pi, pulse_channel=pulse_channel),
            *self._hw_X_pi_2(target, pulse_channel=pulse_channel),
            *self._hw_Z(target, theta=np.pi - theta, pulse_channel=pulse_channel),
            *self._hw_X_pi_2(target, pulse_channel=pulse_channel),
            *self._hw_Z(target, theta=phi, pulse_channel=pulse_channel),
        )

    def ZX(self, target1: Qubit, target2: Qubit, theta: float = np.pi / 4.0):
        """
        Adds a two-qubit interaction gate exp(-i \theta Z x X) to the builder.

        :param target1: The qubit to which Z gets applied to.
        :param target2: The qubit to which X gets applied to.
        :param theta: The applied rotation angle.
        """
        theta = self.constrain(theta)
        if np.isclose(theta, 0):
            return []

        target1_id = self._qubit_index_by_uuid[target1.uuid]
        target2_id = self._qubit_index_by_uuid[target2.uuid]

        cr_pulse_channel = target1.cross_resonance_pulse_channels[target2_id]
        cr_cancellation_pulse_channel = target2.cross_resonance_cancellation_pulse_channels[
            target1_id
        ]

        if np.isclose(theta, np.pi / 4.0):
            return self.add(*self._hw_ZX_pi_4(target1, target2))
        elif np.isclose(theta, -np.pi / 4.0):
            return self.add(
                PhaseShift(targets=cr_pulse_channel.uuid, phase=np.pi),
                PhaseShift(targets=cr_cancellation_pulse_channel.uuid, phase=np.pi),
                *self._hw_ZX_pi_4(target1, target2),
                PhaseShift(targets=cr_pulse_channel.uuid, phase=np.pi),
                PhaseShift(targets=cr_cancellation_pulse_channel.uuid, phase=np.pi),
            )
        else:
            raise NotImplementedError(
                "Generic ZX gate not implemented yet! Please use `theta` = pi/4 or -pi/4."
            )

    def _apply_z_transform_on_operation(
        self,
        target: Qubit,
        gate: list[Pulse],
        theta: float,
        pulse_channel: PulseChannel = None,
    ):
        if np.isclose(theta, 0.0):
            return self.add(*gate)

        return self.add(
            *self._hw_Z(target, theta=-theta, pulse_channel=pulse_channel),
            *gate,
            *self._hw_Z(target, theta=theta, pulse_channel=pulse_channel),
        )

    def _hw_X_pi_2(
        self, target: Qubit, pulse_channel: PulseChannel = None, amp_scale: float = 1.0
    ):
        """Op definition for a X(pi/2) gate.

        Optionally allows a pulse channel to be provided to apply the X(pi/2) pulse down.
        """

        pulse_channel = pulse_channel or self.get_pulse_channel(
            target.drive_pulse_channel.uuid
        )
        pulse_waveform = target.drive_pulse_channel.pulse
        pulse_waveform = pulse_waveform.waveform_type(**pulse_waveform.model_dump())
        pulse_waveform.amp *= amp_scale
        return [Pulse(targets=pulse_channel.uuid, waveform=pulse_waveform)]

    def _hw_X_pi(self, target: Qubit, pulse_channel: PulseChannel = None):
        pulse_channel = pulse_channel or self.get_pulse_channel(
            target.drive_pulse_channel.uuid
        )
        pulse_waveform = getattr(target.drive_pulse_channel, "pulse_x_pi", None)
        if pulse_waveform is None or not getattr(target, "direct_x_pi", False):
            x_pi_2_pulse = deepcopy(self._hw_X_pi_2(target, pulse_channel=pulse_channel))
            return [*x_pi_2_pulse, *x_pi_2_pulse]
        pulse_waveform = pulse_waveform.waveform_type(**pulse_waveform.model_dump())
        return [Pulse(targets=pulse_channel.uuid, waveform=pulse_waveform)]

    def _hw_Z(
        self, target: Qubit, theta: float = np.pi, pulse_channel: PulseChannel = None
    ):
        """Op definition for a Z(theta) gate.

        If a channel is provided, the phase shift is applied to that channel. If none is
        provided, or the channel is the drive channel for the target qubit, the phase shift
        is applied to all pulse channels associated with the qubit.
        """

        if theta == 0:
            return []

        # Rotate drive pulse channel of the qubit.
        qubit_drive_id = target.drive_pulse_channel.uuid
        pulse_channel = pulse_channel or self.get_pulse_channel(qubit_drive_id)
        instr_collection = [PhaseShift(targets=pulse_channel.uuid, phase=theta)]
        # Rotate all cross resonance (cancellation) pulse channels pertaining to the qubit.
        qubit_id = self._qubit_index_by_uuid[target.uuid]
        if pulse_channel.uuid == qubit_drive_id:
            for (
                coupled_qubit_id,
                crc_pulse_channel,
            ) in target.cross_resonance_cancellation_pulse_channels.items():
                coupled_qubit = self.hw.qubit_with_index(coupled_qubit_id)
                cr_pulse_channel = coupled_qubit.cross_resonance_pulse_channels[qubit_id]

                instr_collection.append(
                    PhaseShift(targets=cr_pulse_channel.uuid, phase=theta)
                )
                instr_collection.append(
                    PhaseShift(targets=crc_pulse_channel.uuid, phase=theta)
                )

        return instr_collection

    def _hw_ZX_pi_4(self, target1: Qubit, target2: Qubit):
        """Native op definition for a ZX(pi/4) gate."""

        target1_id = self._qubit_index_by_uuid[target1.uuid]
        target2_id = self._qubit_index_by_uuid[target2.uuid]

        target1_pulse_channel = target1.cross_resonance_pulse_channels[target2_id]
        target2_pulse_channel = target2.cross_resonance_cancellation_pulse_channels[
            target1_id
        ]

        if target1_pulse_channel is None or target2_pulse_channel is None:
            raise ValueError(
                f"Tried to perform cross resonance on {str(target2)} "
                f"that is not linked to {str(target1)}."
            )

        pulse = target1_pulse_channel.zx_pi_4_pulse.model_dump()
        waveform_type = target1_pulse_channel.zx_pi_4_pulse.waveform_type
        if pulse is None:
            raise ValueError(
                f"No `zx_pi_4_pulse` available on {target1} with index {target1_id}."
            )

        sync = Synchronize(
            targets=[
                pulse_ch.uuid
                for pulse_ch in target1.all_qubit_and_resonator_pulse_channels
                + target2.all_qubit_and_resonator_pulse_channels
            ]
        )

        return [
            sync,
            Pulse(target=target1_pulse_channel.uuid, waveform=waveform_type(**pulse)),
            Pulse(target=target2_pulse_channel.uuid, waveform=waveform_type(**pulse)),
            sync,
        ]

    def _generate_measure_acquire(
        self, target: Qubit, mode: AcquireMode, output_variable: str = None
    ):
        # Measure-related info.
        measure_channel = target.measure_pulse_channel
        measure_instruction = Pulse(
            targets=measure_channel.uuid,
            waveform=measure_channel.pulse.waveform_type(
                **measure_channel.pulse.model_dump()
            ),
        )

        # Acquire-related info.
        acquire_channel = target.resonator.acquire_pulse_channel
        acquire_duration = (
            measure_channel.pulse.width
            if acquire_channel.acquire.sync
            else acquire_channel.acquire.width
        )

        if acquire_channel.acquire.use_weights:
            filter = Pulse(
                waveform=SampledWaveform(samples=acquire_channel.acquire.weights),
                duration=acquire_duration,
                target=acquire_channel.uuid,
            )
        else:
            filter = None

        A, B = target.mean_z_map_args[0], target.mean_z_map_args[1]
        mean_g = (1 - B) / A
        mean_e = (-1 - B) / A
        rotation = np.mod(-np.angle(mean_e - mean_g), 2 * np.pi)
        threshold = (np.exp(1j * rotation) * (mean_e + mean_g)).real / 2

        acquire_instruction = Acquire(
            targets=acquire_channel.uuid,
            duration=acquire_duration,
            mode=mode,
            delay=acquire_channel.acquire.delay,
            filter=filter,
            output_variable=output_variable,
            rotation=rotation,
            threshold=threshold,
        )

        return [
            measure_instruction,
            acquire_instruction,
        ]

    def measure(
        self,
        targets: Qubit | set[Qubit],
        mode: AcquireMode = AcquireMode.INTEGRATOR,
        output_variable: str = None,
        sync_qubits: bool = True,
    ) -> QuantumInstructionBuilder:
        """
        Measure one or more qubits.

        :param targets: The qubit(s) to be measured.
        :param mode: The type of acquisition at the level of the control hardware.
        :param sync_qubits: Flag determining whether to align the measurements of all
                            qubits in `targets` or not. Sync between qubits is on by default.
        """
        if isinstance(targets, Qubit):
            targets = [targets]

        target_ids = {self._qubit_index_by_uuid[target.uuid] for target in targets}
        measure_block = MeasureBlock(qubit_targets=target_ids)

        all_pulse_channels = [
            pc.uuid
            for qubit in targets
            for pc in qubit.all_qubit_and_resonator_pulse_channels
        ]

        if sync_qubits:
            # Sync pulse channels of all pulse channels.
            measure_block.add(Synchronize(targets=all_pulse_channels))

        for qubit in targets:
            if not sync_qubits:
                qubit_pulse_channels = [
                    pc.uuid for pc in qubit.all_qubit_and_resonator_pulse_channels
                ]
                measure_block.add(Synchronize(targets=qubit_pulse_channels))

            output_var = output_variable or self._generate_output_variable(qubit)

            # Measure and acquire instructions for the qubit.
            measure, acquire = self._generate_measure_acquire(
                qubit, mode=mode, output_variable=output_var
            )
            duration = max(measure.duration, acquire.delay + acquire.duration)
            measure_block.add(measure, acquire)
            measure_block.duration = max(measure_block.duration, duration)

            if not sync_qubits:
                qubit_pulse_channels = [
                    pc.uuid for pc in qubit.all_qubit_and_resonator_pulse_channels
                ]
                measure_block.add(Synchronize(targets=qubit_pulse_channels))

        # Sync pulse channels of all pulse channels after measurement.
        if sync_qubits:
            measure_block.add(Synchronize(targets=all_pulse_channels))

        return self.add(measure_block)

    def _take_time_mean(
        self,
        acquire_mode: AcquireMode,
        target: Qubit | set[Qubit],
        output_variable: str,
    ):
        """
        Takes the mean of the acquired data.
        If the acquire mode is INTEGRATOR, it returns the builder without any changes

        :param target: The qubit to be measured.
        :param axis: The type of axis which the post-processing of readouts should occur on.
        :param output_variable: The variable where the output of the acquire should be saved.
        """
        if acquire_mode == AcquireMode.INTEGRATOR:
            return self

        return self.post_processing(
            target, output_variable, PostProcessType.MEAN, ProcessAxis.TIME
        )

    def _take_sequence_mean(
        self,
        acquire_mode: AcquireMode,
        target: Qubit | set[Qubit],
        output_variable: str,
    ):
        """
        Applies the mean post-processing to the acquired data along the sequence axis. If the acquire mode is SCOPE,
        the mean post-processing is not needed, so it returns the builder without any changes.

        :param target: The qubit to be measured.
        :param axis: The type of axis which the post-processing of readouts should occur on.
        :param output_variable: The variable where the output of the acquire should be saved.
        """
        if acquire_mode == AcquireMode.SCOPE:
            return self

        return self.post_processing(
            target, output_variable, PostProcessType.MEAN, ProcessAxis.SEQUENCE
        )

    def measure_single_shot_z(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ):
        """
        Measure a single qubit along the z-axis.
        Execution results in an array of floats centred around 0, where anything above 0 shows a bias towards a
        +Z measurement, and below 0 shows a bias towards a -Z measurement.
        The array is an ndarray, where the first dimension is for each of the qubits measured, the second dimension is
        for each shot run.

        :param target: The qubit to be measured.
        :param axis: The type of axis which the post-processing of readouts should occur on.
        :param output_variable: The variable where the output of the acquire should be saved.
        """
        output_variable = output_variable or self._generate_output_variable(target)
        acquire_mode = acq_mode_process_axis[axis]
        self.measure(target, acquire_mode, output_variable)
        self._take_time_mean(acquire_mode, target, output_variable)
        return self.post_processing(
            target, output_variable, PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL
        )

    def measure_single_shot_signal(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ):
        """
        Measure the signal of a single qubit.
        Execution results in an array of complex numbers that show the war signal output form the hardware.
        The array is an ndarray, where the first dimension is for each of the qubits measured, the second dimension is
        for each shot run.

        :param target: The qubit to be measured.
        :param axis: The type of axis which the post-processing of readouts should occur on.
        :param output_variable: The variable where the output of the acquire should be saved.
        """
        output_variable = output_variable or self._generate_output_variable(target)
        acquire_mode = acq_mode_process_axis[axis]
        self.measure(target, acquire_mode, output_variable)
        return self._take_time_mean(acquire_mode, target, output_variable)

    def measure_mean_z(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ):
        """
        Measure a single qubit along the z-axis.
        Execution results in an array of floats centred around 0, where anything above 0 shows a bias towards a
        +Z measurement, and below 0 shows a bias towards a -Z measurement.
        Each entry is for each qubit measured, and is the mean of the results from `measure_single_shot_z`.

        :param target: The qubit to be measured.
        :param axis: The type of axis which the post-processing of readouts should occur on.
        :param output_variable: The variable where the output of the acquire should be saved.
        """
        output_variable = output_variable or self._generate_output_variable(target)
        acquire_mode = acq_mode_process_axis[axis]
        self.measure(target, acquire_mode, output_variable)
        self._take_time_mean(acquire_mode, target, output_variable)
        self._take_sequence_mean(acquire_mode, target, output_variable)
        return self.post_processing(
            target, output_variable, PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL
        )

    def measure_mean_signal(self, target: Qubit, output_variable: str = None):
        """
        Measure the signal of a single qubit.
        Execution results in an array of complex numbers that show the war signal output form the hardware.
        Each entry is for each qubit measured, and is the mean of the results from `measure_single_signal_shot`.

        :param target: The qubit to be measured.
        :param output_variable: The variable where the output of the acquire should be saved.
        """
        output_variable = output_variable or self._generate_output_variable(target)
        acquire_mode = acq_mode_process_axis[ProcessAxis.SEQUENCE]
        return self.measure(target, acquire_mode, output_variable)._take_sequence_mean(
            acquire_mode, target, output_variable
        )

    def measure_scope_mode(self, target: Qubit, output_variable: str = None):
        """
        Measure the scope mode

        :param target: The qubit to be measured.
        :param output_variable: The variable where the output of the acquire should be saved.
        """
        acquire_mode = acq_mode_process_axis[ProcessAxis.TIME]
        return self.measure(target, acquire_mode, output_variable)

    def measure_single_shot_binned(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ):
        """
        Measure a single qubit along the z-axis.
        Execution results in an array of ±1s, where 1 indicates a +Z measurement, and -1 indicates a -Z measurement.
        It takes the results from `measure_single_shot_z` and applies a binning operation to round the results to ±1.
        The array is an ndarray, where the first dimension is for each of the qubits measured, the second dimension is
        for each shot run.

        :param target: The qubit to be measured.
        :param axis: The type of axis which the post-processing of readouts should occur on.
        :param output_variable: The variable where the output of the acquire should be saved.
        """
        output_variable = output_variable or self._generate_output_variable(target)
        acquire_mode = acq_mode_process_axis[axis]
        return (
            self.measure(target, acq_mode_process_axis[axis], output_variable)
            ._take_time_mean(acquire_mode, target, output_variable)
            .post_processing(
                target, output_variable, PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL
            )
            .post_processing(target, output_variable, PostProcessType.DISCRIMINATE)
        )

    def reset(
        self, targets: PulseChannel | Qubit | Iterable[PulseChannel | Qubit], **kwargs
    ):
        if isinstance(targets, list):
            targets = set(targets)
        elif isinstance(targets, PulseChannel | Qubit):
            targets = {targets}
        else:
            raise TypeError(
                f"Invalid type, expected '(PulseChannel | Qubit | list[PulseChannel | Qubit])' but got {type(targets)}."
            )

        pulse_channel_ids = []
        for target in targets:
            if isinstance(target, Qubit):
                self.add(Reset(qubit_target=self._qubit_index_by_uuid[target.uuid]))
                pulse_channel_ids.extend(
                    [pc.uuid for pc in target.all_qubit_and_resonator_pulse_channels]
                )
            else:
                pulse_channel_ids.append(target.uuid)

        for pulse_ch_id in pulse_channel_ids:
            self.add(PhaseReset(target=pulse_ch_id))

        return self

    def _generate_output_variable(self, target: Component):
        return "out_" + target.uuid + f"_{np.random.randint(np.iinfo(np.int32).max)}"

    def _find_valid_measure_block(self, target_ids: QubitId | set[QubitId]):
        """
        Finds the previous :class:`MeasureBlock` where the given qubits are not measured yet.

        :param target_ids: The indices of the qubits to be measured.
        """
        for instruction in reversed(self.instructions):
            if isinstance(instruction, MeasureBlock) and not any(
                target_ids & instruction.qubit_targets
            ):
                instruction.qubit_targets.update(target_ids)
                return instruction
        return None

    def post_processing(
        self,
        target: Qubit,
        output_variable: str,
        process_type: PostProcessType,
        axes: ProcessAxis | list[ProcessAxis] | None = None,
        args=None,
    ):
        axes = axes if axes is not None else []
        args = args if args is not None else []

        # Default the mean z-map args if none supplied.
        if not any(args):
            if process_type == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL:
                args = target.mean_z_map_args

            elif process_type == PostProcessType.DISCRIMINATE:
                args = [target.discriminator]

        axes = axes if isinstance(axes, list) else [axes]

        return self.add(
            PostProcessing(
                output_variable=output_variable,
                process_type=process_type,
                axes=axes,
                args=args,
            )
        )

    def cnot(self, control: Qubit, target: Qubit):
        return (
            self.ECR(control, target)
            .X(control)
            .Z(control, theta=-np.pi / 2)
            .X(target, theta=-np.pi / 2)
        )

    def ECR(self, control: Qubit, target: Qubit):
        if not isinstance(control, Qubit) or not isinstance(target, Qubit):
            raise ValueError("The quantum targets of the ECR node must be qubits!")

        target_id = self._qubit_index_by_uuid[target.uuid]

        if not control.cross_resonance_pulse_channels.get(target_id, None):
            raise ValueError(
                f"Qubits {self._qubit_index_by_uuid[control.uuid]} and {target_id} are not coupled."
            )

        return (
            self.ZX(control, target, theta=np.pi / 4.0)
            .X(control, theta=np.pi)
            .ZX(control, target, theta=-np.pi / 4.0)
        )

    def swap(self, target: Qubit, destination: Qubit):
        raise NotImplementedError("Not available on this hardware model.")

    def pulse(self, **kwargs):
        return self.add(Pulse(**kwargs))

    def acquire(
        self,
        target: Qubit,
        delay: float = 1e-06,
        **kwargs,
    ):
        pulse_channel = target.acquire_pulse_channel

        if delay is None:
            kwargs["delay"] = pulse_channel.acquire.delay

        return self.add(Acquire(target=pulse_channel.uuid, **kwargs))

    def delay(self, target: Qubit | PulseChannel, duration: float):
        delays = []
        if isinstance(target, Qubit):
            delays = [
                Delay(targets=pulse_ch.uuid, duration=duration)
                for pulse_ch in target.all_qubit_and_resonator_pulse_channels
            ]
        else:
            delays = [Delay(targets=target.uuid, duration=duration)]
        return self.add(*delays)

    def synchronize(self, targets: Qubit | list[Qubit | PulseChannel]):
        targets = targets if isinstance(targets, list) else [targets]

        pulse_channel_ids = set()
        for target in targets:
            if isinstance(target, Qubit):
                qubit_pulse_channel_ids = [
                    pulse_channel.uuid
                    for pulse_channel in target.all_qubit_and_resonator_pulse_channels
                ]
                pulse_channel_ids.update(qubit_pulse_channel_ids)
            else:
                pulse_channel_ids.add(target.uuid)
        if len(pulse_channel_ids) > 1:
            return self.add(Synchronize(targets=pulse_channel_ids))
        else:
            log.warning(
                f"Ignored synchronisation of a single pulse channel with id {next(iter(pulse_channel_ids))}."
            )
            return self

    def controlled(self, controllers: Qubit | list[Qubit], builder: InstructionBuilder):
        raise NotImplementedError("Not available on this hardware model.")

    def ccnot(self, controllers: list[Qubit], target: Qubit):
        raise NotImplementedError("Not available on this hardware model.")

    @InstructionBuilder._check_identity_operation
    def phase_shift(self, target: PulseChannel, theta: float):
        return self.add(PhaseShift(targets=target.uuid, phase=theta))

    def frequency_shift(self, target: PulseChannel, frequency: float):
        if np.isclose(frequency, 0.0):
            return self
        return self.add(FrequencyShift(targets=target.uuid, frequency=frequency))

    def get_pulse_channel(
        self,
        id: str,
    ) -> PulseChannel:
        """Given an id, return the corresponding pulse channel.

        Checks internally stored pulse channels, but can pull pulse channels from the
        hardware model if not found.
        """

        pulse_channel = self._pulse_channels.get(id, None)
        if pulse_channel is not None:
            return pulse_channel

        pulse_channel = self.hw.pulse_channel_with_id(id)
        if pulse_channel is None:
            raise ValueError(f"Pulse channel with id '{id}' not found.")

        pulse_channel = self.create_pulse_channel(
            frequency=pulse_channel.frequency,
            physical_channel=self.hw.physical_channel_for_pulse_channel_id(
                pulse_channel.uuid
            ).uuid,
            imbalance=pulse_channel.imbalance,
            phase_iq_offset=pulse_channel.phase_iq_offset,
            scale=pulse_channel.scale,
            uuid=pulse_channel.uuid,
        )
        return pulse_channel

    def create_pulse_channel(
        self,
        frequency: float,
        physical_channel: PhysicalChannel | str,
        imbalance: float = 1.0,
        phase_iq_offset: float = 0.0,
        scale: float | complex = 1.0 + 0.0j,
        uuid: str | None = None,
    ) -> PulseChannel:
        """Creates a pulse channel and adds stores it within the builder.

        The channel can be provided as a physical channel which the logical channel is
        linked too, or use the physical channel of a provided pulse channel.
        """

        if isinstance(physical_channel, PhysicalChannel):
            physical_channel = physical_channel.uuid
        else:
            if self.hw.physical_channel_with_id(physical_channel) is None:
                raise ValueError(
                    f"Physical channel with id '{physical_channel}' not found."
                )

        uuid = uuid if uuid is not None else str(uuid4())
        pulse_channel = PulseChannel(
            uuid=uuid,
            physical_channel_id=physical_channel,
            frequency=frequency,
            imbalance=imbalance,
            phase_iq_offset=phase_iq_offset,
            scale=scale,
        )

        self._pulse_channels[uuid] = pulse_channel
        return pulse_channel

    @staticmethod
    def _build_pulse_channel_mapping(hw: PhysicalHardwareModel) -> dict[str, PulseChannel]:
        pulse_channels = {}
        for qubit in hw.qubits.values():
            for device in [qubit, qubit.resonator]:
                physical_channel_id = device.physical_channel.uuid
                for pc in device.all_pulse_channels:
                    pulse_channels[pc.uuid] = PulseChannel(
                        physical_channel_id=physical_channel_id,
                        frequency=pc.frequency,
                        imbalance=pc.imbalance,
                        phase_iq_offset=pc.phase_iq_offset,
                        scale=pc.scale,
                        uuid=pc.uuid,
                    )
        return pulse_channels

    def _create_empty_builder(self):
        """Creates a new instance of the builder that is free of instructions; passes
        through the expensive mappings created at instantiation."""

        return type(self)(
            hardware_model=self.hw,
            instructions=[],
            _pulse_channel_map=self._pulse_channels,
            _qubit_index_by_uuid=self._qubit_index_by_uuid,
            _qubits_ordered_by_index=self._qubits_ordered_by_index,
        )


PydQuantumInstructionBuilder = QuantumInstructionBuilder
