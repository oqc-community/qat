# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
"""Instruction builder abstractions and a concrete quantum builder.

This module defines the abstract :class:`InstructionBuilder` API and a
concrete :class:`QuantumInstructionBuilder` which implements pulse-level gate
construction and measurement helpers. The builders produce an IR
``InstructionBlock`` suitable for later compilation and scheduling stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from uuid import uuid4

import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.ir.instruction_basetypes import AcquireMode, PostProcessType, ProcessAxis
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
    Discriminate,
    Equalise,
    MeasureBlock,
    PostProcessing,
    PostSelect,
    acq_mode_process_axis,
)
from qat.ir.pulse_channel import PulseChannel
from qat.ir.waveforms import Pulse, SampledWaveform
from qat.model.device import Component, PhysicalChannel, Qubit
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.post_processing import LinearMapToRealMethod, MaxLikelihoodMethod
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
        instructions: list[Instruction] | None = None,
        _qubit_index_by_uuid: dict[str, int] | None = None,
        _qubits_ordered_by_index: list[Qubit] | None = None,
    ):
        if instructions is None:
            instructions = []
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
    ) -> InstructionBuilder: ...

    @abstractmethod
    def Y(
        self, target: Qubit, theta: float = np.pi, pulse_channel: PulseChannel = None
    ) -> InstructionBuilder: ...

    @abstractmethod
    def Z(
        self, target: Qubit, theta: float = np.pi, pulse_channel: PulseChannel = None
    ) -> InstructionBuilder: ...

    @abstractmethod
    def U(
        self,
        target: Qubit,
        theta: float,
        phi: float,
        lamb: float,
        pulse_channel: PulseChannel = None,
    ) -> InstructionBuilder: ...

    @abstractmethod
    def swap(self, target: Qubit, destination: Qubit) -> InstructionBuilder: ...

    def had(self, target: Qubit) -> InstructionBuilder:
        return self.Z(target).Y(target, theta=np.pi / 2.0)

    def SX(self, target: Qubit) -> InstructionBuilder:
        return self.X(target, theta=np.pi / 2.0)

    def SXdg(self, target: Qubit) -> InstructionBuilder:
        return self.X(target, theta=-(np.pi / 2.0))

    def S(self, target: Qubit) -> InstructionBuilder:
        return self.Z(target, theta=np.pi / 2.0)

    def Sdg(self, target: Qubit) -> InstructionBuilder:
        return self.Z(target, theta=-(np.pi / 2))

    def T(self, target: Qubit) -> InstructionBuilder:
        return self.Z(target, theta=np.pi / 4.0)

    def Tdg(self, target: Qubit) -> InstructionBuilder:
        return self.Z(target, -(np.pi / 4.0))

    @abstractmethod
    def controlled(
        self, controllers: Qubit | list[Qubit], builder: InstructionBuilder
    ) -> InstructionBuilder: ...

    def cX(
        self, controllers: Qubit | list[Qubit], target: Qubit, theta: float = np.pi
    ) -> InstructionBuilder:
        return self.controlled(controllers, self.X(target, theta=theta))

    def cY(
        self, controllers: Qubit | list[Qubit], target: Qubit, theta: float = np.pi
    ) -> InstructionBuilder:
        return self.controlled(controllers, self.Y(target, theta=theta))

    def cZ(
        self, controllers: Qubit | list[Qubit], target: Qubit, theta: float = np.pi
    ) -> InstructionBuilder:
        return self.controlled(controllers, self.Z(target, theta=theta))

    def cnot(self, control: Qubit | list[Qubit], target: Qubit) -> InstructionBuilder:
        return self.cX(control, target, theta=np.pi)

    @abstractmethod
    def ccnot(self, controllers: list[Qubit], target: Qubit) -> InstructionBuilder: ...

    def cswap(
        self, controllers: Qubit | list[Qubit], target: Qubit, destination: Qubit
    ) -> InstructionBuilder:
        return self.controlled(controllers, self.swap(target, destination))

    @abstractmethod
    def ECR(self, control: Qubit, target: Qubit) -> InstructionBuilder: ...

    @abstractmethod
    def measure_with_granular_post_processing(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ):
        """Measure a qubit and emit the full granular post-processing pipeline.

        Compiler frontends should call this method when lowering a measurement
        assignment into IR. The granular discrimination chain is emitted for
        both qubits with ``post_process_method`` configured and legacy qubits.
        For configured qubits, the chain (for example ``Equalise`` →
        ``Discriminate``) is derived from the configured
        post-processing method; for legacy qubits, the equivalent granular
        chain is derived from legacy ``mean_z_map_args`` data.

        See :meth:`QuantumInstructionBuilder.measure_with_granular_post_processing` for
        full detail.
        """
        ...

    @abstractmethod
    def reset(
        self, targets: Qubit | list[Qubit], **kwargs
    ) -> QuantumInstructionBuilder: ...

    def repeat(self, repeat_count: int) -> InstructionBuilder:
        return self.add(Repeat(repeat_count=repeat_count))

    def returns(self, variables: list[str] = None) -> InstructionBuilder:
        """Add return statement."""
        variables = variables if variables is not None else []
        return self.add(Return(variables=variables))

    def assign(self, name: str, value) -> InstructionBuilder:
        return self.add(Assign(name=name, value=value))

    def jump(
        self, label: str | Label, condition: BinaryOperator | None = None
    ) -> InstructionBuilder:
        return self.add(Jump(label=label, condition=condition))

    def results_processing(
        self, variable: str, res_format: InlineResultsProcessing
    ) -> InstructionBuilder:
        return self.add(ResultsProcessing(variable=variable, results_processing=res_format))

    def add(self, *instructions: Instruction, flatten: bool = False) -> InstructionBuilder:
        """Add one or more instruction(s) into this builder.

        All methods should use this instead of accessing the instructions tree directly as
        it deals with composite instructions.
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
        """Constrain the rotation angle to avoid redundant rotations around the Bloch
        sphere."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @classmethod
    def _check_identity_operation(cls, f):
        """Wrapper method to constrain the rotation angle and to determine whether to avoid
        redundant rotations around the Bloch sphere."""

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
        """Flatten the instruction builder by removing nested structures like
        InstructionBlocks."""
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

    def pretty_print(self) -> str:
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
        """Adds a gate that drives the qubit with a rotation angle `theta` to the builder.

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
        """Adds a gate that drives the qubit with a rotation angle `theta` to the builder.

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
        """Adds a virtual gate that rotates the reference frame of the qubit with a phase
        `theta` to the builder.

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
        """Adds a two-qubit interaction gate exp(-i \theta Z x X) to the builder.

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
    ) -> QuantumInstructionBuilder:
        if np.isclose(theta, 0.0):
            return self.add(*gate)

        return self.add(
            *self._hw_Z(target, theta=-theta, pulse_channel=pulse_channel),
            *gate,
            *self._hw_Z(target, theta=theta, pulse_channel=pulse_channel),
        )

    def _hw_X_pi_2(
        self, target: Qubit, pulse_channel: PulseChannel = None, amp_scale: float = 1.0
    ) -> list[Pulse]:
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

    def _hw_X_pi(self, target: Qubit, pulse_channel: PulseChannel = None) -> list[Pulse]:
        """Op definition for a X(pi) gate.

        If the qubit has a direct X(pi) pulse, it is used. Otherwise, a X(pi/2) pulse is
        used twice.
        """

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
    ) -> list[PhaseShift]:
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

    def _hw_ZX_pi_4(self, target1: Qubit, target2: Qubit) -> list[Synchronize | Pulse]:
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

        if target1_pulse_channel.zx_pi_4_pulse is None:
            raise ValueError(
                f"No `zx_pi_4_pulse` available for qubit pair {target1_id} and "
                f"{target2_id}."
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
    ) -> list[Pulse | Acquire]:
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

        if target.mean_z_map_args is not None:
            # This is the legacy path
            A, B = target.mean_z_map_args[0], target.mean_z_map_args[1]
            mean_g = (1 - B) / A
            mean_e = (-1 - B) / A
            rotation = np.mod(-np.angle(mean_e - mean_g), 2 * np.pi)
            threshold = (np.exp(1j * rotation) * (mean_e + mean_g)).real / 2
        elif isinstance(target.post_process_method, LinearMapToRealMethod):
            # New-style path: derive rotation/threshold from the configured method's
            # mean_z_map_args so QBlox HW thresholded acquisition is correctly set.
            A, B = (
                target.post_process_method.mean_z_map_args[0],
                target.post_process_method.mean_z_map_args[1],
            )
            mean_g = (1 - B) / A
            mean_e = (-1 - B) / A
            rotation = np.mod(-np.angle(mean_e - mean_g), 2 * np.pi)
            threshold = (np.exp(1j * rotation) * (mean_e + mean_g)).real / 2
        else:
            # MaxLikelihoodMethod or unknown: no linear-map calibration available.
            # Default to zero so hardware backends that unconditionally configure
            # these fields do not fail at runtime.
            rotation = 0.0
            threshold = 0.0

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
        """Measure one or more qubits.

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
        """Emit ``PostProcessing(MEAN, TIME)`` unless the acquire mode is INTEGRATOR.

        For ``INTEGRATOR`` mode the hardware already integrates over time so no
        software mean is needed; the builder is returned unchanged.

        :param acquire_mode: The hardware acquisition mode in use.
        :param target: The qubit being measured.
        :param output_variable: The variable name to attach to the instruction.
        :returns: The builder instance.
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
        """Emit ``PostProcessing(MEAN, SEQUENCE)`` unless the acquire mode is SCOPE.

        For ``SCOPE`` mode the raw time-series waveform is captured directly so
        averaging over the shot sequence is not meaningful; the builder is returned
        unchanged.

        :param acquire_mode: The hardware acquisition mode in use.
        :param target: The qubit being measured.
        :param output_variable: The variable name to attach to the instruction.
        :returns: The builder instance.
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
        """Measure a single qubit along the z-axis (customer-facing API).

        Emits a :class:`~qat.ir.measure.MeasureBlock`, an optional
        ``PostProcessing(MEAN, TIME)`` when ``axis`` is
        :attr:`~qat.ir.instruction_basetypes.ProcessAxis.TIME` (SCOPE mode), and a
        ``PostProcessing(LINEAR_MAP_COMPLEX_TO_REAL)`` to project the complex IQ value
        onto a real z-projection.  Results are an ndarray of floats centred around 0:
        values above 0 indicate a bias towards the +Z state, values below 0 indicate
        a bias towards the −Z state.

        .. note::
            This method always emits a legacy ``PostProcessing`` instruction regardless
            of whether ``post_process_method`` is configured on the qubit.  Compiler
            frontends that need the granular discrimination pipeline (``Equalise`` →
            ``Discriminate`` → ``PostSelect``) should call
            :meth:`measure_with_granular_post_processing` instead (and
            :meth:`emit_post_select` for post-selection insertion).

        :param target: The qubit to be measured.
        :param axis: The axis along which post-processing of readouts should occur.
            :attr:`~ProcessAxis.SEQUENCE` (default) uses ``INTEGRATOR`` acquisition;
            :attr:`~ProcessAxis.TIME` uses ``SCOPE`` acquisition and emits an
            additional ``PostProcessing(MEAN, TIME)``.
        :param output_variable: Name of the variable where the acquire result is saved.
            A unique name is generated if not provided.
        :returns: The builder instance.
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
        """Measure the raw IQ signal of a single qubit (customer-facing API).

        Emits a :class:`~qat.ir.measure.MeasureBlock` and, when ``axis`` is
        :attr:`~qat.ir.instruction_basetypes.ProcessAxis.TIME` (SCOPE mode), an
        additional ``PostProcessing(MEAN, TIME)`` to average over the time axis.
        No z-discrimination is applied; results are an ndarray of complex IQ values,
        one per shot.

        :param target: The qubit to be measured.
        :param axis: The axis along which post-processing of readouts should occur.
        :param output_variable: Name of the variable where the acquire result is saved.
            A unique name is generated if not provided.
        :returns: The builder instance.
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
        """Measure a single qubit along the z-axis and return the shot-averaged result.

        Emits a :class:`~qat.ir.measure.MeasureBlock`, averages first over the time
        axis (SCOPE mode only) then over the sequence axis (INTEGRATOR mode only), and
        finally emits ``PostProcessing(LINEAR_MAP_COMPLEX_TO_REAL)`` to project the
        averaged complex IQ value to a real z-expectation value.

        .. note::
            Like :meth:`measure_single_shot_z`, this method always emits a legacy
            ``PostProcessing`` instruction.  Use
            :meth:`measure_with_granular_post_processing` for the granular
            discrimination pipeline.

        :param target: The qubit to be measured.
        :param axis: The axis along which post-processing of readouts should occur.
        :param output_variable: Name of the variable where the acquire result is saved.
            A unique name is generated if not provided.
        :returns: The builder instance.
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
        """Measure the raw IQ signal of a single qubit and return the shot-averaged result.

        Emits a :class:`~qat.ir.measure.MeasureBlock` using ``INTEGRATOR`` acquisition
        and appends a ``PostProcessing(MEAN, SEQUENCE)`` to average the complex IQ
        values across all shots.  Results are a single complex value per qubit.

        :param target: The qubit to be measured.
        :param output_variable: Name of the variable where the acquire result is saved.
            A unique name is generated if not provided.
        :returns: The builder instance.
        """
        output_variable = output_variable or self._generate_output_variable(target)
        acquire_mode = acq_mode_process_axis[ProcessAxis.SEQUENCE]
        return self.measure(target, acquire_mode, output_variable)._take_sequence_mean(
            acquire_mode, target, output_variable
        )

    def measure_scope_mode(self, target: Qubit, output_variable: str = None):
        """Measure the qubit in scope (waveform-capture) mode.

        Emits a :class:`~qat.ir.measure.MeasureBlock` using ``SCOPE`` acquisition mode.
        No post-processing is applied; the raw time-series IQ waveform is stored
        directly in the output variable.

        :param target: The qubit to be measured.
        :param output_variable: Name of the variable where the acquire result is saved.
            A unique name is generated if not provided.
        :returns: The builder instance.
        """
        acquire_mode = acq_mode_process_axis[ProcessAxis.TIME]
        return self.measure(target, acquire_mode, output_variable)

    def measure_single_shot_binned(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ):
        """Measure a single qubit along the z-axis and discriminate to ±1 (customer-facing
        API).

        Emits a :class:`~qat.ir.measure.MeasureBlock`, an optional
        ``PostProcessing(MEAN, TIME)`` (SCOPE mode only),
        ``PostProcessing(LINEAR_MAP_COMPLEX_TO_REAL)`` to project complex IQ to a
        real z-value, and a ``PostProcessing(DISCRIMINATE)`` that rounds the
        z-projection value to +1 (above threshold) or −1 (at or below threshold).

        .. note::
            This method uses the legacy ``DISCRIMINATE`` post-processing type and does
            not emit the granular :class:`~qat.ir.measure.Discriminate` instruction
            used by the post-selection pipeline.  It is not equivalent to
            :meth:`measure_with_granular_post_processing` (which uses
            ``Discriminate``/``PostSelect``) and should not be substituted
            for it.

        :param target: The qubit to be measured.
        :param axis: The axis along which post-processing of readouts should occur.
        :param output_variable: Name of the variable where the acquire result is saved.
            A unique name is generated if not provided.
        :returns: The builder instance.
        """
        output_variable = output_variable or self._generate_output_variable(target)
        acquire_mode = acq_mode_process_axis[axis]
        self.measure(target, acq_mode_process_axis[axis], output_variable)
        self._take_time_mean(acquire_mode, target, output_variable)
        self.post_processing(
            target, output_variable, PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL
        )
        return self.post_processing(target, output_variable, PostProcessType.DISCRIMINATE)

    def reset(
        self, targets: PulseChannel | Qubit | Iterable[PulseChannel | Qubit], **kwargs
    ) -> QuantumInstructionBuilder:
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

    @staticmethod
    def _generate_output_variable(target: Component) -> str:
        return "out_" + target.uuid + f"_{np.random.randint(np.iinfo(np.int32).max)}"

    def _find_valid_measure_block(
        self, target_ids: QubitId | set[QubitId]
    ) -> MeasureBlock | None:
        """Finds the previous :class:`MeasureBlock` where the given qubits are not measured
        yet.

        :param target_ids: The indices of the qubits to be measured.
        """
        for instruction in reversed(self.instructions):
            if isinstance(instruction, MeasureBlock) and not any(
                target_ids & instruction.qubit_targets
            ):
                instruction.qubit_targets.update(target_ids)
                return instruction
        return None

    @staticmethod
    def build_legacy_equalise_args(
        mean_z_map_args: list,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute ``Equalise`` parameters and discrimination threshold from
        ``mean_z_map_args``.

        Given ``mean_z_map_args = [A, B]`` where the legacy linear map is
        ``z = Re(A * iq + B)``, the affine transform applied to the IQ vector
        ``[I, Q]`` is:

        .. code-block:: text

            [I', Q'] = [[a, -b], [b, a]] @ [I, Q] + [c_I, c_Q]

        where ``a + jb = A`` and ``c_I + jc_Q = B``.  This is identical to
        the transform used by the :class:`~qat.model.post_processing.LinearMapToRealMethod`
        branch of :meth:`emit_granular_post_processing`.

        The discrimination threshold is ``0.0`` — the midpoint on the real
        axis between the rotated state centroids — matching the convention of
        :class:`~qat.ir.measure.Discriminate` (``> 0 → "0"``, ``≤ 0 → "1"``).

        :param mean_z_map_args: Two-element list ``[A, B]`` as stored on the
            legacy qubit.
        :returns: ``(transform, offset, threshold)`` tuple.
        """
        A, B = complex(mean_z_map_args[0]), complex(mean_z_map_args[1])
        a, b = A.real, A.imag
        transform = np.array([[a, -b], [b, a]], dtype=float)
        offset = np.array([B.real, B.imag], dtype=float)
        return transform, offset, 0.0

    @staticmethod
    def build_equalise_discriminate_instrs(
        qubit: Qubit,
        output_variable: str,
    ) -> list[Equalise | Discriminate]:
        """Build the ``Equalise → Discriminate`` instruction pair for *any* qubit type.

        This is the single authoritative implementation of the three-way dispatch used
        by both :meth:`emit_granular_post_processing` and
        :class:`~qat.middleend.passes.transform.InsertPreSelectionMeasurement`.  Both
        pre-selection and circuit-measurement paths produce **identical** pairs for the
        same qubit, differing only in the ``PostSelect.additional_disallowed`` that
        follows.

        Dispatch:

        * :class:`~qat.model.post_processing.MaxLikelihoodMethod` — optional
          :class:`~qat.ir.measure.Equalise` (when ``transform`` and ``offset`` are set)
          + :class:`~qat.ir.measure.Discriminate` with the full ML method.
        * :class:`~qat.model.post_processing.LinearMapToRealMethod` or legacy
          ``mean_z_map_args`` — :class:`~qat.ir.measure.Equalise` (rotation/offset
          from :meth:`build_legacy_equalise_args`) + threshold
          :class:`~qat.ir.measure.Discriminate`.

        :param qubit: Qubit whose post-processing configuration drives dispatch.
        :param output_variable: Variable name to attach to each instruction.
        :returns: List of ``[Equalise, Discriminate]`` (or ``[Discriminate]`` for ML
            without a pre-rotation).
        """
        method = qubit.post_process_method
        instructions: list[Equalise | Discriminate] = []

        if isinstance(method, MaxLikelihoodMethod):
            if method.transform is not None and method.offset is not None:
                instructions.append(
                    Equalise(
                        output_variable=output_variable,
                        transform=np.asarray(method.transform, dtype=float),
                        offset=np.asarray(method.offset, dtype=float),
                    )
                )
            instructions.append(
                Discriminate(output_variable=output_variable, method=method)
            )
        else:
            # LinearMapToRealMethod or legacy mean_z_map_args (post_process_method=None).
            args = (
                method.mean_z_map_args
                if isinstance(method, LinearMapToRealMethod)
                else qubit.mean_z_map_args
            )
            transform, offset, threshold = (
                QuantumInstructionBuilder.build_legacy_equalise_args(args)
            )
            instructions.append(
                Equalise(
                    output_variable=output_variable, transform=transform, offset=offset
                )
            )
            instructions.append(
                Discriminate(output_variable=output_variable, threshold=threshold)
            )

        return instructions

    def emit_granular_post_processing(
        self, target: Qubit, output_variable: str
    ) -> QuantumInstructionBuilder:
        """Emit the granular post-processing instruction chain (without PostSelect).

        Emits ``Equalise`` → ``Discriminate`` based on the qubit's configuration.
        :class:`~qat.ir.measure.Discriminate` outputs **integer state keys** directly.
        Use :meth:`emit_post_select` separately to append a
        :class:`~qat.ir.measure.PostSelect` when disallowed states are configured.

        Dispatches as follows:

        * :class:`~qat.model.post_processing.LinearMapToRealMethod` →
          :class:`~qat.ir.measure.Equalise` + :class:`~qat.ir.measure.Discriminate`
          (threshold path; above-threshold → key ``0``, below → key ``1``).
        * :class:`~qat.model.post_processing.MaxLikelihoodMethod` →
          optional :class:`~qat.ir.measure.Equalise`
          + :class:`~qat.ir.measure.Discriminate` (ML path; emits dict keys directly).
        * Legacy (``post_process_method is None``, ``mean_z_map_args`` set) →
          :class:`~qat.ir.measure.Equalise` (rotation derived from
          ``mean_z_map_args``) + :class:`~qat.ir.measure.Discriminate`.

        **Integer key convention**

        ``Discriminate`` emits the integer dict key from
        :attr:`~qat.model.post_processing.MaxLikelihoodMethod.states` for ML paths,
        or ``0``/``1`` for threshold/legacy paths.  Non-negative keys are allowed
        states written to the classical register; negative keys are disallowed and
        subsequently filtered by :class:`~qat.ir.measure.PostSelect`.

        **Results Format Semantics with Compiler Config**

        Integer output values from ``Discriminate`` flow through the runtime pipeline
        where ``results_format`` flags determine final encoding:

        - **``raw()``**: Complex IQ arrays from
          :class:`~qat.runtime.passes.analysis.EqualiseResult` (post-mask applied).
          For legacy acquires without an ``Equalise`` step, falls back to the mapped
          arrays.

        - **``binary()``**: Per-shot int output-value array from ``Discriminate`` (one
          value per retained shot, e.g. ``[0, 1, 0, ...]``). With post-selection,
          only retained shots are included.

        - **``binary_count()``**: Dictionary of state string → count using
          :func:`~qat.runtime.results_processing.label_count` on the integer keys
          from :class:`~qat.runtime.passes.analysis.DiscriminateResult`.
          **Key difference**: when post-selection is active,
          the repeat count passed to ``binary_count()`` is ``shots_retained`` (not
          ``shots_requested``), so counts reflect only the passing shots.

        **Example: 3-state ML with state keyed as -2 (disallowed)**

        Setup::

            states = {
                0: MLDiscriminateParams(location=1+0j),
                1: MLDiscriminateParams(location=-1+0j),
                -2: MLDiscriminateParams(location=0+1j),  # negative key = disallowed
            }
            # 10 shots: 4 → key 0, 3 → key 1, 3 → key -2 (filtered by post-select)

        After ``Discriminate``::

            [0, 0, 0, 0, 1, 1, 1, -2, -2, -2]

        After ``PostSelect`` + global mask (7 retained)::

            [0, 0, 0, 0, 1, 1, 1]

        Format outcomes::

            raw():          <complex IQ from EqualiseResult, 7 retained shots>
            binary():       [0, 0, 0, 0, 1, 1, 1]
            binary_count(): {"0": 4, "1": 3}  (from 7 retained, not 10 requested)

        **Default Behavior**

        When ``CompilerConfig.results_format`` is not set, it defaults to
        ``ResultsFormatting.DynamicStructureReturn``, which simplifies output
        (removes generated variable names, unwraps single-value results).

        :param target: Qubit whose configuration drives the emitted instructions.
        :param output_variable: The variable name to attach to the instructions.
        :returns: The builder instance.
        """
        for instr in self.build_equalise_discriminate_instrs(target, output_variable):
            self.add(instr)
        return self

    def emit_post_select(
        self,
        output_variable: str,
    ) -> QuantumInstructionBuilder:
        """Emit a PostSelect instruction after the Discriminate for the given variable.

        :class:`~qat.ir.measure.PostSelect` filters shots whose integer state key is
        negative.  When all keys are non-negative (e.g. :class:`LinearMapToRealMethod`
        always produces ``0`` or ``1``) the instruction becomes a no-op at runtime
        (the validity mask is all ``True``).  Callers do not need to guard on whether
        the method has disallowed states.

        :param output_variable: The variable name to attach to the instruction.
        :returns: The builder instance.
        """
        self.add(PostSelect(output_variable=output_variable))
        return self

    def measure_with_granular_post_processing(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ) -> QuantumInstructionBuilder:
        """Measure a qubit and emit the full granular post-processing pipeline.

        This method is intended for use by compiler frontends (QASM2, QASM3, QIR,
        tket) where the result of a measurement assignment must be discriminated into
        state labels. Customer-facing code that only needs raw z-projection floats
        should call :meth:`measure_single_shot_z` instead.

        For qubits with a ``post_process_method`` configured the emitted sequence is:

        * :class:`~qat.ir.measure.MeasureBlock` (via :meth:`measure`)
        * Optional ``PostProcessing(MEAN, TIME)`` via :meth:`_take_time_mean`
          (only for SCOPE / TIME axis).
        * The granular chain from :meth:`emit_granular_post_processing`:
          :class:`~qat.ir.measure.Equalise` → :class:`~qat.ir.measure.Discriminate`.

        Frontends that need post-selection should append
        :meth:`emit_post_select` explicitly based on compiler configuration.

        For legacy qubits (``post_process_method is None``, ``mean_z_map_args`` set),
        :meth:`emit_granular_post_processing` derives the ``Equalise`` rotation from
        ``mean_z_map_args`` and emits the same ``Equalise`` → ``Discriminate``
        chain so that all measurement paths use one unified pipeline.

        :param target: The qubit to be measured.
        :param axis: The axis along which post-processing of readouts should occur.
        :param output_variable: The variable where the acquire result will be saved.
            A unique name is generated if not provided.
        :returns: The builder instance.
        """
        output_variable = output_variable or self._generate_output_variable(target)
        acquire_mode = acq_mode_process_axis[axis]
        self.measure(target, acquire_mode, output_variable)
        self._take_time_mean(acquire_mode, target, output_variable)
        self.emit_granular_post_processing(target, output_variable)
        return self

    def post_processing(
        self,
        target: Qubit,
        output_variable: str,
        process_type: PostProcessType,
        axes: ProcessAxis | list[ProcessAxis] | None = None,
        args=None,
    ) -> QuantumInstructionBuilder:
        """Emit a legacy :class:`~qat.ir.measure.PostProcessing` instruction.

        Populates ``args`` automatically when none are provided:

        * ``LINEAR_MAP_COMPLEX_TO_REAL`` — reads from ``target.mean_z_map_args``
          (legacy) or ``target.post_process_method.mean_z_map_args`` (new-style
          :class:`~qat.model.post_processing.LinearMapToRealMethod`).
        * ``DISCRIMINATE`` — reads from ``target.discriminator``.

        :param target: The qubit whose calibration data is used to fill ``args``.
        :param output_variable: The variable name to attach to the instruction.
        :param process_type: The type of post-processing to apply.
        :param axes: Axis or axes along which the post-processing operates.
        :param args: Explicit arguments for the post-processing; auto-filled when
            ``None`` for ``LINEAR_MAP_COMPLEX_TO_REAL`` and ``DISCRIMINATE`` types.
        :returns: The builder instance.
        """
        axes = axes if axes is not None else []
        args = args if args is not None else []

        # Default the mean z-map args if none supplied.
        if not any(args):
            if process_type == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL:
                # Prefer mean_z_map_args on the qubit directly; fall back to
                # LinearMapToRealMethod.mean_z_map_args when the new-style
                # post_process_method is configured instead.
                if target.mean_z_map_args is not None:
                    args = target.mean_z_map_args
                elif isinstance(target.post_process_method, LinearMapToRealMethod):
                    args = target.post_process_method.mean_z_map_args

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

    def cnot(self, control: Qubit, target: Qubit) -> QuantumInstructionBuilder:
        return (
            self.ECR(control, target)
            .X(control)
            .Z(control, theta=-np.pi / 2)
            .X(target, theta=-np.pi / 2)
        )

    def ECR(self, control: Qubit, target: Qubit) -> QuantumInstructionBuilder:
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

    def swap(self, target: Qubit, destination: Qubit) -> QuantumInstructionBuilder:
        raise NotImplementedError("Not available on this hardware model.")

    def pulse(self, **kwargs) -> QuantumInstructionBuilder:
        return self.add(Pulse(**kwargs))

    def acquire(
        self,
        target: Qubit,
        delay: float = 1e-06,
        output_variable: str | None = None,
        **kwargs,
    ) -> QuantumInstructionBuilder:
        pulse_channel = target.acquire_pulse_channel

        if delay is None:
            kwargs["delay"] = pulse_channel.acquire.delay

        output_variable = output_variable or self._generate_output_variable(target)

        return self.add(
            Acquire(target=pulse_channel.uuid, output_variable=output_variable, **kwargs)
        )

    def delay(
        self, target: Qubit | PulseChannel, duration: float
    ) -> QuantumInstructionBuilder:
        delays = []
        if isinstance(target, Qubit):
            delays = [
                Delay(targets=pulse_ch.uuid, duration=duration)
                for pulse_ch in target.all_qubit_and_resonator_pulse_channels
            ]
        else:
            delays = [Delay(targets=target.uuid, duration=duration)]
        return self.add(*delays)

    def synchronize(
        self, targets: Qubit | list[Qubit | PulseChannel]
    ) -> QuantumInstructionBuilder:
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

    def controlled(
        self, controllers: Qubit | list[Qubit], builder: InstructionBuilder
    ) -> QuantumInstructionBuilder:
        raise NotImplementedError("Not available on this hardware model.")

    def ccnot(self, controllers: list[Qubit], target: Qubit) -> QuantumInstructionBuilder:
        raise NotImplementedError("Not available on this hardware model.")

    @InstructionBuilder._check_identity_operation
    def phase_shift(self, target: PulseChannel, theta: float) -> QuantumInstructionBuilder:
        return self.add(PhaseShift(targets=target.uuid, phase=theta))

    def frequency_shift(
        self, target: PulseChannel, frequency: float
    ) -> QuantumInstructionBuilder:
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

    def _create_empty_builder(self) -> QuantumInstructionBuilder:
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
