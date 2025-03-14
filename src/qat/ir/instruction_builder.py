# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional, Union

import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.ir.instructions import (
    Assign,
    Delay,
    FrequencyShift,
    Instruction,
    InstructionBlock,
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
from qat.ir.waveforms import Pulse, PulseType, SampledWaveform
from qat.model.device import DrivePulseChannel, PulseChannel, Qubit
from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.utils.logger import get_default_logger
from qat.utils.pydantic import QubitId, ValidatedList

log = get_default_logger()


class InstructionBuilder(ABC):
    def __init__(
        self,
        hardware_model: PhysicalHardwareModel,
        instructions: list[Instruction] = [],
    ):
        self.hw = hardware_model
        self._qubit_index_by_uuid = {
            qubit.uuid: idx for (idx, qubit) in hardware_model.qubits.items()
        }
        self._ir = InstructionBlock(instructions=instructions)

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
    def controlled(
        self, controllers: Union[Qubit, list[Qubit]], builder: InstructionBuilder
    ): ...

    def cX(self, controllers: Union[Qubit, list[Qubit]], target: Qubit, theta=np.pi):
        return self.controlled(controllers, self.X(target, theta=theta))

    def cY(self, controllers: Union[Qubit, list[Qubit]], target: Qubit, theta=np.pi):
        return self.controlled(controllers, self.Y(target, theta=theta))

    def cZ(self, controllers: Union[Qubit, list[Qubit]], target: Qubit, theta=np.pi):
        return self.controlled(controllers, self.Z(target, theta=theta))

    def cnot(self, control: Union[Qubit, list[Qubit]], target: Qubit):
        return self.cX(control, target, theta=np.pi)

    @abstractmethod
    def ccnot(self, controllers: list[Qubit], target: Qubit): ...

    def cswap(
        self, controllers: Union[Qubit, list[Qubit]], target: Qubit, destination: Qubit
    ):
        return self.controlled(controllers, self.swap(target, destination))

    @abstractmethod
    def ECR(self, control: Qubit, target: Qubit): ...

    def repeat(self, repeat_count: int, repetition_period: float = None):
        return self.add(
            Repeat(repeat_count=repeat_count, repetition_period=repetition_period)
        )

    def returns(self, variables: list[str] = None):
        """Add return statement."""
        variables = variables if variables is not None else []
        return self.add(Return(variables=variables))

    def reset(
        self, targets: Union[PulseChannel, Qubit] | Iterable[Union[PulseChannel, Qubit]]
    ):
        if isinstance(targets, list):
            targets = set(targets)
        elif isinstance(targets, PulseChannel | Qubit):
            targets = {targets}
        else:
            raise TypeError(
                f"Invalid type, expected '(PulseChannel | Qubit | List[PulseChannel | Qubit])' but got {type(targets)}."
            )

        qubit_ids = []
        pulse_channel_ids = []
        for target in targets:
            if isinstance(target, Qubit):
                qubit_ids.append(self._qubit_index_by_uuid[target.uuid])
                pulse_channel_ids.extend([pc.uuid for pc in target.all_pulse_channels])
            else:
                pulse_channel_ids.append(target.uuid)

        return self.add(
            Reset(qubit_targets=qubit_ids), PhaseReset(targets=pulse_channel_ids)
        )

    def assign(self, name: str, value):
        return self.add(Assign(name=name, value=value))

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
        if isinstance(other, InstructionBuilder):
            self.add(*other.instructions)
        else:
            raise TypeError(
                "Only another `{self.__class__.__name__}` can be added to this builder."
            )
        return self

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

        def wrapper(self, target, theta=np.pi, **kwargs):
            theta = self.constrain(theta)
            return self if np.isclose(theta, 0) else f(self, target, theta, **kwargs)

        return wrapper

    @property
    def instructions(self):
        return self._ir.instructions

    @instructions.setter
    def instructions(self, instructons: ValidatedList[Instruction]):
        self._ir.instructions = instructons

    @property
    def number_of_instructions(self):
        return self._ir.number_of_instructions

    def __iter__(self):
        return self._ir.__iter__()


class QuantumInstructionBuilder(InstructionBuilder):
    @InstructionBuilder._check_identity_operation
    def X(self, target: Qubit, theta: float = np.pi, pulse_channel: PulseChannel = None):
        """
        Adds a gate that drives the qubit with a rotation angle `theta` to the builder.

        :param target: The qubit to be rotated.
        :param theta: The applied rotation angle.
        :param pulse_channel: The pulse channel the pulses get sent to.
        """
        if np.isclose(theta, np.pi / 2.0):
            return self.add(*self._hw_X_pi_2(target, pulse_channel))
        elif np.isclose(theta, -np.pi / 2.0):
            return self.add(
                *self._hw_Z(target, theta=np.pi, pulse_channel=pulse_channel),
                *self._hw_X_pi_2(target, pulse_channel=pulse_channel),
                *self._hw_Z(target, theta=np.pi, pulse_channel=pulse_channel),
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
        if np.isclose(theta, np.pi / 2.0):
            return self.add(
                *self._hw_Z(target, theta=-np.pi / 2.0, pulse_channel=pulse_channel),
                *self._hw_X_pi_2(target, pulse_channel=pulse_channel),
                *self._hw_Z(target, theta=np.pi / 2.0, pulse_channel=pulse_channel),
            )
        elif np.isclose(theta, -np.pi / 2.0):
            return self.add(
                *self._hw_Z(target, theta=np.pi / 2.0, pulse_channel=pulse_channel),
                *self._hw_X_pi_2(target, pulse_channel=pulse_channel),
                *self._hw_Z(target, theta=-np.pi / 2.0, pulse_channel=pulse_channel),
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

    def _hw_X_pi_2(self, target: Qubit, pulse_channel: DrivePulseChannel = None):
        pulse_channel = pulse_channel or target.drive_pulse_channel
        pulse_waveform = pulse_channel.pulse.waveform_type(
            **pulse_channel.pulse.model_dump()
        )

        return [
            Pulse(targets=pulse_channel.uuid, waveform=pulse_waveform, type=PulseType.DRIVE)
        ]

    def _hw_Z(
        self, target: Qubit, theta: float = np.pi, pulse_channel: PulseChannel = None
    ):
        if theta == 0:
            return []

        # Rotate drive pulse channel of the qubit.
        pulse_channel = pulse_channel or target.drive_pulse_channel
        instr_collection = [PhaseShift(targets=pulse_channel.uuid, phase=theta)]
        # Rotate all cross resonance (cancellation) pulse channels pertaining to the qubit.
        qubit_id = self._qubit_index_by_uuid[target.uuid]
        if isinstance(pulse_channel, DrivePulseChannel):
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

        return [
            Synchronize(targets=[target1_pulse_channel.uuid, target2_pulse_channel.uuid]),
            Pulse(
                targets=target1_pulse_channel.uuid,
                type=PulseType.CROSS_RESONANCE,
                waveform=waveform_type(**pulse),
            ),
            Pulse(
                targets=target2_pulse_channel.uuid,
                type=PulseType.CROSS_RESONANCE_CANCEL,
                waveform=waveform_type(**pulse),
            ),
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
            type=PulseType.MEASURE,
        )

        # Acquire-related info.
        acquire_channel = target.resonator.acquire_pulse_channel
        acquire_duration = (
            measure_channel.pulse.width
            if acquire_channel.acquire.sync
            else acquire_channel.acquire.width
        )

        if acquire_channel.acquire.use_weights is False:
            filter = Pulse(
                waveform=SampledWaveform(samples=acquire_channel.acquire.weights),
                duration=acquire_duration,
            )
        else:
            filter = None

        acquire_instruction = Acquire(
            targets=acquire_channel.uuid,
            duration=acquire_duration,
            mode=mode,
            delay=acquire_channel.acquire.delay,
            filter=filter,
            output_variable=output_variable,
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
        target_ids = MeasureBlock.validate_targets(
            {"qubit_targets": target_ids}, field_name="qubit_targets"
        )["qubit_targets"]

        measure_block = MeasureBlock(qubit_targets=target_ids)

        all_pulse_channels = [
            pc.uuid for qubit in targets for pc in qubit.all_pulse_channels
        ]

        if sync_qubits:
            # Sync pulse channels of all pulse channels.
            measure_block.add(Synchronize(targets=all_pulse_channels))

        for qubit in targets:
            if not sync_qubits:
                qubit_pulse_channels = [pc.uuid for pc in qubit.all_pulse_channels]
                measure_block.add(Synchronize(targets=qubit_pulse_channels))

            # Measure and acquire instructions for the qubit.
            measure, acquire = self._generate_measure_acquire(
                qubit, mode=mode, output_variable=output_variable
            )
            duration = max(measure.duration, acquire.delay + acquire.duration)
            measure_block.add(measure, acquire)
            measure_block.duration = max(measure_block.duration, duration)

            if not sync_qubits:
                qubit_pulse_channels = [pc.uuid for pc in qubit.all_pulse_channels]
                measure_block.add(Synchronize(targets=qubit_pulse_channels))

        # Sync pulse channels of all pulse channels after measurement.
        if sync_qubits:
            measure_block.add(Synchronize(targets=all_pulse_channels))

        return self.add(measure_block)

    def measure_single_shot_z(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ):
        """
        Measure a single qubit along the z-axis.

        :param target: The qubit to be measured.
        :param axis: The type of axis which the post-processing of readouts should occur on.
        :param output_variable:
        """
        return (
            self.measure(target, acq_mode_process_axis[axis], output_variable)
            .post_processing(
                target, output_variable, PostProcessType.DOWN_CONVERT, [ProcessAxis.TIME]
            )
            .post_processing(
                target, output_variable, PostProcessType.MEAN, [ProcessAxis.TIME]
            )
            .post_processing(
                target, output_variable, PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL
            )
        )

    def measure_single_shot_signal(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ):
        return (
            self.measure(target, acq_mode_process_axis[axis], output_variable)
            .post_processing(
                target, output_variable, PostProcessType.DOWN_CONVERT, [ProcessAxis.TIME]
            )
            .post_processing(
                target, output_variable, PostProcessType.MEAN, [ProcessAxis.TIME]
            )
        )

    def measure_mean_z(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ):
        return (
            self.measure(target, acq_mode_process_axis[axis], output_variable)
            .post_processing(
                target, output_variable, PostProcessType.DOWN_CONVERT, [ProcessAxis.TIME]
            )
            .post_processing(
                target, output_variable, PostProcessType.MEAN, [ProcessAxis.TIME]
            )
            .post_processing(
                target, output_variable, PostProcessType.MEAN, [ProcessAxis.SEQUENCE]
            )
            .post_processing(
                target, output_variable, PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL
            )
        )

    def measure_mean_signal(self, target: Qubit, output_variable: str = None):
        return (
            self.measure(
                target, acq_mode_process_axis[ProcessAxis.SEQUENCE], output_variable
            )
            .post_processing(
                target, output_variable, PostProcessType.DOWN_CONVERT, [ProcessAxis.TIME]
            )
            .post_processing(
                target, output_variable, PostProcessType.MEAN, [ProcessAxis.TIME]
            )
            .post_processing(
                target, output_variable, PostProcessType.MEAN, [ProcessAxis.SEQUENCE]
            )
        )

    def measure_scope_mode(self, target: Qubit, output_variable: str = None):
        return (
            self.measure(target, acq_mode_process_axis[ProcessAxis.TIME], output_variable)
            .post_processing(
                target, output_variable, PostProcessType.DOWN_CONVERT, [ProcessAxis.TIME]
            )
            .post_processing(
                target, output_variable, PostProcessType.MEAN, [ProcessAxis.SEQUENCE]
            )
        )

    def measure_single_shot_binned(
        self,
        target: Qubit,
        axis: ProcessAxis = ProcessAxis.SEQUENCE,
        output_variable: str = None,
    ):
        return (
            self.measure(target, acq_mode_process_axis[axis], output_variable)
            .post_processing(
                target, output_variable, PostProcessType.DOWN_CONVERT, [ProcessAxis.TIME]
            )
            .post_processing(
                target, output_variable, PostProcessType.MEAN, [ProcessAxis.TIME]
            )
            .post_processing(
                target, output_variable, PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL
            )
            .post_processing(target, output_variable, PostProcessType.DISCRIMINATE)
        )

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
        axes: Optional[Union[ProcessAxis, list[ProcessAxis]]] = None,
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

            elif process_type == PostProcessType.DOWN_CONVERT:
                resonator = target.resonator
                phys_channel = resonator.physical_channel
                bb = phys_channel.baseband
                measure_pulse_ch = resonator.measure_pulse_channel

                if measure_pulse_ch.fixed_if:
                    args = [bb.if_frequency, phys_channel.sample_time]
                else:
                    args = [
                        measure_pulse_ch.frequency - bb.frequency,
                        phys_channel.sample_time,
                    ]

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

        pulse_channels = [
            control.drive_pulse_channel.uuid,
            control.cross_resonance_pulse_channels[target_id].uuid,
            control.cross_resonance_cancellation_pulse_channels[target_id].uuid,
            target.drive_pulse_channel.uuid,
        ]
        sync_instruction = Synchronize(targets=pulse_channels)

        return (
            self.add(sync_instruction)
            .ZX(control, target, theta=np.pi / 4.0)
            .add(sync_instruction)
            .X(control, theta=np.pi)
            .add(sync_instruction)
            .ZX(control, target, theta=-np.pi / 4.0)
            .add(sync_instruction)
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

        return self.add(Acquire(targets=pulse_channel.uuid, **kwargs))

    def delay(self, target: Qubit | PulseChannel, duration: float):
        pulse_channel = target.drive_pulse_channel if isinstance(target, Qubit) else target
        return self.add(Delay(targets=pulse_channel.uuid, duration=duration))

    def synchronize(self, targets: Qubit | PulseChannel | list[Qubit | PulseChannel]):
        targets = targets if isinstance(targets, list) else [targets]

        pulse_channel_ids = set()
        for target in targets:
            if isinstance(target, PulseChannel):
                pulse_channel_ids.add(target.uuid)
            elif isinstance(target, Qubit):
                # TODO: At some point, we might want to implement (pulse channel)
                # getters at the qubit level just for the sake of conveniency. #326
                pulse_channel_ids.add(target.acquire_pulse_channel.uuid)
                pulse_channel_ids.add(target.measure_pulse_channel.uuid)
                qubit_pulse_channel_ids = [
                    pulse_channel.uuid for pulse_channel in target.all_pulse_channels
                ]
                pulse_channel_ids.update(qubit_pulse_channel_ids)
            else:
                raise TypeError(
                    "Please provide :class:`Qubit`s or :class:`PulseChannel`s as targets."
                )

        return self.add(Synchronize(targets=pulse_channel_ids))

    def controlled(
        self, controllers: Union[Qubit, list[Qubit]], builder: InstructionBuilder
    ):
        raise NotImplementedError("Not available on this hardware model.")

    def ccnot(self, controllers: list[Qubit], target: Qubit):
        raise NotImplementedError("Not available on this hardware model.")

    @InstructionBuilder._check_identity_operation
    def phase_shift(self, target: Qubit | PulseChannel, theta: float):
        if isinstance(target, Qubit):
            return self.add(
                PhaseShift(targets=target.drive_pulse_channel.uuid, phase=theta)
            )
        elif isinstance(target, PulseChannel):
            return self.add(PhaseShift(targets=target.uuid, phase=theta))
        else:
            raise TypeError(
                "Please provide a target that is either a `Qubit` or a `PulseChannel`."
            )

    @InstructionBuilder._check_identity_operation
    def frequency_shift(self, target: Qubit, frequency):
        if isinstance(target, Qubit):
            return self.add(
                FrequencyShift(targets=target.drive_pulse_channel.uuid, frequency=frequency)
            )
        elif isinstance(target, PulseChannel):
            return self.add(FrequencyShift(targets=target.uuid, frequency=frequency))
        else:
            raise TypeError(
                "Please provide a target that is either a `Qubit` or a `PulseChannel`."
            )
