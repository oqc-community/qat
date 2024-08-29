# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

from qat.purr.compiler.devices import (
    Calibratable,
    ChannelType,
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    QuantumDevice,
    Qubit,
    QubitCoupling,
    Resonator,
)
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CrossResonanceCancelPulse,
    CrossResonancePulse,
    DrivePulse,
    Instruction,
    PhaseShift,
    Synchronize,
)

if TYPE_CHECKING:
    from qat.purr.compiler.builders import InstructionBuilder, QuantumInstructionBuilder
    from qat.purr.compiler.execution import InstructionExecutionEngine

AnyEngine = TypeVar("AnyEngine")
AnyBuilder = TypeVar("AnyBuilder")


class HardwareModel:
    """
    Base class for all hardware models. Every model should return the builder class that
    should be used to build circuits/pulses for its particular back-end.
    """

    def __init__(self, shot_limit=-1):
        super().__init__()
        self.repeat_limit = shot_limit

    def create_engine(self) -> InstructionExecutionEngine: ...

    def create_runtime(self, existing_engine: InstructionExecutionEngine = None):
        if existing_engine is None:
            existing_engine = self.create_engine()

        from qat.purr.compiler.runtime import QuantumRuntime

        return QuantumRuntime(existing_engine)

    def create_builder(self) -> InstructionBuilder: ...

    def __repr__(self):
        return self.__class__.__name__


def get_cl2qu_index_mapping(instructions: List[Instruction], model: QuantumHardwareModel):
    """
    Returns a Dict[str, str] mapping creg to qreg indices.
    Classical register indices are extracted following the pattern <clreg_name>[<clreg_index>]
    """
    mapping = {}
    pattern = re.compile(r"(.*)\[([0-9]+)\]")

    for instruction in instructions:
        if not isinstance(instruction, Acquire):
            continue

        qubit = next(
            (
                qubit
                for qubit in model.qubits
                if qubit.get_acquire_channel().full_id() == instruction.channel.full_id()
            ),
            None,
        )
        if qubit is None:
            raise ValueError(
                f"Could not find any qubits by acquire channel {instruction.channel}"
            )

        result = pattern.match(instruction.output_variable)
        if result is None:
            raise ValueError(
                f"Could not extract cl register index from {instruction.output_variable}"
            )

        clbit_index = result.group(2)
        mapping[clbit_index] = qubit.index

    return mapping


class ReadoutMitigation(Calibratable):
    """
    Linear maps each individual qubit to its <0/1> given <0/1> probability.
    Note that linear assumes no cross-talk effects and considers each qubit independent.
    linear = {
        <qubit_number>: {
            "0|0": 1,
            "1|0": 1,
            "0|1": 1,
            "1|1": 1,
        }
    }
    Matrix is the entire 2**n x 2**n process matrix of p(<bitstring_1>|<bitstring_2>).
    M3 is a runtime mitigation strategy that builds up the calibration it needs at runtime,
    hence a bool of available or not. For more info https://github.com/Qiskit-Partners/mthree.
    """

    def __init__(
        self,
        linear: [Dict[str, Dict[str, float]]] = None,
        matrix: np.array = None,
        m3: bool = False,
    ):
        super().__init__()
        self.m3_available = m3
        self.linear: [Dict[str, Dict[str, float]]] = linear
        self.matrix: np.array = matrix


class ErrorMitigation(Calibratable):
    def __init__(self, readout_mitigation: Optional[ReadoutMitigation] = None):
        super().__init__()
        self.readout_mitigation = readout_mitigation

    def __setstate__(self, state):
        readout_mitigation = state.pop("readout_mitigation", None)
        if readout_mitigation is None or isinstance(readout_mitigation, ReadoutMitigation):
            self.readout_mitigation = readout_mitigation
        else:
            raise ValueError(
                f"Expected {ReadoutMitigation.__class__} but got {type(readout_mitigation)}"
            )

        self.__dict__.update(state)


class QuantumHardwareModel(HardwareModel, Calibratable):
    """
    Object modelling our superconducting hardware. Holds up-to-date information about a
    current piece of hardware, whether simulated or physical machine.
    """

    def __init__(
        self,
        shot_limit=10000,
        acquire_mode=None,
        repeat_count=1000,
        repetition_period=100e-6,
        error_mitigation: Optional[ErrorMitigation] = None,
    ):
        # Our hardware has a default shot limit of 10,000 right now.
        self.default_acquire_mode = acquire_mode or AcquireMode.RAW
        self.default_repeat_count = repeat_count
        self.default_repetition_period = repetition_period

        self.quantum_devices: Dict[str, QuantumDevice] = {}
        self.pulse_channels: Dict[str, PulseChannel] = {}
        self.physical_channels: Dict[str, PhysicalChannel] = {}
        self.basebands: Dict[str, PhysicalBaseband] = {}
        self.qubit_direction_couplings: List[QubitCoupling] = []
        self.error_mitigation: Optional[ErrorMitigation] = error_mitigation

        # Construct last due to us overriding calibratables fields with properties.
        super().__init__(
            shot_limit=shot_limit,
        )

    def create_engine(self) -> InstructionExecutionEngine:
        from qat.purr.backends.echo import EchoEngine

        return EchoEngine(self)

    def create_builder(self) -> "QuantumInstructionBuilder":
        from qat.purr.compiler.builders import QuantumInstructionBuilder

        return QuantumInstructionBuilder(self)

    def resolve_qb_pulse_channel(
        self, chanbit: Union[Qubit, PulseChannel]
    ) -> Tuple[Qubit, PulseChannel]:
        if isinstance(chanbit, Qubit):
            return chanbit, chanbit.get_default_pulse_channel()
        else:
            for qubit in self.qubits:
                for channel in qubit.pulse_channels.values():
                    if chanbit == channel:
                        return qubit, chanbit
        raise ValueError(f"Cannot resolve {chanbit}")

    @property
    def qubits(self):
        """Returns list of the qubits on this hardware sorted by index."""
        qubits = [qb for qb in self.quantum_devices.values() if isinstance(qb, Qubit)]
        qubits = sorted(qubits, key=lambda x: x.index)
        return qubits

    @property
    def resonators(self):
        return [qd for qd in self.quantum_devices.values() if isinstance(qd, Resonator)]

    def has_qubit(self, id_: Union[int, str]):
        if isinstance(id_, int):
            id_ = f"Q{id_}"
        return id_ in self.quantum_devices

    def get_qubit(self, id_: Union[int, str, Qubit]) -> Qubit:
        """
        Returns a qubit based on id/index. If the passed-in object is already a Qubit
        object just returns that.
        """
        if isinstance(id_, Qubit):
            return id_

        if isinstance(id_, int):
            id_ = f"Q{id_}"

        if id_ not in self.quantum_devices:
            raise ValueError(f"Tried to retrieve a qubit ({str(id_)}) that doesn't exist.")

        found_qubit = self.quantum_devices.get(id_)
        if not isinstance(found_qubit, Qubit):
            raise ValueError(f"Device with id {str(id_)} isn't a qubit.")

        return found_qubit

    def add_quantum_device(self, *devices: QuantumDevice):
        for device in devices:
            existing_dev = self.quantum_devices.get(device.full_id(), None)
            if existing_dev is not None:
                # If we're the same instance just don't throw.
                if existing_dev is device:
                    continue

                raise KeyError(f"Quantum Device with id '{device.id}' already exists.")

            self.quantum_devices[device.full_id()] = device

            # add device pulse channels to hardware if not already present
            for pulse_channel in device.pulse_channels.values():
                self.add_pulse_channel(pulse_channel)

    def get_quantum_device(self, id_: str):
        return self.quantum_devices.get(id_, None)

    def add_pulse_channel(self, *pulse_channels: PulseChannel):
        for pulse_channel in pulse_channels:
            existing_channel = self.pulse_channels.get(pulse_channel.full_id(), None)
            if existing_channel is not None:
                # If we're the same instance just don't throw.
                if existing_channel is pulse_channel:
                    continue

                raise KeyError(
                    f"Pulse channel with id '{pulse_channel.full_id()}' already exists."
                )

            self.pulse_channels[pulse_channel.full_id()] = pulse_channel

        return pulse_channels

    def get_pulse_channel_from_device(
        self,
        ch_type: ChannelType,
        host_device_id: str,
        aux_device_ids: List[str] = None,
    ):
        if aux_device_ids is None:
            aux_device_ids = []
        return self.get_quantum_device(host_device_id).get_pulse_channel(
            ch_type, [self.get_quantum_device(device) for device in aux_device_ids]
        )

    def get_pulse_channels_from_physical_channel(self, physical_channel_id: str):
        # WARNING: this will not get pulse channels created during execution, only ones
        # predefined on the hardware.
        pulse_channels = []
        for pulse_channel in self.pulse_channels.values():
            if pulse_channel.physical_channel_id == physical_channel_id:
                pulse_channels.append(pulse_channel)

        return pulse_channels

    def get_devices_from_pulse_channel(self, id_: str):
        pulse_channel = self.get_pulse_channel_from_id(id_)
        devices = [
            device
            for device in self.quantum_devices.values()
            if pulse_channel in device.pulse_channels.values()
        ]
        return devices

    def get_pulse_channel_from_id(self, id_: str):
        return self.pulse_channels.get(id_)

    def get_devices_from_physical_channel(self, id_: str):
        physical_channel = self.get_physical_channel(id_)
        devices = [
            device
            for device in self.quantum_devices.values()
            if physical_channel == device.physical_channel
        ]
        return devices

    def add_physical_channel(self, *physical_channels: PhysicalChannel):
        for physical_channel in physical_channels:
            existing_channel = self.physical_channels.get(physical_channel.full_id(), None)
            if existing_channel is not None:
                # If we're the same instance just don't throw.
                if existing_channel is physical_channel:
                    continue

                raise KeyError(
                    f"Physical channel with id '{physical_channel.full_id()}' already exists."
                )

            self.physical_channels[physical_channel.full_id()] = physical_channel

    def get_physical_channel(self, id_: str):
        return self.physical_channels.get(id_, None)

    def add_physical_baseband(self, *basebands: PhysicalBaseband):
        for baseband in basebands:
            existing_baseband = self.basebands.get(baseband.full_id(), None)
            if existing_baseband is not None:
                # If we're the same instance just don't throw.
                if existing_baseband is baseband:
                    continue

                raise KeyError(f"Baseband with id '{baseband.full_id()}' already exists.")

            self.basebands[baseband.full_id()] = baseband

    def get_physical_baseband(self, id_: str):
        return self.basebands.get(id_, None)

    def get_device(self, id_):
        if (device := self.get_quantum_device(id_)) is not None:
            return device

        if (device := self.get_physical_channel(id_)) is not None:
            return device

        if (device := self.get_pulse_channel_from_id(id_)) is not None:
            return device

        return self.get_physical_baseband(id_)

    def add_device(self, device):
        if isinstance(device, PhysicalBaseband):
            self.add_physical_baseband(device)
        elif isinstance(device, PhysicalChannel):
            self.add_physical_channel(device)
        else:
            self.add_quantum_device(device)

    def _resolve_qb_pulse_channel(self, chanbit: Union[Qubit, PulseChannel]):
        if isinstance(chanbit, Qubit):
            return chanbit, chanbit.get_default_pulse_channel()
        else:
            quantum_devices = self.get_devices_from_pulse_channel(chanbit.full_id())
            primary_devices = [
                device
                for device in quantum_devices
                if device.get_default_pulse_channel() == chanbit
            ]
            if len(primary_devices) > 1:
                raise NotImplementedError(
                    f"Too many devices use the channel with id {chanbit.full_id()} as "
                    "a default channel to resolve a primary device."
                )
            elif len(primary_devices) == 0:
                return None, chanbit
            else:
                return primary_devices[0], chanbit

    @property
    def is_calibrated(self):
        def check_devices(target_devices):
            return all(
                [
                    device.is_calibrated
                    for device in target_devices.values()
                    if isinstance(device, Calibratable)
                ]
            )

        return (
            check_devices(self.quantum_devices)
            and check_devices(self.physical_channels)
            and check_devices(self.basebands)
        )

    @is_calibrated.setter
    def is_calibrated(self, val):
        def set_devices(target_dict):
            for device in target_dict.values():
                if isinstance(device, Calibratable):
                    device.is_calibrated = val

        set_devices(self.quantum_devices)
        set_devices(self.physical_channels)
        set_devices(self.basebands)

    def constrain(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_hw_x_pi_2(
        self, qubit, pulse_channel: PulseChannel = None, amp_scale: float = None
    ) -> List[Any]:
        if amp_scale is None or np.isclose(amp_scale, 1.0):
            pulse_args = qubit.pulse_hw_x_pi_2
        else:
            pulse_args = deepcopy(qubit.pulse_hw_x_pi_2)
            pulse_args["amp"] *= amp_scale

        # note: this could be a more complicated set of instructions
        return [
            DrivePulse(pulse_channel or qubit.get_default_pulse_channel(), **pulse_args)
        ]

    def get_hw_z(self, qubit, phase, pulse_channel: PulseChannel = None) -> List[Any]:
        if phase == 0:
            return []

        instr_collection = [
            PhaseShift(pulse_channel or qubit.get_default_pulse_channel(), phase)
        ]
        if pulse_channel is None or pulse_channel == qubit.get_default_pulse_channel():
            instr_collection.extend(
                PhaseShift(coupled_qubit.get_cross_resonance_channel(qubit), phase)
                for coupled_qubit in qubit.coupled_qubits
            )
            # TODO: the cross-resonance cancellation channel mirrors the cr channel,
            #   they should be tied together
            instr_collection.extend(
                PhaseShift(
                    qubit.get_cross_resonance_cancellation_channel(coupled_qubit), phase
                )
                for coupled_qubit in qubit.coupled_qubits
            )

        return instr_collection

    def get_hw_zx_pi_4(self, qubit: Qubit, target_qubit: Qubit) -> List[Any]:
        control_channel = qubit.get_cross_resonance_channel(target_qubit)
        target_channel = target_qubit.get_cross_resonance_cancellation_channel(qubit)
        pulse = qubit.pulse_hw_zx_pi_4.get(target_qubit.id, None)
        if pulse is None:
            raise ValueError(
                f"Tried to perform cross resonance on {str(target_qubit)} "
                f"that isn't linked to {str(qubit)}."
            )

        return [
            Synchronize([control_channel, target_channel]),
            CrossResonancePulse(control_channel, **pulse),
            CrossResonanceCancelPulse(target_channel, **pulse),
        ]

    def get_gate_U(self, qubit, theta, phi, lamb, pulse_channel: PulseChannel = None):
        theta = self.constrain(theta)
        return [
            *self.get_hw_z(qubit, lamb + np.pi, pulse_channel),
            *self.get_hw_x_pi_2(qubit, pulse_channel),
            *self.get_hw_z(qubit, np.pi - theta, pulse_channel),
            *self.get_hw_x_pi_2(qubit, pulse_channel),
            *self.get_hw_z(qubit, phi, pulse_channel),
        ]

    def get_gate_X(self, qubit, theta, pulse_channel: PulseChannel = None):
        theta = self.constrain(theta)
        if np.isclose(theta, 0.0):
            return []
        elif np.isclose(theta, np.pi / 2.0):
            return self.get_hw_x_pi_2(qubit, pulse_channel)
        elif np.isclose(theta, -np.pi / 2.0):
            return [
                *self.get_hw_z(qubit, np.pi, pulse_channel),
                *self.get_hw_x_pi_2(qubit, pulse_channel),
                *self.get_hw_z(qubit, -np.pi, pulse_channel),
            ]
        return self.get_gate_U(qubit, theta, -np.pi / 2.0, np.pi / 2.0, pulse_channel)

    def get_gate_Y(self, qubit, theta, pulse_channel: PulseChannel = None):
        theta = self.constrain(theta)
        if np.isclose(theta, 0.0):
            return []
        elif np.isclose(theta, np.pi / 2.0):
            return [
                *self.get_hw_z(qubit, -np.pi / 2.0, pulse_channel),
                *self.get_hw_x_pi_2(qubit, pulse_channel),
                *self.get_hw_z(qubit, np.pi / 2.0, pulse_channel),
            ]
        elif np.isclose(theta, -np.pi / 2.0):
            return [
                *self.get_hw_z(qubit, np.pi / 2.0, pulse_channel),
                *self.get_hw_x_pi_2(qubit, pulse_channel),
                *self.get_hw_z(qubit, -np.pi / 2.0, pulse_channel),
            ]
        return self.get_gate_U(qubit, theta, 0.0, 0.0, pulse_channel)

    def get_gate_Z(self, qubit, theta, pulse_channel: PulseChannel = None):
        theta = self.constrain(theta)
        return self.get_hw_z(qubit, theta, pulse_channel)

    def get_gate_ZX(self, qubit, theta, target_qubit):
        theta = self.constrain(theta)
        cr_pulse_channel = qubit.get_pulse_channel(
            ChannelType.cross_resonance, [target_qubit]
        )
        cr_cancellation_pulse_channel = target_qubit.get_pulse_channel(
            ChannelType.cross_resonance_cancellation, [qubit]
        )
        if np.isclose(theta, 0.0):
            return []
        elif np.isclose(theta, np.pi / 4.0):
            return self.get_hw_zx_pi_4(qubit, target_qubit)
        elif np.isclose(theta, -np.pi / 4.0):
            return [
                PhaseShift(cr_pulse_channel, np.pi),
                PhaseShift(cr_cancellation_pulse_channel, np.pi),
                *self.get_hw_zx_pi_4(qubit, target_qubit),
                PhaseShift(cr_pulse_channel, np.pi),
                PhaseShift(cr_cancellation_pulse_channel, np.pi),
            ]
        else:
            raise ValueError("Generic ZX gate not implemented yet!")

    def __setstate__(self, state):
        error_mitigation = state.pop("error_mitigation", None)
        if error_mitigation is None or isinstance(error_mitigation, ErrorMitigation):
            self.error_mitigation = error_mitigation
        else:
            raise ValueError(
                f"Expected {ErrorMitigation.__class__} but got {type(error_mitigation)}"
            )

        self.__dict__.update(state)
