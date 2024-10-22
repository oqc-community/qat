# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from typing import Dict, List, Optional

import numpy as np

from qat.purr.backends.live_devices import (
    ControlHardware,
    ControlHardwareChannel,
    Instrument,
    LivePhysicalBaseband,
)
from qat.purr.backends.utilities import UPCONVERT_SIGN, get_axis_map
from qat.purr.compiler.devices import (
    ChannelType,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    QubitCoupling,
    Resonator,
)
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import QuantumExecutionEngine, SweepIterator
from qat.purr.compiler.hardware_models import ErrorMitigation, QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    Instruction,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
)
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration

log = get_default_logger()


def build_lucy_hardware(hw: QuantumHardwareModel):
    return apply_setup_to_hardware(hw, 8, 25e-9, 3.18e-07, 2.41e-06)


def apply_setup_to_hardware(
    hw: QuantumHardwareModel,
    qubit_count: int,
    pulse_hw_x_pi_2_width: float,
    pulse_hw_zx_pi_4_width: float,
    pulse_measure_width: float,
) -> QuantumHardwareModel:
    """Apply lucy to the passed-in hardware."""
    qubit_devices = []
    resonator_devices = []
    channel_index = 1

    control_hardware = ControlHardware()

    for primary_index in range(qubit_count):
        bb1 = LivePhysicalBaseband(
            f"LO{channel_index}", 5.5e9, if_frequency=250e6, instrument=control_hardware
        )
        bb2 = LivePhysicalBaseband(
            f"LO{channel_index + 1}", 8.5e9, if_frequency=250e6, instrument=control_hardware
        )
        hw.add_physical_baseband(bb1, bb2)

        ch1 = PhysicalChannel(f"CH{channel_index}", 1.0e-9, bb1, 1)
        ch2 = PhysicalChannel(
            f"CH{channel_index + 1}", 1.0e-9, bb2, 1, acquire_allowed=True
        )
        hw.add_physical_channel(ch1, ch2)

        resonator = Resonator(f"R{primary_index}", ch2)
        resonator.create_pulse_channel(ChannelType.measure, frequency=8.5e9)
        resonator.create_pulse_channel(ChannelType.acquire, frequency=8.5e9)

        # As our main system is a ring architecture we just attach every qubit in the
        # ring to the one on either side.
        # 2 has a connection to 1 and 3. This number wraps around, so we also have
        # 10-0-1 linkages.
        qubit = Qubit(primary_index, resonator, ch1)
        qubit.create_pulse_channel(ChannelType.drive, frequency=5.5e9)

        qubit.pulse_hw_x_pi_2.update({"width": pulse_hw_x_pi_2_width})
        qubit.pulse_hw_zx_pi_4.update({"width": pulse_hw_zx_pi_4_width})
        qubit.pulse_measure.update({"width": pulse_measure_width})

        qubit_devices.append(qubit)
        resonator_devices.append(resonator)
        channel_index = channel_index + 2

    # TODO: For backwards compatability cross resonance pulse channels are fully
    #   connected but coupled qubits are only in a ring architecture. I think it would be
    #   more appropriate for cross resonace channels to also be a ring architecture but
    #   that can be done in a later PR.
    for i, qubit in enumerate(qubit_devices):
        for other_qubit in qubit_devices:
            if qubit != other_qubit:
                qubit.create_pulse_channel(
                    auxiliary_devices=[other_qubit],
                    channel_type=ChannelType.cross_resonance,
                    frequency=5.5e9,
                    scale=50,
                )
                qubit.create_pulse_channel(
                    auxiliary_devices=[other_qubit],
                    channel_type=ChannelType.cross_resonance_cancellation,
                    frequency=5.5e9,
                    scale=0.0,
                )
            qubit.add_coupled_qubit(qubit_devices[(i + 1) % qubit_count])
            qubit.add_coupled_qubit(qubit_devices[(i - 1) % qubit_count])

    hw.add_quantum_device(*qubit_devices, *resonator_devices)
    hw.is_calibrated = True
    return hw


def sync_baseband_frequencies_to_value(hw: QuantumHardwareModel, lo_freq, target_qubits):
    # Round the drive channel frequencies to the multiple of 1kHz
    for qubit in target_qubits:
        hw.get_qubit(qubit).get_drive_channel().frequency = 1e9 * round(
            hw.get_qubit(qubit).get_drive_channel().frequency / 1e9, 6
        )

    for qubit in target_qubits:
        drive_channel = hw.get_qubit(qubit).get_drive_channel()
        physical_channel = drive_channel.physical_channel
        baseband = physical_channel.baseband
        log.info(f"Old baseband IF frequency of qubit '{qubit}': {baseband.if_frequency}")
        log.info(f"Old baseband frequency of qubit '{qubit}': {baseband.frequency}")
        baseband.if_frequency = float(round(drive_channel.frequency - lo_freq))
        baseband.frequency = lo_freq
        log.info(f"Baseband IF frequency of qubit '{qubit}' set to {baseband.if_frequency}")
        log.info(f"Baseband frequency of qubit '{qubit}' set to {baseband.frequency}")


class LiveHardwareModel(QuantumHardwareModel):
    def __init__(
        self,
        control_hardware: ControlHardware = None,
        shot_limit=10000,
        acquire_mode=None,
        repeat_count=1000,
        repetition_period=100e-6,
        error_mitigation: Optional[ErrorMitigation] = None,
    ):
        super().__init__(
            shot_limit=shot_limit,
            acquire_mode=acquire_mode or AcquireMode.INTEGRATOR,
            repeat_count=repeat_count,
            repetition_period=repetition_period,
            error_mitigation=error_mitigation,
        )
        self.control_hardware: ControlHardware = control_hardware
        self.instruments: Dict[str, Instrument] = {}
        self.qubit_direction_couplings: List[QubitCoupling] = []

    def create_engine(self, startup_engine: bool = True):
        return LiveDeviceEngine(self, startup_engine)

    def add_device(self, device):
        if isinstance(device, Instrument):
            self.add_instrument(device)
        else:
            super().add_device(device)

    def add_instrument(self, *instruments: Instrument):
        for instrument in instruments:
            if instrument.id in self.instruments:
                raise KeyError(f"Instrument with id '{instrument.id}' already exists.")
            self.instruments[instrument.id] = instrument

    def get_instrument(self, id_: str):
        return self.instruments.get(id_, None)

    def get_device(self, id_):
        if (device := self.get_instrument(id_)) is not None:
            return device
        return super().get_device(id_)

    def add_physical_baseband(self, *basebands: LivePhysicalBaseband):
        for baseband in basebands:
            if baseband.instrument.id not in self.instruments:
                self.add_instrument(baseband.instrument)
        super().add_physical_baseband(*basebands)

    def add_physical_channel(self, *physical_channels: ControlHardwareChannel):
        if self.control_hardware is not None:
            self.control_hardware.add_physical_channel(*physical_channels)
        super().add_physical_channel(*physical_channels)


class LiveDeviceEngine(QuantumExecutionEngine):
    """
    Backend that hooks up to our QPU's, currently hardcoded to particular fridges.
    This will only work when run on a machine physically connected to a QPU.
    """

    model: LiveHardwareModel

    def __init__(self, model: LiveHardwareModel, startup_engine: bool = True):
        super().__init__(model, startup_engine)

    def __enter__(self):
        return self

    def __del__(self):
        self.shutdown()

    def __exit__(self, exc, value, tb):
        self.shutdown()

    def startup(self):
        aggregate_result = []

        for instrument in self.model.instruments.values():
            instrument.connect()
            aggregate_result.append(instrument.is_connected)
        if self.model.control_hardware is not None:
            self.model.control_hardware.connect()
            aggregate_result.append(self.model.control_hardware.is_connected)
        for baseband in self.model.basebands.values():
            if isinstance(baseband, LivePhysicalBaseband):
                baseband.connect_to_instrument()
                aggregate_result.append(baseband.instrument.is_connected)
        for physical_channel in self.model.physical_channels:
            if isinstance(physical_channel, ControlHardwareChannel):
                for dc_bias_ch in physical_channel.dcbiaschannel_pair.values():
                    dc_bias_ch.connect_to_instrument()
                    aggregate_result.append(dc_bias_ch.instrument.is_connected)

        return all(aggregate_result)

    def shutdown(self):
        # Store the instrument connection statuses.
        # All instruments should be disconnected (represented as a False entry)
        aggregate_result = []
        for instrument in self.model.instruments.values():
            instrument.close()
            aggregate_result.append(instrument.is_connected)
        if self.model.control_hardware is not None:
            self.model.control_hardware.close()
            aggregate_result.append(self.model.control_hardware.is_connected)

        return not any(aggregate_result)

    def process_reset(self, position):
        raise NotImplementedError(
            "Active qubit reset is not currently implemented on live hardware."
        )

    def build_baseband_frequencies(
        self, pulse_channel_buffers: Dict[PulseChannel, np.ndarray]
    ):
        """Find fixed intermediate frequencies for physical channels if they exist."""
        baseband_freqs = {}
        baseband_freqs_fixed_if = {}
        for pulse_channel in pulse_channel_buffers.keys():
            if pulse_channel.fixed_if:
                if (
                    baseband_freqs_fixed_if.get(pulse_channel.physical_channel_id, False)
                    and baseband_freqs[pulse_channel.physical_channel_id]
                    != pulse_channel.frequency
                    - UPCONVERT_SIGN * pulse_channel.baseband_if_frequency
                ):
                    raise ValueError(
                        "Cannot fix the frequency for two pulse channels of different "
                        "frequencies on the same physical channel!"
                    )
                baseband_freqs[pulse_channel.physical_channel_id] = (
                    pulse_channel.frequency
                    - UPCONVERT_SIGN * pulse_channel.baseband_if_frequency
                )
                baseband_freqs_fixed_if[pulse_channel.physical_channel_id] = (
                    pulse_channel.fixed_if
                )
            else:
                if (
                    pulse_channel.physical_channel_id not in baseband_freqs_fixed_if
                    or not baseband_freqs_fixed_if[pulse_channel.physical_channel_id]
                ):
                    baseband_freqs_fixed_if[pulse_channel.physical_channel_id] = (
                        pulse_channel.fixed_if
                    )

        return baseband_freqs

    def _execute_on_hardware(
        self,
        sweep_iterator: SweepIterator,
        package: QatFile,
        interrupt: Interrupt = NullInterrupt(),
    ) -> Dict[str, np.ndarray]:
        if self.model.control_hardware is None:
            raise ValueError("Please add a control hardware first!")

        results = {}
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)
            metadata = {"sweep_iteration": sweep_iterator.get_current_sweep_iteration()}

            position_map = self.create_duration_timeline(package.instructions)
            pulse_channel_buffers = self.build_pulse_channel_buffers(position_map, True)
            buffers = self.build_physical_channel_buffers(pulse_channel_buffers)
            baseband_freqs = self.build_baseband_frequencies(pulse_channel_buffers)
            aq_map = self.build_acquire_list(position_map)

            data = {}
            for ch, buffer in buffers.items():
                if len(buffer) > 0:
                    data[ch] = buffer
                    if ch in baseband_freqs:
                        physical_channel = self.model.get_physical_channel(ch)
                        physical_channel.baseband.set_frequency(int(baseband_freqs[ch]))

            self.model.control_hardware.set_data(data)

            repetitions = package.repeat.repeat_count
            repetition_time = package.repeat.repetition_period

            for aqs in aq_map.values():
                if len(aqs) > 1:
                    raise ValueError(
                        "Multiple acquisitions are not supported on the same channel "
                        "in one sweep step."
                    )
                for aq in aqs:
                    physical_channel = aq.physical_channel
                    dt = physical_channel.sample_time
                    physical_channel.readout_start = aq.start * dt + (
                        aq.delay if aq.delay else 0.0
                    )
                    physical_channel.readout_length = aq.samples * dt
                    physical_channel.acquire_mode_integrator = (
                        aq.mode == AcquireMode.INTEGRATOR
                    )

            interrupt.if_triggered(metadata, throw=True)

            playback_results = self.model.control_hardware.start_playback(
                repetitions=repetitions, repetition_time=repetition_time
            )

            for channel, aqs in aq_map.items():
                if len(aqs) > 1:
                    raise ValueError(
                        "Multiple acquisitions are not supported on the same channel "
                        "in one sweep step."
                    )
                for aq in aqs:
                    response = playback_results[aq.physical_channel.id]
                    response_axis = get_axis_map(aq.mode, response)
                    for pp in package.get_pp_for_variable(aq.output_variable):
                        response, response_axis = self.run_post_processing(
                            pp, response, response_axis
                        )
                    var_result = results.setdefault(
                        aq.output_variable,
                        np.empty(
                            sweep_iterator.get_results_shape(response.shape),
                            response.dtype,
                        ),
                    )
                    sweep_iterator.insert_result_at_sweep_position(var_result, response)
        return results

    def optimize(self, instructions):
        with log_duration("Instructions optimized in {} seconds."):
            instructions = super().optimize(instructions)
            for pp in [val for val in instructions if isinstance(val, PostProcessing)]:
                if pp.acquire.mode == AcquireMode.SCOPE:
                    if (
                        pp.process == PostProcessType.MEAN
                        and ProcessAxis.SEQUENCE in pp.axes
                        and len(pp.axes) <= 1
                    ):
                        instructions.remove(pp)

                elif pp.acquire.mode == AcquireMode.INTEGRATOR:
                    if (
                        pp.process == PostProcessType.DOWN_CONVERT
                        and ProcessAxis.TIME in pp.axes
                        and len(pp.axes) <= 1
                    ):
                        instructions.remove(pp)

                    if (
                        pp.process == PostProcessType.MEAN
                        and ProcessAxis.TIME in pp.axes
                        and len(pp.axes) <= 1
                    ):
                        instructions.remove(pp)

            return instructions

    def validate(self, instructions: List[Instruction]):
        with log_duration("Instructions validated in {} seconds."):
            super().validate(instructions)

            consumed_qubits: List[str] = []
            for inst in instructions:
                if isinstance(inst, PostProcessing):
                    if (
                        inst.acquire.mode == AcquireMode.SCOPE
                        and ProcessAxis.SEQUENCE in inst.axes
                    ):
                        raise ValueError(
                            "Invalid post-processing! Post-processing over SEQUENCE is "
                            "not possible after the result is returned from hardware "
                            "in SCOPE mode!"
                        )
                    elif (
                        inst.acquire.mode == AcquireMode.INTEGRATOR
                        and ProcessAxis.TIME in inst.axes
                    ):
                        raise ValueError(
                            "Invalid post-processing! Post-processing over TIME is not "
                            "possible after the result is returned from hardware in "
                            "INTEGRATOR mode!"
                        )
                    elif inst.acquire.mode == AcquireMode.RAW:
                        raise ValueError(
                            "Invalid acquire mode! The live hardware doesn't support "
                            "RAW acquire mode!"
                        )

                # Check if we've got a measure in the middle of the circuit somewhere.
                elif isinstance(inst, Acquire):
                    for qbit in self.model.qubits:
                        if qbit.get_measure_channel() == inst.channel:
                            consumed_qubits.append(qbit)
                elif isinstance(inst, Pulse):
                    # Find target qubit from instruction and check whether it's been
                    # measured already.
                    acquired_qubits = [
                        self.model._resolve_qb_pulse_channel(chanbit)[0] in consumed_qubits
                        for chanbit in inst.quantum_targets
                        if isinstance(chanbit, (Qubit, PulseChannel))
                    ]

                    if any(acquired_qubits):
                        raise ValueError(
                            "Mid-circuit measurements currently unable to be used."
                        )
