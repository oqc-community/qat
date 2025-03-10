# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from copy import deepcopy

from qat.model.device import (
    AcquirePulseChannel,
    CalibratableAcquire,
    CalibratablePulse,
    CrossResonanceCancellationPulseChannel,
    CrossResonancePulseChannel,
    DrivePulseChannel,
    FreqShiftPulseChannel,
    MeasurePulseChannel,
    PhysicalBaseband,
    PhysicalChannel,
    Qubit,
    QubitPulseChannels,
    Resonator,
    ResonatorPulseChannels,
    SecondStatePulseChannel,
)
from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.compiler.devices import ChannelType, PulseChannelView
from qat.utils.pydantic import FrozenDict


def convert_legacy_echo_hw_to_pydantic(legacy_hw):
    new_qubits = {}
    logical_connectivity = defaultdict(set)

    for qubit in legacy_hw.qubits:
        # Add topology of qubit
        coupled_q_indices = [q.index for q in qubit.coupled_qubits]
        logical_connectivity[qubit.index].update(coupled_q_indices)

        # Physical baseband
        phys_bb_q = qubit.physical_channel.baseband
        new_phys_bb_q = PhysicalBaseband(
            frequency=phys_bb_q.frequency, if_frequency=phys_bb_q.if_frequency
        )

        phys_bb_r = qubit.measure_device.physical_channel.baseband
        new_phys_bb_r = PhysicalBaseband(
            frequency=phys_bb_r.frequency, if_frequency=phys_bb_r.if_frequency
        )

        # Physical channel
        phys_channel_q = qubit.physical_channel
        new_phys_ch_q = PhysicalChannel(
            baseband=new_phys_bb_q,
            sample_time=phys_channel_q.sample_time,
            block_size=phys_channel_q.block_size,
        )

        phys_channel_r = qubit.measure_device.physical_channel
        new_phys_ch_r = PhysicalChannel(
            baseband=new_phys_bb_r,
            sample_time=phys_channel_r.sample_time,
            block_size=phys_channel_r.block_size,
        )

        # Resonator
        measure_pulse_channel = qubit.measure_device.get_pulse_channel(ChannelType.measure)
        pulse_measure = deepcopy(qubit.pulse_measure)
        pulse_measure.pop("shape")
        new_measure_pulse_channel = MeasurePulseChannel(
            frequency=measure_pulse_channel.frequency,
            imbalance=phys_channel_r.imbalance,
            scale=measure_pulse_channel.scale,
            fixed_if=measure_pulse_channel.fixed_if,
            phase_iq_offset=phys_channel_r.phase_offset,
            pulse=CalibratablePulse(**pulse_measure),
        )

        if qubit.measure_acquire["weights"] is None:
            qubit.measure_acquire["weights"] = []

        acquire_pulse_channel = qubit.measure_device.get_pulse_channel(ChannelType.acquire)
        new_acquire_pulse_channel = AcquirePulseChannel(
            frequency=acquire_pulse_channel.frequency,
            imbalance=phys_channel_r.imbalance,
            scale=acquire_pulse_channel.scale,
            fixed_if=acquire_pulse_channel.fixed_if,
            phase_iq_offset=phys_channel_r.phase_offset,
            acquire=CalibratableAcquire(**qubit.measure_acquire),
        )

        new_res_pulse_channels = ResonatorPulseChannels(
            measure=new_measure_pulse_channel, acquire=new_acquire_pulse_channel
        )

        resonator = Resonator(
            physical_channel=new_phys_ch_r,
            pulse_channels=new_res_pulse_channels,
        )

        # Qubit pulse channels
        drive_pulse_channel = qubit.get_pulse_channel(ChannelType.drive)
        pulse_hw_x_pi_2 = deepcopy(qubit.pulse_hw_x_pi_2)
        pulse_hw_x_pi_2.pop("shape")
        new_drive_pulse_channel = DrivePulseChannel(
            frequency=drive_pulse_channel.frequency,
            imbalance=phys_channel_q.imbalance,
            scale=drive_pulse_channel.scale,
            fixed_if=drive_pulse_channel.fixed_if,
            phase_iq_offset=phys_channel_q.phase_offset,
            pulse=CalibratablePulse(**pulse_hw_x_pi_2),
        )

        try:
            freqshift_pulse_channel = qubit.get_pulse_channel(ChannelType.freq_shift)
            new_freqshift_pulse_channel = FreqShiftPulseChannel(
                frequency=freqshift_pulse_channel.frequency,
                imbalance=phys_channel_q.imbalance,
                scale=freqshift_pulse_channel.scale,
                fixed_if=freqshift_pulse_channel.fixed_if,
                phase_iq_offset=phys_channel_q.phase_offset,
            )
        except KeyError:
            new_freqshift_pulse_channel = FreqShiftPulseChannel()

        try:
            secondstate_pulse_channel = qubit.get_pulse_channel(ChannelType.second_state)
            new_secondstate_pulse_channel = SecondStatePulseChannel(
                frequency=secondstate_pulse_channel.frequency,
                imbalance=phys_channel_q.imbalance,
                scale=secondstate_pulse_channel.scale,
                fixed_if=secondstate_pulse_channel.fixed_if,
                phase_iq_offset=phys_channel_q.phase_offset,
            )
        except KeyError:
            new_secondstate_pulse_channel = SecondStatePulseChannel()

        new_cross_resonance_pulse_channels = {}
        new_cross_resonance_cancellation_pulse_channels = {}

        for pulse_channel in qubit.pulse_channels.values():
            if (
                isinstance(pulse_channel, PulseChannelView)
                and pulse_channel.auxiliary_devices
            ):
                aux_qubit = pulse_channel.auxiliary_devices[0].index

                if pulse_channel.channel_type == ChannelType.cross_resonance:
                    new_cr_pulse_channel = CrossResonancePulseChannel(
                        auxiliary_qubit=aux_qubit,
                        frequency=pulse_channel.frequency,
                        imbalance=phys_channel_q.imbalance,
                        scale=pulse_channel.scale,
                        fixed_if=pulse_channel.fixed_if,
                        phase_iq_offset=phys_channel_q.phase_offset,
                    )
                    new_cross_resonance_pulse_channels[aux_qubit] = new_cr_pulse_channel

                elif pulse_channel.channel_type == ChannelType.cross_resonance_cancellation:
                    new_crc_pulse_channel = CrossResonanceCancellationPulseChannel(
                        auxiliary_qubit=aux_qubit,
                        frequency=pulse_channel.frequency,
                        imbalance=phys_channel_q.imbalance,
                        scale=pulse_channel.scale,
                        fixed_if=pulse_channel.fixed_if,
                        phase_iq_offset=phys_channel_q.phase_offset,
                    )
                    new_cross_resonance_cancellation_pulse_channels[aux_qubit] = (
                        new_crc_pulse_channel
                    )

        new_cross_resonance_pulse_channels = FrozenDict(new_cross_resonance_pulse_channels)

        new_cross_resonance_cancellation_pulse_channels = FrozenDict(
            new_cross_resonance_cancellation_pulse_channels
        )
        new_qubit_pulse_channels = QubitPulseChannels(
            drive=new_drive_pulse_channel,
            freq_shift=new_freqshift_pulse_channel,
            second_state=new_secondstate_pulse_channel,
            cross_resonance_channels=new_cross_resonance_pulse_channels,
            cross_resonance_cancellation_channels=new_cross_resonance_cancellation_pulse_channels,
        )

        new_qubit = Qubit(
            physical_channel=new_phys_ch_q,
            pulse_channels=new_qubit_pulse_channels,
            resonator=resonator,
        )
        new_qubits[qubit.index] = new_qubit

    new_qubits = FrozenDict(new_qubits)
    physical_connectivity = defaultdict(set)
    for q1_index in logical_connectivity:
        for q2_index in logical_connectivity[q1_index]:
            physical_connectivity[q1_index].add(q2_index)
            physical_connectivity[q2_index].add(q1_index)

    if legacy_hw.qubit_direction_couplings:
        logical_connectivity_quality = {}
        for coupling in legacy_hw.qubit_direction_couplings:
            logical_connectivity_quality[coupling.direction] = (
                coupling.quality
            )  # We assume that the quality is in [0, 1].
    else:
        logical_connectivity_quality = None

    new_hw = PhysicalHardwareModel(
        logical_connectivity=logical_connectivity,
        logical_connectivity_quality=logical_connectivity_quality,
        physical_connectivity=physical_connectivity,
        qubits=new_qubits,
    )

    return new_hw
