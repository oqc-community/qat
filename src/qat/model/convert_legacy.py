from collections import defaultdict

from qat.model.device import (
    AcquirePulseChannel,
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
from qat.model.hardware_base import FrozenDict
from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.compiler.devices import ChannelType, PulseChannelView


def convert_legacy_hw_to_pydantic(legacy_hw):
    new_qubits = {}
    logical_connectivity = defaultdict(set)

    for qubit in legacy_hw.qubits:
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
            phase_iq_offset=phys_channel_q.phase_offset,
            bias=phys_channel_q.imbalance,
        )

        phys_channel_r = qubit.measure_device.physical_channel
        new_phys_ch_r = PhysicalChannel(
            baseband=new_phys_bb_r,
            sample_time=phys_channel_r.sample_time,
            block_size=phys_channel_r.block_size,
            phase_iq_offset=phys_channel_r.phase_offset,
            bias=phys_channel_r.imbalance,
        )

        # Resonator
        measure_pulse_channel = qubit.measure_device.get_pulse_channel(ChannelType.measure)
        new_measure_pulse_channel = MeasurePulseChannel(
            frequency=measure_pulse_channel.frequency,
            bias=measure_pulse_channel.bias,
            scale=measure_pulse_channel.scale,
            fixed_if=measure_pulse_channel.fixed_if,
        )

        acquire_pulse_channel = qubit.measure_device.get_pulse_channel(ChannelType.acquire)
        new_acquire_pulse_channel = AcquirePulseChannel(
            frequency=acquire_pulse_channel.frequency,
            bias=acquire_pulse_channel.bias,
            scale=acquire_pulse_channel.scale,
            fixed_if=acquire_pulse_channel.fixed_if,
        )

        new_res_pulse_channels = ResonatorPulseChannels(
            measure=new_measure_pulse_channel, acquire=new_acquire_pulse_channel
        )
        resonator = Resonator(
            physical_channel=new_phys_ch_r, pulse_channels=new_res_pulse_channels
        )

        # Qubit pulse channels
        drive_pulse_channel = qubit.get_pulse_channel(ChannelType.drive)
        new_drive_pulse_channel = DrivePulseChannel(
            frequency=drive_pulse_channel.frequency,
            bias=drive_pulse_channel.bias,
            scale=drive_pulse_channel.scale,
            fixed_if=drive_pulse_channel.fixed_if,
        )

        try:
            freqshift_pulse_channel = qubit.get_pulse_channel(ChannelType.freq_shift)
            new_freqshift_pulse_channel = FreqShiftPulseChannel(
                frequency=freqshift_pulse_channel.frequency,
                bias=freqshift_pulse_channel.bias,
                scale=freqshift_pulse_channel.scale,
                fixed_if=freqshift_pulse_channel.fixed_if,
            )
        except KeyError:
            new_freqshift_pulse_channel = FreqShiftPulseChannel()

        try:
            secondstate_pulse_channel = qubit.get_pulse_channel(ChannelType.second_state)
            new_secondstate_pulse_channel = SecondStatePulseChannel(
                frequency=secondstate_pulse_channel.frequency,
                bias=secondstate_pulse_channel.bias,
                scale=secondstate_pulse_channel.scale,
                fixed_if=secondstate_pulse_channel.fixed_if,
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
                        bias=pulse_channel.bias,
                        scale=pulse_channel.scale,
                        fixed_if=pulse_channel.fixed_if,
                    )
                    new_cross_resonance_pulse_channels[aux_qubit] = new_cr_pulse_channel

                elif pulse_channel.channel_type == ChannelType.cross_resonance_cancellation:
                    new_crc_pulse_channel = CrossResonanceCancellationPulseChannel(
                        auxiliary_qubit=aux_qubit,
                        frequency=pulse_channel.frequency,
                        bias=pulse_channel.bias,
                        scale=pulse_channel.scale,
                        fixed_if=pulse_channel.fixed_if,
                    )
                    new_cross_resonance_cancellation_pulse_channels[aux_qubit] = (
                        new_crc_pulse_channel
                    )

                logical_connectivity[qubit.index].add(aux_qubit)

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

    new_hw = PhysicalHardwareModel(
        logical_connectivity=logical_connectivity,
        physical_connectivity=physical_connectivity,
        qubits=new_qubits,
    )

    return new_hw
