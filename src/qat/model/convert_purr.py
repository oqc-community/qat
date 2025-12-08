# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import re
from collections import defaultdict
from copy import deepcopy

import numpy as np

from qat.ir.waveforms import SofterSquareWaveform, SoftSquareWaveform, SquareWaveform
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
    Qubit,
    QubitPhysicalChannel,
    QubitPulseChannels,
    Resonator,
    ResonatorPhysicalChannel,
    ResonatorPulseChannels,
    SecondStatePulseChannel,
)
from qat.model.error_mitigation import (
    CalibratableUnitInterval2x2Array,
    ErrorMitigation,
    ReadoutMitigation,
)
from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.compiler.devices import ChannelType, PulseChannelView, PulseShapeType
from qat.purr.compiler.instructions import CustomPulse
from qat.utils.pydantic import FrozenDict
from qat.utils.uuid import uuid_randomiser

number_mask = re.compile("[0-9]+")


def get_number_from_string(s: str) -> int | None:
    number_match = number_mask.search(s)
    if number_match is not None:
        return int(number_match.group())
    return None


def convert_purr_echo_hw_to_pydantic(legacy_hw, seed_uuid: bool = True):
    new_qubits = {}
    logical_connectivity = defaultdict(set)

    if seed_uuid:
        uuid_randomiser.seed(legacy_hw.calibration_id)

    for qubit in legacy_hw.qubits:
        # Add topology of qubit. Since the topology is not always stored both in
        # the `qubit_direction_couplings`, look into `coupled_qubits` as well.
        if couplings := getattr(legacy_hw, "qubit_direction_couplings", None):
            coupled_q_indices = [
                c.direction[1] for c in couplings if c.direction[0] == qubit.index
            ]
        else:
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
        new_phys_ch_q = QubitPhysicalChannel(
            baseband=new_phys_bb_q,
            block_size=phys_channel_q.block_size,
            name_index=get_number_from_string(phys_channel_q.id),
        )

        phys_channel_r = qubit.measure_device.physical_channel
        new_phys_ch_r = ResonatorPhysicalChannel(
            baseband=new_phys_bb_r,
            block_size=phys_channel_r.block_size,
            swap_readout_iq=getattr(phys_channel_r, "swap_readout_IQ", False),
            name_index=get_number_from_string(phys_channel_r.id),
        )

        # Resonator
        measure_pulse_channel = qubit.measure_device.get_pulse_channel(ChannelType.measure)
        pulse_measure = deepcopy(qubit.pulse_measure)
        pulse_measure.pop("shape")
        new_measure_pulse_channel = MeasurePulseChannel(
            frequency=measure_pulse_channel.frequency,
            imbalance=measure_pulse_channel.imbalance,
            scale=_process_real_or_complex(measure_pulse_channel.scale),
            phase_iq_offset=measure_pulse_channel.phase_offset,
            pulse=CalibratablePulse(waveform_type=SquareWaveform, **pulse_measure),
        )

        meas_acq = deepcopy(qubit.measure_acquire)
        if meas_acq.get("weights") is None:
            meas_acq["weights"] = []
        elif isinstance(meas_acq.get("weights"), CustomPulse):
            meas_acq["weights"] = meas_acq["weights"].samples

        acquire_pulse_channel = qubit.measure_device.get_pulse_channel(ChannelType.acquire)
        new_acquire_pulse_channel = AcquirePulseChannel(
            frequency=acquire_pulse_channel.frequency,
            imbalance=acquire_pulse_channel.imbalance,
            scale=_process_real_or_complex(acquire_pulse_channel.scale),
            phase_iq_offset=acquire_pulse_channel.phase_offset,
            acquire=CalibratableAcquire(**meas_acq),
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
        pulse_hw_x_pi = getattr(qubit, "pulse_hw_x_pi", None)
        if pulse_hw_x_pi is not None:
            pulse_hw_x_pi = deepcopy(pulse_hw_x_pi)
            pulse_hw_x_pi.pop("shape")
            pulse_hw_x_pi = CalibratablePulse(**pulse_hw_x_pi)

        new_drive_pulse_channel = DrivePulseChannel(
            frequency=drive_pulse_channel.frequency,
            imbalance=drive_pulse_channel.imbalance,
            scale=_process_real_or_complex(drive_pulse_channel.scale),
            phase_iq_offset=drive_pulse_channel.phase_offset,
            pulse=CalibratablePulse(**pulse_hw_x_pi_2),
            pulse_x_pi=pulse_hw_x_pi,
        )

        try:
            freqshift_pulse_channel = qubit.get_pulse_channel(ChannelType.freq_shift)
            new_freqshift_pulse_channel = FreqShiftPulseChannel(
                frequency=freqshift_pulse_channel.frequency,
                imbalance=freqshift_pulse_channel.imbalance,
                scale=_process_real_or_complex(freqshift_pulse_channel.scale),
                phase_iq_offset=freqshift_pulse_channel.phase_offset,
                active=freqshift_pulse_channel.active,
                amp=freqshift_pulse_channel.amp,
                phase=getattr(freqshift_pulse_channel.pulse_channel, "phase", 0.0),
            )
        except KeyError:
            new_freqshift_pulse_channel = FreqShiftPulseChannel()

        try:
            secondstate_pulse_channel = qubit.get_pulse_channel(ChannelType.second_state)
            new_secondstate_pulse_channel = SecondStatePulseChannel(
                frequency=secondstate_pulse_channel.frequency,
                imbalance=secondstate_pulse_channel.imbalance,
                scale=_process_real_or_complex(secondstate_pulse_channel.scale),
                phase_iq_offset=secondstate_pulse_channel.phase_offset,
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
                    zx_pi_4 = deepcopy(qubit.pulse_hw_zx_pi_4.get(f"Q{aux_qubit}", None))

                    waveform_type = (
                        SofterSquareWaveform
                        if zx_pi_4["shape"] == PulseShapeType.SOFTER_SQUARE
                        else SoftSquareWaveform
                    )
                    zx_pi_4.pop("shape")

                    new_cr_pulse_channel = CrossResonancePulseChannel(
                        auxiliary_qubit=aux_qubit,
                        frequency=pulse_channel.frequency,
                        imbalance=pulse_channel.imbalance,
                        scale=_process_real_or_complex(pulse_channel.scale),
                        phase_iq_offset=pulse_channel.phase_offset,
                        zx_pi_4_pulse=CalibratablePulse(
                            waveform_type=waveform_type, **zx_pi_4
                        ),
                    )
                    new_cross_resonance_pulse_channels[aux_qubit] = new_cr_pulse_channel

                elif pulse_channel.channel_type == ChannelType.cross_resonance_cancellation:
                    new_crc_pulse_channel = CrossResonanceCancellationPulseChannel(
                        auxiliary_qubit=aux_qubit,
                        frequency=pulse_channel.frequency,
                        imbalance=pulse_channel.imbalance,
                        scale=_process_real_or_complex(pulse_channel.scale),
                        phase_iq_offset=pulse_channel.phase_offset,
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
            mean_z_map_args=qubit.mean_z_map_args,
            discriminator=qubit.discriminator[0],
            direct_x_pi=getattr(qubit, "direct_x_pi", False),
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
                coupling.quality / 100.0
            )  # We assume that the quality is in [0, 100] in the legacy hw model.
    else:
        logical_connectivity_quality = None

    # Convert Error Mitigation if present
    if (error_mitigation := legacy_hw.error_mitigation) is not None and (
        readout_mitigation := getattr(
            legacy_hw.error_mitigation, "readout_mitigation", None
        )
    ) is not None:
        linear = {}
        if readout_mitigation.linear:
            for qubit_idx, qubit_map in readout_mitigation.linear.items():
                linear[qubit_idx] = CalibratableUnitInterval2x2Array(
                    np.array(
                        [
                            [qubit_map["0|0"], qubit_map["0|1"]],
                            [qubit_map["1|0"], qubit_map["1|1"]],
                        ]
                    )
                )
        linear = FrozenDict(linear)

        readout_mitigation = ReadoutMitigation(
            linear=linear,
            matrix=getattr(readout_mitigation, "matrix", None),
            m3_available=getattr(readout_mitigation, "m3_available", False),
        )
        error_mitigation = ErrorMitigation(readout_mitigation=readout_mitigation)
    else:
        error_mitigation = ErrorMitigation()

    calibration_id = getattr(legacy_hw, "calibration_id", "")
    new_hw = PhysicalHardwareModel(
        logical_connectivity=logical_connectivity,
        logical_connectivity_quality=logical_connectivity_quality,
        physical_connectivity=physical_connectivity,
        qubits=new_qubits,
        calibration_id=calibration_id,
        error_mitigation=error_mitigation,
    )

    if seed_uuid:
        uuid_randomiser.seed()

    return new_hw


def _process_real_or_complex(value: float | complex) -> float | complex:
    if isinstance(value, complex) and value.imag == 0.0:
        return value.real

    return value
