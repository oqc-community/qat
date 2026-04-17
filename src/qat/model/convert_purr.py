# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import re
from collections import defaultdict
from contextlib import nullcontext as not_seeded
from copy import deepcopy
from warnings import warn

import numpy as np

import qat.purr.compiler.devices as purr_devices
from qat.frontend.converters.purr import WAVEFORM_MAP
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
from qat.purr.compiler.devices import ChannelType
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import CustomPulse
from qat.purr.utils.logger import get_default_logger
from qat.utils.pydantic import FrozenDict
from qat.utils.uuid import temporary_uuid_seed

number_mask = re.compile("[0-9]+")

log = get_default_logger()


def get_number_from_string(s: str) -> int | None:
    """Returns the first number found in a string, or None if no number is found.

    Used to extract the index from IDs.
    """
    number_match = number_mask.search(s)
    if number_match is not None:
        return int(number_match.group())
    return None


def convert_purr_echo_hw_to_pydantic(
    legacy_hw: QuantumHardwareModel, seed_uuid: bool = True
) -> PhysicalHardwareModel:
    """Converts a PuRR QuantumHardwareModel into a PhysicalHardwareModel.

    :param legacy_hw: The PuRR QuantumHardwareModel to convert.
    :param seed_uuid: Whether to seed the UUID generation for pulse channels that don't
        exist in the PuRR model. Defaults to ``True`` so conversions generate
        deterministic UUIDs across runs; set to ``False`` to use unseeded UUID
        generation instead.
    """
    if seed_uuid:
        seed_context = temporary_uuid_seed(legacy_hw.calibration_id)
    else:
        seed_context = not_seeded()

    with seed_context:
        qubits = {}

        physical_connectivity = _build_physical_topology(legacy_hw)
        inverse_physical_connectivity = _build_inversed_physical_topology(
            physical_connectivity
        )
        logical_connectivity = _build_logical_topology(legacy_hw, physical_connectivity)
        logical_connectivity_quality = _build_coupling_qualities(
            legacy_hw, logical_connectivity
        )
        error_mitigation = _build_error_mitigation(legacy_hw)

        for qubit in legacy_hw.qubits:
            resonator = _build_resonator(
                qubit.measure_device,
                _build_pulse(qubit.pulse_measure),
                _build_acquire(qubit.measure_acquire),
            )
            cr_qubits = [
                legacy_hw.get_qubit(idx) for idx in physical_connectivity[qubit.index]
            ]
            crc_qubits = [
                legacy_hw.get_qubit(idx)
                for idx in inverse_physical_connectivity[qubit.index]
            ]
            qubits[qubit.index] = _build_qubit(qubit, resonator, cr_qubits, crc_qubits)

        qubits = FrozenDict(qubits)

        calibration_id = getattr(legacy_hw, "calibration_id", "")
        hw = PhysicalHardwareModel(
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
            physical_connectivity=physical_connectivity,
            qubits=qubits,
            calibration_id=calibration_id,
            error_mitigation=error_mitigation,
        )
    return hw


def _process_real_or_complex(value: float | complex) -> float | complex:
    if isinstance(value, complex) and value.imag == 0.0:
        return value.real

    return value


def _get_pulse_channel(
    qubit: purr_devices.Qubit,
    channel_type: ChannelType,
    auxiliary_qubit: purr_devices.Qubit | None = None,
) -> purr_devices.PulseChannelView | None:
    """PuRR implementation raises an error if we query for a channel that doesn't exist.

    We want to be able to see if the channel exists, and handle exceptions otherwise. This
    is a helper function to do that.
    """

    auxiliary_qubit = [auxiliary_qubit] if auxiliary_qubit is not None else None

    try:
        return qubit.get_pulse_channel(channel_type, auxiliary_qubit)
    except KeyError:
        return None


def _build_physical_topology(legacy_hw: QuantumHardwareModel) -> dict[int, set[int]]:
    """Assemble the physical connectivity from the legacy hardware model.

    Connectivity is
    inferred from the ``coupled_qubits`` attribute of each qubit. For each pair of coupled
    qubits, a physical edge is added only if both a cross-resonance pulse channel and a
    corresponding cross-resonance-cancellation pulse channel are defined between the two
    qubits in the PuRR hardware model.
    """

    physical_connectivity = defaultdict(set)
    invalid_couplings = set()
    for qubit in legacy_hw.qubits:
        for coupled_qubit in qubit.coupled_qubits:
            cr_channel = _get_pulse_channel(
                qubit, purr_devices.ChannelType.cross_resonance, coupled_qubit
            )
            crc_channel = _get_pulse_channel(
                coupled_qubit, purr_devices.ChannelType.cross_resonance_cancellation, qubit
            )
            if cr_channel is None or crc_channel is None:
                invalid_couplings.add((qubit.index, coupled_qubit.index))
                continue

            physical_connectivity[qubit.index].add(coupled_qubit.index)

    if invalid_couplings:
        log.warning(
            "The following physical couplings are present in the PuRR hardware model, but "
            "will be ignored in the new hardware model due to missing pulse channel "
            f"definitions: {invalid_couplings}."
        )
    return physical_connectivity


def _build_inversed_physical_topology(
    physical_connectivity: dict[int, set[int]],
) -> dict[int, set[int]]:
    """Inverses the physical connectivity to get the incoming couplings for each qubit.

    This is used to check the logical connectivity against the physical connectivity, and to
    build the qubits with their incoming couplings.
    """

    inversed_physical_connectivity = defaultdict(set)
    for qubit_idx, coupled_qubits in physical_connectivity.items():
        for coupled_qubit_idx in coupled_qubits:
            inversed_physical_connectivity[coupled_qubit_idx].add(qubit_idx)

    return inversed_physical_connectivity


def _build_logical_topology(
    legacy_hw: QuantumHardwareModel, physical_connectivity: dict[int, set[int]]
) -> dict[int, set[int]] | None:
    """Assembles the logical connectivity from the legacy hardware model and the established
    physical connectivity.

    This is done by looking at the qubit direction couplings if they exist, otherwise we
    assume the logical connectivity is the same as the physical connectivity. Filters out
    any couplings that do not include a zx_pi_4 gate.
    """

    couplings = getattr(legacy_hw, "qubit_direction_couplings", None)

    invalid_couplings = set()
    if couplings:
        logical_connectivity = defaultdict(set)
        for coupling in couplings:
            if coupling.direction[1] not in physical_connectivity[coupling.direction[0]]:
                invalid_couplings.add((coupling.direction[0], coupling.direction[1]))
                continue
            logical_connectivity[coupling.direction[0]].add(coupling.direction[1])
    else:
        logical_connectivity = deepcopy(physical_connectivity)

    removed_couplings = set()
    for qubit_idx, coupled_qubits in logical_connectivity.items():
        qubit = legacy_hw.get_qubit(qubit_idx)
        for coupled_qubit_idx in coupled_qubits:
            coupled_qubit = legacy_hw.get_qubit(coupled_qubit_idx)
            if qubit.pulse_hw_zx_pi_4.get(coupled_qubit.id, None) is None:
                invalid_couplings.add((qubit_idx, coupled_qubit_idx))
                removed_couplings.add((qubit_idx, coupled_qubit_idx))

    for qubit_idx, coupled_qubit_index in removed_couplings:
        logical_connectivity[qubit_idx].remove(coupled_qubit_index)

    if invalid_couplings:
        warn(
            "The following logical couplings are present in the PuRR hardware model, but "
            "will be ignored in the new hardware model due to missing pulse channels or "
            f"zx_pi_4 definitions: {invalid_couplings}."
        )

    return logical_connectivity


def _build_coupling_qualities(
    legacy_hw: QuantumHardwareModel, logical_connectivity: dict[int, set[int]]
) -> dict[tuple[int, int], float] | None:
    """Assembles the coupling qualities from the legacy hardware model, if they are present
    in the logical connectivity."""

    coupling_qualities = getattr(legacy_hw, "qubit_direction_couplings", None)
    if coupling_qualities is None:
        return None

    new_coupling_qualities = {}
    for coupling in coupling_qualities:
        if coupling.direction[1] not in logical_connectivity.get(
            coupling.direction[0], set()
        ):
            continue
        # it's assumed the quality is between [0, 100], but we want to normalize onto [0, 1]
        new_coupling_qualities[coupling.direction] = coupling.quality / 100.0

    return new_coupling_qualities


def _build_error_mitigation(legacy_hw: QuantumHardwareModel) -> ErrorMitigation:
    """Builds the error mitigation strategy from the legacy hardware model, if it is
    present."""

    error_mitigation = getattr(legacy_hw, "error_mitigation", None)
    if error_mitigation is None:
        return ErrorMitigation()

    readout_mitigation = getattr(error_mitigation, "readout_mitigation", None)
    if readout_mitigation is None:
        return ErrorMitigation()

    linear = {}
    missing_qubits = set()
    valid_qubit_indices = [qubit.index for qubit in legacy_hw.qubits]
    if readout_mitigation.linear:
        for qubit_idx, qubit_map in readout_mitigation.linear.items():
            qubit_idx = int(qubit_idx)
            if qubit_idx not in valid_qubit_indices:
                missing_qubits.add(qubit_idx)
                continue
            linear[qubit_idx] = CalibratableUnitInterval2x2Array(
                np.array(
                    [
                        [qubit_map["0|0"], qubit_map["0|1"]],
                        [qubit_map["1|0"], qubit_map["1|1"]],
                    ]
                )
            )

    if missing_qubits:
        log.warning(
            "The following qubits have linear readout mitigation defined in the PuRR "
            "hardware model, but will be ignored in the new hardware model due to missing "
            f"qubit definitions: {missing_qubits}."
        )

    linear = FrozenDict(linear)
    readout_mitigation = ReadoutMitigation(
        linear=linear,
        matrix=getattr(readout_mitigation, "matrix", None),
        m3_available=getattr(readout_mitigation, "m3_available", False),
    )
    return ErrorMitigation(readout_mitigation=readout_mitigation)


def _build_pulse(pulse_params: dict) -> CalibratablePulse:
    """Builds a pulse from a dictionary of parameters."""

    pulse_params = deepcopy(pulse_params)
    purr_waveform_type = pulse_params.pop("shape", purr_devices.PulseShapeType.GAUSSIAN)
    waveform_type = WAVEFORM_MAP.get(purr_waveform_type, None)
    if waveform_type is None:
        raise ValueError(
            f"Unsupported waveform shape {purr_waveform_type} found when converting "
            "Pulse data dictionary."
        )
    return CalibratablePulse(waveform_type=waveform_type, **pulse_params)


def _build_acquire(measure_acquire: dict) -> CalibratableAcquire:
    """Builds an acquire from a dictionary of parameters."""

    measure_acquire = deepcopy(measure_acquire)
    if measure_acquire.get("weights") is None:
        measure_acquire["weights"] = []
    elif isinstance(measure_acquire.get("weights"), CustomPulse):
        measure_acquire["weights"] = measure_acquire["weights"].samples

    return CalibratableAcquire(**measure_acquire)


def _build_resonator(
    resonator: purr_devices.Resonator,
    measure_pulse: CalibratablePulse,
    measure_acquire: CalibratableAcquire,
) -> Resonator:
    """Builds a resonator from a PuRR resonator definition."""

    phys_bb_r = resonator.physical_channel.baseband
    new_phys_bb_r = PhysicalBaseband(
        uuid=phys_bb_r.full_id(),
        frequency=phys_bb_r.frequency,
        if_frequency=phys_bb_r.if_frequency,
    )

    phys_channel_r = resonator.physical_channel
    new_phys_ch_r = ResonatorPhysicalChannel(
        uuid=phys_channel_r.full_id(),
        baseband=new_phys_bb_r,
        block_size=phys_channel_r.block_size,
        swap_readout_iq=getattr(phys_channel_r, "swap_readout_IQ", False),
        name_index=get_number_from_string(phys_channel_r.id),
    )

    measure_pulse_channel = resonator.get_pulse_channel(ChannelType.measure)
    new_measure_pulse_channel = MeasurePulseChannel(
        uuid=measure_pulse_channel.full_id(),
        frequency=measure_pulse_channel.frequency,
        imbalance=measure_pulse_channel.imbalance,
        scale=_process_real_or_complex(measure_pulse_channel.scale),
        phase_iq_offset=measure_pulse_channel.phase_offset,
        pulse=measure_pulse,
    )

    acquire_pulse_channel = resonator.get_pulse_channel(ChannelType.acquire)
    new_acquire_pulse_channel = AcquirePulseChannel(
        uuid=acquire_pulse_channel.full_id(),
        frequency=acquire_pulse_channel.frequency,
        imbalance=acquire_pulse_channel.imbalance,
        scale=_process_real_or_complex(acquire_pulse_channel.scale),
        phase_iq_offset=acquire_pulse_channel.phase_offset,
        acquire=measure_acquire,
    )

    new_res_pulse_channels = ResonatorPulseChannels(
        measure=new_measure_pulse_channel, acquire=new_acquire_pulse_channel
    )

    return Resonator(
        uuid=resonator.full_id(),
        physical_channel=new_phys_ch_r,
        pulse_channels=new_res_pulse_channels,
    )


def _build_qubit(
    qubit: purr_devices.Qubit,
    resonator: Resonator,
    cr_qubits: list[purr_devices.Qubit],
    crc_qubits: list[purr_devices.Qubit],
) -> Qubit:
    """Builds a qubit from a PuRR qubit definition."""

    # Physical baseband
    phys_bb_q = qubit.physical_channel.baseband
    new_phys_bb_q = PhysicalBaseband(
        uuid=phys_bb_q.full_id(),
        frequency=phys_bb_q.frequency,
        if_frequency=phys_bb_q.if_frequency,
    )

    # Physical channel
    phys_channel_q = qubit.physical_channel
    new_phys_ch_q = QubitPhysicalChannel(
        uuid=phys_channel_q.full_id(),
        baseband=new_phys_bb_q,
        block_size=phys_channel_q.block_size,
        name_index=get_number_from_string(phys_channel_q.id),
    )

    # Qubit pulse channels
    drive_pulse_channel = qubit.get_pulse_channel(ChannelType.drive)
    pulse_hw_x_pi = getattr(qubit, "pulse_hw_x_pi", None)
    if pulse_hw_x_pi is not None:
        pulse_hw_x_pi = _build_pulse(pulse_hw_x_pi)

    new_drive_pulse_channel = DrivePulseChannel(
        uuid=drive_pulse_channel.full_id(),
        frequency=drive_pulse_channel.frequency,
        imbalance=drive_pulse_channel.imbalance,
        scale=_process_real_or_complex(drive_pulse_channel.scale),
        phase_iq_offset=drive_pulse_channel.phase_offset,
        pulse=_build_pulse(qubit.pulse_hw_x_pi_2),
        pulse_x_pi=pulse_hw_x_pi,
    )

    freqshift_pulse_channel = _get_pulse_channel(qubit, ChannelType.freq_shift)
    if freqshift_pulse_channel is not None:
        new_freqshift_pulse_channel = FreqShiftPulseChannel(
            uuid=freqshift_pulse_channel.full_id(),
            frequency=freqshift_pulse_channel.frequency,
            imbalance=freqshift_pulse_channel.imbalance,
            scale=_process_real_or_complex(freqshift_pulse_channel.scale),
            phase_iq_offset=freqshift_pulse_channel.phase_offset,
            active=freqshift_pulse_channel.active,
            amp=freqshift_pulse_channel.amp,
            phase=getattr(freqshift_pulse_channel.pulse_channel, "phase", 0.0),
        )
    else:
        new_freqshift_pulse_channel = FreqShiftPulseChannel()

    secondstate_pulse_channel = _get_pulse_channel(qubit, ChannelType.second_state)
    if secondstate_pulse_channel is not None:
        new_secondstate_pulse_channel = SecondStatePulseChannel(
            uuid=secondstate_pulse_channel.full_id(),
            frequency=secondstate_pulse_channel.frequency,
            imbalance=secondstate_pulse_channel.imbalance,
            scale=_process_real_or_complex(secondstate_pulse_channel.scale),
            phase_iq_offset=secondstate_pulse_channel.phase_offset,
        )
    else:
        new_secondstate_pulse_channel = SecondStatePulseChannel()

    new_cr_pulse_channels = {}
    new_crc_pulse_channels = {}

    for cr_qubit in cr_qubits:
        pulse_channel = qubit.get_pulse_channel(ChannelType.cross_resonance, cr_qubit)
        zx_pi_4 = deepcopy(qubit.pulse_hw_zx_pi_4.get(f"Q{cr_qubit.index}", None))

        if zx_pi_4 is not None:
            zx_pi_4 = _build_pulse(zx_pi_4)

        new_cr_pulse_channel = CrossResonancePulseChannel(
            uuid=pulse_channel.full_id(),
            auxiliary_qubit=cr_qubit.index,
            frequency=pulse_channel.frequency,
            imbalance=pulse_channel.imbalance,
            scale=_process_real_or_complex(pulse_channel.scale),
            phase_iq_offset=pulse_channel.phase_offset,
            zx_pi_4_pulse=zx_pi_4,
        )
        new_cr_pulse_channels[cr_qubit.index] = new_cr_pulse_channel

    for crc_qubit in crc_qubits:
        pulse_channel = qubit.get_pulse_channel(
            ChannelType.cross_resonance_cancellation, crc_qubit
        )
        new_crc_pulse_channel = CrossResonanceCancellationPulseChannel(
            uuid=pulse_channel.full_id(),
            auxiliary_qubit=crc_qubit.index,
            frequency=pulse_channel.frequency,
            imbalance=pulse_channel.imbalance,
            scale=_process_real_or_complex(pulse_channel.scale),
            phase_iq_offset=pulse_channel.phase_offset,
        )
        new_crc_pulse_channels[crc_qubit.index] = new_crc_pulse_channel

    new_cr_pulse_channels = FrozenDict(new_cr_pulse_channels)
    new_crc_pulse_channels = FrozenDict(new_crc_pulse_channels)

    new_qubit_pulse_channels = QubitPulseChannels(
        drive=new_drive_pulse_channel,
        freq_shift=new_freqshift_pulse_channel,
        second_state=new_secondstate_pulse_channel,
        cross_resonance_channels=new_cr_pulse_channels,
        cross_resonance_cancellation_channels=new_crc_pulse_channels,
    )

    return Qubit(
        uuid=qubit.full_id(),
        physical_channel=new_phys_ch_q,
        pulse_channels=new_qubit_pulse_channels,
        resonator=resonator,
        mean_z_map_args=qubit.mean_z_map_args,
        discriminator=qubit.discriminator[0],
        direct_x_pi=getattr(qubit, "direct_x_pi", False),
    )
