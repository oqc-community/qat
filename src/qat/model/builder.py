# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from qat.model.device import (
    AcquirePulseChannel,
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
from qat.model.hardware_model import PhysicalHardwareModel
from qat.utils.pydantic import CalibratableUnitInterval, FrozenDict, FrozenSet, QubitId


class PhysicalHardwareModelBuilder:
    """
    A builder class that builds a physical hardware model based on the given connectivity.

    :param physical_connectivity: The connectivities of the physical qubits on the QPU
        (undirected graph).
    :param logical_connectivity: The connectivities (directed graph) of the qubits used for
        compilation, which can be a subgraph of `physical_connectivity`.
    :param logical_connectivity_quality: Quality of the connections between the qubits.
    """

    def __init__(
        self,
        physical_connectivity: dict[int, FrozenSet[int]],
        logical_connectivity: dict[int, FrozenSet[int]] | None = None,
        logical_connectivity_quality: FrozenDict[
            tuple[QubitId, QubitId], CalibratableUnitInterval
        ] = None,
    ):
        self._current_model = self._build_uncalibrated_hardware_model(
            physical_connectivity=physical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
            logical_connectivity=logical_connectivity,
        )

    @property
    def model(self) -> PhysicalHardwareModel:
        return self._current_model

    def _build_uncalibrated_hardware_model(
        self,
        physical_connectivity: dict[int, set[int]],
        logical_connectivity: dict[int, set[int]] = None,
        logical_connectivity_quality: None = None,
    ):
        logical_connectivity = logical_connectivity or physical_connectivity

        unique_qubit_indices = set(physical_connectivity.keys()).union(
            *physical_connectivity.values()
        )

        unused_chan_indices = set(range(len(unique_qubit_indices) * 2))

        qubits = {}
        for qubit_id in unique_qubit_indices:
            qubit_connectivity = physical_connectivity.get(qubit_id, set())

            bb_q = self._build_uncalibrated_baseband()
            bb_r = self._build_uncalibrated_baseband()

            physical_channel_q = self._build_uncalibrated_physical_channel(
                baseband=bb_q, target_device=Qubit, name_index=unused_chan_indices.pop()
            )
            physical_channel_r = self._build_uncalibrated_physical_channel(
                baseband=bb_r, target_device=Resonator, name_index=unused_chan_indices.pop()
            )

            pulse_channels_q = self._build_uncalibrated_qubit_pulse_channels(
                qubit_connectivity=qubit_connectivity
            )
            pulse_channels_r = self._build_uncalibrated_resonator_pulse_channels()

            resonator = Resonator(
                physical_channel=physical_channel_r, pulse_channels=pulse_channels_r
            )
            qubit = Qubit(
                physical_channel=physical_channel_q,
                pulse_channels=pulse_channels_q,
                resonator=resonator,
            )

            qubits[qubit_id] = qubit

        return PhysicalHardwareModel(
            qubits=qubits,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
            physical_connectivity=physical_connectivity,
        )

    def _build_uncalibrated_baseband(self):
        return PhysicalBaseband()

    def _build_uncalibrated_physical_channel(
        self,
        baseband: PhysicalBaseband,
        target_device: type[Qubit | Resonator],
        name_index: int,
    ):
        if target_device is Qubit:
            return QubitPhysicalChannel(baseband=baseband, name_index=name_index)
        elif target_device is Resonator:
            return ResonatorPhysicalChannel(baseband=baseband, name_index=name_index)
        else:
            raise ValueError(f"Unsupported target device type: {target_device}")

    def _build_uncalibrated_qubit_pulse_channels(self, qubit_connectivity: list[QubitId]):
        cross_resonance_channels = FrozenDict(
            {
                q_other: CrossResonancePulseChannel(auxiliary_qubit=q_other)
                for q_other in qubit_connectivity
            }
        )
        cross_resonance_cancellation_channels = FrozenDict(
            {
                q_other: CrossResonanceCancellationPulseChannel(auxiliary_qubit=q_other)
                for q_other in qubit_connectivity
            }
        )

        pulse_channels = QubitPulseChannels(
            drive=DrivePulseChannel(),
            freq_shift=FreqShiftPulseChannel(),
            second_state=SecondStatePulseChannel(),
            cross_resonance_channels=cross_resonance_channels,
            cross_resonance_cancellation_channels=cross_resonance_cancellation_channels,
        )
        return pulse_channels

    def _build_uncalibrated_resonator_pulse_channels(self):
        pulse_channels_r = ResonatorPulseChannels(
            measure=MeasurePulseChannel(), acquire=AcquirePulseChannel()
        )
        return pulse_channels_r

    def model_dump(self):
        return self.model.model_dump()
