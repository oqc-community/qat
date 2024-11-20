from typing import Optional

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
from qat.model.hardware_base import QubitId
from qat.model.hardware_model import PhysicalHardwareModel


class PhysicalHardwareModelBuilder:
    """
    A builder class that builds a physical hardware model based on the given connectivity.

    :param physical_connectivity: The connectivities of the physical qubits on the QPU.
    :param logical_connectivity: The connectivities of the qubits used for compilation,
                            which is equal to `physical_connectivity` or a subset thereof.
    """

    def __init__(
        self,
        physical_connectivity: dict[int, set[int]],
        logical_connectivity: Optional[dict[int, set[int]]] = None,
    ):
        self._current_model = self._build_uncalibrated_hardware_model(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
        )

    @property
    def model(self):
        return self._current_model

    def _build_uncalibrated_hardware_model(
        self,
        physical_connectivity: dict[int, set[int]],
        logical_connectivity: dict[int, set[int]] = None,
    ):
        logical_connectivity = logical_connectivity or physical_connectivity

        unique_qubit_indices = set(physical_connectivity.keys()).union(
            *physical_connectivity.values()
        )

        qubits = {}
        for qubit_id in unique_qubit_indices:
            qubit_connectivity = physical_connectivity.get(qubit_id, set())

            bb_q = self._build_uncalibrated_baseband()
            bb_r = self._build_uncalibrated_baseband()

            physical_channel_q = self._build_uncalibrated_physical_channel(baseband=bb_q)
            physical_channel_r = self._build_uncalibrated_physical_channel(baseband=bb_r)

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
            physical_connectivity=physical_connectivity,
        )

    def _build_uncalibrated_baseband(self):
        return PhysicalBaseband()

    def _build_uncalibrated_physical_channel(self, baseband: PhysicalBaseband):
        return PhysicalChannel(baseband=baseband)

    def _build_uncalibrated_qubit_pulse_channels(self, qubit_connectivity: list[QubitId]):
        cross_resonance_channels = tuple(
            [
                CrossResonancePulseChannel(auxiliary_qubit=q_other)
                for q_other in qubit_connectivity
            ]
        )
        cross_resonance_cancellation_channels = tuple(
            [
                CrossResonanceCancellationPulseChannel(auxiliary_qubit=q_other)
                for q_other in qubit_connectivity
            ]
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
