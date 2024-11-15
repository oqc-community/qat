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
    QubitId,
    QubitPulseChannels,
    Resonator,
    ResonatorPulseChannels,
    SecondStatePulseChannel,
)
from qat.model.hardware_model import QuantumHardwareModel


class QuantumHardwareModelBuilder:
    def __init__(
        self,
        topology: dict[int, set[int]],
        constrained_topology: dict[int, set[int]] = None,
    ):
        self._current_model = self._build_uncalibrated_hardware_model_from_topology(
            topology, constrained_topology
        )

    @property
    def model(self):
        return self._current_model

    def _build_uncalibrated_hardware_model_from_topology(
        self,
        topology: dict[int, set[int]],
        constrained_topology: dict[int, set[int]] = None,
    ):
        qubits = {}
        for qubit_id, qubit_connectivity in topology.items():
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

        return QuantumHardwareModel(
            qubits=qubits, topology=topology, constrained_topology=constrained_topology
        )

    def _build_uncalibrated_baseband(self):
        return PhysicalBaseband()

    def _build_uncalibrated_physical_channel(self, baseband: PhysicalBaseband):
        return PhysicalChannel(baseband=baseband)

    def _build_uncalibrated_qubit_pulse_channels(self, qubit_connectivity: list[QubitId]):
        cross_resonance_channels = tuple(
            [
                CrossResonancePulseChannel(target_qubit=q_other)
                for q_other in qubit_connectivity
            ]
        )
        cross_resonance_cancellation_channels = tuple(
            [
                CrossResonanceCancellationPulseChannel(target_qubit=q_other)
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
