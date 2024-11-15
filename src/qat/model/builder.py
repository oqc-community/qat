from qat.model.device import (
    AcquirePulseChannel,
    CrossResonanceCancellationPulseChannel,
    CrossResonancePulseChannel,
    DrivePulseChannel,
    FreqShiftPulseChannel,
    MeasurePulseChannel,
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
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


import itertools as it
import random

import networkx as nx


def random_topology(n, max_degree=3, seed=42):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    edges = list(it.combinations(range(n), 2))
    random.Random(seed).shuffle(edges)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for node_edges in edges:
        if (
            len(G.edges(node_edges[0])) < max_degree
            and len(G.edges(node_edges[1])) < max_degree
        ):
            G.add_edge(*node_edges)

    return {node: set(neighbors) for node, neighbors in G.adjacency()}


def randomly_calibrate(hardware_model: QuantumHardwareModel, seed=42):
    for qubit in hardware_model.qubits.values():
        # Calibrate physical channel.
        for physical_channel in [qubit.physical_channel, qubit.resonator.physical_channel]:
            physical_channel.sample_time = random.Random(seed).uniform(1e-08, 1e-10)
            physical_channel.baseband.frequency = random.Random(seed).uniform(1e05, 1e07)
            physical_channel.baseband.if_frequency = random.Random(seed).uniform(1e05, 1e07)

        # Calibrate qubit and resonator pulse channels.
        for pulse_channels in [qubit.pulse_channels, qubit.resonator.pulse_channels]:
            for pulse_channel_name in pulse_channels.model_fields:
                pulse_channel = getattr(pulse_channels, pulse_channel_name)
                if isinstance(pulse_channel, PulseChannel):
                    pulse_channel.frequency = random.Random(seed).uniform(1e08, 1e10)
                elif isinstance(pulse_channel, tuple):
                    for sub_pulse_channel in pulse_channel:
                        sub_pulse_channel.frequency = random.Random(seed).uniform(
                            1e08, 1e10
                        )

    return hardware_model


n_qubits = 8
seed = 42

hw1 = QuantumHardwareModelBuilder(
    topology=random_topology(n=n_qubits, max_degree=3, seed=seed)
).model
hw1 = randomly_calibrate(hardware_model=hw1, seed=seed)

hw2 = QuantumHardwareModel(**hw1.model_dump())

eqs = hw1 == hw2
