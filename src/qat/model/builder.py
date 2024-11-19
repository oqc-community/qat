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
    QubitId,
    QubitPulseChannels,
    Resonator,
    ResonatorPulseChannels,
    SecondStatePulseChannel,
)
from qat.model.hardware_model import PhysicalHardwareModel


class PhysicalHardwareModelBuilder:
    """
    A builder class that builds a physical hardware model based on the given topology.

    :param physical_topology: The connectivities of the physical qubits on the QPU.
    :param logical_topology: The connectivities of the qubits used for compilation,
                            which is equal to `physical_topology` or a subset thereof.
    """

    def __init__(
        self,
        physical_topology: dict[int, set[int]],
        logical_topology: Optional[dict[int, set[int]]] = None,
    ):
        self._current_model = self._build_uncalibrated_hardware_model_from_topology(
            physical_topology=physical_topology,
            logical_topology=logical_topology,
        )

    @property
    def model(self):
        return self._current_model

    def _build_uncalibrated_hardware_model_from_topology(
        self,
        physical_topology: dict[int, set[int]],
        logical_topology: dict[int, set[int]] = None,
    ):
        logical_topology = logical_topology or physical_topology

        qubits = {}
        for qubit_id, qubit_connectivity in physical_topology.items():
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
            logical_topology=logical_topology,
            physical_topology=physical_topology,
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


topology = random_topology(n=4, max_degree=3, seed=0)
builder = PhysicalHardwareModelBuilder(physical_topology=topology)
hw = builder.model

PhysicalHardwareModel(**hw.model_dump())
