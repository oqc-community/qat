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
from qat.model.hardware_base import CalibratableUnitInterval, FrozenDict, FrozenSet, QubitId
from qat.model.hardware_model import PhysicalHardwareModel


class PhysicalHardwareModelBuilder:
    """
    A builder class that builds a physical hardware model based on the given connectivity.

    :param physical_connectivity: The connectivities of the physical qubits on the QPU (undirected graph).
    :param logical_connectivity: The connectivities (directed graph) of the qubits used for compilation, which can be a subgraph of `physical_connectivity`.
    :param logical_connectivity_quality: Quality of the connections between the qubits.
    """

    def __init__(
        self,
        physical_connectivity: dict[int, FrozenSet[int]],
        logical_connectivity: Optional[dict[int, FrozenSet[int]]] = None,
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
    def model(self):
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

        qubits = {}
        for qubit_id in unique_qubit_indices:
            qubit_connectivity = logical_connectivity.get(qubit_id, set())

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
            logical_connectivity_quality=logical_connectivity_quality,
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


import itertools as it
import random
from copy import deepcopy

import networkx as nx


def random_connectivity(n, max_degree=3, seed=42):
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


def random_quality_map(connectivity, seed=42):
    coupling_map = {}
    for q1_index, connected_qubits in connectivity.items():
        for q2_index in connected_qubits:
            coupling_map[(q1_index, q2_index)] = random.Random(seed).uniform(0.0, 1.0)
    return coupling_map


def pick_subconnectivity(connectivity, n, seed=42):
    sub_connectivity = deepcopy(connectivity)
    sub_qubits = random.Random(seed).sample(list(connectivity.keys()), n)
    for qubit in sub_qubits:
        popped_node = sub_connectivity[qubit].pop()
        sub_connectivity[popped_node].remove(qubit)

    return sub_connectivity


def generate_connectivity_data(n_qubits, n_logical_qubits, seed=42):
    physical_connectivity = random_connectivity(n=n_qubits, seed=seed)
    logical_connectivity = pick_subconnectivity(
        physical_connectivity, n=n_logical_qubits, seed=seed
    )
    logical_connectivity_quality = random_quality_map(
        connectivity=logical_connectivity, seed=seed
    )
    return (physical_connectivity, logical_connectivity, logical_connectivity_quality)


n_qubits = 2

physical_connectivity, logical_connectivity, logical_connectivity_quality = (
    generate_connectivity_data(n_qubits, 0)
)

builder = PhysicalHardwareModelBuilder(
    physical_connectivity=physical_connectivity,
    logical_connectivity=logical_connectivity,
    logical_connectivity_quality=logical_connectivity_quality,
)

hw1 = builder.model
