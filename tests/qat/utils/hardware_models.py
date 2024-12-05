import itertools as it
import random
from copy import deepcopy

import networkx as nx

from qat.purr.compiler.devices import (
    ChannelType,
    PhysicalBaseband,
    PhysicalChannel,
    Qubit,
    QubitCoupling,
    Resonator,
)
from qat.purr.compiler.hardware_models import QuantumHardwareModel


def random_connectivity(n, max_degree=3, seed=42):
    """
    Generates a random undirected graph but enforcing that the resulting graph is connected.
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


def random_directed_connectivity(n, max_degree=3, seed=42):
    """
    Generates a random directed graph but enforcing that the resulting graph is connected.
    """
    edges = list(it.combinations(range(n), 2))
    random.Random(seed).shuffle(edges)
    G = nx.DiGraph()
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


def apply_setup_to_echo_hardware(qubit_count: int, connectivity) -> QuantumHardwareModel:
    qubit_devices = []
    resonator_devices = []
    channel_index = 1

    qubit_devices = []
    resonator_devices = []
    channel_index = 1
    hw = QuantumHardwareModel()
    for primary_index in range(qubit_count):
        bb1 = PhysicalBaseband(f"LO{channel_index}", 5.5e9)
        bb2 = PhysicalBaseband(f"LO{channel_index + 1}", 8.5e9)
        hw.add_physical_baseband(bb1, bb2)

        ch1 = PhysicalChannel(f"CH{channel_index}", 1.0e-9, bb1, 1)
        ch2 = PhysicalChannel(
            f"CH{channel_index + 1}", 1.0e-9, bb2, 1, acquire_allowed=True
        )
        hw.add_physical_channel(ch1, ch2)

        resonator = Resonator(f"R{primary_index}", ch2)
        resonator.create_pulse_channel(ChannelType.measure, frequency=8.5e9)
        resonator.create_pulse_channel(ChannelType.acquire, frequency=8.5e9)

        qubit = Qubit(primary_index, resonator, ch1)
        qubit.create_pulse_channel(ChannelType.drive, frequency=5.5e9)

        qubit_devices.append(qubit)
        resonator_devices.append(resonator)
        channel_index = channel_index + 2

    qubits_by_index = {qb.index: qb for qb in qubit_devices}
    for connection in connectivity:
        left_index, right_index = connection
        qubit_left = qubits_by_index.get(left_index)
        qubit_right = qubits_by_index.get(right_index)

        qubit_left.create_pulse_channel(
            auxiliary_devices=[qubit_right],
            channel_type=ChannelType.cross_resonance,
            frequency=5.5e9,
            scale=50,
        )
        qubit_left.create_pulse_channel(
            auxiliary_devices=[qubit_right],
            channel_type=ChannelType.cross_resonance_cancellation,
            frequency=5.5e9,
            scale=0.0,
        )
        qubit_left.add_coupled_qubit(qubit_right)
        hw.qubit_direction_couplings.append(QubitCoupling(connection))

    hw.add_quantum_device(*qubit_devices, *resonator_devices)
    hw.is_calibrated = True
    return hw
