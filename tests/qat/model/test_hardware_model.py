import itertools as it
import json
import random
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest
from pydantic import ValidationError

from qat.model.builder import PhysicalHardwareModelBuilder
from qat.model.device import PulseChannel
from qat.model.hardware_model import VERSION, PhysicalHardwareModel


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
        popped_edge = sub_connectivity[qubit].pop()
        sub_connectivity[popped_edge].remove(qubit)

    return sub_connectivity


def generate_connectivity_data(n_qubits, n_logical_qubits, seed=42):
    physical_connectivity = random_connectivity(n=n_qubits, seed=seed)
    physical_connectivity_quality = random_quality_map(
        connectivity=physical_connectivity, seed=seed
    )
    logical_connectivity = pick_subconnectivity(
        physical_connectivity, n=n_logical_qubits, seed=seed
    )
    return (physical_connectivity, physical_connectivity_quality, logical_connectivity)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 10, 32])
@pytest.mark.parametrize("n_logical_qubits", [0, 2, 4])
@pytest.mark.parametrize("seed", [1, 2, 3])
class Test_HW_Serialisation:
    def test_built_model_serialises(self, n_qubits, n_logical_qubits, seed):
        physical_connectivity, physical_connectivity_quality, logical_connectivity = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        builder = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            physical_connectivity_quality=physical_connectivity_quality,
        )

        hw1 = builder.model
        hw2 = PhysicalHardwareModel(**hw1.model_dump())
        assert hw1 == hw2

    def test_built_logical_model_serialises(self, n_qubits, n_logical_qubits, seed):
        physical_connectivity, physical_connectivity_quality, logical_connectivity = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        builder = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            physical_connectivity_quality=physical_connectivity_quality,
        )

        hw1 = builder.model
        hw2 = PhysicalHardwareModel(**hw1.model_dump())
        assert hw1 == hw2

    def test_dump_load_eq(self, n_qubits, n_logical_qubits, seed):
        physical_connectivity, physical_connectivity_quality, logical_connectivity = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        hw1 = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            physical_connectivity_quality=physical_connectivity_quality,
        ).model
        blob = hw1.model_dump()

        hw2 = PhysicalHardwareModel(**blob)
        assert hw1 == hw2

        hw3 = PhysicalHardwareModelBuilder(
            physical_connectivity=random_connectivity(n=n_qubits, max_degree=3, seed=54389)
        ).model
        assert hw1 != hw3

    def test_dump_eq(self, n_qubits, n_logical_qubits, seed):
        physical_connectivity, physical_connectivity_quality, logical_connectivity = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        hw1 = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            physical_connectivity_quality=physical_connectivity_quality,
        ).model
        blob1 = hw1.model_dump()

        hw2 = PhysicalHardwareModel(**blob1)
        blob2 = hw2.model_dump()
        assert blob1 == blob2

        hw3 = PhysicalHardwareModelBuilder(
            physical_connectivity=random_connectivity(n=n_qubits, max_degree=3, seed=seed)
        ).model
        blob3 = hw3.model_dump()
        assert blob1 != blob3

    def test_deep_equals(self, n_qubits, n_logical_qubits, seed):
        physical_connectivity, physical_connectivity_quality, logical_connectivity = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        hw1 = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            physical_connectivity_quality=physical_connectivity_quality,
        ).model
        hw2 = deepcopy(hw1)

        assert hw1 == hw2

        index = random.Random(seed).choice(list(hw2.qubits.keys()))
        hw2.qubit_with_index(index).pulse_channels.drive.frequency = random.Random(
            seed
        ).uniform(1e08, 1e10)
        assert hw1 != hw2

    def test_deserialise_version(self, n_qubits, n_logical_qubits, seed):
        physical_connectivity, physical_connectivity_quality, logical_connectivity = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        hw1 = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            physical_connectivity_quality=physical_connectivity_quality,
        ).model
        assert hw1.version == VERSION

        hw2 = PhysicalHardwareModel(**hw1.model_dump())
        assert hw2.version == VERSION


def randomly_calibrate(hardware_model: PhysicalHardwareModel, seed=42):
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


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 10, 32])
@pytest.mark.parametrize("seed", [41, 42, 43])
class Test_HW_Calibration:
    def test_model_calibration(self, n_qubits, seed):
        hw = PhysicalHardwareModelBuilder(
            physical_connectivity=random_connectivity(n=n_qubits, max_degree=3, seed=seed)
        ).model
        assert hw.number_of_qubits == n_qubits
        assert not hw.is_calibrated

        hw2 = randomly_calibrate(hardware_model=hw, seed=seed)
        assert hw2.is_calibrated

    def test_model_calibration_serialises(self, n_qubits, seed):
        physical_connectivity, physical_connectivity_quality, logical_connectivity = (
            generate_connectivity_data(
                n_qubits, min(int(np.sqrt(n_qubits - 1)), n_qubits // 2), seed=seed
            )
        )

        hw1 = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            physical_connectivity_quality=physical_connectivity_quality,
        ).model
        hw1 = randomly_calibrate(hardware_model=hw1, seed=seed)

        hw2 = PhysicalHardwareModel(**hw1.model_dump())
        assert hw1 == hw2


@pytest.mark.parametrize("n_qubits", [8, 16, 32, 64])
@pytest.mark.parametrize("seed", [1, 2, 3])
class Test_HW_Connectivity:
    def test_constrained_connectivity_subgraph(self, n_qubits, seed):
        physical_connectivity, _, logical_connectivity = generate_connectivity_data(
            n_qubits, min(int(np.sqrt(n_qubits - 1)), n_qubits // 2), seed=seed
        )

        hw = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
        ).model
        assert hw.physical_connectivity == physical_connectivity
        assert hw.logical_connectivity == logical_connectivity

        wrong_connectivity = deepcopy(physical_connectivity)
        wrong_connectivity[0].add(n_qubits)
        wrong_connectivity[n_qubits] = {0}
        with pytest.raises(ValidationError):
            PhysicalHardwareModelBuilder(
                physical_connectivity=physical_connectivity,
                logical_connectivity=wrong_connectivity,
            )

    def test_invalid_connectivity_quality(self, n_qubits, seed):
        physical_connectivity, physical_connectivity_quality, logical_connectivity = (
            generate_connectivity_data(
                n_qubits, min(int(np.sqrt(n_qubits - 1)), n_qubits // 2), seed=seed
            )
        )

        hw = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            physical_connectivity_quality=physical_connectivity_quality,
        ).model

        for q_index in hw.physical_connectivity_quality:
            with pytest.raises(ValueError):
                hw.physical_connectivity_quality.update(
                    {q_index: random.Random(seed).uniform(-1.0, -0.001)}
                )

            with pytest.raises(ValueError):
                hw.physical_connectivity_quality.update(
                    {q_index: random.Random(seed).uniform(1.001, 100.0)}
                )


@pytest.mark.parametrize("seed", [500, 501, 502])
@pytest.mark.parametrize("n_removed_qubits", [1, 2, 3, 4])
class Test_OQC_Hardware:
    def test_lucy(self, seed, n_removed_qubits):
        n_qubits = 8
        qubit_indices = list(range(0, n_qubits))
        ring_architecture = {
            i: {qubit_indices[i - 1], qubit_indices[i % (n_qubits - 1) + 1]}
            for i in range(0, n_qubits)
        }
        ring_connectivity_quality = random_quality_map(ring_architecture, seed=seed)

        # Randomly remove 3 qubits from the GPU connectivity.
        removed_qubits = set(
            random.Random(seed).sample(tuple(qubit_indices), n_removed_qubits)
        )
        logical_connectivity = {
            k: deepcopy(v) for k, v in ring_architecture.items() if k not in removed_qubits
        }
        for connected_qubits in logical_connectivity.values():
            connected_qubits -= removed_qubits

        hw = PhysicalHardwareModelBuilder(
            physical_connectivity=ring_architecture,
            physical_connectivity_quality=ring_connectivity_quality,
        ).model
        assert hw.number_of_qubits == n_qubits
        assert not hw.is_calibrated

        hw = randomly_calibrate(hardware_model=hw, seed=seed)
        assert hw.is_calibrated

    def test_toshiko(self, seed, n_removed_qubits):
        lattice_connectivity = defaultdict(set)
        qubit_indices = set()

        filepath = "tests/qat/files/hardware/toshiko_lattice_connections.json"
        with open(filepath, "r") as f:
            connections = json.load(f)
            for c in connections["connections"]:
                lattice_connectivity[c[0]].add(c[1])
                qubit_indices.update([c[0], c[1]])
        lattice_connectivity = dict(lattice_connectivity)
        lattice_connectivity_quality = random_quality_map(lattice_connectivity, seed=seed)

        # Randomly remove 3 qubits from the GPU connectivity.
        removed_qubits = set(
            random.Random(seed).sample(tuple(qubit_indices), n_removed_qubits)
        )
        logical_connectivity = {
            k: deepcopy(v)
            for k, v in lattice_connectivity.items()
            if k not in removed_qubits
        }
        for connected_qubits in logical_connectivity.values():
            connected_qubits -= removed_qubits

        n_physical_qubits = len(
            set(lattice_connectivity.keys()).union(*lattice_connectivity.values())
        )
        n_logical_qubits = len(
            set(logical_connectivity.keys()).union(*logical_connectivity.values())
        )
        assert n_physical_qubits > n_logical_qubits

        builder = PhysicalHardwareModelBuilder(
            physical_connectivity=lattice_connectivity,
            logical_connectivity=logical_connectivity,
            physical_connectivity_quality=lattice_connectivity_quality,
        )
        hw = builder.model
        assert hw.number_of_qubits == n_physical_qubits
        assert hw.physical_connectivity == lattice_connectivity
        assert hw.logical_connectivity == logical_connectivity
        assert hw.physical_connectivity_quality == lattice_connectivity_quality
