import itertools as it
import random
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from qat.model.builder import QuantumHardwareModelBuilder
from qat.model.hardware_model import VERSION, QuantumHardwareModel


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

    return {node: list(neighbors) for node, neighbors in G.adjacency()}


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 10, 32])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
class Test_HW_Builder:

    def test_built_model_serialises(self, n_qubits, seed):
        builder = QuantumHardwareModelBuilder(
            topology=random_topology(n=n_qubits, max_degree=3, seed=seed)
        )

        hw1 = builder.model
        hw2 = QuantumHardwareModel(**hw1.model_dump())

        assert (
            hw1 == hw2
        ), "Serialised and deserialised version of the hardware model must be equal."

        builder = QuantumHardwareModelBuilder(topology=random_topology(n_qubits))

    def test_dump_load_eq(self, n_qubits, seed):
        hw1 = QuantumHardwareModelBuilder(
            topology=random_topology(n=n_qubits, max_degree=3, seed=seed)
        ).model
        blob = hw1.model_dump()

        hw2 = QuantumHardwareModel(**blob)
        assert hw1 == hw2

        hw3 = QuantumHardwareModelBuilder(
            topology=random_topology(n=n_qubits, max_degree=3, seed=54389)
        ).model
        assert hw1 != hw3

    def test_dump_eq(self, n_qubits, seed):
        hw1 = QuantumHardwareModelBuilder(
            topology=random_topology(n=n_qubits, max_degree=3, seed=seed)
        ).model
        blob1 = hw1.model_dump()

        hw2 = QuantumHardwareModel(**blob1)
        blob2 = hw2.model_dump()

        hw3 = QuantumHardwareModelBuilder(
            topology=random_topology(n=n_qubits, max_degree=3, seed=seed)
        ).model
        blob3 = hw3.model_dump()

        assert blob1 == blob2
        assert blob1 != blob3

    def test_deep_equals(self, n_qubits, seed):
        hw1 = QuantumHardwareModelBuilder(
            topology=random_topology(n=n_qubits, max_degree=3, seed=seed)
        ).model
        hw2 = deepcopy(hw1)

        assert hw1 == hw2

        index = random.Random(seed).choice(list(hw2.qubits.keys()))
        hw2.qubit_with_index(index).pulse_channels.drive.frequency = random.Random(
            seed
        ).uniform(1e08, 1e10)
        assert hw1 != hw2

    def test_deserialise_version(self, n_qubits, seed):
        hw1 = QuantumHardwareModelBuilder(
            topology=random_topology(n=n_qubits, max_degree=3, seed=seed)
        ).model
        assert hw1.version == VERSION

        hw2 = QuantumHardwareModel(**hw1.model_dump())
        assert hw2.version == VERSION

    def test_built_model_calibration(self, n_qubits, seed):
        hw = QuantumHardwareModelBuilder(
            topology=random_topology(n=n_qubits, max_degree=3, seed=seed)
        ).model
        assert not hw.calibrated, "Default hardware model must be uncalibrated."

        for qubit in hw.qubits.values():

            for field_name in qubit.physical_channel.model_fields:
                field_value = getattr(qubit.physical_channel, field_name)
                if isinstance(field_value, float) and np.isnan(field_value):
                    qubit.physical_channel.model_fields
