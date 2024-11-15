import itertools as it
import random
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest
from pydantic import ValidationError

from qat.model.builder import QuantumHardwareModelBuilder
from qat.model.device import PulseChannel
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

    return {node: set(neighbors) for node, neighbors in G.adjacency()}


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 10, 32])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
class Test_HW_Serialisation:
    def test_built_model_serialises(self, n_qubits, seed):
        builder = QuantumHardwareModelBuilder(
            topology=random_topology(n=n_qubits, max_degree=3, seed=seed)
        )

        hw1 = builder.model
        hw2 = QuantumHardwareModel(**hw1.model_dump())

        assert hw1 == hw2

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


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 10, 32])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
class Test_HW_Calibration:
    def test_model_calibration(self, n_qubits, seed):
        hw = QuantumHardwareModelBuilder(
            topology=random_topology(n=n_qubits, max_degree=3, seed=seed)
        ).model
        assert hw.number_of_qubits == n_qubits
        assert not hw.calibrated

        hw2 = randomly_calibrate(hardware_model=hw, seed=seed)
        assert hw2.calibrated

    def test_model_calibration_serialises(self, n_qubits, seed):
        hw1 = QuantumHardwareModelBuilder(
            topology=random_topology(n=n_qubits, max_degree=3, seed=seed)
        ).model
        hw1 = randomly_calibrate(hardware_model=hw1, seed=seed)

        hw2 = QuantumHardwareModel(**hw1.model_dump())
        assert hw1 == hw2


@pytest.mark.parametrize("n_qubits", [8, 16, 32, 64])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
class Test_HW_Topology:
    def test_constrained_topology(self, n_qubits, seed):
        topology = random_topology(n=n_qubits, max_degree=3, seed=seed)
        constrained_topology = deepcopy(topology)

        constrained_qubits = random.Random(seed).sample(
            list(range(0, n_qubits)), int(np.sqrt(n_qubits))
        )
        for qubit in constrained_qubits:
            popped_edge = constrained_topology[qubit].pop()
            constrained_topology[popped_edge].remove(qubit)

        QuantumHardwareModelBuilder(topology=topology, constrained_topology=topology)

        QuantumHardwareModelBuilder(
            topology=topology, constrained_topology=constrained_topology
        )

        wrong_topology = deepcopy(topology)
        wrong_topology[0].add(n_qubits)
        wrong_topology[n_qubits] = {0}
        with pytest.raises(ValidationError):
            QuantumHardwareModelBuilder(
                topology=topology, constrained_topology=wrong_topology
            )

    def test_valid_topology(self, n_qubits, seed):
        valid_topology = random_topology(n=n_qubits, max_degree=3, seed=seed)

        wrong_qubit = random.Random(seed).sample(list(range(0, n_qubits)), 1)[0]
        wrong_topology = deepcopy(valid_topology)
        wrong_topology[wrong_qubit].pop()

        with pytest.raises(ValidationError):
            QuantumHardwareModelBuilder(topology=wrong_topology)

        with pytest.raises(ValidationError):
            QuantumHardwareModelBuilder(
                topology=valid_topology, constrained_topology=wrong_topology
            )
