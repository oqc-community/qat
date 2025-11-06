# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import hashlib
import itertools as it
import random
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from qat.model.builder import PhysicalHardwareModelBuilder
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.compiler.devices import (
    ChannelType,
    PhysicalBaseband,
    PhysicalChannel,
    PulseShapeType,
    Qubit,
    QubitCoupling,
    Resonator,
)
from qat.purr.compiler.hardware_models import (
    ErrorMitigation,
    QuantumHardwareModel,
    ReadoutMitigation,
)
from qat.utils.pydantic import CalibratableUnitInterval2x2Array


def random_connectivity(n, max_degree=3, seed=42):
    """
    Generates a random undirected graph but enforcing that the resulting graph is connected.
    """
    seeded_random = seed if isinstance(seed, random.Random) else random.Random(seed)
    edges = list(it.combinations(range(n), 2))
    seeded_random.shuffle(edges)
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


def random_quality_map(connectivity, seed=42, min_quality=0.0, max_quality=1.0):
    seeded_random = seed if isinstance(seed, random.Random) else random.Random(seed)
    coupling_map = {}
    for q1_index, connected_qubits in connectivity.items():
        for q2_index in connected_qubits:
            coupling_map[(q1_index, q2_index)] = seeded_random.uniform(
                min_quality, max_quality
            )
    return coupling_map


def random_error_mitigation(physicaal_indices, seed: int | None = None) -> ErrorMitigation:
    seeded_random = random.Random(seed)
    linear: dict = {}
    for q_id in physicaal_indices:
        p00 = seeded_random.uniform(0.8, 1.0)
        p11 = seeded_random.uniform(0.8, 1.0)
        linear[str(q_id)] = {
            "0|0": p00,
            "1|0": 1 - p00,
            "1|1": p11,
            "0|1": 1 - p11,
        }

    return ErrorMitigation(readout_mitigation=ReadoutMitigation(linear=linear))


def pick_subconnectivity(connectivity, n, seed=42):
    seeded_random = seed if isinstance(seed, random.Random) else random.Random(seed)
    sub_qubits = seeded_random.sample(list(connectivity.keys()), n)
    sub_connectivity = defaultdict(set)
    for qubit in sub_qubits:
        for connected_qubit in connectivity[qubit]:
            if qubit not in sub_connectivity[connected_qubit]:
                # TODO: Discussed that this should possibly be something like:
                # if connected_qubit in sub_qubits:
                # Evaluate if it should be changed
                # COMPILER-822
                sub_connectivity[qubit].add(connected_qubit)

    return sub_connectivity


def ensure_connected_connectivity(connectivity: dict, qubit_indices: list | set) -> dict:
    """Ensures that all `qubit_indices` are connected together.

    Looks at only the selected qubit indices and ensures that they are all part of a
    single connected graph. This does not mean an all-to-all connection.
    E.g. While the connectivity dict below has all qubits linked, in the subset of indices
    `{1, 2, 3}` qubit `0` is not linked with the other two.

    .. code:: python

        connectivity = {
            0: {1, 2},
            1: {0},
            2: {0, 3},
            3: {2},
        }
        qubit_indices = {1, 2, 3}
        new_connectivity = ensure_connected_connectivity(connectivity, qubit_indices)
        # new_connectivity == {
        #     0: {1, 2},
        #     1: {0, 2},
        #     2: {0, 1, 3},
        #     3: {2}
        # }

    :param connectivity: Base connectivity dictionary.
    :param qubit_indices: Selected qubits to ensure are a connected graph.
    """
    new_connectivity = {k: set(v) for k, v in connectivity.items()}
    G = nx.Graph()
    G.add_nodes_from(qubit_indices)
    for q1_index in qubit_indices:
        for q2_index in filter(
            lambda x: x in qubit_indices, new_connectivity.get(q1_index, [])
        ):
            G.add_edge(q1_index, q2_index)
    if not nx.is_connected(G):
        connected_generators = nx.connected_components(G)
        main_component = next(connected_generators)
        tail = list(main_component)[-1]
        for component in connected_generators:
            head = list(component)[0]
            new_connectivity[tail].add(head)
            new_connectivity[head].add(tail)
            G.add_edge(tail, head)
            G.add_edge(head, tail)
            tail = list(component)[-1]
        if not nx.is_connected(G):
            raise ValueError("The provided connectivity is not connected.")
    return new_connectivity


def generate_connectivity_data(n_qubits, n_logical_qubits, seed=42):
    physical_connectivity = random_connectivity(n=n_qubits, seed=seed)
    logical_connectivity = pick_subconnectivity(
        physical_connectivity, n=n_logical_qubits, seed=seed
    )
    logical_connectivity_quality = random_quality_map(
        connectivity=logical_connectivity, seed=seed
    )
    return (physical_connectivity, logical_connectivity, logical_connectivity_quality)


def generate_hw_model(n_qubits, seed=42):
    physical_connectivity, _, _ = generate_connectivity_data(n_qubits, n_qubits, seed=seed)

    builder = PhysicalHardwareModelBuilder(physical_connectivity=physical_connectivity)
    return builder.model


def apply_setup_to_echo_hardware(
    qubit_count: int, connectivity, qubit_indices: set = None
) -> QuantumHardwareModel:
    qubit_devices = []
    resonator_devices = []
    channel_index = 1
    hw = QuantumHardwareModel()
    if qubit_indices is not None:
        if (no_qubits_indices := len(qubit_indices)) > qubit_count:
            qubit_indices = it.islice(qubit_indices, qubit_count)
        elif no_qubits_indices < qubit_count:
            raise ValueError(
                f"Not enough qubit indices provided: len({qubit_indices}) = {no_qubits_indices} < {qubit_count}."
            )
    else:
        qubit_indices = range(qubit_count)
    for primary_index in qubit_indices:
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
        qubit.pulse_hw_x_pi_2 = {
            "shape": PulseShapeType.GAUSSIAN,
            "width": random.Random().uniform(50e-9, 1000e-9),
            "rise": random.Random().uniform(1 / 6, 1 / 2),
            "amp": random.Random().uniform(1e6, 5e6),
            "drag": random.Random().uniform(0, 0.5),
            "phase": random.Random().uniform(0, 1),
        }

        qubit_devices.append(qubit)
        resonator_devices.append(resonator)
        channel_index = channel_index + 2

    qubits_by_index = {qb.index: qb for qb in qubit_devices}
    if isinstance(connectivity, list):
        connectivity = dict.fromkeys(connectivity, None)
    for connection, quality in connectivity.items():
        left_index, right_index = connection
        qubit_left = qubits_by_index.get(left_index, None)
        qubit_right = qubits_by_index.get(right_index, None)

        if not all([qubit_left, qubit_right]):
            continue

        qubit_left.create_pulse_channel(
            auxiliary_devices=[qubit_right],
            channel_type=ChannelType.cross_resonance,
            frequency=5.5e9,
            scale=np.complex128(50 + 50j),
        )
        qubit_left.create_pulse_channel(
            auxiliary_devices=[qubit_right],
            channel_type=ChannelType.cross_resonance_cancellation,
            frequency=5.5e9,
            scale=np.complex128(5 + 5j),
        )
        qubit_left.add_coupled_qubit(qubit_right)
        qubit_left.pulse_hw_zx_pi_4[qubit_right.full_id()] = {
            "shape": random.choice(
                [PulseShapeType.SOFT_SQUARE, PulseShapeType.SOFTER_SQUARE]
            ),
            "width": random.Random().uniform(100e-9, 250e-9),
            "rise": random.Random().uniform(1e-9, 100e-9),
            "amp": random.Random().uniform(1e6, 5e6),
            "drag": random.Random().uniform(0, 0.5),
            "phase": random.Random().uniform(0, 1),
        }
        hw.qubit_direction_couplings.append(
            QubitCoupling(
                direction=connection, quality=quality or random.Random().uniform(1, 100)
            )
        )

    hw.add_quantum_device(*qubit_devices, *resonator_devices)
    hw.is_calibrated = True
    return hw


def generate_random_linear(qubit_indices):
    output = {}
    for index in qubit_indices:
        random_0 = random.random()
        random_1 = random.random()
        output[index] = CalibratableUnitInterval2x2Array(
            [[random_0, 1 - random_1], [1 - random_0, random_1]]
        )
    return output


def check_type_legacy_or_pydantic(hw_model: QuantumHardwareModel | PydHardwareModel):
    if not isinstance(hw_model, QuantumHardwareModel | PydHardwareModel | None):
        raise TypeError(
            f"Invalid type for the hardware model: {hw_model.__class__.__name__}. Please provide a `QuantumHardwareModel` or `PhysicalHardwareModel`."
        )

    return hw_model


def hash_calibration_file(file_path: str | Path, algorithm="md5", chunk_size=8_192) -> str:
    """Compute the hash of a calibration file."""
    file_path = Path(file_path)

    hash_func = hashlib.new(algorithm)
    with file_path.open("rb") as file:
        # Byte chunks from the file and update hash.
        while chunk := file.read(chunk_size):
            hash_func.update(chunk)

    return hash_func.hexdigest()
