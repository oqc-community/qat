# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import networkx as nx

from qat.purr.compiler.hardware_models import ErrorMitigation, ReadoutMitigation
from qat.utils.hardware_model import ensure_connected_connectivity, random_error_mitigation


def _graph_from_connectivity(connectivity: dict) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(connectivity.keys())
    for q1_index in connectivity.keys():
        for q2_index in filter(
            lambda x: x in connectivity.keys(), connectivity.get(q1_index, [])
        ):
            G.add_edge(q1_index, q2_index)
    return G


def test_ensure_connected_connectivity_fixes_unconnected():
    connectivity = {
        0: {1, 2},
        1: {0},
        2: {0, 3},
        3: {2},
    }
    qubit_indices = {1, 2, 3}
    initial_graph = _graph_from_connectivity({k: connectivity[k] for k in qubit_indices})
    assert not nx.is_connected(initial_graph)
    new_connectivity = ensure_connected_connectivity(connectivity, qubit_indices)
    connected_graph = _graph_from_connectivity(
        {k: new_connectivity[k] for k in qubit_indices}
    )
    assert nx.is_connected(connected_graph)
    assert connectivity != new_connectivity


def test_ensure_connected_connectivity_preserves_connected():
    connectivity = {
        0: {1, 2},
        1: {0},
        2: {0, 3},
        3: {2},
    }
    qubit_indices = {0, 1, 2}
    initial_graph = _graph_from_connectivity({k: connectivity[k] for k in qubit_indices})
    assert nx.is_connected(initial_graph)
    new_connectivity = ensure_connected_connectivity(connectivity, qubit_indices)
    connected_graph = _graph_from_connectivity(
        {k: new_connectivity[k] for k in qubit_indices}
    )
    assert nx.is_connected(connected_graph)
    assert connectivity == new_connectivity


def test_random_error_mitigation_without_seed():
    physical_indices = {1, 4, 5}
    error_mitigation = random_error_mitigation(physical_indices)
    assert isinstance(error_mitigation, ErrorMitigation)
    assert isinstance(error_mitigation.readout_mitigation, ReadoutMitigation)
    assert {
        int(key) for key in error_mitigation.readout_mitigation.linear.keys()
    } == physical_indices


def test_random_error_mitigation_with_same_seed():
    physical_indices = {1, 4, 5}
    error_mitigation_1 = random_error_mitigation(physical_indices, seed=42)
    error_mitigation_2 = random_error_mitigation(physical_indices, seed=42)
    assert (
        error_mitigation_1.readout_mitigation.linear
        == error_mitigation_2.readout_mitigation.linear
    )


def test_random_error_mitigation_with_different_seeds():
    physical_indices = {1, 4, 5}
    error_mitigation_1 = random_error_mitigation(physical_indices, seed=42)
    error_mitigation_2 = random_error_mitigation(physical_indices, seed=24)
    assert (
        error_mitigation_1.readout_mitigation.linear
        != error_mitigation_2.readout_mitigation.linear
    )
