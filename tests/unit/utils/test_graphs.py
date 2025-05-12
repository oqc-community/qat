# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import json
import random

import networkx as nx
import numpy as np
import pytest

from qat.purr.integrations.tket import get_coupling_subgraphs
from qat.utils.graphs import get_connected_subgraphs


@pytest.mark.parametrize("n", [0, 1, 2, 8, 32, 64, 256])
class TestLegacyAndNetworkxSubgraphsMatch:
    def test_single_connected_subgraph(self, n):
        G_connected = nx.circular_ladder_graph(n=n)
        connected_edges = list(G_connected.edges)

        legacy_subgraphs = get_coupling_subgraphs(connected_edges)
        nx_subgraphs_nodes, nx_subgraphs_edges = get_connected_subgraphs(connected_edges)

        assert len(legacy_subgraphs) == len(nx_subgraphs_nodes)
        assert len(legacy_subgraphs) == len(nx_subgraphs_edges)

        for edges in nx_subgraphs_edges:
            G = nx.from_edgelist(edges)
            assert nx.is_connected(G)

    def test_two_connected_subgraphs(self, n):
        G1_connected = nx.gnm_random_graph(n, n * 2)
        G2_connected = nx.gnm_random_graph(n, n * 4)
        G = nx.disjoint_union_all([G1_connected, G2_connected])

        disconnected_edges = list(G.edges)

        legacy_subgraphs = get_coupling_subgraphs(disconnected_edges)
        nx_subgraphs_nodes, nx_subgraphs_edges = get_connected_subgraphs(disconnected_edges)

        assert len(legacy_subgraphs) == len(nx_subgraphs_nodes)
        assert len(legacy_subgraphs) == len(nx_subgraphs_edges)

        for edges in nx_subgraphs_edges:
            G = nx.from_edgelist(edges)
            assert nx.is_connected(G)

    def test_caveman_graph(self, n):
        G_caveman = nx.caveman_graph(l=int(np.sqrt(n)), k=int(np.sqrt(n)))
        disconnected_edges = list(G_caveman.edges)

        legacy_subgraphs = get_coupling_subgraphs(disconnected_edges)
        nx_subgraphs_nodes, nx_subgraphs_edges = get_connected_subgraphs(disconnected_edges)

        assert len(legacy_subgraphs) == len(nx_subgraphs_nodes)
        assert len(legacy_subgraphs) == len(nx_subgraphs_edges)

        for edges in nx_subgraphs_edges:
            G = nx.from_edgelist(edges)
            assert nx.is_connected(G)


with open("tests/files/hardware/toshiko_lattice_connections.json", "r") as f:
    toshiko_edges = []

    connections = json.load(f)
    for c in connections["connections"]:
        toshiko_edges.append((c[0], c[1]))


class TestToshiko:
    def test_toshiko_graph(self):
        legacy_subgraphs = get_coupling_subgraphs(toshiko_edges)
        nx_subgraphs_nodes, nx_subgraphs_edges = get_connected_subgraphs(toshiko_edges)

        assert len(legacy_subgraphs) == 1
        assert len(nx_subgraphs_nodes) == 1
        assert len(nx_subgraphs_edges) == 1

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 8, 10])
    def test_defective_toshiko(self, k):
        defective_edges = list(random.choices(toshiko_edges, k=len(toshiko_edges) - k))

        legacy_subgraphs = get_coupling_subgraphs(defective_edges)
        nx_subgraphs_nodes, nx_subgraphs_edges = get_connected_subgraphs(defective_edges)

        assert len(legacy_subgraphs) == len(nx_subgraphs_nodes)
        assert len(legacy_subgraphs) == len(nx_subgraphs_edges)

        for nx_subgr in nx_subgraphs_edges:
            for edge in nx_subgr:
                assert edge in toshiko_edges

        # Couplings
        for leg_subgr in legacy_subgraphs:
            found_match = False
            for nx_subgr in nx_subgraphs_edges:
                if set(leg_subgr) == set(nx_subgr):
                    found_match = True
            assert found_match
