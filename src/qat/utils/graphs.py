# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from collections import defaultdict

import networkx as nx


def get_connected_subgraphs(edges: list[tuple[int, int]]):
    """
    Given a list of edges in a graph, which can be disconnected,
    construct the connected subgraph(s) within a given graph.
    """
    G = nx.DiGraph(edges)

    subgraphs_nodes = []
    subgraphs_edges = []
    for subgr_nodes in nx.weakly_connected_components(G):
        subgraphs_nodes.append(subgr_nodes)
        subgraphs_edges.append(list(G.subgraph(subgr_nodes).edges()))

    return subgraphs_nodes, subgraphs_edges


def convert_edges_into_connectivity_dict(edges: list[tuple[int, int]]):
    connectivity = defaultdict(set)
    for edge in edges:
        connectivity[edge[0]].add(edge[1])
        connectivity[edge[1]].add(edge[0])

    return connectivity


def generate_cyclic_connectivity(n_nodes: int):
    G = nx.cycle_graph(n_nodes)
    return convert_edges_into_connectivity_dict(list(G.edges()))


def generate_complete_connectivity(n_nodes: int):
    G = nx.complete_graph(n_nodes)
    return convert_edges_into_connectivity_dict(list(G.edges()))
