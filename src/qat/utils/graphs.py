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
