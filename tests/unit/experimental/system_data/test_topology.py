# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import networkx as nx
import numpy as np
import pytest

from qat.experimental.system_data.canonical.schema import (
    CanonicalSystemData,
    QubitCouplingData,
    QubitData,
    TwoQubitGateFidelityData,
)
from qat.experimental.system_data.topology import ScipyTopologyView, TopologyView


def test_canonical_graph_view_from_canonical_with_single_directed_coupling():
    """Build a minimal topology view from two qubits and one directed coupling."""

    canonical = CanonicalSystemData(
        qubits=(
            QubitData(id="q0", index=0),
            QubitData(id="q1", index=1),
        ),
        couplings=(
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="q1",
                gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.99),),
            ),
        ),
    )

    graph = TopologyView.from_canonical(canonical)

    assert graph.node_ids == ("q0", "q1")
    assert np.array_equal(graph.indptr, np.array([0, 1, 1], dtype=np.int_))
    assert np.array_equal(graph.indices, np.array([1], dtype=np.int_))

    row_indices = np.repeat(
        np.arange(len(graph.node_ids), dtype=np.int_),
        np.diff(graph.indptr),
    )
    # checks the csr representation is consistent with the graph structure
    adjacency_from_csr = np.zeros((len(graph.node_ids), len(graph.node_ids)), dtype=np.int_)
    adjacency_from_csr[row_indices, graph.indices] = 1
    assert np.array_equal(
        adjacency_from_csr,
        np.array([[0, 1], [0, 0]], dtype=np.int_),
    )

    assert len(graph.edge_couplings_data) == 1
    assert graph.edge_metadata == (("cx",),)
    assert np.allclose(graph.edge_couplings_data, ((0.99,),))
    assert graph.row_by_node_id["q0"] == 0
    assert graph.row_by_node_id["q1"] == 1
    assert graph.are_coupled("q0", "q1")
    assert not graph.are_coupled("q1", "q0")
    assert len(graph.sorted_gate_fidelities) == 1
    assert graph.sorted_gate_fidelities[0][0] == pytest.approx(0.99)
    assert graph.sorted_gate_fidelities[0][1:] == (("q0", "q1"), "cx")


def test_canonical_graph_view_raises_on_missing_qubit_data():
    """Verify that TopologyView raises an error if no qubit data is provided."""

    canonical = CanonicalSystemData(
        qubits=(),  # Empty qubits
        couplings=(
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="q1",
                gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.99),),
            ),
        ),
    )

    with pytest.raises(ValueError, match="must have qubit information"):
        TopologyView.from_canonical(canonical)


def test_canonical_graph_view_raises_on_missing_coupling_data():
    """Verify that TopologyView raises an error if no coupling data is provided."""

    canonical = CanonicalSystemData(
        qubits=(
            QubitData(id="q0", index=0),
            QubitData(id="q1", index=1),
        ),
        couplings=(),  # Empty couplings
    )

    with pytest.raises(ValueError, match="must have coupling information"):
        TopologyView.from_canonical(canonical)


@pytest.mark.parametrize(
    ("source_id", "target_id", "match"),
    [
        (
            "q0",
            "q_missing",
            r"Coupling target_qubit_id .* not found in qubits",
        ),
        (
            "q_missing",
            "q0",
            r"Coupling source_qubit_id .* not found in qubits",
        ),
    ],
    ids=["missing-target", "missing-source"],
)
def test_topology_view_raises_on_coupling_referencing_unknown_qubit(
    source_id: str,
    target_id: str,
    match: str,
):
    canonical = CanonicalSystemData(
        qubits=(QubitData(id="q0", index=0),),
        couplings=(
            QubitCouplingData(
                source_qubit_id=source_id,
                target_qubit_id=target_id,
                gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.99),),
            ),
        ),
    )

    with pytest.raises(ValueError, match=match):
        TopologyView.from_canonical(canonical)


def _build_canonical_from_directed_edges(
    node_ids: tuple[str, ...],
    directed_edges: set[tuple[str, str]],
) -> CanonicalSystemData:
    """Build canonical data from node IDs and directed coupling edges."""

    id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
    qubits = tuple(QubitData(id=node_id, index=i) for i, node_id in enumerate(node_ids))
    couplings = tuple(
        QubitCouplingData(
            source_qubit_id=source,
            target_qubit_id=target,
            gate_fidelities=(
                TwoQubitGateFidelityData(
                    gate="cx",
                    fidelity=0.90 + 0.01 * ((source_i + target_i) % 10),
                ),
                TwoQubitGateFidelityData(
                    gate="cz",
                    fidelity=0.88 + 0.01 * ((source_i + target_i) % 10),
                ),
            ),
        )
        for source, target in sorted(directed_edges)
        for source_i in (id_to_index[source],)
        for target_i in (id_to_index[target],)
    )
    return CanonicalSystemData(qubits=qubits, couplings=couplings)


def test_canonical_graph_view_from_canonical_with_four_qubit_connectivity_pattern():
    """Build a 4-qubit topology view and validate CSR, metadata, and connectivity."""

    node_ids = ("q1", "q2", "q3", "q4")
    directed_edges = {
        ("q1", "q2"),
        ("q2", "q1"),
        ("q1", "q4"),
        ("q4", "q1"),
        ("q2", "q3"),
        ("q3", "q2"),
        ("q2", "q4"),
        ("q4", "q2"),
    }
    canonical = _build_canonical_from_directed_edges(node_ids, directed_edges)

    graph = TopologyView.from_canonical(canonical)

    assert graph.node_ids == node_ids
    assert np.array_equal(graph.indptr, np.array([0, 2, 5, 6, 8], dtype=np.int_))
    assert np.array_equal(graph.indices, np.array([1, 3, 0, 2, 3, 1, 0, 1], dtype=np.int_))

    row_indices = np.repeat(
        np.arange(len(graph.node_ids), dtype=np.int_),
        np.diff(graph.indptr),
    )
    adjacency_from_csr = np.zeros((len(graph.node_ids), len(graph.node_ids)), dtype=np.int_)
    adjacency_from_csr[row_indices, graph.indices] = 1
    assert np.array_equal(
        adjacency_from_csr,
        np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=np.int_,
        ),
    )

    assert len(graph.edge_couplings_data) == 8
    assert graph.edge_metadata == (
        ("cx", "cz"),
        ("cx", "cz"),
        ("cx", "cz"),
        ("cx", "cz"),
        ("cx", "cz"),
        ("cx", "cz"),
        ("cx", "cz"),
        ("cx", "cz"),
    )
    expected_edge_data = (
        (0.91, 0.89),
        (0.93, 0.91),
        (0.91, 0.89),
        (0.93, 0.91),
        (0.94, 0.92),
        (0.93, 0.91),
        (0.93, 0.91),
        (0.94, 0.92),
    )
    assert np.allclose(graph.edge_couplings_data, expected_edge_data)
    assert graph.row_by_node_id["q1"] == 0
    assert graph.row_by_node_id["q2"] == 1
    assert graph.row_by_node_id["q3"] == 2
    assert graph.row_by_node_id["q4"] == 3
    assert graph.are_coupled("q2", "q4")
    assert not graph.are_coupled("q1", "q3")
    assert len(graph.sorted_gate_fidelities) == 16
    assert graph.sorted_gate_fidelities[0][0] == pytest.approx(0.94)
    assert graph.sorted_gate_fidelities[0][1:] == (("q2", "q4"), "cx")
    assert graph.nnz == 8


def test_sorted_gate_fidelities_by_type_grouping_and_ordering():
    """Validate gate-type grouping and descending fidelity order within each type."""

    node_ids = ("q1", "q2", "q3", "q4")
    directed_edges = {
        ("q1", "q2"),
        ("q2", "q1"),
        ("q1", "q4"),
        ("q4", "q1"),
        ("q2", "q3"),
        ("q3", "q2"),
        ("q2", "q4"),
        ("q4", "q2"),
    }
    canonical = _build_canonical_from_directed_edges(node_ids, directed_edges)
    graph = TopologyView.from_canonical(canonical)

    grouped = graph.sorted_gate_fidelities_by_type
    assert tuple(gate_name for gate_name, _ in grouped) == ("cx", "cz")

    for _, entries in grouped:
        assert all(entries[i][0] >= entries[i + 1][0] for i in range(len(entries) - 1))

    cx_entries = dict(grouped)["cx"]
    cz_entries = dict(grouped)["cz"]
    assert len(cx_entries) == 8
    assert len(cz_entries) == 8
    assert cx_entries[0][0] == pytest.approx(0.94)
    assert cx_entries[0][1] == ("q2", "q4")
    assert cz_entries[0][0] == pytest.approx(0.92)
    assert cz_entries[0][1] == ("q2", "q4")

    subset = graph.gate_fidelities_by_type(("cx",))
    assert tuple(gate_name for gate_name, _ in subset) == ("cx",)
    assert len(dict(subset)["cx"]) == 8


def test_scipy_topology_view_uses_last_duplicate_gate_entry_without_warning():
    """Use last occurrence when duplicate gate labels appear on one edge."""

    canonical = CanonicalSystemData(
        qubits=(
            QubitData(id="q0", index=0),
            QubitData(id="q1", index=1),
        ),
        couplings=(
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="q1",
                gate_fidelities=(
                    TwoQubitGateFidelityData(gate="cx", fidelity=0.90),
                    TwoQubitGateFidelityData(gate="cx", fidelity=0.95),
                    TwoQubitGateFidelityData(gate="cz", fidelity=0.85),
                ),
            ),
        ),
    )

    canonical_graph = TopologyView.from_canonical(canonical)
    derived = ScipyTopologyView.from_derived(canonical_graph)

    assert float(derived.gate_fidelity_matrices["cx"][0, 1]) == pytest.approx(0.95)
    assert float(derived.gate_fidelity_matrices["cz"][0, 1]) == pytest.approx(0.85)


def test_topology_view_raises_on_duplicate_qubit_ids():
    """Duplicate qubit identifiers should be rejected during node mapping."""

    canonical = CanonicalSystemData(
        qubits=(
            QubitData(id="q0", index=0),
            QubitData(id="q0", index=1),
        ),
        couplings=(
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="q0",
                gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.99),),
            ),
        ),
    )

    with pytest.raises(ValueError, match="Duplicate qubit id"):
        TopologyView.from_canonical(canonical)


def test_topology_view_raises_on_duplicate_directed_couplings():
    """Duplicate directed couplings should be rejected during CSR assembly."""

    canonical = CanonicalSystemData(
        qubits=(
            QubitData(id="q0", index=0),
            QubitData(id="q1", index=1),
        ),
        couplings=(
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="q1",
                gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.90),),
            ),
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="q1",
                gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.95),),
            ),
        ),
    )

    with pytest.raises(ValueError, match="Duplicate directed couplings"):
        TopologyView.from_canonical(canonical)


def test_derived_graph_view_from_canonical_with_four_qubit_connectivity_pattern():
    """Build a 4-qubit topology view and validate CSR, metadata, and connectivity."""

    node_ids = ("q1", "q2", "q3", "q4")
    directed_edges = {
        ("q1", "q2"),
        ("q2", "q1"),
        ("q1", "q4"),
        ("q4", "q1"),
        ("q2", "q3"),
        ("q3", "q2"),
        ("q2", "q4"),
        ("q4", "q2"),
    }
    canonical = _build_canonical_from_directed_edges(node_ids, directed_edges)

    canonical_graph = TopologyView.from_canonical(canonical)
    derived = ScipyTopologyView.from_derived(canonical_graph)

    assert derived.adjacency_matrix.shape == (4, 4)
    assert derived.adjacency_matrix.nnz == 8
    assert np.array_equal(
        derived.adjacency_matrix.toarray(),
        np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=np.int_,
        ),
    )
    assert len(derived.gate_fidelity_matrices) == 2
    assert derived.gate_fidelity_matrices["cx"].shape == (4, 4)
    assert derived.gate_fidelity_matrices["cz"].shape == (4, 4)
    assert derived.gate_fidelity_matrices["cx"].nnz == 8
    assert derived.gate_fidelity_matrices["cz"].nnz == 8

    nx_graph_derived = derived.networkx_graph
    assert isinstance(nx_graph_derived, nx.DiGraph)
    assert nx_graph_derived.number_of_nodes() == 4
    assert nx_graph_derived.number_of_edges() == 8
    assert nx_graph_derived.has_edge(0, 1)
    assert nx_graph_derived.has_edge(0, 3)
    assert nx_graph_derived.has_edge(1, 2)
    assert nx_graph_derived.has_edge(1, 3)
    assert not nx_graph_derived.has_edge(0, 2)
    assert nx_graph_derived.edges[0, 1]["cx"] == pytest.approx(
        float(derived.gate_fidelity_matrices["cx"][0, 1])
    )
    assert nx_graph_derived.edges[0, 1]["cz"] == pytest.approx(
        float(derived.gate_fidelity_matrices["cz"][0, 1])
    )


def test_scipy_topology_view_uses_per_gate_sparsity_for_gate_matrices():
    """Gate matrices store only edges where that gate exists."""

    canonical = CanonicalSystemData(
        qubits=(
            QubitData(id="q0", index=0),
            QubitData(id="q1", index=1),
            QubitData(id="q2", index=2),
        ),
        couplings=(
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="q1",
                gate_fidelities=(
                    TwoQubitGateFidelityData(gate="cx", fidelity=0.91),
                    TwoQubitGateFidelityData(gate="cz", fidelity=0.87),
                ),
            ),
            QubitCouplingData(
                source_qubit_id="q1",
                target_qubit_id="q2",
                gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.93),),
            ),
        ),
    )

    canonical_graph = TopologyView.from_canonical(canonical)
    derived = ScipyTopologyView.from_derived(canonical_graph)

    assert derived.adjacency_matrix.nnz == 2
    assert derived.gate_fidelity_matrices["cx"].nnz == 2
    assert derived.gate_fidelity_matrices["cz"].nnz == 1
    assert float(derived.gate_fidelity_matrices["cx"][0, 1]) == pytest.approx(0.91)
    assert float(derived.gate_fidelity_matrices["cx"][1, 2]) == pytest.approx(0.93)
    assert float(derived.gate_fidelity_matrices["cz"][0, 1]) == pytest.approx(0.87)
    assert float(derived.gate_fidelity_matrices["cz"][1, 2]) == pytest.approx(0.0)


def test_scipy_topology_view_networkx_graph_returns_a_fresh_graph():
    """Mutating an exported graph does not affect later exports."""

    canonical = CanonicalSystemData(
        qubits=(
            QubitData(id="q0", index=0),
            QubitData(id="q1", index=1),
        ),
        couplings=(
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="q1",
                gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.91),),
            ),
        ),
    )

    canonical_graph = TopologyView.from_canonical(canonical)
    derived = ScipyTopologyView.from_derived(canonical_graph)

    mutated_graph = derived.networkx_graph
    mutated_graph.remove_edge(0, 1)

    fresh_graph = derived.networkx_graph
    assert fresh_graph.has_edge(0, 1)
    assert fresh_graph.edges[0, 1]["cx"] == pytest.approx(0.91)


# In the following tests we consider various lattice and hypercube graphs from NetworkX,
# and validate the conversion to canonical and derived graph views, along with connectivity
# queries and NetworkX export.


def _build_from_networkx_graph(
    graph: nx.Graph,
) -> tuple[CanonicalSystemData, dict[object, str]]:
    """Build canonical data from a NetworkX undirected graph.

    Returns canonical data plus mapping from original NetworkX node labels to generated
    qubit IDs.
    """

    nx_nodes = tuple(sorted(graph.nodes(), key=repr))
    node_ids = tuple(f"q{i}" for i in range(len(nx_nodes)))
    nx_to_qid = {node: node_ids[i] for i, node in enumerate(nx_nodes)}

    directed_edges: set[tuple[str, str]] = set()
    for source, target in graph.edges():
        source_q = nx_to_qid[source]
        target_q = nx_to_qid[target]
        directed_edges.add((source_q, target_q))
        directed_edges.add((target_q, source_q))

    canonical = _build_canonical_from_directed_edges(node_ids, directed_edges)
    return canonical, nx_to_qid


def _triangular_lattice(lattice_site: int) -> nx.Graph:
    """Build a triangular lattice graph with ``lattice_site``"""

    return nx.triangular_lattice_graph(lattice_site, lattice_site)


def _hexagonal_lattice(lattice_site: int) -> nx.Graph:
    """Build a hexagonal lattice graph with ``lattice_site``"""

    return nx.hexagonal_lattice_graph(lattice_site, lattice_site)


def _sqr_lattice_2d(lattice_site: int) -> nx.Graph:
    """Build a 2D square lattice graph."""

    return nx.grid_2d_graph(lattice_site, lattice_site)


def _hypercube_3d(dimension: int) -> nx.Graph:
    """Build a 3D hypercube graph."""

    return nx.hypercube_graph(dimension)


@pytest.mark.parametrize(
    ("builder", "lattice_site"),
    [
        (_triangular_lattice, 4),
        (_hexagonal_lattice, 4),
        (_sqr_lattice_2d, 4),
        (_hypercube_3d, 3),
    ],
    ids=["triangular", "hexagonal", "square-2d", "hypercube-3d"],
)
def test_canonical_graph_view_from_networkx_lattice_and_hypercube(
    builder,
    lattice_site: int,
):
    """Validate topology conversion and connectivity queries for NetworkX graphs."""

    nx_graph = builder(lattice_site)
    canonical, nx_to_qid = _build_from_networkx_graph(nx_graph)

    canonical_graph = TopologyView.from_canonical(canonical)

    # Validate CSR structure
    n_qubits = len(canonical_graph.node_ids)
    assert n_qubits == nx_graph.number_of_nodes()
    assert canonical_graph.indptr.size == n_qubits + 1
    assert canonical_graph.nnz == 2 * nx_graph.number_of_edges()  # Bidirectional

    # Validate node mappings
    assert len(canonical_graph.node_ids) == n_qubits
    assert set(canonical_graph.row_by_node_id.keys()) == set(canonical_graph.node_ids)

    # Validate connectivity: sample some edges from original graph
    for nx_node1, nx_node2 in list(nx_graph.edges())[: min(5, nx_graph.number_of_edges())]:
        qid1 = nx_to_qid[nx_node1]
        qid2 = nx_to_qid[nx_node2]
        assert canonical_graph.are_coupled(qid1, qid2)
        assert canonical_graph.are_coupled(qid2, qid1)

    # Validate gate fidelities exist
    assert len(canonical_graph.sorted_gate_fidelities) > 0


@pytest.mark.parametrize(
    ("builder", "lattice_site"),
    [
        (_triangular_lattice, 3),
        (_hexagonal_lattice, 2),
        (_sqr_lattice_2d, 2),
        (_hypercube_3d, 3),
    ],
    ids=["triangular", "hexagonal", "square-2d", "hypercube-3d"],
)
def test_scipy_topology_view_from_networkx_lattice_and_hypercube(
    builder,
    lattice_site: int,
):
    """Validate ScipyTopologyView CSR conversion and NetworkX export per graph family."""

    nx_graph = builder(lattice_site)
    canonical, nx_to_qid = _build_from_networkx_graph(nx_graph)

    canonical_graph = TopologyView.from_canonical(canonical)
    derived = ScipyTopologyView.from_derived(canonical_graph)

    n_qubits = nx_graph.number_of_nodes()
    assert derived.adjacency_matrix.shape == (n_qubits, n_qubits)
    assert derived.adjacency_matrix.nnz == 2 * nx_graph.number_of_edges()

    assert len(derived.gate_fidelity_matrices) > 0
    for gate_name, gate_matrix in derived.gate_fidelity_matrices.items():
        assert gate_matrix.shape == (n_qubits, n_qubits)
        assert gate_matrix.nnz <= 2 * nx_graph.number_of_edges()
        assert gate_name in {"cx", "cz"}

    nx_derived = derived.networkx_graph
    assert isinstance(nx_derived, nx.DiGraph)
    assert nx_derived.number_of_nodes() == n_qubits
    assert nx_derived.number_of_edges() == 2 * nx_graph.number_of_edges()

    edges = sorted(nx_graph.edges(), key=repr)
    for nx_node1, nx_node2 in edges[: min(3, len(edges))]:
        row1 = canonical_graph.row_by_node_id[nx_to_qid[nx_node1]]
        row2 = canonical_graph.row_by_node_id[nx_to_qid[nx_node2]]
        assert nx_derived.has_edge(row1, row2)
        assert nx_derived.has_edge(row2, row1)

        edge_attrs = nx_derived.edges[row1, row2]
        assert any(gate in edge_attrs for gate in ("cx", "cz"))


def test_scipy_topology_view_networkx_graph_is_mutable_but_regenerated():
    """Verify exported NetworkX graphs are mutable snapshots regenerated on access."""

    canonical = _build_canonical_from_directed_edges(
        ("q0", "q1", "q2"),
        {("q0", "q1")},
    )

    canonical_graph = TopologyView.from_canonical(canonical)
    derived = ScipyTopologyView.from_derived(canonical_graph)

    graph = derived.networkx_graph
    assert graph.has_edge(0, 1)
    assert not graph.has_edge(0, 2)

    original_cx = float(derived.gate_fidelity_matrices["cx"][0, 1])
    original_cz = float(derived.gate_fidelity_matrices["cz"][0, 1])

    graph.add_edge(0, 2, cx=0.95, cz=0.98)
    graph.edges[0, 1]["cx"] = 0.99
    graph.remove_edge(0, 1)

    assert graph.has_edge(0, 2)
    assert not graph.has_edge(0, 1)
    assert graph.edges[0, 2]["cx"] == pytest.approx(0.95)
    assert graph.edges[0, 2]["cz"] == pytest.approx(0.98)

    regenerated = derived.networkx_graph

    assert regenerated is not graph
    assert regenerated.has_edge(0, 1)
    assert not regenerated.has_edge(0, 2)
    assert regenerated.edges[0, 1]["cx"] == pytest.approx(original_cx)
    assert regenerated.edges[0, 1]["cz"] == pytest.approx(original_cz)


def test_scipy_topology_view_networkx_graph_returns_deepcopy():
    """Each exported NetworkX graph is a deep copy with independent edge attrs."""

    canonical = _build_canonical_from_directed_edges(
        ("q0", "q1"),
        {("q0", "q1")},
    )
    canonical_graph = TopologyView.from_canonical(canonical)
    derived = ScipyTopologyView.from_derived(canonical_graph)

    graph_a = derived.networkx_graph
    graph_b = derived.networkx_graph

    assert graph_a is not graph_b
    assert graph_a.edges[0, 1] is not graph_b.edges[0, 1]

    graph_a.edges[0, 1]["nested"] = {"history": [0.1, 0.2]}
    graph_a.edges[0, 1]["nested"]["history"].append(0.3)

    graph_c = derived.networkx_graph
    assert "nested" not in graph_b.edges[0, 1]
    assert "nested" not in graph_c.edges[0, 1]
