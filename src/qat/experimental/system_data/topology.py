# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Topology view assembled from canonical system data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.derived.interface import DerivedViewInterface


@dataclass(frozen=True)
class TopologyView(DerivedViewInterface):
    """Minimal canonical topology view encoded in a CSR-like graph structure.

    Connectivity is stored structurally by ``indptr`` and ``indices``. Here,
    ``n_rows == n_qubits`` (one row per source qubit), and the implied adjacency/fidelity
    matrix is square with shape ``(n_qubits, n_qubits)``. For a source row ``row``, the
    destination rows for outgoing directed couplings are
    ``indices[indptr[row]:indptr[row + 1]]``. These entries indicate
    logical qubit connectivity.

        For example, with node rows ``0 -> q0``, ``1 -> q1``, ``2 -> q2`` and directed
        couplings ``q0 -> q1``, ``q0 -> q2``, ``q1 -> q2``:

        - ``indptr = [0, 2, 3, 3]``
        - ``indices = [1, 2, 2]``

        This means:

        - row ``0`` spans ``indices[indptr[0]:indptr[1]] == indices[0:2] == [1, 2]``
            -> ``q0`` connects to ``q1`` and ``q2``
        - row ``1`` spans ``indices[indptr[1]:indptr[2]] == indices[2:3] == [2]``
            -> ``q1`` connects to ``q2``
        - row ``2`` spans ``indices[indptr[2]:indptr[3]] == indices[3:3] == []``
            -> ``q2`` has no outgoing couplings


    This representation intentionally keeps only topology-relevant canonical content.

    :ivar indptr: CSR row pointer array of length ``n_qubits + 1``.
    :ivar indices: CSR destination-row indices for each structural non-zero entry.
    :ivar edge_couplings_data: Per-edge fidelity values in CSR edge order.
    :ivar edge_metadata: Per-edge gate-label entries aligned with
        ``edge_couplings_data`` and CSR edge order.

    :ivar node_ids: Node identifiers in row order.
    :ivar node_index: Device qubit indices in row order.
    :ivar row_by_node_id: Read-only map from node identifier to row index. Derived from
        ``node_ids`` and computed once on first access. Not accepted as a constructor
        argument, preventing inconsistent manual construction.
    """

    indptr: NDArray[np.int_]  #  row index pointer array
    indices: NDArray[np.int_]  # column index pointer array

    node_ids: tuple[str, ...]
    node_index: NDArray[np.uint32]

    edge_couplings_data: tuple[tuple[float, ...], ...]
    edge_metadata: tuple[tuple[str, ...], ...]

    @cached_property
    def row_by_node_id(self) -> Mapping[str, int]:
        """Read-only map from node identifier to row index, derived from ``node_ids``.

        Computed once on first access and cached. Not accepted as a constructor argument to
        prevent mismatched manual construction such as supplying node ids that contradict
        the mapping.

        :returns: Immutable mapping from node id to row index.
        """
        return MappingProxyType({node_id: i for i, node_id in enumerate(self.node_ids)})

    @classmethod
    def from_canonical(cls, canonical: CanonicalSystemData) -> TopologyView:
        """Constructs a TopologyView from a CanonicalSystemData object, for topology
        relevant data. This is a lightweight representation of the system connectivity,
        stored as a custom CSR matrix form.

        :param canonical: The CanonicalSystemData object to convert.
        :returns: Topology view with CSR connectivity and aligned fidelity data with
            associated metadata.
        :raises ValueError: If qubit or coupling information is missing.
        """
        cls._validate_canonical(canonical)
        node_ids, node_index = cls._build_node_data(canonical)
        row_by_node_id = {node_id: i for i, node_id in enumerate(node_ids)}
        raw_edges = cls._build_raw_edges(canonical, row_by_node_id)
        indptr, indices, edge_couplings_data, edge_metadata = cls._build_csr_arrays(
            raw_edges, len(node_ids)
        )
        return cls(
            indptr=indptr,
            indices=indices,
            node_ids=node_ids,
            node_index=node_index,
            edge_couplings_data=tuple(edge_couplings_data),
            edge_metadata=tuple(edge_metadata),
        )

    @staticmethod
    def _validate_canonical(canonical: CanonicalSystemData) -> None:
        """Raises ValueError if the canonical data is missing qubits or couplings."""
        if not canonical.qubits:
            raise ValueError(
                "CanonicalSystemData must have qubit information to construct TopologyView"
            )
        if not canonical.couplings:
            raise ValueError(
                "CanonicalSystemData must have coupling information to construct "
                "TopologyView"
            )

    @staticmethod
    def _build_node_data(
        canonical: CanonicalSystemData,
    ) -> tuple[tuple[str, ...], NDArray[np.uint32]]:
        """Returns node identifiers and device qubit indices.

        :raises ValueError: If duplicate qubit identifiers are present.
        """
        node_ids_list: list[str] = []
        node_index_list: list[int] = []
        seen_qubit_ids: set[str] = set()

        for qubit in canonical.qubits:
            if qubit.id in seen_qubit_ids:
                raise ValueError(
                    f"Duplicate qubit id {qubit.id!r} found in canonical qubit data."
                )
            seen_qubit_ids.add(qubit.id)
            node_ids_list.append(qubit.id)
            node_index_list.append(qubit.index)

        return (
            tuple(node_ids_list),
            np.asarray(node_index_list, dtype=np.uint32),
        )

    @staticmethod
    def _build_raw_edges(
        canonical: CanonicalSystemData,
        row_by_node_id: dict[str, int],
    ) -> list[tuple[int, int, tuple[tuple[str, float], ...]]]:
        """Returns sorted raw edges built from coupling data.

        :raises ValueError: If a coupling references a qubit not present in the node map.
        """
        # Each raw edge is:
        #   (source_row, target_row, ((gate_name, fidelity), ...))
        # Example:
        #   (0, 2, (("cx", 0.991), ("cz", 0.973)))
        raw_edges: list[tuple[int, int, tuple[tuple[str, float], ...]]] = []
        for coupling in canonical.couplings:
            if coupling.source_qubit_id not in row_by_node_id:
                raise ValueError(
                    f"Coupling source_qubit_id {coupling.source_qubit_id!r} not found "
                    "in qubits"
                )
            if coupling.target_qubit_id not in row_by_node_id:
                raise ValueError(
                    f"Coupling target_qubit_id {coupling.target_qubit_id!r} not found "
                    "in qubits"
                )
            src = row_by_node_id[coupling.source_qubit_id]
            dst = row_by_node_id[coupling.target_qubit_id]
            gate_fidelities = tuple(
                (gf.gate, float(gf.fidelity)) for gf in coupling.gate_fidelities
            )
            raw_edges.append((src, dst, gate_fidelities))
        # Group all edges for the same source row together.
        raw_edges.sort(key=lambda e: (e[0], e[1]))
        return raw_edges

    @staticmethod
    def _build_csr_arrays(
        raw_edges: list[tuple[int, int, tuple[tuple[str, float], ...]]],
        n_nodes: int,
    ) -> tuple[
        NDArray[np.int_],
        NDArray[np.int_],
        list[tuple[float, ...]],
        list[tuple[str, ...]],
    ]:
        """Builds CSR indptr/indices arrays and aligned edge data from sorted raw edges.

        Duplicate directed couplings for the same ``(source_row, target_row)`` pair
        should not occur in validated canonical data. If encountered, a
        :class:`ValueError` is raised to avoid ambiguous downstream sparse-matrix
        aggregation.
        """
        duplicate_found = any(
            raw_edges[i][0] == raw_edges[i - 1][0]
            and raw_edges[i][1] == raw_edges[i - 1][1]
            for i in range(1, len(raw_edges))
        )
        if duplicate_found:
            raise ValueError(
                "Duplicate directed couplings found in canonical coupling data."
            )

        n_edges = len(raw_edges)
        indptr = np.zeros(n_nodes + 1, dtype=np.int_)
        indices = np.zeros(n_edges, dtype=np.int_)
        edge_couplings_data: list[tuple[float, ...]] = []
        edge_metadata: list[tuple[str, ...]] = []

        k = 0
        for row in range(n_nodes):
            indptr[row] = k
            while k < n_edges and raw_edges[k][0] == row:
                _, dst, gate_fidelities = raw_edges[k]
                indices[k] = dst
                edge_couplings_data.append(
                    tuple(fidelity for _, fidelity in gate_fidelities)
                )
                edge_metadata.append(tuple(gate_name for gate_name, _ in gate_fidelities))
                k += 1
        indptr[n_nodes] = n_edges
        return indptr, indices, edge_couplings_data, edge_metadata

    @cached_property
    def nnz(self) -> int:
        """Returns number of non-zero (nnz) entries in this CSR view.

        In this topology representation, this equals the number of directed edges.

        :returns: Number of non-zero CSR entries.
        """
        return int(self.indices.size)

    def are_coupled(self, source_qubit_id: str, target_qubit_id: str) -> bool:
        """Checks whether a directed edge exists from source to target qubit.

        :param source_qubit_id: Source qubit identifier.
        :param target_qubit_id: Target qubit identifier.
        :returns: ``True`` if the directed coupling exists, otherwise ``False``.
        """
        source_row = self.row_by_node_id[source_qubit_id]
        target_row = self.row_by_node_id[target_qubit_id]

        start = int(self.indptr[source_row])
        end = int(self.indptr[source_row + 1])
        return bool(np.any(self.indices[start:end] == target_row))

    def coupled_qubit_indices(self, qubit_id: str) -> tuple[int, ...]:
        """Returns outgoing neighbour qubit indices for a qubit.

        :param qubit_id: Qubit identifier.
        :returns: Tuple of neighbour qubit indices reachable by outgoing edges.
        """
        source_row = self.row_by_node_id[qubit_id]
        start = int(self.indptr[source_row])
        end = int(self.indptr[source_row + 1])
        neighbour_rows = self.indices[start:end]
        return tuple(int(self.node_index[row]) for row in neighbour_rows)

    @cached_property
    def sorted_gate_fidelities(self) -> tuple[tuple[float, tuple[str, str], str], ...]:
        """Returns all gate fidelities sorted from best to worst.

        Each entry is ``(fidelity, (source_qubit_id, target_qubit_id), gate_name)``.
        Sorted in descending order by fidelity (best first).

        :returns: Tuple of ``(fidelity, qubits, gate_name)`` entries sorted best to worst,
            or empty tuple if no gate fidelities exist.
        """
        fidelities: list[tuple[float, tuple[str, str], str]] = []
        for edge_index, gate_name, fidelity in self._iter_gate_fidelities():
            src_row = self._source_row_for_edge_index(edge_index)
            dst_row = int(self.indices[edge_index])
            fidelities.append(
                (
                    fidelity,
                    (self.node_ids[src_row], self.node_ids[dst_row]),
                    gate_name,
                )
            )
        return tuple(sorted(fidelities, key=lambda x: x[0], reverse=True))

    @cached_property
    def sorted_gate_fidelities_by_type(
        self,
    ) -> tuple[tuple[str, tuple[tuple[float, tuple[str, str]], ...]], ...]:
        """Returns gate fidelities grouped by gate type.

        The returned structure is ``((gate_name, gate_entries), ...)`` where each
        ``gate_entries`` is a tuple of ``(fidelity, (source_qubit_id, target_qubit_id))``.
        Gate groups are sorted alphabetically by gate name, and entries within each gate
        are sorted in descending fidelity order.

        :returns: Grouped gate fidelities sorted by gate name and fidelity.
        """
        grouped: dict[str, list[tuple[float, tuple[str, str]]]] = {}
        for edge_index, gate_name, fidelity in self._iter_gate_fidelities():
            src_row = self._source_row_for_edge_index(edge_index)
            dst_row = int(self.indices[edge_index])
            grouped.setdefault(gate_name, []).append(
                (fidelity, (self.node_ids[src_row], self.node_ids[dst_row]))
            )

        return tuple(
            (
                gate_name,
                tuple(sorted(entries, key=lambda entry: entry[0], reverse=True)),
            )
            for gate_name, entries in sorted(grouped.items(), key=lambda item: item[0])
        )

    def gate_fidelities_by_type(
        self,
        gate_names: set[str] | tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, tuple[tuple[float, tuple[str, str]], ...]], ...]:
        """Returns grouped gate fidelities, optionally filtered to selected gate names.

        When ``gate_names`` is provided, only matching gate types are grouped and sorted,
        avoiding work for unrelated gate types.

        :param gate_names: Optional gate names to include. If ``None``, all gate types are
            returned.
        :returns: Grouped gate fidelities sorted by gate name and fidelity.
        """
        if gate_names is None:
            return self.sorted_gate_fidelities_by_type

        allowed = set(gate_names)
        if not allowed:
            return ()

        grouped: dict[str, list[tuple[float, tuple[str, str]]]] = defaultdict(list)
        for edge_index, gate_name, fidelity in self._iter_gate_fidelities():
            if gate_name not in allowed:
                continue

            src_row = self._source_row_for_edge_index(edge_index)
            dst_row = int(self.indices[edge_index])
            grouped[gate_name].append(
                (fidelity, (self.node_ids[src_row], self.node_ids[dst_row]))
            )

        return tuple(
            (
                gate_name,
                tuple(sorted(entries, key=lambda entry: entry[0], reverse=True)),
            )
            for gate_name, entries in sorted(grouped.items(), key=lambda item: item[0])
        )

    def _iter_gate_fidelities(self):
        """Yields ``(edge_index, gate_name, fidelity)`` for all edge gates."""
        for edge_index, (gate_names, gate_fidelities) in enumerate(
            zip(self.edge_metadata, self.edge_couplings_data, strict=True)
        ):
            for gate_name, fidelity in zip(gate_names, gate_fidelities, strict=True):
                yield edge_index, gate_name, fidelity

    def _source_row_for_edge_index(self, edge_index: int) -> int:
        """Returns CSR source row for an edge index."""
        return int(np.searchsorted(self.indptr, edge_index, side="right") - 1)


@dataclass(frozen=True)
class ScipyTopologyView:
    """Topology view derived from canonical data as SciPy CSR matrices, as well as further
    allowing for return to a NetworkX graph representation.

    The derived view stores binary logical connectivity and per-gate fidelity matrices.

    :ivar adjacency_matrix: Binary directed adjacency matrix (1 if logically connected, no
        entry otherwise).
    :ivar gate_fidelity_matrices: Mapping from gate name to directed fidelity matrix. Each
        matrix uses per-gate sparsity and only stores directed edges where that gate is
        available.
    """

    adjacency_matrix: csr_matrix
    gate_fidelity_matrices: Mapping[str, csr_matrix]

    @classmethod
    def from_derived(
        cls,
        canonical_graph: TopologyView,
    ) -> ScipyTopologyView:
        """Constructs a :class:`ScipyTopologyView` from canonical graph data.

        :param canonical_graph: Topology view to convert.
        :returns: Derived view with binary adjacency and per-gate fidelity matrices.
        """
        n = len(canonical_graph.node_ids)
        shape = (n, n)
        nnz = int(canonical_graph.indices.size)

        adjacency_matrix = cls._build_adjacency_matrix(canonical_graph, shape, nnz)
        gate_fidelity_matrices = cls._build_gate_fidelity_matrices(canonical_graph, shape)
        return cls(
            adjacency_matrix=adjacency_matrix,
            gate_fidelity_matrices=MappingProxyType(gate_fidelity_matrices),
        )

    @staticmethod
    def _build_adjacency_matrix(
        canonical_graph: TopologyView,
        shape: tuple[int, int],
        nnz: int,
    ) -> csr_matrix:
        """Returns a binary directed adjacency CSR matrix."""
        return csr_matrix(
            (
                np.ones(nnz, dtype=np.int8),
                canonical_graph.indices,
                canonical_graph.indptr,
            ),
            shape=shape,
        )

    @staticmethod
    def _build_gate_fidelity_matrices(
        canonical_graph: TopologyView,
        shape: tuple[int, int],
    ) -> dict[str, csr_matrix]:
        """Returns per-gate fidelity CSR matrices."""
        gate_row_indices_by_name: dict[str, list[int]] = defaultdict(list)
        gate_col_indices_by_name: dict[str, list[int]] = defaultdict(list)
        gate_data_by_name: dict[str, list[float]] = defaultdict(list)

        for source_row in range(len(canonical_graph.node_ids)):
            start = int(canonical_graph.indptr[source_row])
            end = int(canonical_graph.indptr[source_row + 1])
            for edge_index in range(start, end):
                dst = int(canonical_graph.indices[edge_index])
                gate_labels = canonical_graph.edge_metadata[edge_index]
                gate_fidelities = canonical_graph.edge_couplings_data[edge_index]
                edge_gate_fidelities: dict[str, float] = {}

                for gate_label, fidelity in zip(gate_labels, gate_fidelities, strict=True):
                    edge_gate_fidelities[gate_label] = float(fidelity)

                for gate_label, fidelity in edge_gate_fidelities.items():
                    gate_row_indices_by_name[gate_label].append(source_row)
                    gate_col_indices_by_name[gate_label].append(dst)
                    gate_data_by_name[gate_label].append(fidelity)

        return {
            gate_name: csr_matrix(
                (
                    gate_data_by_name[gate_name],
                    (
                        gate_row_indices_by_name[gate_name],
                        gate_col_indices_by_name[gate_name],
                    ),
                ),
                shape=shape,
                dtype=np.float64,
            )
            for gate_name in sorted(gate_data_by_name)
        }

    @cached_property
    def _networkx_graph_internal(self) -> nx.DiGraph:
        """Compute the NetworkX graph once and cache it internally.

        This internal copy is never exposed directly. Callers always receive a
        deep copy via :attr:`networkx_graph`.

        :returns: Directed graph derived from adjacency and gate-fidelity matrices.
        """
        graph = nx.from_scipy_sparse_array(
            self.adjacency_matrix,
            create_using=nx.DiGraph,
        )

        # Add per-gate fidelities as edge attributes
        for gate_name, gate_matrix in self.gate_fidelity_matrices.items():
            gate_coo = gate_matrix.tocoo()
            for row, col, value in zip(
                gate_coo.row,
                gate_coo.col,
                gate_coo.data,
                strict=True,
            ):
                source = int(row)
                target = int(col)
                if graph.has_edge(source, target):
                    graph.edges[source, target][gate_name] = float(value)

        return graph

    @property
    def networkx_graph(self) -> nx.DiGraph:
        """Return a deep copy of the cached NetworkX graph.

        The internal graph is computed once on first access and cached. Each call returns a
        new deep copy so the caller may freely mutate the returned graph without affecting
        the cached original.

        Node labels are integer row indices. Edge metadata includes per-gate fidelity
        attributes for available gates.

        :returns: Mutable deep copy of the directed graph.
        """
        return deepcopy(self._networkx_graph_internal)
