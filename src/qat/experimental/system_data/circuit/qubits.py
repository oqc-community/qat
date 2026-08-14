# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Qubit level derived view assembled from canonical data."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from qat.experimental.system_data.canonical.schema import (
    CanonicalSystemData,
    OperationData,
    ReadoutProbabilityData,
)
from qat.experimental.system_data.derived.interface import DerivedViewInterface


def _measurement_fidelity_from_readout(
    readout: ReadoutProbabilityData | None,
) -> float | None:
    """Compute scalar measurement fidelity from readout confusion probabilities.

    .. math::

        F = \\frac{1}{2}\\bigl[P(0|0) + P(1|1)\\bigr]

    where :math:`P(m|p)` is the probability of measuring state :math:`m` given that
    state :math:`p` was prepared. Equivalent to
    :math:`1 - \\frac{1}{2}[P(1|0) + P(0|1)]` when each row sums to 1.

    :param readout: Readout confusion data, or ``None``.
    :returns: Fidelity scalar, or ``None`` if data is absent or has no diagonal entries.
    """
    if readout is None:
        return None
    diagonal = tuple(
        e for e in readout.probability_entries if e.prepared_state == e.measured_state
    )
    return sum(e.probability for e in diagonal) / len(diagonal) if diagonal else None


def _flatten_operation_ids(operation: OperationData) -> tuple[str, ...]:
    """Recursively collect ids of leaf operations within an operation tree, depth-first.

    Walks the ``operation_steps`` of each variant. Any step that is itself an
    :class:`~qat.experimental.system_data.canonical.schema.OperationData` is recursed
    into. A node is considered a leaf when none of its steps are ``OperationData``;
    at that point its ``id`` is collected.

    :param operation: Root operation to flatten.
    :returns: Tuple of leaf operation ids in depth-first traversal order.
    """
    all_steps = [step for variant in operation.variants for step in variant.operation_steps]
    nested_ops = [s for s in all_steps if isinstance(s, OperationData)]
    if not nested_ops:
        return (operation.id,)
    ids: list[str] = []
    for nested in nested_ops:
        ids.extend(_flatten_operation_ids(nested))
    return tuple(ids)


@dataclass(frozen=True)
class OperationSet:
    """Operations available on a qubit.

    :ivar operation_type: Identifiers of the operations supported on this qubit.
    :ivar fidelity: Optional per-gate fidelity mapping. Stubbed as ``None`` until
        the schema exposes per-operation fidelity data.
    :ivar duration: Optional per-gate duration mapping in picoseconds. Stubbed as
        ``None`` until the schema exposes per-operation duration data.
    """

    operation_type: tuple[str, ...]
    fidelity: Mapping[str, float | None] | None = None
    duration: Mapping[str, float | None] | None = None


@dataclass(frozen=True)
class QubitProperties:
    """Per-qubit properties within a derived view.

    :ivar index: Qubit index.
    :ivar supported_operations: Operations available on this qubit.
    :ivar measurement_fidelity: Scalar measurement fidelity, or ``None`` if unavailable.
    """

    index: int
    supported_operations: OperationSet
    measurement_fidelity: float | None


@dataclass(frozen=True)
class Interaction:
    """Resolved directed coupling between two qubits in a :class:`QubitView`.

    Positions refer to the ordinal index of each qubit in the :attr:`QubitView.qubits`
    mapping (0-based, in canonical order). They are view-local and have no meaning
    outside of the :class:`QubitView` that produced them.

    :ivar source_position: Position of the source qubit in :attr:`QubitView.qubits`.
    :ivar target_position: Position of the target qubit in :attr:`QubitView.qubits`.
    :ivar gate_fidelities: Per-gate fidelity values keyed by gate name.
    """

    source_position: int
    target_position: int
    gate_fidelities: Mapping[str, float]


@dataclass(frozen=True)
class QubitView(DerivedViewInterface[CanonicalSystemData]):
    """Derived qubit-level view for gate-level compilation.

    By default all qubits in the canonical data are included. Pass ``qubit_ids``
    to :meth:`derive` to restrict the view to a specific subset; couplings are
    automatically filtered to pairs where both endpoints are within that subset.

    :ivar qubits: Per-qubit properties keyed by qubit identifier.
    :ivar interactions: Directed interactions between qubits, with positions resolved
        against :attr:`qubits` iteration order and gate fidelities pre-extracted.
    """

    qubits: Mapping[str, QubitProperties]
    interactions: tuple[Interaction, ...]

    @classmethod
    def derive(
        cls,
        parent: CanonicalSystemData,
        qubit_ids: set[str] | None = None,
        **kwargs,
    ) -> QubitView:
        """Constructs a qubit view over a subset of qubits in the canonical data.

        When ``qubit_ids`` is ``None`` all qubits are included. Interactions are filtered
        to pairs where both endpoints are within the included set.

        :param parent: Canonical system data.
        :param qubit_ids: Optional set of qubit ids to include. Defaults to all
            qubits when ``None``.
        :returns: :class:`QubitView` restricted to the given qubit ids.
        """
        selected = cls._select_qubits(parent, qubit_ids)
        qubits = cls._unpack_qubit_properties(selected)
        position_by_id = {qid: i for i, qid in enumerate(qubits)}
        interactions = cls._unpack_interactions(parent, position_by_id)
        return cls(
            qubits=MappingProxyType(qubits),
            interactions=interactions,
        )

    @staticmethod
    def _select_qubits(
        parent: CanonicalSystemData,
        qubit_ids: set[str] | None,
    ) -> dict:
        """Returns the selected canonical qubit records in canonical order.

        Warns and drops any requested ids not present in the canonical data.

        :param parent: Canonical system data.
        :param qubit_ids: Optional set of qubit ids to include. ``None`` means all.
        :returns: Dict of canonical qubit records keyed by id, in canonical order.
        """
        qubit_by_id = {q.id: q for q in parent.qubits}
        if qubit_ids is None:
            return qubit_by_id
        unknown = qubit_ids - qubit_by_id.keys()
        if unknown:
            warnings.warn(
                f"Requested qubit ids not found in canonical data and will be "
                f"ignored: {sorted(unknown)}",
                stacklevel=3,
            )
        # Preserve canonical ordering for deterministic positions.
        return {q.id: q for q in parent.qubits if q.id in qubit_ids}

    @staticmethod
    def _unpack_qubit_properties(selected: dict) -> dict[str, QubitProperties]:
        """Unpacks per-qubit properties from the selected canonical qubit records.

        :param selected: Canonical qubit records keyed by id.
        :returns: Dict of :class:`QubitProperties` keyed by qubit id.
        """
        return {
            qid: QubitProperties(
                index=qubit.index,
                supported_operations=OperationSet(
                    operation_type=tuple(
                        op_id
                        for operation in qubit.operations
                        for op_id in _flatten_operation_ids(operation)
                    )
                ),
                measurement_fidelity=_measurement_fidelity_from_readout(
                    qubit.readout_probability
                ),
            )
            for qid, qubit in selected.items()
        }

    @staticmethod
    def _unpack_interactions(
        parent: CanonicalSystemData,
        position_by_id: dict[str, int],
    ) -> tuple[Interaction, ...]:
        """Unpacks sorted interactions from canonical couplings filtered to included qubits.

        :param parent: Canonical system data.
        :param position_by_id: Map from qubit id to its position in the view.
        :returns: Tuple of :class:`Interaction` sorted by ``(source_position, target_position)``.
        """
        return tuple(
            sorted(
                (
                    Interaction(
                        source_position=position_by_id[c.source_qubit_id],
                        target_position=position_by_id[c.target_qubit_id],
                        gate_fidelities=MappingProxyType(
                            {gf.gate: float(gf.fidelity) for gf in c.gate_fidelities}
                        ),
                    )
                    for c in parent.couplings
                    if c.source_qubit_id in position_by_id
                    and c.target_qubit_id in position_by_id
                ),
                key=lambda i: (i.source_position, i.target_position),
            )
        )
