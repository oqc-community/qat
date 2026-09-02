# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Qubit level derived view assembled from canonical data."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from qat.experimental.system_data.canonical.schema import (
    AcquireOperationStepData,
    CanonicalSystemData,
    DelayOperationStepData,
    ErrorOperationStepData,
    OperationData,
    OperationReferenceStepData,
    PulseOperationStepData,
    QubitData,
    ReadoutProbabilityData,
    SyncOperationStepData,
    WaveformData,
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


def _resolve_waveform_width(step: PulseOperationStepData, qubit: QubitData) -> int | None:
    """Return the waveform width in picoseconds for a pulse step, or ``None`` if
    unresolvable."""
    waveform_def = step.waveform_definition
    if isinstance(waveform_def, WaveformData):
        return waveform_def.width
    mode = next((m for m in qubit.modes if m.id == step.mode_id), None)
    if mode is None:
        return None
    waveform = next((w for w in mode.waveform_definitions if w.id == waveform_def), None)
    return waveform.width if waveform is not None else None


def _pulse_duration_modes(
    operation: OperationData,
    qubit: QubitData,
    qubit_by_id: dict[str, QubitData],
    _visited: frozenset[tuple[str, str]],
) -> dict[tuple[str, str], int] | None:
    """Accumulate per-mode durations for the default variant; ``None`` if unresolvable.

    Keys are ``(qubit_id, mode_id)`` so that identically-named modes on different
    qubits remain distinct. :class:`SyncOperationStepData` aligns all named modes
    on the current qubit to the same value (barrier semantics).
    """
    key = (qubit.id, operation.id)
    if key in _visited:
        return None
    default_variant = next((v for v in operation.variants if v.when is None), None)
    if default_variant is None:
        return None
    visited = _visited | {key}
    modes: dict[tuple[str, str], int] = {}
    for step in default_variant.operation_steps:
        match step:
            case PulseOperationStepData() as pulse_step:
                width = _resolve_waveform_width(pulse_step, qubit)
                if width is None:
                    return None
                mk = (qubit.id, pulse_step.mode_id)
                modes[mk] = modes.get(mk, 0) + width
            case DelayOperationStepData() as delay_step:
                duration = delay_step.duration
                # Only integral int/float durations are statically resolvable; bools and
                # fractional picosecond values (which cannot occur physically) are not.
                if (
                    isinstance(duration, bool)
                    or not isinstance(duration, int | float)
                    or (isinstance(duration, float) and not duration.is_integer())
                ):
                    return None
                mk = (qubit.id, delay_step.mode_id)
                modes[mk] = modes.get(mk, 0) + int(duration)
            case SyncOperationStepData() as sync_step:
                # Each mode_ref carries an optional qubit_id; default to the current qubit.
                mks = [
                    (ref.qubit_id if ref.qubit_id else qubit.id, ref.mode_id)
                    for ref in sync_step.mode_refs
                ]
                synced = max((modes.get(mk, 0) for mk in mks), default=0)
                for mk in mks:
                    modes[mk] = synced
            case OperationReferenceStepData() as ref_step:
                target_qubit = (
                    qubit_by_id.get(ref_step.qubit_id) if ref_step.qubit_id else qubit
                )
                if target_qubit is None:
                    return None
                ref_op = next(
                    (o for o in target_qubit.operations if o.id == ref_step.operation_id),
                    None,
                )
                if ref_op is None:
                    return None
                sub = _pulse_duration_modes(ref_op, target_qubit, qubit_by_id, visited)
                if sub is None:
                    return None
                for mk, d in sub.items():
                    modes[mk] = modes.get(mk, 0) + d
            case OperationData() as operation_step:
                sub = _pulse_duration_modes(operation_step, qubit, qubit_by_id, visited)
                if sub is None:
                    return None
                for mk, d in sub.items():
                    modes[mk] = modes.get(mk, 0) + d
            case ErrorOperationStepData():
                return None
            case AcquireOperationStepData():
                # Acquisition timing is not yet modelled; treat as unresolvable.
                return None
            case _:
                pass
    return modes


def _pulse_duration_for_operation(
    operation: OperationData,
    qubit: QubitData,
    qubit_by_id: dict[str, QubitData],
) -> int | None:
    """Resolve the pulse duration of the default variant; ``None`` if unresolvable.

    Per-mode durations are accumulated independently; sync barriers align the named
    modes to the same value. The total is the maximum across all modes, which correctly
    models parallel execution on different modes or qubits.

    Only the unconditional variant (``when=None``) is considered. Virtual steps
    (phase shifts, phase sets) contribute zero. :class:`ErrorOperationStepData` and
    :class:`AcquireOperationStepData` (acquisition timing is not yet modelled), along
    with any missing references, propagate ``None``.

    :param operation: Root operation to evaluate.
    :param qubit: Qubit that owns the operation.
    :param qubit_by_id: All qubits in the system, keyed by id.
    :returns: Total pulse duration in picoseconds, or ``None``.
    """
    modes = _pulse_duration_modes(operation, qubit, qubit_by_id, frozenset())
    if modes is None:
        return None
    return max(modes.values(), default=0)


@dataclass(frozen=True)
class OperationSet:
    """Operations available on a qubit.

    :ivar operation_type: Identifiers of the operations supported on this qubit.
    :ivar fidelity: Optional per-gate fidelity mapping. Stubbed as ``None`` until
        the schema exposes per-operation fidelity data.
    :ivar duration: Per-gate duration in picoseconds, keyed by operation id.
        ``None`` for individual entries when the duration cannot be statically resolved.
    """

    operation_type: tuple[str, ...]
    fidelity: Mapping[str, float | None] | None = None
    duration: Mapping[str, int | None] | None = None


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
        qubit_by_id = {q.id: q for q in parent.qubits}
        qubits = cls._unpack_qubit_properties(selected, qubit_by_id)
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
    def _unpack_qubit_properties(
        selected: dict,
        qubit_by_id: dict[str, QubitData],
    ) -> dict[str, QubitProperties]:
        """Unpacks per-qubit properties from the selected canonical qubit records.

        :param selected: Canonical qubit records keyed by id.
        :param qubit_by_id: All qubits in the system, used to resolve cross-qubit references.
        :returns: Dict of :class:`QubitProperties` keyed by qubit id.
        """
        return {
            qid: QubitProperties(
                index=qubit.index,
                supported_operations=OperationSet(
                    operation_type=tuple(
                        op.id for op in qubit.operations if op.interface == "public"
                    ),
                    duration=MappingProxyType(
                        {
                            op.id: _pulse_duration_for_operation(op, qubit, qubit_by_id)
                            for op in qubit.operations
                            if op.interface == "public"
                        }
                    ),
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
