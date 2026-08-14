# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import warnings

import pytest

from qat.experimental.system_data.canonical.schema import (
    CanonicalSystemData,
    OperationData,
    OperationVariantData,
    ProbabilityEntry,
    QubitCouplingData,
    QubitData as CanonicalQubitData,
    ReadoutProbabilityData,
    TwoQubitGateFidelityData,
)
from qat.experimental.system_data.circuit.qubits import (
    OperationSet,
    QubitProperties,
    QubitView,
    _flatten_operation_ids,
)


def _make_op(op_id: str, *steps) -> OperationData:
    return OperationData(
        id=op_id,
        variants=(OperationVariantData(operation_steps=tuple(steps)),),
    )


def _make_readout(*pairs: tuple[int, int, float]) -> ReadoutProbabilityData:
    """Build ReadoutProbabilityData from (prepared, measured, probability) triples."""
    return ReadoutProbabilityData(
        probability_entries=tuple(
            ProbabilityEntry(prepared_state=p, measured_state=m, probability=prob)
            for p, m, prob in pairs
        )
    )


def _two_qubit_canonical() -> CanonicalSystemData:
    op_x = _make_op("x")
    op_h = _make_op("h")
    return CanonicalSystemData(
        qubits=(
            CanonicalQubitData(id="q0", index=0, operations=(op_x,)),
            CanonicalQubitData(id="q1", index=1, operations=(op_h,)),
        ),
        couplings=(
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="q1",
                gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.99),),
            ),
        ),
    )


def test_flatten_operation_ids_primitive_op_returns_own_id():
    """A primitive operation with no nested OperationData returns its own id."""
    assert _flatten_operation_ids(_make_op("x")) == ("x",)


def test_flatten_operation_ids_single_level_nested_ops():
    """Direct nested OperationData children are returned as leaf ids."""
    op = _make_op("cx", _make_op("x"), _make_op("phase"))
    assert _flatten_operation_ids(op) == ("x", "phase")


def test_flatten_operation_ids_single_nested_operation():
    """A single nested OperationData child is returned as its id."""
    op = _make_op("outer", _make_op("inner"))
    assert _flatten_operation_ids(op) == ("inner",)


def test_flatten_operation_ids_deep_nesting():
    """Deep nesting is resolved depth-first to the leaf operation ids."""
    op = _make_op("top", _make_op("mid", _make_op("deep")))
    assert _flatten_operation_ids(op) == ("deep",)


def test_flatten_operation_ids_mixed_leaf_and_composite():
    """A mix of leaf and composite children produces ids in depth-first order."""
    op = _make_op("top", _make_op("a"), _make_op("composite", _make_op("b"), _make_op("c")))
    assert _flatten_operation_ids(op) == ("a", "b", "c")


def test_supported_operation_fidelity_defaults_to_none():
    """OperationSet fidelity is None until the schema provides real data."""
    op = OperationSet(operation_type=("x",))
    assert op.fidelity is None


def test_supported_operation_duration_defaults_to_none():
    """OperationSet duration is None until the schema provides real data."""
    op = OperationSet(operation_type=("x",))
    assert op.duration is None


def test_qubit_view_from_canonical_qubit_indices():
    """Qubit indices are correctly extracted from canonical qubit records."""
    view = QubitView.derive(_two_qubit_canonical())
    assert view.qubits["q0"].index == 0
    assert view.qubits["q1"].index == 1


def test_qubit_view_from_canonical_all_qubits_present():
    """All canonical qubits appear as keys in the qubits mapping."""
    view = QubitView.derive(_two_qubit_canonical())
    assert set(view.qubits) == {"q0", "q1"}


def test_qubit_view_from_canonical_qubit_data_instances():
    """Each entry in qubits is a QubitProperties instance."""
    view = QubitView.derive(_two_qubit_canonical())
    assert all(isinstance(q, QubitProperties) for q in view.qubits.values())


def test_qubit_view_from_canonical_operations_mapped_to_correct_qubit():
    """Each qubit's operations are keyed to the right qubit id."""
    view = QubitView.derive(_two_qubit_canonical())
    assert view.qubits["q0"].supported_operations.operation_type == ("x",)
    assert view.qubits["q1"].supported_operations.operation_type == ("h",)


def test_qubit_view_from_canonical_operation_fidelity_is_stubbed_none():
    """All operation fidelity entries are None as the schema has no fidelity data yet."""
    view = QubitView.derive(_two_qubit_canonical())
    for qubit in view.qubits.values():
        assert qubit.supported_operations.fidelity is None


def test_qubit_view_from_canonical_interactions_resolved_with_positions():
    """Interactions carry positions resolved against QubitView.qubits ordering."""
    view = QubitView.derive(_two_qubit_canonical())
    assert len(view.interactions) == 1
    assert view.interactions[0].source_position == 0  # q0 is at position 0
    assert view.interactions[0].target_position == 1  # q1 is at position 1


def test_qubit_view_from_canonical_measurement_fidelity_none_when_no_readout_data():
    """Measurement fidelity is None for qubits with no readout data in canonical."""
    view = QubitView.derive(_two_qubit_canonical())
    assert view.qubits["q0"].measurement_fidelity is None
    assert view.qubits["q1"].measurement_fidelity is None


def test_qubit_view_from_canonical_measurement_fidelity_computed_when_readout_provided():
    """Measurement fidelity is computed when readout data is present in canonical."""
    rp = _make_readout((0, 0, 0.95), (1, 1, 0.97))
    canonical = CanonicalSystemData(
        qubits=(CanonicalQubitData(id="q0", index=0, readout_probability=rp),),
    )
    view = QubitView.derive(canonical)
    assert view.qubits["q0"].measurement_fidelity == pytest.approx(0.96)


def test_qubit_view_from_canonical_empty_system_data():
    """An empty CanonicalSystemData produces an empty qubits mapping and no couplings."""
    view = QubitView.derive(CanonicalSystemData())
    assert view.qubits == {}
    assert view.interactions == ()


def test_qubit_view_from_canonical_qubit_with_no_operations():
    """A qubit with no operations has an empty supported_operations tuple."""
    canonical = CanonicalSystemData(qubits=(CanonicalQubitData(id="q0", index=0),))
    view = QubitView.derive(canonical)
    assert view.qubits["q0"].supported_operations.operation_type == ()


def test_qubit_view_measurement_fidelity_none_when_no_readout_data():
    """Measurement fidelity is None for a qubit with no readout probability data."""
    canonical = CanonicalSystemData(qubits=(CanonicalQubitData(id="q0", index=0),))
    view = QubitView.derive(canonical)
    assert view.qubits["q0"].measurement_fidelity is None


def test_qubit_view_measurement_fidelity_computed_from_diagonal_entries():
    """Fidelity is the mean of diagonal P(s|s) entries from the confusion matrix."""
    # P(0|0)=0.95, P(1|1)=0.97 -> mean = 0.96
    rp = _make_readout((0, 0, 0.95), (1, 0, 0.05), (0, 1, 0.03), (1, 1, 0.97))
    canonical = CanonicalSystemData(
        qubits=(CanonicalQubitData(id="q0", index=0, readout_probability=rp),)
    )
    view = QubitView.derive(canonical)
    assert view.qubits["q0"].measurement_fidelity == pytest.approx(0.96)


def test_qubit_view_measurement_fidelity_none_when_no_diagonal_entries():
    """Fidelity is None when only off-diagonal confusion entries are present."""
    rp = _make_readout((0, 1, 0.05), (1, 0, 0.03))
    canonical = CanonicalSystemData(
        qubits=(CanonicalQubitData(id="q0", index=0, readout_probability=rp),)
    )
    view = QubitView.derive(canonical)
    assert view.qubits["q0"].measurement_fidelity is None


@pytest.mark.parametrize(
    ("rp0_entries", "rp1_entries", "expected_f0", "expected_f1"),
    [
        (
            ((0, 0, 0.90), (1, 1, 0.90)),
            ((0, 0, 1.00), (1, 1, 1.00)),
            0.90,
            1.00,
        ),
        (
            ((0, 0, 0.80), (1, 1, 0.60)),
            ((0, 0, 0.95), (1, 1, 0.99)),
            0.70,
            0.97,
        ),
    ],
    ids=["symmetric", "asymmetric"],
)
def test_qubit_view_measurement_fidelity_per_qubit_independent(
    rp0_entries, rp1_entries, expected_f0, expected_f1
):
    """Fidelity for each qubit is computed independently from its own readout data."""
    canonical = CanonicalSystemData(
        qubits=(
            CanonicalQubitData(
                id="q0", index=0, readout_probability=_make_readout(*rp0_entries)
            ),
            CanonicalQubitData(
                id="q1", index=1, readout_probability=_make_readout(*rp1_entries)
            ),
        )
    )
    view = QubitView.derive(canonical)
    assert view.qubits["q0"].measurement_fidelity == pytest.approx(expected_f0)
    assert view.qubits["q1"].measurement_fidelity == pytest.approx(expected_f1)


def test_qubit_subset_view_only_includes_requested_qubits():
    """from_canonical with qubit_ids restricts the view to the given qubit ids."""
    canonical = _two_qubit_canonical()
    view = QubitView.derive(canonical, qubit_ids={"q0"})
    assert set(view.qubits) == {"q0"}


def test_qubit_subset_view_interactions_filtered_to_subset():
    """Interactions only include pairs where both endpoints are in the subset."""
    canonical = _two_qubit_canonical()
    view = QubitView.derive(canonical, qubit_ids={"q0"})
    assert view.interactions == ()


def test_qubit_subset_view_interactions_included_when_both_endpoints_in_subset():
    """Interactions are present when both source and target are in the requested subset."""
    canonical = _two_qubit_canonical()
    view = QubitView.derive(canonical, qubit_ids={"q0", "q1"})
    assert len(view.interactions) == 1
    assert view.interactions[0].source_position == 0  # q0 first in canonical order
    assert view.interactions[0].target_position == 1  # q1 second


def test_qubit_subset_view_returns_qubit_subset_view_instance():
    """from_canonical with qubit_ids returns a QubitView instance."""
    canonical = _two_qubit_canonical()
    view = QubitView.derive(canonical, qubit_ids={"q0"})
    assert isinstance(view, QubitView)


def test_qubit_subset_view_from_canonical_includes_all_qubits_when_no_ids_given():
    """from_canonical with no qubit_ids builds a view over all qubits."""
    canonical = _two_qubit_canonical()
    view = QubitView.derive(canonical)
    assert set(view.qubits) == {"q0", "q1"}


def test_qubit_view_warns_on_unknown_qubit_ids():
    """A warning is issued for requested qubit ids not present in canonical data."""
    canonical = _two_qubit_canonical()
    with pytest.warns(UserWarning, match="q99"):
        view = QubitView.derive(canonical, qubit_ids={"q0", "q99"})
    assert set(view.qubits) == {"q0"}


def test_qubit_view_no_warning_when_all_qubit_ids_valid():
    """No warning is issued when all requested qubit ids exist in canonical data."""
    canonical = _two_qubit_canonical()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        view = QubitView.derive(canonical, qubit_ids={"q0", "q1"})
    assert set(view.qubits) == {"q0", "q1"}
