# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import warnings

import pytest

from qat.experimental.system_data.canonical.schema import (
    AcquireOperationStepData,
    CanonicalSystemData,
    DelayOperationStepData,
    ErrorOperationStepData,
    ModeData,
    OperationCapabilityPredicateData,
    OperationData,
    OperationModeReferenceData,
    OperationParameterRefData,
    OperationReferenceStepData,
    OperationVariantData,
    PhaseShiftOperationStepData,
    ProbabilityEntry,
    PulseOperationStepData,
    QubitCouplingData,
    QubitData as CanonicalQubitData,
    ReadoutProbabilityData,
    SyncOperationStepData,
    TwoQubitGateFidelityData,
    WaveformData,
)
from qat.experimental.system_data.circuit.qubits import (
    OperationSet,
    QubitProperties,
    QubitView,
    _pulse_duration_for_operation,
    _resolve_waveform_width,
)


def _make_op(op_id: str, *steps, interface: str = "public") -> OperationData:
    return OperationData(
        id=op_id,
        interface=interface,
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


def test_supported_operation_fidelity_defaults_to_none():
    """OperationSet fidelity is None until the schema provides real data."""
    op = OperationSet(operation_type=("x",))
    assert op.fidelity is None


def _make_qubit_with_mode(
    mode_id: str,
    waveform_id: str,
    width: int | None,
    *,
    qubit_id: str = "q0",
    index: int = 0,
    operations: tuple = (),
) -> CanonicalQubitData:
    return CanonicalQubitData(
        id=qubit_id,
        index=index,
        modes=(
            ModeData(
                id=mode_id,
                channel_id="ch0",
                waveform_definitions=(WaveformData(id=waveform_id, width=width),),
            ),
        ),
        operations=operations,
    )


def test_resolve_waveform_width_from_mode_lookup():
    """Width is resolved by looking up the waveform id in the mode's definitions."""
    qubit = _make_qubit_with_mode("drive", "x_pi", 100)
    step = PulseOperationStepData(mode_id="drive", waveform_definition="x_pi")
    assert _resolve_waveform_width(step, qubit) == 100


def test_resolve_waveform_width_from_inline_waveform_data():
    """Width is returned directly when waveform_definition is an inline WaveformData."""
    qubit = CanonicalQubitData(id="q0", index=0)
    step = PulseOperationStepData(
        mode_id="drive", waveform_definition=WaveformData(id="x_pi", width=80)
    )
    assert _resolve_waveform_width(step, qubit) == 80


def test_resolve_waveform_width_none_when_mode_missing():
    """None is returned when the mode_id does not exist on the qubit."""
    qubit = CanonicalQubitData(id="q0", index=0)
    step = PulseOperationStepData(mode_id="drive", waveform_definition="x_pi")
    assert _resolve_waveform_width(step, qubit) is None


def test_resolve_waveform_width_none_when_waveform_missing():
    """None is returned when the waveform id is not in the mode's definitions."""
    qubit = _make_qubit_with_mode("drive", "other", 100)
    step = PulseOperationStepData(mode_id="drive", waveform_definition="x_pi")
    assert _resolve_waveform_width(step, qubit) is None


def test_resolve_waveform_width_none_when_width_unset():
    """None is returned when WaveformData.width is None."""
    qubit = _make_qubit_with_mode("drive", "x_pi", None)
    step = PulseOperationStepData(mode_id="drive", waveform_definition="x_pi")
    assert _resolve_waveform_width(step, qubit) is None


def _make_pulse_op(
    op_id: str,
    mode_id: str,
    waveform_id: str,
    interface: str = "public",
) -> OperationData:
    """Primitive operation with a single pulse step."""
    return OperationData(
        id=op_id,
        interface=interface,
        variants=(
            OperationVariantData(
                operation_steps=(
                    PulseOperationStepData(
                        mode_id=mode_id, waveform_definition=waveform_id
                    ),
                ),
            ),
        ),
    )


def _make_ref_op(op_id: str, ref_id: str, interface: str = "public") -> OperationData:
    """Composite operation delegating to a single referenced operation."""
    return OperationData(
        id=op_id,
        interface=interface,
        variants=(
            OperationVariantData(
                operation_steps=(OperationReferenceStepData(operation_id=ref_id),),
            ),
        ),
    )


def test_pulse_duration_direct_pulse_step():
    """A primitive op with one PulseOperationStepData returns the waveform width."""
    op = _make_pulse_op("x", "drive", "x_pi")
    qubit = _make_qubit_with_mode("drive", "x_pi", 100, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) == 100


def test_pulse_duration_inline_waveform_data():
    """Width is resolved correctly from an inline WaveformData in the pulse step."""
    op = OperationData(
        id="x",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(
                    PulseOperationStepData(
                        mode_id="drive",
                        waveform_definition=WaveformData(id="x_pi", width=90),
                    ),
                ),
            ),
        ),
    )
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) == 90


def test_pulse_duration_reference_chain():
    """Duration is the recursive sum of pulse widths through OperationReferenceStepData."""
    x_pi_2 = _make_pulse_op("x_pi_2", "drive", "x_pi_2")
    u_gate = _make_ref_op("u", "x_pi_2")
    rx_gate = _make_ref_op("rx", "u")
    qubit = _make_qubit_with_mode(
        "drive", "x_pi_2", 50, operations=(x_pi_2, u_gate, rx_gate)
    )
    assert _pulse_duration_for_operation(rx_gate, qubit, {"q0": qubit}) == 50


def test_pulse_duration_inline_nested_operation():
    """Inline OperationData steps (not references) contribute their pulse width."""
    inner = _make_pulse_op("x_pi_2", "drive", "x_pi_2")
    outer = OperationData(
        id="outer",
        interface="public",
        variants=(OperationVariantData(operation_steps=(inner,)),),
    )
    qubit = _make_qubit_with_mode("drive", "x_pi_2", 50, operations=(outer,))
    assert _pulse_duration_for_operation(outer, qubit, {"q0": qubit}) == 50


def test_pulse_duration_two_pulse_steps_summed():
    """Operations with two pulse steps in one variant return their combined width."""
    op = OperationData(
        id="cx_local",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(
                    PulseOperationStepData(mode_id="drive", waveform_definition="x_pi_2"),
                    PulseOperationStepData(mode_id="drive", waveform_definition="x_pi_2"),
                ),
            ),
        ),
    )
    qubit = _make_qubit_with_mode("drive", "x_pi_2", 50, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) == 100


def test_pulse_duration_virtual_z_is_zero():
    """An op with only PhaseShiftOperationStepData contributes zero pulse duration."""
    op = OperationData(
        id="rz",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(
                    PhaseShiftOperationStepData(
                        mode_ref=OperationModeReferenceData(mode_id="drive"), phase=1.0
                    ),
                ),
            ),
        ),
    )
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) == 0


def test_pulse_duration_literal_delay_adds_to_total():
    """A DelayOperationStepData with a literal int duration contributes to the total."""
    op = OperationData(
        id="delay",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(DelayOperationStepData(mode_id="drive", duration=200),),
            ),
        ),
    )
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) == 200


def test_pulse_duration_literal_float_delay_adds_to_total():
    """A DelayOperationStepData with a literal float duration is resolved to int ps."""
    op = OperationData(
        id="delay",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(DelayOperationStepData(mode_id="drive", duration=200.0),),
            ),
        ),
    )
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) == 200


def test_pulse_duration_fractional_float_delay_returns_none():
    """A DelayOperationStepData with a non-integral float duration is unresolvable."""
    op = OperationData(
        id="delay",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(DelayOperationStepData(mode_id="drive", duration=200.4),),
            ),
        ),
    )
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) is None


def test_pulse_duration_symbolic_delay_returns_none():
    """A DelayOperationStepData with a parameter-ref duration is unresolvable."""
    op = OperationData(
        id="delay",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(
                    DelayOperationStepData(
                        mode_id="drive",
                        duration=OperationParameterRefData(parameter="t"),
                    ),
                ),
            ),
        ),
    )
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) is None


def test_pulse_duration_sync_contributes_zero():
    """A SyncOperationStepData carries no inherent duration."""
    op = OperationData(
        id="sync_op",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(
                    SyncOperationStepData(
                        mode_refs=frozenset(
                            {
                                OperationModeReferenceData(mode_id="drive"),
                                OperationModeReferenceData(mode_id="readout"),
                            }
                        )
                    ),
                ),
            ),
        ),
    )
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) == 0


def test_pulse_duration_error_step_returns_none():
    """ErrorOperationStepData signals an unimplemented gate and returns None."""
    op = OperationData(
        id="cx",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(ErrorOperationStepData(),),
            ),
        ),
    )
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) is None


def test_pulse_duration_acquire_step_returns_none():
    """AcquireOperationStepData timing is not modelled, so duration is unresolvable."""
    op = OperationData(
        id="measure",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(
                    AcquireOperationStepData(
                        mode_id="readout", acquire_definition="default"
                    ),
                ),
            ),
        ),
    )
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) is None


def test_pulse_duration_missing_referenced_operation_returns_none():
    """None is returned when the referenced operation id does not exist on the qubit."""
    op = _make_ref_op("rx", "x_pi_2")
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) is None


def test_pulse_duration_missing_cross_qubit_target_returns_none():
    """None is returned when the target qubit for a cross-qubit reference is absent."""
    op = OperationData(
        id="cx",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(
                    OperationReferenceStepData(operation_id="x_pi_2", qubit_id="q1"),
                ),
            ),
        ),
    )
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) is None


def test_pulse_duration_cycle_returns_none():
    """A cycle in the operation reference graph returns None rather than infinite
    recursion."""
    # a -> b -> a
    op_a = _make_ref_op("a", "b")
    op_b = _make_ref_op("b", "a")
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op_a, op_b))
    assert _pulse_duration_for_operation(op_a, qubit, {"q0": qubit}) is None


def test_pulse_duration_cross_qubit_reused_id_is_not_a_cycle():
    """A cross-qubit reference to an op with the same id as the caller is not a cycle."""
    q1_x = _make_pulse_op("x", "drive", "x_pi")
    q0_x = OperationData(
        id="x",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(
                    OperationReferenceStepData(operation_id="x", qubit_id="q1"),
                ),
            ),
        ),
    )
    q0 = CanonicalQubitData(id="q0", index=0, operations=(q0_x,))
    q1 = _make_qubit_with_mode(
        "drive", "x_pi", 100, qubit_id="q1", index=1, operations=(q1_x,)
    )
    assert _pulse_duration_for_operation(q0_x, q0, {"q0": q0, "q1": q1}) == 100


def test_pulse_duration_no_default_variant_returns_none():
    """None is returned when every variant has a non-None predicate (no default)."""
    op = OperationData(
        id="rx",
        interface="public",
        variants=(
            OperationVariantData(
                when=OperationCapabilityPredicateData(capability="has_x_pi"),
                operation_steps=(),
            ),
        ),
    )
    qubit = CanonicalQubitData(id="q0", index=0, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) is None


def test_pulse_duration_waveform_width_none_propagates():
    """None WaveformData.width propagates as None for the whole operation."""
    op = _make_pulse_op("x", "drive", "x_pi")
    qubit = _make_qubit_with_mode("drive", "x_pi", None, operations=(op,))
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) is None


def test_pulse_duration_parallel_pulses_on_different_modes():
    """Pulses on different modes run in parallel; duration is the max, not the sum."""
    op = OperationData(
        id="parallel",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(
                    PulseOperationStepData(mode_id="drive", waveform_definition="a"),
                    PulseOperationStepData(mode_id="readout", waveform_definition="b"),
                ),
            ),
        ),
    )
    qubit = CanonicalQubitData(
        id="q0",
        index=0,
        modes=(
            ModeData(
                id="drive",
                channel_id="ch0",
                waveform_definitions=(WaveformData(id="a", width=100),),
            ),
            ModeData(
                id="readout",
                channel_id="ch1",
                waveform_definitions=(WaveformData(id="b", width=60),),
            ),
        ),
        operations=(op,),
    )
    assert _pulse_duration_for_operation(op, qubit, {"q0": qubit}) == 100


def test_pulse_duration_cross_qubit_parallel_execution():
    """A pulse on the control and a simultaneous reference on the target return the max."""
    cancellation = _make_pulse_op("cancel", "cr_cancel", "tone", interface="private")
    target = _make_qubit_with_mode(
        "cr_cancel", "tone", 200, qubit_id="q1", index=1, operations=(cancellation,)
    )
    op = OperationData(
        id="zx",
        interface="private",
        variants=(
            OperationVariantData(
                operation_steps=(
                    PulseOperationStepData(mode_id="cr", waveform_definition="zx_pulse"),
                    OperationReferenceStepData(operation_id="cancel", qubit_id="q1"),
                ),
            ),
        ),
    )
    control = CanonicalQubitData(
        id="q0",
        index=0,
        modes=(
            ModeData(
                id="cr",
                channel_id="ch0",
                waveform_definitions=(WaveformData(id="zx_pulse", width=200),),
            ),
        ),
        operations=(op,),
    )
    assert _pulse_duration_for_operation(op, control, {"q0": control, "q1": target}) == 200


def test_pulse_duration_cross_qubit_same_mode_name_is_parallel():
    """Cross-qubit pulses on modes that share the same id are parallel, not serial."""
    q1_op = _make_pulse_op("prim", "drive", "a", interface="private")
    q1 = _make_qubit_with_mode(
        "drive", "a", 200, qubit_id="q1", index=1, operations=(q1_op,)
    )
    op = OperationData(
        id="cross",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(
                    PulseOperationStepData(mode_id="drive", waveform_definition="b"),
                    OperationReferenceStepData(operation_id="prim", qubit_id="q1"),
                ),
            ),
        ),
    )
    q0 = CanonicalQubitData(
        id="q0",
        index=0,
        modes=(
            ModeData(
                id="drive",
                channel_id="ch0",
                waveform_definitions=(WaveformData(id="b", width=200),),
            ),
        ),
        operations=(op,),
    )
    assert _pulse_duration_for_operation(op, q0, {"q0": q0, "q1": q1}) == 200


def _make_qubit_for_view(
    qubit_id: str,
    index: int,
    mode_id: str,
    waveform_id: str,
    width: int | None,
) -> CanonicalQubitData:
    """Canonical qubit with one public pulse operation on a single mode."""
    op = _make_pulse_op("x", mode_id, waveform_id)
    return _make_qubit_with_mode(
        mode_id, waveform_id, width, qubit_id=qubit_id, index=index, operations=(op,)
    )


def test_qubit_view_duration_populated_for_public_operations():
    """Duration mapping is populated for public operations with resolvable pulse widths."""
    qubit = _make_qubit_for_view("q0", 0, "drive", "x_pi", 120)
    canonical = CanonicalSystemData(qubits=(qubit,))
    view = QubitView.derive(canonical)
    assert view.qubits["q0"].supported_operations.duration == {"x": 120}


def test_qubit_view_duration_none_for_unresolvable_operation():
    """Duration is None for a public operation whose waveform width cannot be resolved."""
    qubit = _make_qubit_for_view("q0", 0, "drive", "x_pi", None)
    canonical = CanonicalSystemData(qubits=(qubit,))
    view = QubitView.derive(canonical)
    assert view.qubits["q0"].supported_operations.duration == {"x": None}


def test_qubit_view_private_operations_excluded_from_operation_type():
    """Private operations are not included in operation_type."""
    private_op = _make_op("x_pi_2", interface="private")
    public_op = _make_op("rx")
    canonical = CanonicalSystemData(
        qubits=(CanonicalQubitData(id="q0", index=0, operations=(private_op, public_op)),)
    )
    view = QubitView.derive(canonical)
    assert view.qubits["q0"].supported_operations.operation_type == ("rx",)


def test_qubit_view_private_operations_excluded_from_duration():
    """Private operations are not included in the duration mapping."""
    private_op = _make_op("x_pi_2", interface="private")
    public_op = _make_op("rx")
    canonical = CanonicalSystemData(
        qubits=(CanonicalQubitData(id="q0", index=0, operations=(private_op, public_op)),)
    )
    view = QubitView.derive(canonical)
    assert "x_pi_2" not in view.qubits["q0"].supported_operations.duration
    assert "rx" in view.qubits["q0"].supported_operations.duration


def test_qubit_view_duration_is_mapping_not_none():
    """Duration is a Mapping (not None) once the qubit has at least one public operation."""
    canonical = CanonicalSystemData(
        qubits=(CanonicalQubitData(id="q0", index=0, operations=(_make_op("x"),)),)
    )
    view = QubitView.derive(canonical)
    assert view.qubits["q0"].supported_operations.duration is not None


def test_qubit_view_error_stub_operation_has_none_duration():
    """Multi-qubit stub operations (ErrorOperationStepData) get None duration."""
    op = OperationData(
        id="cx",
        interface="public",
        variants=(
            OperationVariantData(
                operation_steps=(ErrorOperationStepData(),),
            ),
        ),
    )
    canonical = CanonicalSystemData(
        qubits=(CanonicalQubitData(id="q0", index=0, operations=(op,)),)
    )
    view = QubitView.derive(canonical)
    assert view.qubits["q0"].supported_operations.duration["cx"] is None


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
