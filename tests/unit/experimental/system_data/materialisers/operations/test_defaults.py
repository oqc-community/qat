# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.canonical.schema import (
    AcquireOperationStepData,
    AttributeEntry,
    DelayOperationStepData,
    ErrorOperationStepData,
    OperationBinaryExprData,
    OperationCapabilityPredicateData,
    OperationComparisonPredicateData,
    OperationData,
    OperationModeReferenceData,
    OperationNamedConstantData,
    OperationPredicateClauseData,
    OperationReferenceStepData,
    OperationUnaryExprData,
    PhaseShiftOperationStepData,
    PulseOperationStepData,
    ResetData,
    SyncOperationStepData,
)
from qat.experimental.system_data.materialisers.operations.defaults import (
    DefaultOperationBuilder,
    _get_attribute_value,
    _resolve_reset_methods,
    make_ccnot_operation,
    make_cnot_operation,
    make_cswap_operation,
    make_cx_operation,
    make_cy_operation,
    make_cz_operation,
    make_ddrop_reset_operation,
    make_default_operations,
    make_ecr_operation,
    make_had_operation,
    make_initiate_operation,
    make_measure_operation,
    make_passive_reset_operation,
    make_reset_operation,
    make_rx_gate,
    make_ry_gate,
    make_rz_gate,
    make_s_operation,
    make_sdg_operation,
    make_swap_operation,
    make_sx_operation,
    make_sxdg_operation,
    make_t_operation,
    make_tdg_operation,
    make_u_gate,
    make_x_gate,
    make_x_pi_2_operation,
    make_x_pi_operation,
    make_y_gate,
    make_z_gate,
    make_zx_neg_pi_4_cancellation_operation,
    make_zx_operation,
    make_zx_pi_4_cancellation_operation,
)

# ── Expected operation-set constants ─────────────────────────────────────────

_SINGLE_QUBIT_PUBLIC_IDS = frozenset(
    {
        "Z",
        "X",
        "Y",
        "U",
        "H",
        "SX",
        "SXdg",
        "S",
        "Sdg",
        "T",
        "Tdg",
        "rx",
        "ry",
        "rz",
        "u1",
        "u2",
        "id",
        "delay",
        "measure",
        "initiate",
        "reset",
    }
)
_SINGLE_QUBIT_PRIVATE_IDS = frozenset({"X_pi_2", "X_pi", "passive_reset"})
_SINGLE_QUBIT_ALL_IDS = _SINGLE_QUBIT_PUBLIC_IDS | _SINGLE_QUBIT_PRIVATE_IDS


def _coupling_ids(target: str) -> frozenset[str]:
    return frozenset({f"zx_{target}", f"ecr_{target}", f"cnot_{target}"})


def _cancellation_ids(control: str) -> frozenset[str]:
    return frozenset(
        {f"zx_pi_4_cancellation_{control}", f"zx_neg_pi_4_cancellation_{control}"}
    )


# ── Cross-gate parametrized sweeps ────────────────────────────────────────────


@pytest.mark.parametrize(
    "make_fn, expected_id, expected_kind, expected_interface",
    [
        (make_x_pi_2_operation, "X_pi_2", "pulse_primitive", "private"),
        (make_x_pi_operation, "X_pi", "pulse_primitive", "private"),
        (make_z_gate, "Z", "gate", "public"),
        (make_x_gate, "X", "gate", "public"),
        (make_y_gate, "Y", "gate", "public"),
        (make_u_gate, "U", "gate", "public"),
        (make_had_operation, "H", "gate", "public"),
        (make_sx_operation, "SX", "gate", "public"),
        (make_sxdg_operation, "SXdg", "gate", "public"),
        (make_s_operation, "S", "gate", "public"),
        (make_sdg_operation, "Sdg", "gate", "public"),
        (make_t_operation, "T", "gate", "public"),
        (make_tdg_operation, "Tdg", "gate", "public"),
        (make_measure_operation, "measure", "gate", "public"),
        (make_initiate_operation, "initiate", "utility", "public"),
        (make_reset_operation, "reset", "utility", "public"),
        (
            lambda: make_passive_reset_operation(duration_ps=1_000_000_000),
            "passive_reset",
            "utility",
            "private",
        ),
        (make_ddrop_reset_operation, "ddrop_reset", "utility", "private"),
    ],
)
def test_operation_basic_properties(
    make_fn, expected_id, expected_kind, expected_interface
):
    """Verify that each operation returns the expected id, kind, and interface."""
    op = make_fn()
    assert op.id == expected_id
    assert op.kind == expected_kind
    assert op.interface == expected_interface


@pytest.mark.parametrize(
    "make_fn",
    [
        make_x_pi_2_operation,
        make_x_pi_operation,
        make_z_gate,
        make_u_gate,
        make_had_operation,
        make_sx_operation,
        make_sxdg_operation,
        make_s_operation,
        make_sdg_operation,
        make_t_operation,
        make_tdg_operation,
        make_measure_operation,
        make_initiate_operation,
        lambda: make_passive_reset_operation(duration_ps=1_000_000_000),
        make_ddrop_reset_operation,
    ],
)
def test_single_variant_operations_are_unconditional(make_fn):
    """Verify that fixed-structure operations produce exactly one unconditional variant
    (when=None)."""
    op = make_fn()
    assert len(op.variants) == 1
    assert op.variants[0].when is None


# ── Internal helper unit tests ────────────────────────────────────────────────


def test_get_attribute_value_returns_none_for_absent_key():
    """_get_attribute_value returns None when the key is not present in attributes."""
    attrs = (AttributeEntry(key="duration", value=500),)
    assert _get_attribute_value(attrs, "nonexistent") is None


def test_resolve_reset_methods_falls_back_to_first_type_when_no_passive_or_default():
    """When no matching default and no passive method present, the first method type is
    used."""
    reset_methods = (ResetData(type="ddrop", operation_name="ddrop_reset", attributes=()),)
    _, resolved_default = _resolve_reset_methods(reset_methods, None)
    assert resolved_default == "ddrop"


@pytest.mark.parametrize(
    "make_fn",
    [
        make_had_operation,
        make_sx_operation,
        make_sxdg_operation,
        make_s_operation,
        make_sdg_operation,
        make_t_operation,
        make_tdg_operation,
        make_measure_operation,
        make_initiate_operation,
    ],
)
def test_derived_operations_have_no_parameters(make_fn):
    """Verify that alias gates that delegate with a fixed angle expose no parameters of
    their own."""
    assert make_fn().parameters == ()


@pytest.mark.parametrize(
    "make_fn, base_op",
    [
        (make_sx_operation, "rx"),
        (make_sxdg_operation, "rx"),
        (make_s_operation, "rz"),
        (make_sdg_operation, "rz"),
        (make_t_operation, "rz"),
        (make_tdg_operation, "rz"),
    ],
)
def test_clifford_t_gate_delegates_to_base(make_fn, base_op):
    """Verify that Clifford/T alias gates delegate to their base rotation operation with a
    theta argument."""
    (step,) = make_fn().variants[0].operation_steps
    assert isinstance(step, OperationReferenceStepData)
    assert step.operation_id == base_op
    assert len(step.arguments) == 1
    assert step.arguments[0][0] == "theta"


@pytest.mark.parametrize(
    "make_fn, expected_sign",
    [
        (make_sx_operation, "positive"),
        (make_sxdg_operation, "negative"),
        (make_s_operation, "positive"),
        (make_sdg_operation, "negative"),
        (make_t_operation, "positive"),
        (make_tdg_operation, "negative"),
    ],
)
def test_clifford_t_gate_theta_sign(make_fn, expected_sign):
    """Verify that dagger variants use a negated theta and standard variants use a positive
    theta."""
    (step,) = make_fn().variants[0].operation_steps
    theta = step.arguments[0][1]
    if expected_sign == "positive":
        assert not isinstance(theta, OperationUnaryExprData)
    else:
        assert isinstance(theta, OperationUnaryExprData) and theta.op == "neg"


@pytest.mark.parametrize(
    "make_fn, expected_id",
    [
        (lambda: make_swap_operation("q1"), "swap_q1"),
        (lambda: make_cx_operation("q1"), "cx_q1"),
        (lambda: make_cy_operation("q1"), "cy_q1"),
        (lambda: make_cz_operation("q1"), "cz_q1"),
        (lambda: make_ccnot_operation("q1", "q2"), "ccnot_q1_q2"),
        (lambda: make_cswap_operation("q1", "q2"), "cswap_q1_q2"),
    ],
)
def test_stub_operation_has_error_step(make_fn, expected_id):
    """Each not-yet-implemented stub returns OperationData with an ErrorOperationStepData
    variant."""
    op = make_fn()
    assert op.id == expected_id
    assert len(op.variants) == 1
    (step,) = op.variants[0].operation_steps
    assert isinstance(step, ErrorOperationStepData)
    assert step.error_type == "NotImplementedError"


# ── Pulse primitives ──────────────────────────────────────────────────────────


class TestXPi2Operation:
    @pytest.fixture(scope="class")
    def op(self):
        return make_x_pi_2_operation()

    def test_fires_drive_mode(self, op):
        """Verify that X(π/2) emits a pulse on the drive mode using the x_pi_2 waveform
        definition."""
        (step,) = op.variants[0].operation_steps
        assert isinstance(step, PulseOperationStepData)
        assert step.mode_id == "drive"
        assert step.waveform_definition == "x_pi_2"


class TestXPiOperation:
    @pytest.fixture(scope="class")
    def op(self):
        return make_x_pi_operation()

    def test_fires_drive_mode(self, op):
        """Verify that X(π) emits a pulse on the drive mode using the x_pi waveform
        definition."""
        (step,) = op.variants[0].operation_steps
        assert isinstance(step, PulseOperationStepData)
        assert step.mode_id == "drive"
        assert step.waveform_definition == "x_pi"


# ── Rz gate ───────────────────────────────────────────────────────────────────


class TestRzGate:
    @pytest.fixture(scope="class")
    def op(self):
        return make_rz_gate()

    @pytest.fixture(scope="class")
    def op_coupled(self):
        return make_rz_gate(own_qubit_id="q0", coupled_qubit_ids=("q1",))

    @pytest.fixture(scope="class")
    def op_two_coupled(self):
        return make_rz_gate(own_qubit_id="q0", coupled_qubit_ids=("q1", "q2"))

    @pytest.fixture(scope="class")
    def op_no_own_id(self):
        return make_rz_gate(coupled_qubit_ids=("q1",))

    def test_fires_drive_mode_as_phase_shift(self, op):
        """Verify that the bare rz gate emits a single drive-mode
        PhaseShiftOperationStepData."""
        (step,) = op.variants[0].operation_steps
        assert isinstance(step, PhaseShiftOperationStepData)
        assert step.mode_ref == OperationModeReferenceData(mode_id="drive")

    def test_has_optional_theta_defaulting_to_pi(self, op):
        """Verify that rz has a single optional theta parameter whose default value is π."""
        assert len(op.parameters) == 1
        p = op.parameters[0]
        assert p.name == "theta"
        assert p.optional is True
        assert isinstance(p.default_value, OperationNamedConstantData)
        assert p.default_value.name == "pi"

    def test_coupled_qubit_adds_crc_and_cr_shifts(self, op_coupled):
        """Verify that rz with one coupled qubit includes drive, CRC, and cross-qubit CR
        phase shifts."""
        drive, crc, cr = op_coupled.variants[0].operation_steps
        assert isinstance(drive, PhaseShiftOperationStepData)
        assert drive.mode_ref == OperationModeReferenceData(mode_id="drive")
        assert isinstance(crc, PhaseShiftOperationStepData)
        assert crc.mode_ref == OperationModeReferenceData(
            mode_id="q1.cross_resonance_cancellation"
        )
        assert isinstance(cr, PhaseShiftOperationStepData)
        assert cr.mode_ref == OperationModeReferenceData(
            mode_id="q0.cross_resonance", qubit_id="q1"
        )

    def test_two_coupled_qubits_step_count(self, op_two_coupled):
        """Verify that rz with two coupled qubits emits 5 steps: drive + 2×(CRC + CR)."""
        # drive + 2*(CRC + CR) = 5 steps
        steps = op_two_coupled.variants[0].operation_steps
        assert len(steps) == 5
        assert steps[0].mode_ref == OperationModeReferenceData(mode_id="drive")

    def test_all_steps_are_phase_shifts(self, op_two_coupled):
        """Verify that every step in a topology-aware rz variant is a
        PhaseShiftOperationStepData."""
        for step in op_two_coupled.variants[0].operation_steps:
            assert isinstance(step, PhaseShiftOperationStepData)

    def test_without_own_qubit_id_omits_cr_shifts(self, op_no_own_id):
        """Verify that omitting own_qubit_id suppresses the cross-qubit CR shift, leaving
        only drive and CRC."""
        steps = op_no_own_id.variants[0].operation_steps
        # drive + CRC only (no cross-qubit CR without own_qubit_id)
        assert len(steps) == 2
        assert steps[1].mode_ref == OperationModeReferenceData(
            mode_id="q1.cross_resonance_cancellation"
        )


# ── Rx gate ───────────────────────────────────────────────────────────────────


class TestRxGate:
    @pytest.fixture(scope="class")
    def op(self):
        return make_rx_gate()

    @pytest.fixture(scope="class")
    def op_no_x_pi(self):
        return make_rx_gate(has_x_pi=False)

    def test_has_optional_theta_defaulting_to_pi(self, op):
        """Verify that rx has a single optional theta parameter whose default value is π."""
        p = op.parameters[0]
        assert p.name == "theta"
        assert p.optional is True
        assert isinstance(p.default_value, OperationNamedConstantData)
        assert p.default_value.name == "pi"

    def test_has_five_variants(self, op):
        """Verify that rx(has_x_pi=True) produces exactly 5 variants."""
        assert len(op.variants) == 5

    def test_first_four_variants_are_conditional(self, op):
        """Verify that the first four rx variants all carry a predicate (none are
        unconditional)."""
        for variant in op.variants[:4]:
            assert variant.when is not None

    def test_last_variant_is_default_via_u(self, op):
        """Verify that the unconditional fallback delegates to the U gate with correct phi
        and lambda expressions."""
        default = op.variants[4]
        assert default.when is None
        (step,) = default.operation_steps
        assert step.operation_id == "U"
        args = dict(step.arguments)
        assert set(args) == {"theta", "phi", "lambda"}
        assert isinstance(args["phi"], OperationUnaryExprData) and args["phi"].op == "neg"
        assert (
            isinstance(args["lambda"], OperationBinaryExprData)
            and args["lambda"].op == "div"
        )

    def test_pi_half_variants_are_bare_isclose(self, op):
        """±π/2 variants need no direct_x_pi capability."""
        for variant in op.variants[:2]:
            assert isinstance(variant.when, OperationComparisonPredicateData)

    def test_pi_variants_require_direct_x_pi(self, op):
        """±π variants require the direct_x_pi capability guard."""
        for variant in op.variants[2:4]:
            assert isinstance(variant.when, OperationPredicateClauseData)
            assert variant.when.op == "all"
            cap = variant.when.predicates[1]
            assert isinstance(cap, OperationCapabilityPredicateData)
            assert cap.capability == "direct_x_pi"

    @pytest.mark.parametrize("variant_idx, expected_op", [(0, "X_pi_2"), (2, "X_pi")])
    def test_direct_variants_target_correct_primitive(self, op, variant_idx, expected_op):
        """Verify that positive-angle variants reference the appropriate pulse primitive
        directly."""
        (step,) = op.variants[variant_idx].operation_steps
        assert isinstance(step, OperationReferenceStepData)
        assert step.operation_id == expected_op

    @pytest.mark.parametrize("variant_idx, expected_core_op", [(1, "X_pi_2"), (3, "X_pi")])
    def test_negative_variants_are_z_wrapped(self, op, variant_idx, expected_core_op):
        """Θ < 0 uses rz(−π) → X → rz(π), matching builder z-transform behavior."""
        z_pre, x_step, z_post = op.variants[variant_idx].operation_steps
        assert isinstance(z_pre, OperationReferenceStepData) and z_pre.operation_id == "rz"
        assert (
            isinstance(x_step, OperationReferenceStepData)
            and x_step.operation_id == expected_core_op
        )
        assert (
            isinstance(z_post, OperationReferenceStepData) and z_post.operation_id == "rz"
        )
        assert isinstance(z_pre.arguments[0][1], OperationUnaryExprData)
        assert z_pre.arguments[0][1].op == "neg"
        assert isinstance(z_post.arguments[0][1], OperationNamedConstantData)
        assert z_post.arguments[0][1].name == "pi"

    def test_without_x_pi_has_three_variants(self, op_no_x_pi):
        """Verify that rx(has_x_pi=False) produces exactly 3 variants."""
        assert len(op_no_x_pi.variants) == 3

    def test_without_x_pi_no_x_pi_references(self, op_no_x_pi):
        """Verify that no step in any variant of rx(has_x_pi=False) references X_pi."""
        for variant in op_no_x_pi.variants:
            for step in variant.operation_steps:
                if isinstance(step, OperationReferenceStepData):
                    assert step.operation_id != "X_pi"

    def test_without_x_pi_default_variant_is_last(self, op_no_x_pi):
        """Verify that the fallback U variant is the last variant when has_x_pi=False."""
        assert op_no_x_pi.variants[-1].when is None


# ── Ry gate ───────────────────────────────────────────────────────────────────


class TestRyGate:
    @pytest.fixture(scope="class")
    def op(self):
        return make_ry_gate()

    @pytest.fixture(scope="class")
    def op_no_x_pi(self):
        return make_ry_gate(has_x_pi=False)

    def test_has_optional_theta_defaulting_to_pi(self, op):
        """Verify that ry has a single optional theta parameter whose default value is π."""
        p = op.parameters[0]
        assert p.name == "theta"
        assert p.optional is True
        assert isinstance(p.default_value, OperationNamedConstantData)
        assert p.default_value.name == "pi"

    def test_has_five_variants(self, op):
        """Verify that ry(has_x_pi=True) produces exactly 5 variants."""
        assert len(op.variants) == 5

    def test_pi_over_2_variant_has_z_wrap(self, op):
        """Θ ≈ π/2: rz(−π/2) → X_pi_2 → rz(π/2)."""
        variant = op.variants[0]
        assert isinstance(variant.when, OperationComparisonPredicateData)
        assert variant.when.op == "isclose"
        z_pre, x, z_post = variant.operation_steps
        assert isinstance(z_pre, OperationReferenceStepData) and z_pre.operation_id == "rz"
        assert isinstance(x, OperationReferenceStepData) and x.operation_id == "X_pi_2"
        assert (
            isinstance(z_post, OperationReferenceStepData) and z_post.operation_id == "rz"
        )
        assert isinstance(z_pre.arguments[0][1], OperationUnaryExprData)
        assert z_pre.arguments[0][1].op == "neg"
        assert isinstance(z_post.arguments[0][1], OperationBinaryExprData)
        assert z_post.arguments[0][1].op == "div"

    def test_pi_variants_require_direct_x_pi(self, op):
        """Verify that the ±π variants of ry require the direct_x_pi capability guard."""
        for variant in op.variants[2:4]:
            assert isinstance(variant.when, OperationPredicateClauseData)
            assert variant.when.op == "all"
            cap = variant.when.predicates[1]
            assert isinstance(cap, OperationCapabilityPredicateData)
            assert cap.capability == "direct_x_pi"

    def test_default_variant_delegates_to_u_with_zero_phi_lambda(self, op):
        """Verify that the ry fallback U variant uses phi=0 and lambda=0."""
        default = op.variants[4]
        assert default.when is None
        (step,) = default.operation_steps
        assert step.operation_id == "U"
        args = dict(step.arguments)
        assert args["phi"] == 0.0
        assert args["lambda"] == 0.0

    def test_without_x_pi_has_three_variants(self, op_no_x_pi):
        """Verify that ry(has_x_pi=False) produces exactly 3 variants."""
        assert len(op_no_x_pi.variants) == 3

    def test_without_x_pi_no_x_pi_references(self, op_no_x_pi):
        """Verify that no step in any variant of ry(has_x_pi=False) references X_pi."""
        for variant in op_no_x_pi.variants:
            for step in variant.operation_steps:
                if isinstance(step, OperationReferenceStepData):
                    assert step.operation_id != "X_pi"


# ── U gate ────────────────────────────────────────────────────────────────────


class TestUGate:
    @pytest.fixture(scope="class")
    def op(self):
        return make_u_gate()

    def test_has_three_parameters(self, op):
        """Verify that U has exactly three optional parameters: theta, phi, and lambda."""
        names = [p.name for p in op.parameters]
        assert names == ["theta", "phi", "lambda"]
        for p in op.parameters:
            assert p.optional is True
            assert p.default_value is None

    def test_decomposition_steps(self, op):
        """Rz(λ+π) → X_pi_2 → rz(π−θ) → X_pi_2 → rz(φ)."""
        z_lamb, x1, z_theta, x2, z_phi = op.variants[0].operation_steps

        assert (
            isinstance(z_lamb, OperationReferenceStepData) and z_lamb.operation_id == "rz"
        )
        assert isinstance(z_lamb.arguments[0][1], OperationBinaryExprData)
        assert z_lamb.arguments[0][1].op == "add"

        assert isinstance(x1, OperationReferenceStepData) and x1.operation_id == "X_pi_2"

        assert (
            isinstance(z_theta, OperationReferenceStepData) and z_theta.operation_id == "rz"
        )
        assert isinstance(z_theta.arguments[0][1], OperationBinaryExprData)
        assert z_theta.arguments[0][1].op == "sub"

        assert isinstance(x2, OperationReferenceStepData) and x2.operation_id == "X_pi_2"
        assert isinstance(z_phi, OperationReferenceStepData) and z_phi.operation_id == "rz"


# ── Hadamard, measure, initiate ───────────────────────────────────────────────


class TestHadamardGate:
    @pytest.fixture(scope="class")
    def op(self):
        return make_had_operation()

    def test_decomposes_as_z_then_ry_pi_over_2(self, op):
        """Verify that H decomposes as Z → ry(π/2), matching the builder had()
        implementation."""
        z_step, y_step = op.variants[0].operation_steps
        assert isinstance(z_step, OperationReferenceStepData) and z_step.operation_id == "Z"
        assert (
            isinstance(y_step, OperationReferenceStepData) and y_step.operation_id == "ry"
        )
        theta_val = y_step.arguments[0][1]
        assert isinstance(theta_val, OperationBinaryExprData) and theta_val.op == "div"
        assert isinstance(theta_val.left, OperationNamedConstantData)
        assert theta_val.right == 2


class TestMeasureOperation:
    @pytest.fixture(scope="class")
    def op(self):
        return make_measure_operation()

    def test_fires_pulse_then_acquire(self, op):
        """Verify that measure emits a pulse on the measure mode followed by an acquire
        step."""
        pulse, acquire = op.variants[0].operation_steps
        assert isinstance(pulse, PulseOperationStepData)
        assert pulse.mode_id == "measure"
        assert pulse.waveform_definition == "measure"
        assert isinstance(acquire, AcquireOperationStepData)
        assert acquire.mode_id == "acquire"
        assert acquire.acquire_definition == "acquire"


class TestInitiateOperation:
    @pytest.fixture(scope="class")
    def op(self):
        return make_initiate_operation()

    def test_is_no_op(self, op):
        """Verify that initiate is a utility operation with one unconditional variant
        containing no steps."""
        assert op.kind == "utility"
        assert len(op.variants) == 1
        assert op.variants[0].operation_steps == ()


class TestResetOperation:
    @pytest.fixture(scope="class")
    def passive_op(self):
        return make_passive_reset_operation(duration_ps=1_000_000_000)

    @pytest.fixture(scope="class")
    def ddrop_op(self):
        return make_ddrop_reset_operation()

    @pytest.fixture(scope="class")
    def ddrop_op_with_delay(self):
        return make_ddrop_reset_operation(delay_ps=50_000)

    @pytest.fixture(scope="class")
    def reset_op(self):
        return make_reset_operation()

    def test_passive_reset_uses_drive_delay(self, passive_op):
        """Verify passive reset is represented as a delay on drive mode."""
        (step,) = passive_op.variants[0].operation_steps
        assert isinstance(step, DelayOperationStepData)
        assert step.mode_id == "drive"

    def test_ddrop_reset_uses_reset_pulse(self, ddrop_op):
        """Verify DDROP reset without delay emits only the two pulse steps."""
        qubit_pulse, res_pulse = ddrop_op.variants[0].operation_steps
        assert isinstance(qubit_pulse, PulseOperationStepData)
        assert qubit_pulse.mode_id == "reset"
        assert qubit_pulse.waveform_definition == "ddrop_reset"
        assert isinstance(res_pulse, PulseOperationStepData)
        assert res_pulse.mode_id == "readout_reset"
        assert res_pulse.waveform_definition == "ddrop_reset"

    def test_ddrop_reset_with_delay_appends_delay_steps(self, ddrop_op_with_delay):
        """Verify DDROP reset with delay_ps appends integer delay steps on each mode."""
        qubit_pulse, res_pulse, qubit_delay, res_delay = ddrop_op_with_delay.variants[
            0
        ].operation_steps

        assert isinstance(qubit_pulse, PulseOperationStepData)
        assert qubit_pulse.mode_id == "reset"
        assert isinstance(res_pulse, PulseOperationStepData)
        assert res_pulse.mode_id == "readout_reset"

        assert isinstance(qubit_delay, DelayOperationStepData)
        assert qubit_delay.mode_id == "reset"
        assert qubit_delay.duration == 50_000

        assert isinstance(res_delay, DelayOperationStepData)
        assert res_delay.mode_id == "readout_reset"
        assert res_delay.duration == 50_000

    def test_reset_defaults_to_passive_when_no_metadata(self, reset_op):
        """Verify reset defaults to passive reset when no reset metadata is supplied."""
        assert len(reset_op.variants) == 1
        (step,) = reset_op.variants[0].operation_steps
        assert isinstance(step, OperationReferenceStepData)
        assert step.operation_id == "passive_reset"
        assert reset_op.parameters == ()

    def test_reset_uses_default_reset_method_operation_name(self):
        """Verify reset resolves the default method via operation_name metadata."""
        reset_methods = (
            ResetData(
                type="passive",
                operation_name="passive_custom",
                attributes=(AttributeEntry(key="duration", value=1234),),
            ),
            ResetData(
                type="ddrop",
                operation_name="ddrop_custom",
                attributes=(),
            ),
        )

        op = make_reset_operation(
            reset_methods=reset_methods,
            default_reset_method="ddrop",
        )

        assert op.parameters == ()
        assert len(op.variants) == 1
        (step,) = op.variants[0].operation_steps
        assert isinstance(step, OperationReferenceStepData)
        assert step.operation_id == "ddrop_custom"

    def test_reset_passive_uses_operation_name_from_reset_metadata(self):
        """Verify passive reset forwards to the operation named in reset metadata."""
        reset_methods = (
            ResetData(
                type="passive",
                operation_name="passive_custom",
                attributes=(AttributeEntry(key="duration", value=4321),),
            ),
        )

        op = make_reset_operation(
            reset_methods=reset_methods,
            default_reset_method="passive",
        )

        assert op.parameters == ()
        (step,) = op.variants[0].operation_steps
        assert isinstance(step, OperationReferenceStepData)
        assert step.operation_id == "passive_custom"


# ── ZX / ECR / CNOT / cancellations ──────────────────────────────────────────


class TestZxOperation:
    @pytest.fixture(scope="class")
    def op(self):
        return make_zx_operation(target_qubit_id="q1", own_qubit_id="q0")

    def test_properties(self, op):
        """Verify that the ZX gate has the expected id, kind, and interface."""
        assert op.id == "zx_q1"
        assert op.kind == "gate"
        assert op.interface == "private"

    def test_pi_4_variant_steps(self, op):
        """Verify the π/4 variant: sync → CR pulse → cancellation-tone reference → sync."""
        sync_pre, pulse, cancel_ref, sync_post = op.variants[0].operation_steps
        assert isinstance(sync_pre, SyncOperationStepData)
        assert sync_pre.mode_refs == frozenset(
            {
                OperationModeReferenceData(mode_id="q1.cross_resonance", qubit_id="q0"),
                OperationModeReferenceData(
                    mode_id="q0.cross_resonance_cancellation", qubit_id="q1"
                ),
            }
        )
        assert isinstance(pulse, PulseOperationStepData)
        assert pulse.mode_id == "q1.cross_resonance"
        assert pulse.waveform_definition == "zx_pi_4"
        assert isinstance(cancel_ref, OperationReferenceStepData)
        assert cancel_ref.operation_id == "zx_pi_4_cancellation_q0"
        assert cancel_ref.qubit_id == "q1"
        assert isinstance(sync_post, SyncOperationStepData)
        assert sync_post.mode_refs == sync_pre.mode_refs

    def test_neg_pi_4_variant_uses_neg_waveform(self, op):
        """Verify that the −π/4 variant references the zx_neg_pi_4 waveform and matching
        cancellation op."""
        _, pulse, cancel_ref, _ = op.variants[1].operation_steps
        assert pulse.waveform_definition == "zx_neg_pi_4"
        assert cancel_ref.operation_id == "zx_neg_pi_4_cancellation_q0"

    def test_fallback_variant_is_error(self, op):
        """Verify that the unconditional fallback contains an ErrorOperationStepData
        signalling NotImplementedError."""
        fallback = op.variants[2]
        assert fallback.when is None
        (step,) = fallback.operation_steps
        assert isinstance(step, ErrorOperationStepData)
        assert step.error_type == "NotImplementedError"


class TestZxCancellationOperations:
    @pytest.fixture(scope="class")
    def pi_4_op(self):
        return make_zx_pi_4_cancellation_operation(control_qubit_id="q0")

    @pytest.fixture(scope="class")
    def neg_pi_4_op(self):
        return make_zx_neg_pi_4_cancellation_operation(control_qubit_id="q0")

    def test_pi_4_cancellation_properties(self, pi_4_op):
        """Verify the ZX(π/4) cancellation tone: id, private interface, CR-cancellation
        mode, and waveform."""
        assert pi_4_op.id == "zx_pi_4_cancellation_q0"
        assert pi_4_op.interface == "private"
        (step,) = pi_4_op.variants[0].operation_steps
        assert isinstance(step, PulseOperationStepData)
        assert step.mode_id == "q0.cross_resonance_cancellation"
        assert step.waveform_definition == "zx_pi_4"

    def test_neg_pi_4_cancellation_uses_neg_waveform(self, neg_pi_4_op):
        """Verify that the ZX(−π/4) cancellation tone references the zx_neg_pi_4
        waveform."""
        assert neg_pi_4_op.id == "zx_neg_pi_4_cancellation_q0"
        (step,) = neg_pi_4_op.variants[0].operation_steps
        assert step.waveform_definition == "zx_neg_pi_4"


class TestEcrOperation:
    @pytest.fixture(scope="class")
    def op(self):
        return make_ecr_operation(target_qubit_id="q1")

    def test_decomposes_via_zx_with_correct_angles(self, op):
        """Verify that ECR decomposes as ZX(π/4) → X → ZX(−π/4) with correct theta argument
        expressions."""
        assert op.id == "ecr_q1"
        assert op.interface == "private"
        zx_fwd, x, zx_rev = op.variants[0].operation_steps
        assert (
            isinstance(zx_fwd, OperationReferenceStepData)
            and zx_fwd.operation_id == "zx_q1"
        )
        assert dict(zx_fwd.arguments)["theta"].op == "div"  # π/4 = π÷4
        assert isinstance(x, OperationReferenceStepData) and x.operation_id == "X"
        assert (
            isinstance(zx_rev, OperationReferenceStepData)
            and zx_rev.operation_id == "zx_q1"
        )
        assert dict(zx_rev.arguments)["theta"].op == "neg"  # −π/4


class TestCnotOperation:
    @pytest.fixture(scope="class")
    def op(self):
        return make_cnot_operation(target_qubit_id="q1")

    def test_decomposes_via_ecr(self, op):
        """Verify that CNOT decomposes as ECR → X(ctrl) → rz(ctrl, −π/2) → rx(tgt, −π/2)."""
        assert op.id == "cnot_q1"
        assert op.interface == "public"
        ecr, x_ctrl, z_ctrl, x_tgt = op.variants[0].operation_steps
        assert isinstance(ecr, OperationReferenceStepData) and ecr.operation_id == "ecr_q1"
        assert isinstance(x_ctrl, OperationReferenceStepData) and x_ctrl.operation_id == "X"
        assert (
            isinstance(z_ctrl, OperationReferenceStepData) and z_ctrl.operation_id == "rz"
        )
        assert isinstance(x_tgt, OperationReferenceStepData)
        assert x_tgt.operation_id == "rx"
        assert x_tgt.qubit_id == "q1"


# ── DefaultOperationBuilder class ────────────────────────────────────────────


class TestDefaultOperationBuilder:
    @pytest.fixture(scope="class")
    def builder(self):
        return DefaultOperationBuilder(qubit_id="q0")

    @pytest.fixture(scope="class")
    def builder_coupled(self):
        return DefaultOperationBuilder(
            qubit_id="q0", coupled_qubit_ids=("q1",), control_qubit_ids=("q2",)
        )

    @pytest.fixture(scope="class")
    def builder_no_x_pi(self):
        return DefaultOperationBuilder(qubit_id="q0", has_x_pi=False)

    @pytest.fixture(scope="class")
    def single_qubit_ops(self, builder):
        return builder.build_single_qubit_operations()

    @pytest.fixture(scope="class")
    def full_ops(self, builder_coupled):
        return builder_coupled.build()

    def test_stores_topology_params(self):
        """Verify that the constructor stores qubit_id, coupled_qubit_ids,
        control_qubit_ids, and has_x_pi."""
        b = DefaultOperationBuilder(
            qubit_id="q0",
            coupled_qubit_ids=("q1",),
            control_qubit_ids=("q2",),
            has_x_pi=False,
        )
        assert b.qubit_id == "q0"
        assert b.coupled_qubit_ids == ("q1",)
        assert b.control_qubit_ids == ("q2",)
        assert b.has_x_pi is False

    def test_build_single_qubit_ids(self, single_qubit_ops):
        """Verify that build_single_qubit_operations() returns the expected operation
        IDs."""
        assert {op.id for op in single_qubit_ops} == _SINGLE_QUBIT_ALL_IDS

    def test_build_with_topology_ids_match_module_level(self, full_ops):
        """Verify that build() with full topology returns the same IDs as
        make_default_operations."""
        expected = {
            op.id
            for op in make_default_operations(
                qubit_id="q0", coupled_qubit_ids=("q1",), control_qubit_ids=("q2",)
            )
        }
        assert {op.id for op in full_ops} == expected

    def test_make_rz_gate_passes_topology(self):
        """Verify that make_rz_gate() on a topologically-configured builder includes the CRC
        and CR phase shifts."""
        builder = DefaultOperationBuilder(qubit_id="q0", coupled_qubit_ids=("q1",))
        op = builder.make_rz_gate()
        assert len(op.variants[0].operation_steps) == 3  # drive + CRC + CR

    def test_make_rx_propagates_has_x_pi_false(self, builder_no_x_pi):
        """Verify that make_rx_gate() uses the builder's has_x_pi=False setting, yielding 3
        variants."""
        assert len(builder_no_x_pi.make_rx_gate().variants) == 3

    def test_make_ry_propagates_has_x_pi_false(self, builder_no_x_pi):
        """Verify that make_ry_gate() uses the builder's has_x_pi=False setting, yielding 3
        variants."""
        assert len(builder_no_x_pi.make_ry_gate().variants) == 3

    def test_constructor_requires_qubit_id(self):
        """Verify that qubit_id is a required non-empty string."""
        with pytest.raises(ValueError, match="qubit_id"):
            DefaultOperationBuilder(qubit_id=None)  # type: ignore[arg-type]

    def test_constructor_rejects_empty_qubit_id(self):
        """Verify that an empty qubit_id is rejected."""
        with pytest.raises(ValueError, match="qubit_id"):
            DefaultOperationBuilder(qubit_id="")

    def test_private_single_qubit_operations_include_primitives(self, builder):
        """Verify that default private support ops include pulse and reset helpers."""
        ids = {op.id for op in builder.make_private_single_qubit_operations()}
        assert ids == {"X_pi_2", "X_pi", "passive_reset"}

    def test_private_single_qubit_operations_respect_has_x_pi(self, builder_no_x_pi):
        """Verify has_x_pi=False removes X_pi while retaining reset support ops."""
        ids = {op.id for op in builder_no_x_pi.make_private_single_qubit_operations()}
        assert ids == {"X_pi_2", "passive_reset"}

    def test_private_single_qubit_operations_follow_reset_method_operation_name(self):
        """Verify private reset helper operation IDs come from reset metadata."""
        builder = DefaultOperationBuilder(
            qubit_id="q0",
            reset_methods=(
                ResetData(
                    type="passive",
                    operation_name="passive_custom",
                    attributes=(AttributeEntry(key="duration", value=2000),),
                ),
                ResetData(
                    type="ddrop",
                    operation_name="ddrop_custom",
                    attributes=(),
                ),
            ),
            default_reset_method="passive",
            ddrop_delay_ps=75_000,
        )
        private_ops = {op.id: op for op in builder.make_private_single_qubit_operations()}
        assert "passive_custom" in private_ops
        assert "ddrop_custom" in private_ops
        # Verify delay is threaded into the ddrop operation's steps.
        ddrop_steps = private_ops["ddrop_custom"].variants[0].operation_steps
        assert len(ddrop_steps) == 4
        delay_durations = {
            s.duration for s in ddrop_steps if isinstance(s, DelayOperationStepData)
        }
        assert delay_durations == {75_000}

    def test_private_single_qubit_operations_raises_on_unknown_reset_type(self):
        """Verify an unsupported reset method type raises ValueError."""
        builder = DefaultOperationBuilder(
            qubit_id="q0",
            reset_methods=(
                ResetData(
                    type="unknown_type",
                    operation_name="unknown_reset",
                    attributes=(),
                ),
            ),
            default_reset_method="unknown_type",
        )
        with pytest.raises(ValueError, match="Unsupported reset method type"):
            builder.make_private_single_qubit_operations()

    def test_make_reset_operation_uses_builder_default_reset_method(self):
        """Verify builder reset operation points at the default reset method operation."""
        builder = DefaultOperationBuilder(
            qubit_id="q0",
            reset_methods=(
                ResetData(
                    type="passive",
                    operation_name="passive_custom",
                    attributes=(AttributeEntry(key="duration", value=9000),),
                ),
                ResetData(
                    type="ddrop",
                    operation_name="ddrop_custom",
                    attributes=(),
                ),
            ),
            default_reset_method="ddrop",
        )
        reset_op = builder.make_reset_operation()
        (step,) = reset_op.variants[0].operation_steps
        assert isinstance(step, OperationReferenceStepData)
        assert step.operation_id == "ddrop_custom"

    def test_extra_operations_override_default(self, builder):
        """Verify that an extra OperationData with a matching id replaces the default in
        build_single_qubit_operations."""
        custom_h = OperationData(id="H", kind="gate", interface="public")
        ops = builder.build_single_qubit_operations(extra_operations=(custom_h,))
        h_ops = [op for op in ops if op.id == "H"]
        assert len(h_ops) == 1 and h_ops[0] is custom_h

    def test_build_extra_operations_appended(self, builder):
        """Verify that an extra OperationData with a new id is appended by build."""
        custom_op = OperationData(id="my_custom_gate", kind="gate", interface="public")
        ops = builder.build(extra_operations=(custom_op,))
        assert any(op.id == "my_custom_gate" for op in ops)


# ── Aggregate: make_default_operations ───────────────────────────────────────


class TestDefaultOperations:
    @pytest.fixture(scope="class")
    def ops(self):
        return make_default_operations(qubit_id="q0")

    @pytest.fixture(scope="class")
    def ops_coupled(self):
        return make_default_operations(qubit_id="q0", coupled_qubit_ids=("q1",))

    @pytest.fixture(scope="class")
    def ops_as_target(self):
        return make_default_operations(qubit_id="q1", control_qubit_ids=("q0",))

    @pytest.fixture(scope="class")
    def ops_no_x_pi(self):
        return make_default_operations(qubit_id="q0", has_x_pi=False)

    def test_minimal_args_equals_single_qubit_set(self, ops):
        """Verify that calling with a qubit_id returns the single-qubit operation set."""
        assert {op.id for op in ops} == _SINGLE_QUBIT_ALL_IDS

    def test_with_coupling_adds_expected_ids(self, ops_coupled):
        """Verify that coupled_qubit_ids appends ZX, ECR, and CNOT operations for the target
        qubit."""
        assert {op.id for op in ops_coupled} == _SINGLE_QUBIT_ALL_IDS | _coupling_ids("q1")

    def test_coupling_public_private_split(self, ops_coupled):
        """Verify that CNOT is public and ZX/ECR are private when a coupled qubit is
        present."""
        public_ids = {op.id for op in ops_coupled if op.interface == "public"}
        private_ids = {op.id for op in ops_coupled if op.interface == "private"}
        assert "cnot_q1" in public_ids
        assert {"zx_q1", "ecr_q1"} <= private_ids

    def test_z_gate_includes_crc_and_cr_shifts(self, ops_coupled):
        """Verify that the rz operation in a coupled set contains drive, CRC, and cross-
        qubit CR modes."""
        z_gate = next(op for op in ops_coupled if op.id == "rz")
        mode_refs = [s.mode_ref for s in z_gate.variants[0].operation_steps]
        assert OperationModeReferenceData(mode_id="drive") in mode_refs
        assert (
            OperationModeReferenceData(mode_id="q1.cross_resonance_cancellation")
            in mode_refs
        )
        assert (
            OperationModeReferenceData(mode_id="q0.cross_resonance", qubit_id="q1")
            in mode_refs
        )

    def test_as_target_adds_cancellation_ops(self, ops_as_target):
        """Verify that control_qubit_ids appends the expected ZX cancellation-tone
        operations."""
        assert {op.id for op in ops_as_target} == _SINGLE_QUBIT_ALL_IDS | _cancellation_ids(
            "q0"
        )

    def test_cancellation_ops_are_private(self, ops_as_target):
        """Verify that all ZX cancellation-tone operations have a private interface."""
        for op in ops_as_target:
            if "cancellation" in op.id:
                assert op.interface == "private"

    def test_multiple_couplings(self):
        """Verify that multiple coupled and control qubit IDs all produce the correct
        combined operation set."""
        ops = make_default_operations(
            qubit_id="q0", coupled_qubit_ids=("q1", "q2"), control_qubit_ids=("q3",)
        )
        expected = (
            _SINGLE_QUBIT_ALL_IDS
            | _coupling_ids("q1")
            | _coupling_ids("q2")
            | _cancellation_ids("q3")
        )
        assert {op.id for op in ops} == expected

    def test_no_duplicates(self):
        """Verify that no operation id appears more than once across the full topology-aware
        set."""
        ops = make_default_operations(
            qubit_id="q0", coupled_qubit_ids=("q1",), control_qubit_ids=("q2",)
        )
        ids = [op.id for op in ops]
        assert len(ids) == len(set(ids))

    def test_qubit_id_is_required(self):
        """Verify that make_default_operations requires qubit_id."""
        with pytest.raises(
            TypeError,
            match=r"missing 1 required positional argument: 'qubit_id'",
        ):
            make_default_operations()  # type: ignore[call-arg]

    def test_without_x_pi_excludes_x_pi(self, ops_no_x_pi):
        """Verify that has_x_pi=False removes X_pi from the full operation set."""
        assert "X_pi" not in {op.id for op in ops_no_x_pi}

    def test_without_x_pi_rx_has_three_variants(self, ops_no_x_pi):
        """Verify that has_x_pi=False reduces rx to 3 variants in the full set."""
        rx = next(op for op in ops_no_x_pi if op.id == "rx")
        assert len(rx.variants) == 3

    def test_without_x_pi_ry_has_three_variants(self, ops_no_x_pi):
        """Verify that has_x_pi=False reduces ry to 3 variants in the full set."""
        ry = next(op for op in ops_no_x_pi if op.id == "ry")
        assert len(ry.variants) == 3

    def test_extra_operations_append_new(self):
        """Verify that an extra OperationData with a new id is appended to the full
        topology-aware set."""
        custom_op = OperationData(id="my_custom_gate", kind="gate", interface="public")
        ops = make_default_operations(qubit_id="q0", extra_operations=(custom_op,))
        assert any(op.id == "my_custom_gate" for op in ops)

    def test_all_default_parameters_use_valid_type_exprs(self):
        """Verify that every OperationParameterData in the full default set uses a type_expr
        that is a valid Python type-annotation expression.

        Delegates to :func:`type_expr_checker.is_valid_type_expr` which accepts names,
        subscripts, ``|`` unions, and tuple slices while rejecting literals, arithmetic,
        and call expressions.
        """
        from tests.unit.experimental.system_data.materialisers.operations.type_expr_checker import (
            is_valid_type_expr,
        )

        ops = make_default_operations(
            qubit_id="q0", coupled_qubit_ids=("q1",), control_qubit_ids=("q2",)
        )
        invalid = [
            (op.id, p.name, p.type_expr)
            for op in ops
            for p in op.parameters
            if not is_valid_type_expr(p.type_expr)
        ]
        assert invalid == [], f"Parameters with invalid type_expr: {invalid}"
