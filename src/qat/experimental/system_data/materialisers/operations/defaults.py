# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Default operation construction functions for standard superconducting-qubit gate sets.

This module provides functions that construct canonical
:class:`~qat.experimental.system_data.canonical.schema.OperationData` instances for
the standard single-qubit gate set (X_pi_2, X_pi, Z, X, U, measure) and multi-qubit
gates (CNOT via ECR decomposition).

These factories are designed for use by materialiser plugins such as the PuRR
materialiser, but are not tied to any specific source format.

Mode IDs
--------

The following mode IDs are assumed to exist on the target qubit:

- ``drive`` — main qubit drive channel, used for X pulses and virtual Z frame shifts.
- ``measure`` — measurement pulse channel.
- ``acquire`` — signal acquisition channel.

When constructing topology-aware two-qubit operations, the following coupling mode IDs
are also expected:

- ``<target>.cross_resonance`` — cross-resonance drive mode addressed on the coupled
    target qubit.
- ``<control>.cross_resonance_cancellation`` — cancellation-tone mode on the control
    qubit.

Waveform and acquire-definition IDs
------------------------------------

The following identifiers are assumed to match waveform and acquire definitions present
in the mode's calibration data, as built by the PuRR qubit materialiser:

- ``x_pi_2`` — half-pi X pulse.
- ``x_pi`` — full-pi X pulse.
- ``zx_pi_4`` — ZX(π/4) cross-resonance pulse.
- ``zx_neg_pi_4`` — ZX(−π/4) cross-resonance pulse (phase-inverted).
- ``measure`` — measurement pulse.
- ``acquire`` — acquisition definition.
"""

from typing import Any

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
    OperationParameterData,
    OperationParameterRefData,
    OperationPredicateClauseData,
    OperationReferenceStepData,
    OperationUnaryExprData,
    OperationVariantData,
    PhaseShiftOperationStepData,
    PulseOperationStepData,
    ResetData,
    SyncOperationStepData,
)
from qat.experimental.system_data.materialisers.operations.operation_builder import (
    AbstractOperationBuilder,
)

# ── Shared symbolic values ────────────────────────────────────────────────────

_PI = OperationNamedConstantData(name="pi")

_PI_OVER_2 = OperationBinaryExprData(op="div", left=_PI, right=2)

_NEG_PI = OperationUnaryExprData(op="neg", operand=_PI)

_NEG_PI_OVER_2 = OperationUnaryExprData(op="neg", operand=_PI_OVER_2)

_PI_OVER_4 = OperationBinaryExprData(op="div", left=_PI, right=4)

_NEG_PI_OVER_4 = OperationUnaryExprData(op="neg", operand=_PI_OVER_4)

_RADIAN_ISCLOSE_TOLERANCE = 1e-8
_DEFAULT_PASSIVE_RESET_DURATION_PS = 1_000_000_000

# ── Internal helpers ──────────────────────────────────────────────────────────


def _unconditional(*steps: Any) -> OperationVariantData:
    """Single unconditional variant wrapping a sequence of operation steps."""
    return OperationVariantData(when=None, operation_steps=steps)


def _simple_operation(
    id: str,
    kind: str,
    interface: str,
    *steps: Any,
    parameters: tuple[OperationParameterData, ...] = (),
) -> OperationData:
    """OperationData with a single unconditional variant and optional parameters."""
    return OperationData(
        id=id,
        kind=kind,
        interface=interface,
        parameters=parameters,
        variants=(_unconditional(*steps),),
    )


def _get_attribute_value(
    attributes: tuple[AttributeEntry, ...],
    key: str,
) -> Any | None:
    """Return the value for ``key`` from ``attributes`` if present."""
    for attribute in attributes:
        if attribute.key == key:
            return attribute.value
    return None


def _resolve_reset_methods(
    reset_methods: tuple[ResetData, ...],
    default_reset_method: str | None,
) -> tuple[tuple[ResetData, ...], str]:
    """Resolve reset methods and default method for operation generation."""
    if not reset_methods:
        reset_methods = (
            ResetData(
                type="passive",
                operation_name="passive_reset",
                attributes=(
                    AttributeEntry(
                        key="duration", value=_DEFAULT_PASSIVE_RESET_DURATION_PS
                    ),
                ),
            ),
        )

    methods_by_type = {method.type: method for method in reset_methods}

    if default_reset_method in methods_by_type:
        return reset_methods, default_reset_method
    if "passive" in methods_by_type:
        return reset_methods, "passive"
    return reset_methods, reset_methods[0].type


def _make_reset_private_operations(
    reset_methods: tuple[ResetData, ...],
    default_reset_method: str | None,
    ddrop_delay_ps: int | None = None,
) -> tuple[tuple[ResetData, ...], str, list[OperationData]]:
    """Build resolved reset methods and their corresponding private operations.

    :param reset_methods: Supported reset strategies (top-level canonical metadata).
    :param default_reset_method: Default reset method type selected from ``reset_methods``.
    :returns: Tuple of (resolved_reset_methods, resolved_default_reset_method, reset_private_ops).
    """
    resolved_reset_methods, resolved_default = _resolve_reset_methods(
        reset_methods,
        default_reset_method,
    )

    reset_private_ops: list[OperationData] = []
    for method in resolved_reset_methods:
        operation_name = method.operation_name
        if method.type == "passive":
            duration = int(_get_attribute_value(method.attributes, "duration"))
            reset_private_ops.append(
                make_passive_reset_operation(
                    operation_id=operation_name,
                    duration_ps=duration,
                )
            )
        elif method.type == "ddrop":
            reset_private_ops.append(
                make_ddrop_reset_operation(
                    operation_id=operation_name,
                    delay_ps=ddrop_delay_ps,
                )
            )
        else:
            raise ValueError(f"Unsupported reset method type: {method.type}")

    return resolved_reset_methods, resolved_default, reset_private_ops


def _theta_ref_gate(id: str, base_op: str, theta: Any) -> OperationData:
    """Public gate that delegates to ``base_op`` with a fixed theta argument."""
    return _simple_operation(
        id,
        "gate",
        "public",
        OperationReferenceStepData(operation_id=base_op, arguments=(("theta", theta),)),
    )


def _z_ref(theta: Any) -> OperationReferenceStepData:
    """Rz rotation reference step with a symbolic theta argument."""
    return OperationReferenceStepData(operation_id="rz", arguments=(("theta", theta),))


def _z_wrap_variant(
    when: Any,
    op_id: str,
    pre_z: Any,
    post_z: Any,
) -> OperationVariantData:
    """Variant: Z(pre_z) → op → Z(post_z).

    Mirrors ``_apply_z_transform_on_operation`` in ``QuantumInstructionBuilder``.
    """
    return OperationVariantData(
        when=when,
        operation_steps=(
            _z_ref(pre_z),
            OperationReferenceStepData(operation_id=op_id),
            _z_ref(post_z),
        ),
    )


def _isclose_and_direct_x_pi(
    theta_ref: OperationParameterRefData,
    target: Any,
    tol: float = _RADIAN_ISCLOSE_TOLERANCE,
) -> OperationPredicateClauseData:
    """``all(isclose(theta, target), direct_x_pi)`` predicate clause."""
    return OperationPredicateClauseData(
        op="all",
        predicates=(
            OperationComparisonPredicateData(
                op="isclose", left=theta_ref, right=target, tolerance=tol
            ),
            OperationCapabilityPredicateData(capability="direct_x_pi"),
        ),
    )


def _cr_cancellation_primitive(
    id: str,
    control_qubit_id: str,
    waveform: str,
) -> OperationData:
    """Cross-resonance cancellation-tone primitive owned by the target qubit."""
    return _simple_operation(
        id,
        "pulse_primitive",
        "private",
        PulseOperationStepData(
            mode_id=f"{control_qubit_id}.cross_resonance_cancellation",
            waveform_definition=waveform,
        ),
    )


# ── Pulse primitives ──────────────────────────────────────────────────────────


def make_x_pi_2_operation() -> OperationData:
    """Return the X(π/2) pulse primitive.

    Applies a calibrated half-pi X pulse on the ``drive`` mode using the ``x_pi_2``
    waveform definition.
    """
    return _simple_operation(
        "X_pi_2",
        "pulse_primitive",
        "private",
        PulseOperationStepData(mode_id="drive", waveform_definition="x_pi_2"),
    )


def make_x_pi_operation() -> OperationData:
    """Return the X(π) pulse primitive.

    Applies a calibrated full-pi X pulse on the ``drive`` mode using the ``x_pi``
    waveform definition.
    """
    return _simple_operation(
        "X_pi",
        "pulse_primitive",
        "private",
        PulseOperationStepData(mode_id="drive", waveform_definition="x_pi"),
    )


# ── Parameterized rotation gates ─────────────────────────────────────────────
# These are the primary parameterized gates. Named gates (X, Y, Z, SX, …) are
# fixed-angle aliases defined in the next section.


def make_rz_gate(
    own_qubit_id: str | None = None,
    coupled_qubit_ids: tuple[str, ...] = (),
) -> OperationData:
    """Return the Rz(θ) virtual rotation gate.

    The canonical parameterized Z-axis rotation. All named Z-axis gates (``Z``,
    ``S``, ``Sdg``, ``T``, ``Tdg``) are fixed-angle aliases of this operation.

    Applies a virtual Rz rotation as a sequence of reference-frame phase shifts,
    mirroring :meth:`QuantumInstructionBuilder._hw_Z`:

    - ``drive`` mode on the owning qubit (always).
    - ``{target}.cross_resonance_cancellation`` mode on the owning qubit, for each
      coupled target qubit.
    - ``{own_qubit_id}.cross_resonance`` mode on each coupled target qubit (cross-qubit
      reference, only when ``own_qubit_id`` is provided).

    No physical pulse is emitted; the runtime implements each step as a frame-offset
    update.

    :param own_qubit_id: Identifier of the qubit that owns this operation. Required
        for generating the cross-qubit CR mode phase shifts.
    :param coupled_qubit_ids: Identifiers of target qubits this qubit drives as control.
    """
    theta_param = OperationParameterData(
        name="theta", type_expr="float", optional=True, default_value=_PI, units="radians"
    )
    _theta_ref = OperationParameterRefData(parameter="theta")

    steps: list = [
        PhaseShiftOperationStepData(
            mode_ref=OperationModeReferenceData(mode_id="drive"), phase=_theta_ref
        )
    ]
    for target_id in coupled_qubit_ids:
        steps.append(
            PhaseShiftOperationStepData(
                mode_ref=OperationModeReferenceData(
                    mode_id=f"{target_id}.cross_resonance_cancellation"
                ),
                phase=_theta_ref,
            )
        )
        if own_qubit_id is not None:
            steps.append(
                PhaseShiftOperationStepData(
                    mode_ref=OperationModeReferenceData(
                        mode_id=f"{own_qubit_id}.cross_resonance",
                        qubit_id=target_id,
                    ),
                    phase=_theta_ref,
                )
            )

    return _simple_operation(
        "rz",
        "gate",
        "public",
        *steps,
        parameters=(theta_param,),
    )


def make_rx_gate(has_x_pi: bool = True) -> OperationData:
    """Return the Rx(θ) gate with conditional hardware-native variants.

    The canonical parameterized X-axis rotation. All named X-axis gates (``X``,
    ``SX``, ``SXdg``) are fixed-angle aliases of this operation.

    Variant selection:

    - θ ≈  π/2 → ``X_pi_2`` (always available; no ``direct_x_pi`` guard)
    - θ ≈ −π/2 → rz(−π) → ``X_pi_2`` → rz(π)
        (always available; no ``direct_x_pi`` guard)
    - θ ≈  π,   ``direct_x_pi`` capability present → ``X_pi`` (only when ``has_x_pi``)
    - θ ≈ −π,   ``direct_x_pi`` capability present → rz(−π) → ``X_pi`` → rz(π)
        (only when ``has_x_pi``)
    - default → decompose via ``U`` gate

    Default angle is π.

    :param has_x_pi: Whether a calibrated X(π) pulse is available on the qubit. When
        ``False`` the two ``X_pi``-referencing variants are omitted, leaving 3 variants.
    """
    theta_param = OperationParameterData(
        name="theta", type_expr="float", optional=True, default_value=_PI, units="radians"
    )
    _theta_ref = OperationParameterRefData(parameter="theta")

    _x_pi_variants = (
        (
            OperationVariantData(
                when=_isclose_and_direct_x_pi(_theta_ref, _PI, _RADIAN_ISCLOSE_TOLERANCE),
                operation_steps=(OperationReferenceStepData(operation_id="X_pi"),),
            ),
            _z_wrap_variant(
                when=_isclose_and_direct_x_pi(
                    _theta_ref, _NEG_PI, _RADIAN_ISCLOSE_TOLERANCE
                ),
                op_id="X_pi",
                pre_z=_NEG_PI,
                post_z=_PI,
            ),
        )
        if has_x_pi
        else ()
    )

    return OperationData(
        id="rx",
        kind="gate",
        interface="public",
        parameters=(theta_param,),
        variants=(
            OperationVariantData(
                when=OperationComparisonPredicateData(
                    op="isclose",
                    left=_theta_ref,
                    right=_PI_OVER_2,
                    tolerance=_RADIAN_ISCLOSE_TOLERANCE,
                ),
                operation_steps=(OperationReferenceStepData(operation_id="X_pi_2"),),
            ),
            OperationVariantData(
                when=OperationComparisonPredicateData(
                    op="isclose",
                    left=_theta_ref,
                    right=_NEG_PI_OVER_2,
                    tolerance=_RADIAN_ISCLOSE_TOLERANCE,
                ),
                operation_steps=(
                    _z_ref(_NEG_PI),
                    OperationReferenceStepData(operation_id="X_pi_2"),
                    _z_ref(_PI),
                ),
            ),
            *_x_pi_variants,
            _unconditional(
                OperationReferenceStepData(
                    operation_id="U",
                    arguments=(
                        ("theta", _theta_ref),
                        ("phi", _NEG_PI_OVER_2),
                        ("lambda", _PI_OVER_2),
                    ),
                ),
            ),
        ),
    )


def make_ry_gate(has_x_pi: bool = True) -> OperationData:
    """Return the Ry(θ) gate with conditional hardware-native variants.

    The canonical parameterized Y-axis rotation. The named gate ``Y`` is a
    fixed-angle alias of this operation.

    Variant selection mirrors the ``QuantumInstructionBuilder.Y`` dispatch:

    - θ ≈  π/2 → rz(−π/2) → X_pi_2 → rz(π/2)
    - θ ≈ −π/2 → rz( π/2) → X_pi_2 → rz(−π/2)
    - θ ≈  π,   ``direct_x_pi`` present → rz(−π/2) → X_pi → rz(π/2)  (only when ``has_x_pi``)
    - θ ≈ −π,   ``direct_x_pi`` present → rz( π/2) → X_pi → rz(−π/2) (only when ``has_x_pi``)
    - default → decompose via ``U(theta, phi=0, lambda=0)``

    Default angle is π.

    :param has_x_pi: Whether a calibrated X(π) pulse is available on the qubit. When
        ``False`` the two ``X_pi``-referencing variants are omitted, leaving 3 variants.
    """
    theta_param = OperationParameterData(
        name="theta",
        type_expr="float",
        optional=True,
        default_value=_PI,
        units="radians",
    )
    _theta_ref = OperationParameterRefData(parameter="theta")

    _x_pi_variants = (
        (
            # ry(±π): direct_x_pi guard required for X_pi.
            _z_wrap_variant(
                when=_isclose_and_direct_x_pi(_theta_ref, _PI, _RADIAN_ISCLOSE_TOLERANCE),
                op_id="X_pi",
                pre_z=_NEG_PI_OVER_2,
                post_z=_PI_OVER_2,
            ),
            _z_wrap_variant(
                when=_isclose_and_direct_x_pi(
                    _theta_ref, _NEG_PI, _RADIAN_ISCLOSE_TOLERANCE
                ),
                op_id="X_pi",
                pre_z=_PI_OVER_2,
                post_z=_NEG_PI_OVER_2,
            ),
        )
        if has_x_pi
        else ()
    )

    return OperationData(
        id="ry",
        kind="gate",
        interface="public",
        parameters=(theta_param,),
        variants=(
            # ry(±π/2): no direct_x_pi guard — X_pi_2 is always available.
            _z_wrap_variant(
                when=OperationComparisonPredicateData(
                    op="isclose",
                    left=_theta_ref,
                    right=_PI_OVER_2,
                    tolerance=_RADIAN_ISCLOSE_TOLERANCE,
                ),
                op_id="X_pi_2",
                pre_z=_NEG_PI_OVER_2,
                post_z=_PI_OVER_2,
            ),
            _z_wrap_variant(
                when=OperationComparisonPredicateData(
                    op="isclose",
                    left=_theta_ref,
                    right=_NEG_PI_OVER_2,
                    tolerance=_RADIAN_ISCLOSE_TOLERANCE,
                ),
                op_id="X_pi_2",
                pre_z=_PI_OVER_2,
                post_z=_NEG_PI_OVER_2,
            ),
            *_x_pi_variants,
            _unconditional(
                OperationReferenceStepData(
                    operation_id="U",
                    arguments=(("theta", _theta_ref), ("phi", 0.0), ("lambda", 0.0)),
                ),
            ),
        ),
    )


# ── Named gate aliases ────────────────────────────────────────────────────────
# Fixed-angle aliases matching the QASM2 qelib1.inc gate set.
# These delegate to the parameterized rotation gates above.


def make_x_gate() -> OperationData:
    """Return the X (Pauli-X) gate: rx(π)."""
    return _theta_ref_gate("X", "rx", _PI)


def make_y_gate() -> OperationData:
    """Return the Y (Pauli-Y) gate: ry(π)."""
    return _theta_ref_gate("Y", "ry", _PI)


def make_z_gate() -> OperationData:
    """Return the Z (Pauli-Z) gate: rz(π)."""
    return _theta_ref_gate("Z", "rz", _PI)


def make_sx_operation() -> OperationData:
    """Return the SX (√X) gate: rx(π/2)."""
    return _theta_ref_gate("SX", "rx", _PI_OVER_2)


def make_sxdg_operation() -> OperationData:
    """Return the SXdg (√X†) gate: rx(−π/2)."""
    return _theta_ref_gate("SXdg", "rx", _NEG_PI_OVER_2)


def make_s_operation() -> OperationData:
    """Return the S gate: rz(π/2)."""
    return _theta_ref_gate("S", "rz", _PI_OVER_2)


def make_sdg_operation() -> OperationData:
    """Return the Sdg gate: rz(−π/2)."""
    return _theta_ref_gate("Sdg", "rz", _NEG_PI_OVER_2)


def make_t_operation() -> OperationData:
    """Return the T gate: rz(π/4)."""
    return _theta_ref_gate("T", "rz", _PI_OVER_4)


def make_tdg_operation() -> OperationData:
    """Return the Tdg gate: rz(−π/4)."""
    return _theta_ref_gate("Tdg", "rz", _NEG_PI_OVER_4)


def make_u_gate() -> OperationData:
    """Return the U(θ, φ, λ) gate.

    Decomposition matches :meth:`QuantumInstructionBuilder.U` and is expressed
    in terms of ``rz`` and ``X_pi_2`` primitives.
    """
    theta_param = OperationParameterData(
        name="theta", type_expr="float", optional=True, units="radians"
    )
    phi_param = OperationParameterData(
        name="phi", type_expr="float", optional=True, units="radians"
    )
    lambda_param = OperationParameterData(
        name="lambda", type_expr="float", optional=True, units="radians"
    )
    _theta_ref = OperationParameterRefData(parameter="theta")
    _phi_ref = OperationParameterRefData(parameter="phi")
    _lambda_ref = OperationParameterRefData(parameter="lambda")

    return _simple_operation(
        "U",
        "gate",
        "public",
        _z_ref(OperationBinaryExprData(op="add", left=_lambda_ref, right=_PI)),
        OperationReferenceStepData(operation_id="X_pi_2"),
        _z_ref(OperationBinaryExprData(op="sub", left=_PI, right=_theta_ref)),
        OperationReferenceStepData(operation_id="X_pi_2"),
        _z_ref(_phi_ref),
        parameters=(theta_param, phi_param, lambda_param),
    )


def make_had_operation() -> OperationData:
    """Return the Hadamard gate.

    Implemented as the sequence Z then ry(π/2), i.e. rz(π) followed by ry(π/2), matching
    :meth:`InstructionBuilder.had`.
    """
    return _simple_operation(
        "H",
        "gate",
        "public",
        OperationReferenceStepData(operation_id="Z"),
        OperationReferenceStepData(operation_id="ry", arguments=(("theta", _PI_OVER_2),)),
    )


# ── Readout ───────────────────────────────────────────────────────────────────


def make_measure_operation() -> OperationData:
    """Return the measure gate.

    Sends a measurement pulse on the ``measure`` mode (using the ``measure`` waveform
    definition) and captures the qubit response on the ``acquire`` mode (using the
    ``acquire`` acquire definition).
    """
    return _simple_operation(
        "measure",
        "gate",
        "public",
        PulseOperationStepData(mode_id="measure", waveform_definition="measure"),
        AcquireOperationStepData(mode_id="acquire", acquire_definition="acquire"),
    )


def make_passive_reset_operation(
    operation_id: str = "passive_reset",
    *,
    duration_ps: int,
) -> OperationData:
    """Return a passive reset operation.

    Passive reset is represented as a delay on the ``drive`` mode, with a
    fixed duration in picoseconds sourced from the canonical reset metadata.
    """
    return _simple_operation(
        operation_id,
        "utility",
        "private",
        DelayOperationStepData(mode_id="drive", duration=duration_ps),
    )


def make_ddrop_reset_operation(
    operation_id: str = "ddrop_reset",
    *,
    delay_ps: int | None = None,
) -> OperationData:
    """Return a DDROP reset operation.

    Fires a simultaneous pulse on the qubit-side ``reset`` mode and the resonator-side
    ``readout_reset`` mode. When ``delay_ps`` is provided, a delay step is appended to
    each mode after the pulse.

    :param delay_ps: Post-pulse settling delay in picoseconds, sourced from the
        ddrop_reset calibration payload. Omit when no delay is calibrated.
    """
    steps: list[Any] = [
        PulseOperationStepData(mode_id="reset", waveform_definition="ddrop_reset"),
        PulseOperationStepData(mode_id="readout_reset", waveform_definition="ddrop_reset"),
    ]
    if delay_ps is not None:
        steps.append(DelayOperationStepData(mode_id="reset", duration=delay_ps))
        steps.append(DelayOperationStepData(mode_id="readout_reset", duration=delay_ps))
    return _simple_operation(operation_id, "utility", "private", *steps)


def make_reset_operation(
    reset_methods: tuple[ResetData, ...] = (),
    default_reset_method: str | None = None,
) -> OperationData:
    """Return the public reset operation.

    The reset method is selected from top-level reset metadata: ``default_reset_method``
    identifies the method type, which is resolved against ``reset_methods`` and mapped to
    the method's ``operation_name`` attribute.
    """
    resolved_methods, resolved_default = _resolve_reset_methods(
        reset_methods,
        default_reset_method,
    )
    method = next(item for item in resolved_methods if item.type == resolved_default)

    return _simple_operation(
        "reset",
        "utility",
        "public",
        OperationReferenceStepData(operation_id=method.operation_name),
    )


# ── Multi-qubit gates ─────────────────────────────────────────────────────────


def make_zx_operation(
    target_qubit_id: str,
    own_qubit_id: str,
) -> OperationData:
    """Return the ZX(θ) cross-resonance entangling gate owned by the control qubit.

    Supports θ = π/4 and θ = −π/4, which are the only angles available in the
    :class:`~qat.ir.instruction_builder.QuantumInstructionBuilder`. For any other
    angle the unconditional fallback variant contains an :class:`ErrorOperationStepData`
    step that signals ``NotImplementedError`` at execution time.

    Variants (in evaluation order):

    1. ``isclose(θ, π/4)`` — fires the ``zx_pi_4`` CR pulse and references the
       matching cancellation tone on the target qubit.
    2. ``isclose(θ, −π/4)`` — fires the ``zx_neg_pi_4`` CR pulse and references the
       matching cancellation tone on the target qubit.
    3. (unconditional fallback) — raises ``NotImplementedError``; general ZX(θ) is not
       yet supported.

    :param target_qubit_id: Identifier of the target (driven) qubit (e.g. ``"q1"``).
    :param own_qubit_id: Identifier of the owning control qubit (e.g. ``"q0"``).
    """
    _theta_ref = OperationParameterRefData(parameter="theta")
    cr_mode = OperationModeReferenceData(
        qubit_id=own_qubit_id,
        mode_id=f"{target_qubit_id}.cross_resonance",
    )
    cancellation_mode = OperationModeReferenceData(
        qubit_id=target_qubit_id,
        mode_id=f"{own_qubit_id}.cross_resonance_cancellation",
    )

    def _zx_variant(
        when_value: Any,
        waveform: str,
        cancellation_op_id: str,
    ) -> OperationVariantData:
        return OperationVariantData(
            when=OperationComparisonPredicateData(
                op="isclose",
                left=_theta_ref,
                right=when_value,
                tolerance=_RADIAN_ISCLOSE_TOLERANCE,
            ),
            operation_steps=(
                SyncOperationStepData(mode_refs=frozenset({cr_mode, cancellation_mode})),
                PulseOperationStepData(
                    mode_id=cr_mode.mode_id, waveform_definition=waveform
                ),
                OperationReferenceStepData(
                    operation_id=cancellation_op_id, qubit_id=target_qubit_id
                ),
                SyncOperationStepData(mode_refs=frozenset({cr_mode, cancellation_mode})),
            ),
        )

    return OperationData(
        id=f"zx_{target_qubit_id}",
        kind="gate",
        interface="private",
        parameters=(
            OperationParameterData(name="theta", type_expr="float", units="radians"),
        ),
        variants=(
            _zx_variant(
                _PI_OVER_4,
                "zx_pi_4",
                f"zx_pi_4_cancellation_{own_qubit_id}",
            ),
            _zx_variant(
                _NEG_PI_OVER_4,
                "zx_neg_pi_4",
                f"zx_neg_pi_4_cancellation_{own_qubit_id}",
            ),
            OperationVariantData(
                operation_steps=(
                    ErrorOperationStepData(
                        error_type="NotImplementedError",
                        message=(
                            "General ZX(θ) gate not yet implemented; "
                            "only ZX(±π/4) are supported."
                        ),
                    ),
                ),
            ),
        ),
    )


def make_zx_pi_4_cancellation_operation(control_qubit_id: str) -> OperationData:
    """Return the ZX(π/4) cancellation-tone primitive (owned by the target qubit).

    Fires the ``zx_pi_4`` waveform on the ``{control_qubit_id}.cross_resonance_cancellation``
    mode to suppress leakage during the control qubit's CR drive.

    :param control_qubit_id: Identifier of the driving control qubit (e.g. ``"q0"``).
    """
    return _cr_cancellation_primitive(
        f"zx_pi_4_cancellation_{control_qubit_id}", control_qubit_id, "zx_pi_4"
    )


def make_zx_neg_pi_4_cancellation_operation(control_qubit_id: str) -> OperationData:
    """Return the ZX(−π/4) cancellation-tone primitive (owned by the target qubit).

    :param control_qubit_id: Identifier of the driving control qubit (e.g. ``"q0"``).
    """
    return _cr_cancellation_primitive(
        f"zx_neg_pi_4_cancellation_{control_qubit_id}", control_qubit_id, "zx_neg_pi_4"
    )


def make_ecr_operation(target_qubit_id: str) -> OperationData:
    """Return the ECR (echoed cross-resonance) gate targeting ``target_qubit_id``.

    Decomposes as::

        ZX(π/4, ctrl→tgt) → X(ctrl, π) → ZX(−π/4, ctrl→tgt)

    :param target_qubit_id: Identifier of the target qubit (e.g. ``"q1"``).
    """
    return _simple_operation(
        f"ecr_{target_qubit_id}",
        "gate",
        "private",
        OperationReferenceStepData(
            operation_id=f"zx_{target_qubit_id}",
            arguments=(("theta", _PI_OVER_4),),
        ),
        OperationReferenceStepData(operation_id="X"),
        OperationReferenceStepData(
            operation_id=f"zx_{target_qubit_id}",
            arguments=(("theta", _NEG_PI_OVER_4),),
        ),
    )


def make_cnot_operation(target_qubit_id: str) -> OperationData:
    """Return a CNOT gate owned by the control qubit, targeting ``target_qubit_id``.

    Decomposes as::

        ECR(ctrl, tgt) → X(ctrl) → rz(ctrl, −π/2) → rx(tgt, −π/2)

    The ECR operation is referenced as ``ecr_{target_qubit_id}`` and must be defined
    as a separately-owned operation on the same control qubit.

    :param target_qubit_id: Identifier of the target qubit (e.g. ``"q1"``).
    """
    return _simple_operation(
        f"cnot_{target_qubit_id}",
        "gate",
        "public",
        OperationReferenceStepData(operation_id=f"ecr_{target_qubit_id}"),
        OperationReferenceStepData(operation_id="X"),
        _z_ref(_NEG_PI_OVER_2),
        OperationReferenceStepData(
            operation_id="rx",
            qubit_id=target_qubit_id,
            arguments=(("theta", _NEG_PI_OVER_2),),
        ),
    )


# ── Not yet implemented (builder raises NotImplementedError) ──────────────────


def make_swap_operation(target_qubit_id: str) -> OperationData:
    """Return a SWAP gate placeholder.

    Not yet implemented in :class:`~qat.ir.instruction_builder.QuantumInstructionBuilder`.
    The builder raises ``NotImplementedError`` for ``swap`` operations.

    :param target_qubit_id: Identifier of the target qubit.
    """
    return OperationData(
        id=f"swap_{target_qubit_id}",
        kind="gate",
        interface="public",
        parameters=(),
        variants=(
            OperationVariantData(
                operation_steps=(
                    ErrorOperationStepData(
                        error_type="NotImplementedError",
                        message="SWAP gate not yet implemented in QuantumInstructionBuilder.",
                    ),
                ),
            ),
        ),
    )


def make_cx_operation(target_qubit_id: str) -> OperationData:
    """Return a CX (controlled-X) gate placeholder.

    Not yet implemented in :class:`~qat.ir.instruction_builder.QuantumInstructionBuilder`.
    The builder raises ``NotImplementedError`` for ``cX`` operations.

    :param target_qubit_id: Identifier of the target qubit.
    """
    return OperationData(
        id=f"cx_{target_qubit_id}",
        kind="gate",
        interface="public",
        parameters=(),
        variants=(
            OperationVariantData(
                operation_steps=(
                    ErrorOperationStepData(
                        error_type="NotImplementedError",
                        message="CX (controlled-X) gate not yet implemented in QuantumInstructionBuilder.",
                    ),
                ),
            ),
        ),
    )


def make_cy_operation(target_qubit_id: str) -> OperationData:
    """Return a CY (controlled-Y) gate placeholder.

    Not yet implemented in :class:`~qat.ir.instruction_builder.QuantumInstructionBuilder`.
    The builder raises ``NotImplementedError`` for ``cY`` operations.

    :param target_qubit_id: Identifier of the target qubit.
    """
    return OperationData(
        id=f"cy_{target_qubit_id}",
        kind="gate",
        interface="public",
        parameters=(),
        variants=(
            OperationVariantData(
                operation_steps=(
                    ErrorOperationStepData(
                        error_type="NotImplementedError",
                        message="CY (controlled-Y) gate not yet implemented in QuantumInstructionBuilder.",
                    ),
                ),
            ),
        ),
    )


def make_cz_operation(target_qubit_id: str) -> OperationData:
    """Return a CZ (controlled-Z) gate placeholder.

    Not yet implemented in :class:`~qat.ir.instruction_builder.QuantumInstructionBuilder`.
    The builder raises ``NotImplementedError`` for ``cZ`` operations.

    :param target_qubit_id: Identifier of the target qubit.
    """
    return OperationData(
        id=f"cz_{target_qubit_id}",
        kind="gate",
        interface="public",
        parameters=(),
        variants=(
            OperationVariantData(
                operation_steps=(
                    ErrorOperationStepData(
                        error_type="NotImplementedError",
                        message="CZ (controlled-Z) gate not yet implemented in QuantumInstructionBuilder.",
                    ),
                ),
            ),
        ),
    )


def make_ccnot_operation(target_qubit_id: str, second_control_id: str) -> OperationData:
    """Return a CCNOT (Toffoli) gate placeholder.

    Not yet implemented in :class:`~qat.ir.instruction_builder.QuantumInstructionBuilder`.
    The builder raises ``NotImplementedError`` for ``ccnot`` operations.

    :param target_qubit_id: Identifier of the target qubit.
    :param second_control_id: Identifier of the second control qubit.
    """
    return OperationData(
        id=f"ccnot_{target_qubit_id}_{second_control_id}",
        kind="gate",
        interface="public",
        parameters=(),
        variants=(
            OperationVariantData(
                operation_steps=(
                    ErrorOperationStepData(
                        error_type="NotImplementedError",
                        message="CCNOT (Toffoli) gate not yet implemented in QuantumInstructionBuilder.",
                    ),
                ),
            ),
        ),
    )


def make_cswap_operation(target1_id: str, target2_id: str) -> OperationData:
    """Return a CSWAP (Fredkin) gate placeholder.

    Not yet implemented in :class:`~qat.ir.instruction_builder.QuantumInstructionBuilder`.
    The builder raises ``NotImplementedError`` for ``cswap`` operations.

    :param target1_id: Identifier of the first target qubit.
    :param target2_id: Identifier of the second target qubit.
    """
    return OperationData(
        id=f"cswap_{target1_id}_{target2_id}",
        kind="gate",
        interface="public",
        parameters=(),
        variants=(
            OperationVariantData(
                operation_steps=(
                    ErrorOperationStepData(
                        error_type="NotImplementedError",
                        message="CSWAP (Fredkin) gate not yet implemented in QuantumInstructionBuilder.",
                    ),
                ),
            ),
        ),
    )


# ── QASM2 alias gates ─────────────────────────────────────────────────────────
# These are the additional single-qubit gates defined in qelib1.inc that are not
# already covered by the core gate set above. All delegate to existing operations
# and add no new hardware requirements.


def make_u1_gate() -> OperationData:
    """Return the U1(λ) gate: rz(λ).

    ``u1(λ)`` in QASM2 ``qelib1.inc`` is equivalent to ``U(0, 0, λ)``, which
    reduces to a pure Z rotation. Implemented by delegating directly to ``rz``.

    :returns: A ``u1`` operation that forwards ``lambda`` to ``rz`` as the θ argument.
    """
    _lam_param = OperationParameterData(name="lambda", type_expr="float", units="radians")
    _lam_ref = OperationParameterRefData(parameter="lambda")
    return OperationData(
        id="u1",
        kind="gate",
        interface="public",
        parameters=(_lam_param,),
        variants=(
            _unconditional(
                OperationReferenceStepData(
                    operation_id="rz",
                    arguments=(("theta", _lam_ref),),
                )
            ),
        ),
    )


def make_u2_gate() -> OperationData:
    """Return the U2(φ, λ) gate: U(π/2, φ, λ).

    ``u2(φ, λ)`` in QASM2 ``qelib1.inc`` is equivalent to ``U(π/2, φ, λ)``.

    :returns: A ``u2`` operation that delegates to ``U`` with a fixed ``theta=π/2``.
    """
    _phi_param = OperationParameterData(name="phi", type_expr="float", units="radians")
    _lam_param = OperationParameterData(name="lambda", type_expr="float", units="radians")
    _phi_ref = OperationParameterRefData(parameter="phi")
    _lam_ref = OperationParameterRefData(parameter="lambda")
    return OperationData(
        id="u2",
        kind="gate",
        interface="public",
        parameters=(_phi_param, _lam_param),
        variants=(
            _unconditional(
                OperationReferenceStepData(
                    operation_id="U",
                    arguments=(
                        ("theta", _PI_OVER_2),
                        ("phi", _phi_ref),
                        ("lambda", _lam_ref),
                    ),
                )
            ),
        ),
    )


def make_id_gate() -> OperationData:
    """Return the identity gate.

    ``id`` in QASM2 ``qelib1.inc`` is a no-op. Represented as an unconditional
    variant with no operation steps.
    """
    return OperationData(
        id="id",
        kind="gate",
        interface="public",
        parameters=(),
        variants=(_unconditional(),),
    )


def make_delay_operation() -> OperationData:
    """Return the ``delay`` operation.

    Delays the ``drive`` mode by ``duration`` picoseconds. Corresponds to the
    QASM3 ``delay`` statement and the
    :meth:`~qat.ir.instruction_builder.QuantumInstructionBuilder.delay` builder
    method.

    Duration is expressed in picoseconds as an integer, matching the canonical
    schema convention used by
    :class:`~qat.experimental.system_data.canonical.schema.DelayOperationStepData`.

    :returns: A ``delay`` operation with a single ``duration`` parameter forwarded
        to a :class:`~qat.experimental.system_data.canonical.schema.DelayOperationStepData`
        on the ``drive`` mode.
    """
    _duration_param = OperationParameterData(
        name="duration", type_expr="int", units="picoseconds"
    )
    _duration_ref = OperationParameterRefData(parameter="duration")
    return OperationData(
        id="delay",
        kind="gate",
        interface="public",
        parameters=(_duration_param,),
        variants=(
            _unconditional(
                DelayOperationStepData(mode_id="drive", duration=_duration_ref),
            ),
        ),
    )


# ── Operation builder class ───────────────────────────────────────────────────


class DefaultOperationBuilder(AbstractOperationBuilder):
    """Builds the default transmon gate-set operation set for a single qubit.

    Each operation is a method that returns an :class:`~qat.experimental.system_data.canonical.schema.OperationData`
    instance.  Subclasses can override individual methods to customise or replace
    specific operations without touching the rest of the set — no knowledge of
    operation IDs is required.

    Topology parameters are supplied at construction time so all methods share
    consistent qubit context.

    Usage — default set::

        ops = DefaultOperationBuilder(
            qubit_id="q0",
            coupled_qubit_ids=("q1",),
            control_qubit_ids=("q2",),
            has_x_pi=True,
        ).build()

    Usage — subclass override::

        class CustomOperationBuilder(DefaultOperationBuilder):
            def make_z_operation(self) -> OperationData:
                return ...  # hardware-specific Z decomposition

        ops = CustomOperationBuilder(qubit_id="q0").build()

    Usage — data-level extension at the call site::

        ops = DefaultOperationBuilder(qubit_id="q0").build(
            extra_operations=(my_custom_gate,)
        )

    Constructor parameters are inherited from
    :class:`~qat.experimental.system_data.materialisers.operations.operation_builder.AbstractOperationBuilder`.
    """

    def __init__(
        self,
        qubit_id: str,
        coupled_qubit_ids: tuple[str, ...] = (),
        control_qubit_ids: tuple[str, ...] = (),
        has_x_pi: bool = True,
        reset_methods: tuple[ResetData, ...] = (),
        default_reset_method: str | None = None,
        ddrop_delay_ps: int | None = None,
    ) -> None:
        super().__init__(
            qubit_id=qubit_id,
            coupled_qubit_ids=coupled_qubit_ids,
            control_qubit_ids=control_qubit_ids,
            has_x_pi=has_x_pi,
        )
        self.reset_methods, self.default_reset_method = _resolve_reset_methods(
            reset_methods,
            default_reset_method,
        )
        self.ddrop_delay_ps = ddrop_delay_ps

    # ── Private/support single-qubit operations ──────────────────────────────

    def make_private_single_qubit_operations(self) -> tuple[OperationData, ...]:
        """Return private pulse primitives used by default gate decompositions."""
        _, _, reset_ops = _make_reset_private_operations(
            self.reset_methods,
            self.default_reset_method,
            self.ddrop_delay_ps,
        )

        return (
            make_x_pi_2_operation(),
            *((make_x_pi_operation(),) if self.has_x_pi else ()),
            *tuple(reset_ops),
        )

    def make_z_operation(self) -> OperationData:
        """Return the fixed-angle Z gate alias (rz(π))."""
        return make_z_gate()

    # ── Single-qubit gates ────────────────────────────────────────────────────

    def make_x_gate(self) -> OperationData:
        """Return the X (Pauli-X) fixed-angle alias gate: rx(π)."""
        return make_x_gate()

    def make_y_gate(self) -> OperationData:
        """Return the Y (Pauli-Y) fixed-angle alias gate: ry(π)."""
        return make_y_gate()

    def make_u_gate(self) -> OperationData:
        """Return the U(θ,φ,λ) gate."""
        return make_u_gate()

    def make_had_operation(self) -> OperationData:
        """Return the Hadamard gate."""
        return make_had_operation()

    def make_sx_operation(self) -> OperationData:
        """Return the SX (√X) gate."""
        return make_sx_operation()

    def make_sxdg_operation(self) -> OperationData:
        """Return the SXdg (√X†) gate."""
        return make_sxdg_operation()

    def make_s_operation(self) -> OperationData:
        """Return the S gate."""
        return make_s_operation()

    def make_sdg_operation(self) -> OperationData:
        """Return the Sdg gate."""
        return make_sdg_operation()

    def make_t_operation(self) -> OperationData:
        """Return the T gate."""
        return make_t_operation()

    def make_tdg_operation(self) -> OperationData:
        """Return the Tdg gate."""
        return make_tdg_operation()

    # ── QASM2 aliases ─────────────────────────────────────────────────────────

    def make_rx_gate(self) -> OperationData:
        """Return the Rx(θ) gate."""
        return make_rx_gate(has_x_pi=self.has_x_pi)

    def make_ry_gate(self) -> OperationData:
        """Return the Ry(θ) gate."""
        return make_ry_gate(has_x_pi=self.has_x_pi)

    def make_rz_gate(self) -> OperationData:
        """Return the Rz(θ) gate."""
        return make_rz_gate(
            own_qubit_id=self.qubit_id,
            coupled_qubit_ids=self.coupled_qubit_ids,
        )

    def make_u1_gate(self) -> OperationData:
        """Return the U1(λ) gate."""
        return make_u1_gate()

    def make_u2_gate(self) -> OperationData:
        """Return the U2(φ,λ) gate."""
        return make_u2_gate()

    def make_id_gate(self) -> OperationData:
        """Return the identity gate."""
        return make_id_gate()

    def make_delay_operation(self) -> OperationData:
        """Return the delay operation (duration in picoseconds)."""
        return make_delay_operation()

    # ── Readout / lifecycle ───────────────────────────────────────────────────

    def make_measure_operation(self) -> OperationData:
        """Return the measure operation."""
        return make_measure_operation()

    def make_initiate_operation(self) -> OperationData:
        """Return the initiate operation."""
        return make_initiate_operation()

    def make_reset_operation(self) -> OperationData:
        """Return the reset operation."""
        return make_reset_operation(
            reset_methods=self.reset_methods,
            default_reset_method=self.default_reset_method,
        )

    # ── Multi-qubit gates ─────────────────────────────────────────────────────

    def make_two_qubit_operations(self) -> tuple[OperationData, ...]:
        """Return topology-derived two-qubit operations for this qubit.

        Includes control-side ZX/ECR/CNOT for ``coupled_qubit_ids`` and
        target-side cancellation operations for ``control_qubit_ids``.
        """
        operations: list[OperationData] = []
        for target_id in self.coupled_qubit_ids:
            operations.append(self.make_zx_operation(target_id))
            operations.append(self.make_ecr_operation(target_id))
            operations.append(self.make_cnot_operation(target_id))
        for ctrl_id in self.control_qubit_ids:
            operations.append(self.make_zx_pi_4_cancellation_operation(ctrl_id))
            operations.append(self.make_zx_neg_pi_4_cancellation_operation(ctrl_id))
        return tuple(operations)

    def make_zx_operation(self, target_qubit_id: str) -> OperationData:
        """Return the ZX(θ) gate targeting ``target_qubit_id``."""
        if self.qubit_id is None:
            raise ValueError(
                "qubit_id must be set on the builder to construct ZX operations."
            )
        return make_zx_operation(
            target_qubit_id=target_qubit_id, own_qubit_id=self.qubit_id
        )

    def make_ecr_operation(self, target_qubit_id: str) -> OperationData:
        """Return the ECR gate targeting ``target_qubit_id``."""
        return make_ecr_operation(target_qubit_id=target_qubit_id)

    def make_cnot_operation(self, target_qubit_id: str) -> OperationData:
        """Return the CNOT gate targeting ``target_qubit_id``."""
        return make_cnot_operation(target_qubit_id=target_qubit_id)

    def make_zx_pi_4_cancellation_operation(self, control_qubit_id: str) -> OperationData:
        """Return the ZX(π/4) cancellation-tone primitive for ``control_qubit_id``."""
        return make_zx_pi_4_cancellation_operation(control_qubit_id=control_qubit_id)

    def make_zx_neg_pi_4_cancellation_operation(
        self, control_qubit_id: str
    ) -> OperationData:
        """Return the ZX(−π/4) cancellation-tone primitive for ``control_qubit_id``."""
        return make_zx_neg_pi_4_cancellation_operation(control_qubit_id=control_qubit_id)


# ── Convenience aggregates ────────────────────────────────────────────────────


def make_initiate_operation() -> OperationData:
    """Return the ``initiate`` operation.

    For the default transmon gate set this is a no-op: the qubit is assumed to
    be in a known ground state before a circuit begins and requires no explicit
    initialisation pulse.  Qubit types that do require active initialisation
    (e.g. spin qubits, cat qubits, or mid-circuit reset strategies) should
    override this with a concrete variant that drives the appropriate hardware
    sequence.
    """
    return OperationData(
        id="initiate",
        kind="utility",
        interface="public",
        parameters=(),
        variants=(
            OperationVariantData(
                when=None,
                operation_steps=(),
            ),
        ),
    )


def make_default_operations(
    qubit_id: str,
    coupled_qubit_ids: tuple[str, ...] = (),
    control_qubit_ids: tuple[str, ...] = (),
    has_x_pi: bool = True,
    reset_methods: tuple[ResetData, ...] = (),
    default_reset_method: str | None = None,
    ddrop_delay_ps: int | None = None,
    extra_operations: tuple[OperationData, ...] = (),
) -> tuple[OperationData, ...]:
    """Return the full default operation set for a qubit.

    :param qubit_id: Identifier of the qubit that will own these operations.
    :param coupled_qubit_ids: Identifiers of qubits this qubit drives as the control
        qubit (e.g. ``("q1", "q2")``). Generates ZX(±π/4), ECR, and CNOT for each.
    :param control_qubit_ids: Identifiers of qubits that drive this qubit (i.e. this
        qubit is the target). Generates ZX cancellation-tone primitives for each.
    :param has_x_pi: Whether a calibrated X(π) pulse is available on the qubit.
        Controls inclusion of ``X_pi`` and the corresponding variants in
        parameterised ``rx`` and ``ry``.
    :param reset_methods: Supported reset strategies (top-level canonical metadata).
    :param default_reset_method: Default reset method type selected from
        ``reset_methods``.
    :param ddrop_delay_ps: Post-pulse settling delay in picoseconds for DDROP reset,
        sourced from the ddrop_reset calibration payload. Omit when uncalibrated.
    :param extra_operations: Additional or replacement operations applied after the full
        default set (including topology-derived multi-qubit operations) is assembled.
        Any operation whose ``id`` matches a default replaces it in-place (last-wins);
        new IDs are appended.
    :returns: Tuple of canonical :class:`OperationData` instances.
    """
    return DefaultOperationBuilder(
        qubit_id=qubit_id,
        coupled_qubit_ids=coupled_qubit_ids,
        control_qubit_ids=control_qubit_ids,
        has_x_pi=has_x_pi,
        reset_methods=reset_methods,
        default_reset_method=default_reset_method,
        ddrop_delay_ps=ddrop_delay_ps,
    ).build(extra_operations=extra_operations)
