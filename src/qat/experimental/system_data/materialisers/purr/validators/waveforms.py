# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Waveform and acquire-payload validation rules for PuRR ingress payloads."""

from __future__ import annotations

import math
from typing import Any

from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.system_data.materialisers.purr.validators.common import (
    _is_numeric,
    _iter_indexed_quantum_devices,
    _raise_validation_error,
)

_SHAPE_PARAMETERS_ALWAYS_REQUIRED = ("width", "amp")

_SHAPE_PARAMETERS_ALWAYS_OPTIONAL = ("phase", "drag")

_SHAPE_PARAMETER_REQUIREMENTS = {
    "blackman": (),
    "cos": (),
    "drag_gaussian": ("rise",),
    "extra_soft_square": ("rise", "std_dev"),
    "gaussian": ("rise",),
    "gaussian_zero_edges": ("rise",),
    "gaussian_square": ("rise", "std_dev"),
    "rounded_square": ("rise", "std_dev"),
    "sech": ("std_dev",),
    "setup_hold": ("rise", "amp_setup"),
    "sin": (),
    "soft_square": ("rise",),
    "softer_gaussian": ("rise",),
    "softer_square": ("rise", "std_dev"),
    "square": (),
}

_SHAPE_PARAMETER_OPTIONAL = {
    "blackman": (),
    "cos": ("frequency", "internal_phase"),
    "drag_gaussian": ("beta", "zero_at_edges"),
    "extra_soft_square": (),
    "gaussian": (),
    "gaussian_zero_edges": ("zero_at_edges",),
    "gaussian_square": (),
    "rounded_square": (),
    "sech": ("zero_at_edges",),
    "setup_hold": (),
    "sin": ("frequency", "internal_phase"),
    "soft_square": (),
    "softer_gaussian": (),
    "softer_square": (),
    "square": (),
}


def _validate_finite_positive_param(
    *,
    param: str,
    value: Any,
    required_params: tuple[str, ...],
    optional_params: tuple[str, ...],
    path: str,
):
    """Raise if *param* is required but missing/invalid, or optional but invalid.

    Valid means a finite number strictly greater than zero.
    """
    invalid = (
        value is None
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    )
    if (param in required_params and invalid) or (
        param in optional_params and value is not None and invalid
    ):
        _raise_validation_error(
            f"Waveform {param} must be a finite positive number.",
            path=path,
            details={"value": value},
        )


def _validate_finite_non_negative_param(
    *,
    param: str,
    value: Any,
    required_params: tuple[str, ...],
    optional_params: tuple[str, ...],
    path: str,
) -> None:
    """Raise if *param* is required but missing/invalid, or optional but invalid.

    Valid means a finite number greater than or equal to zero.
    """
    invalid = (
        value is None
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    )
    if (param in required_params and invalid) or (
        param in optional_params and value is not None and invalid
    ):
        _raise_validation_error(
            f"Waveform {param} must be a finite non-negative number.",
            path=path,
            details={"value": value},
        )


def _validate_finite_param(
    *,
    param: str,
    value: Any,
    required_params: tuple[str, ...],
    optional_params: tuple[str, ...],
    path: str,
) -> None:
    """Raise if *param* is required but missing/invalid, or optional but invalid.

    Valid means a finite number.
    """
    invalid = (
        value is None or not isinstance(value, int | float) or not math.isfinite(value)
    )
    if (param in required_params and invalid) or (
        param in optional_params and value is not None and invalid
    ):
        _raise_validation_error(
            f"Waveform {param} must be a finite number.",
            path=path,
            details={"value": value},
        )


def _validate_bool_int_param(
    *,
    param: str,
    value: Any,
    required_params: tuple[str, ...],
    optional_params: tuple[str, ...],
    path: str,
) -> None:
    """Raise if *param* is required but missing/invalid, or optional but invalid.

    Valid means a boolean or integer.
    """
    invalid = value is None or not isinstance(value, bool | int)
    if (param in required_params and invalid) or (
        param in optional_params and value is not None and invalid
    ):
        _raise_validation_error(
            f"Waveform {param} must be a boolean or integer.",
            path=path,
            details={"value": value, "value_type": type(value).__name__},
        )


def _validate_waveform_field_bounds(
    *,
    device_id: str,
    field_name: str,
    waveform: dict[str, Any],
) -> None:
    """Validates the fields of a waveform are within their allowed bounds.

    ``shape``, ``width``, and ``amp`` are required. ``phase`` and ``drag`` are optional
    parameters shared across waveforms but must be finite when present.

    ``rise`` and ``std_dev`` are shape-dependent and must be positive (> 0) where
    required or when provided. ``amp_setup`` is required for some shapes and must be
    finite. Other shape-dependent parameters (``frequency``, ``internal_phase``, ``beta``,
    ``zero_at_edges``) are validated for finiteness when present.
    """

    shape = waveform.get("shape")
    if not isinstance(shape, str) or shape.lower() not in _SHAPE_PARAMETER_REQUIREMENTS:
        _raise_validation_error(
            "Waveform shape must be one of the allowed shapes.",
            path=f"$.quantum_devices.{device_id}.{field_name}.shape",
            details={"value": shape, "allowed_shapes": list(_SHAPE_PARAMETER_REQUIREMENTS)},
        )
    shape_lower = shape.lower()
    required_params = _SHAPE_PARAMETERS_ALWAYS_REQUIRED + _SHAPE_PARAMETER_REQUIREMENTS.get(
        shape_lower, ()
    )
    optional_params = _SHAPE_PARAMETERS_ALWAYS_OPTIONAL + _SHAPE_PARAMETER_OPTIONAL.get(
        shape_lower, ()
    )

    # Width and amp always required
    _validate_finite_non_negative_param(
        param="width",
        value=waveform.get("width"),
        required_params=required_params,
        optional_params=optional_params,
        path=f"$.quantum_devices.{device_id}.{field_name}.width",
    )
    _validate_finite_param(
        param="amp",
        value=waveform.get("amp"),
        required_params=required_params,
        optional_params=optional_params,
        path=f"$.quantum_devices.{device_id}.{field_name}.amp",
    )

    # Phase and drag are for all shapes, but optional
    _validate_finite_param(
        param="phase",
        value=waveform.get("phase"),
        required_params=required_params,
        optional_params=optional_params,
        path=f"$.quantum_devices.{device_id}.{field_name}.phase",
    )
    _validate_finite_param(
        param="drag",
        value=waveform.get("drag"),
        required_params=required_params,
        optional_params=optional_params,
        path=f"$.quantum_devices.{device_id}.{field_name}.drag",
    )

    # Other parameters are shape-dependent, sometimes required, sometimes optional,
    # sometimes not relevant
    _validate_finite_positive_param(
        param="rise",
        value=waveform.get("rise"),
        required_params=required_params,
        optional_params=optional_params,
        path=f"$.quantum_devices.{device_id}.{field_name}.rise",
    )
    _validate_finite_positive_param(
        param="std_dev",
        value=waveform.get("std_dev"),
        required_params=required_params,
        optional_params=optional_params,
        path=f"$.quantum_devices.{device_id}.{field_name}.std_dev",
    )
    _validate_finite_param(
        param="amp_setup",
        value=waveform.get("amp_setup"),
        required_params=required_params,
        optional_params=optional_params,
        path=f"$.quantum_devices.{device_id}.{field_name}.amp_setup",
    )
    _validate_bool_int_param(
        param="zero_at_edges",
        value=waveform.get("zero_at_edges"),
        required_params=required_params,
        optional_params=optional_params,
        path=f"$.quantum_devices.{device_id}.{field_name}.zero_at_edges",
    )
    _validate_finite_param(
        param="beta",
        value=waveform.get("beta"),
        required_params=required_params,
        optional_params=optional_params,
        path=f"$.quantum_devices.{device_id}.{field_name}.beta",
    )
    _validate_finite_param(
        param="frequency",
        value=waveform.get("frequency"),
        required_params=required_params,
        optional_params=optional_params,
        path=f"$.quantum_devices.{device_id}.{field_name}.frequency",
    )
    _validate_finite_param(
        param="internal_phase",
        value=waveform.get("internal_phase"),
        required_params=required_params,
        optional_params=optional_params,
        path=f"$.quantum_devices.{device_id}.{field_name}.internal_phase",
    )


def _validate_cross_resonance_waveform(
    *,
    device_id: str,
    aux_id: str,
    waveform: dict[str, Any],
) -> None:
    """Validate one cross-resonance waveform payload used for ZX pulses."""

    width = waveform.get("width")
    if width is not None and (
        not isinstance(width, int | float) or not math.isfinite(width) or width < 0
    ):
        _raise_validation_error(
            "Cross-resonance waveform width must be a finite non-negative number when provided.",
            path=f"$.quantum_devices.{device_id}.pulse_hw_zx_pi_4.{aux_id}.width",
            details={"value": width},
        )


def _validate_waveform_payloads(dto: PurrIngressV010) -> None:
    """Validate waveform timing fields that map into canonical waveform definitions."""

    for device_id, payload in _iter_indexed_quantum_devices(dto):
        for field_name in ("pulse_hw_x_pi_2", "pulse_hw_x_pi", "pulse_measure"):
            waveform = payload.get(field_name)
            if not isinstance(waveform, dict):
                continue
            _validate_waveform_field_bounds(
                device_id=device_id,
                field_name=field_name,
                waveform=waveform,
            )

        zx_waveforms = payload.get("pulse_hw_zx_pi_4")
        if isinstance(zx_waveforms, dict):
            for aux_id, waveform in zx_waveforms.items():
                if not isinstance(waveform, dict):
                    continue
                _validate_cross_resonance_waveform(
                    device_id=device_id,
                    aux_id=aux_id,
                    waveform=waveform,
                )


def _validate_measure_acquire_payloads(dto: PurrIngressV010) -> None:
    """Validate acquisition delay, width, and weights used by readout modes."""

    for device_id, payload in _iter_indexed_quantum_devices(dto):
        acquire_payload = payload.get("measure_acquire")
        if not isinstance(acquire_payload, dict):
            continue

        for field_name in ("delay", "width"):
            value = acquire_payload.get(field_name)
            if value is not None and (
                not isinstance(value, int | float) or not math.isfinite(value) or value < 0
            ):
                _raise_validation_error(
                    f"Acquire {field_name} must be a finite non-negative number when provided.",
                    path=f"$.quantum_devices.{device_id}.measure_acquire.{field_name}",
                    details={"value": value},
                )

        weights = acquire_payload.get("weights")
        if weights is not None and not isinstance(weights, list | dict):
            _raise_validation_error(
                "Acquire weights must be a list, dictionary, or null.",
                path=f"$.quantum_devices.{device_id}.measure_acquire.weights",
                details={"value_type": type(weights).__name__},
            )
        if isinstance(weights, list) and not all(_is_numeric(value) for value in weights):
            _raise_validation_error(
                "Acquire weights entries must be numeric or complex.",
                path=f"$.quantum_devices.{device_id}.measure_acquire.weights",
                details={"value_types": [type(value).__name__ for value in weights]},
            )
        elif isinstance(weights, dict):
            if not (
                weights.get("object_type").rsplit(".", 1)[-1] == "CustomPulse"
                and "samples" in weights
            ):
                _raise_validation_error(
                    "Acquire weights dictionary must be a CustomPulse with samples.",
                    path=f"$.quantum_devices.{device_id}.measure_acquire.weights",
                    details={"value": weights},
                )


def _validate_waveform_numeric_fields(dto: PurrIngressV010) -> None:
    """Validate mapped waveform numeric fields and reject NaN or Inf values.

    Covers ``amp``, ``drag``, and ``phase`` which map to canonical WaveformData.
    """

    finite_fields = ("amp", "drag", "phase")

    for device_id, payload in _iter_indexed_quantum_devices(dto):
        for waveform_field in ("pulse_hw_x_pi_2", "pulse_hw_x_pi", "pulse_measure"):
            waveform = payload.get(waveform_field)
            if not isinstance(waveform, dict):
                continue

            for field_name in finite_fields:
                value = waveform.get(field_name)
                if value is not None and (
                    not isinstance(value, int | float) or not math.isfinite(value)
                ):
                    _raise_validation_error(
                        f"Waveform {field_name} must be a finite number when provided.",
                        path=(
                            f"$.quantum_devices.{device_id}.{waveform_field}.{field_name}"
                        ),
                        details={"value": value},
                    )

        zx_waveforms = payload.get("pulse_hw_zx_pi_4")
        if isinstance(zx_waveforms, dict):
            for aux_id, waveform in zx_waveforms.items():
                if not isinstance(waveform, dict):
                    continue
                for field_name in finite_fields:
                    value = waveform.get(field_name)
                    if value is not None and (
                        not isinstance(value, int | float) or not math.isfinite(value)
                    ):
                        _raise_validation_error(
                            f"Cross-resonance waveform {field_name} must be a finite "
                            "number when provided.",
                            path=(
                                f"$.quantum_devices.{device_id}"
                                f".pulse_hw_zx_pi_4.{aux_id}.{field_name}"
                            ),
                            details={"value": value},
                        )


def _validate_acquire_sync_field(dto: PurrIngressV010) -> None:
    """Validate that acquisition sync is boolean when present.

    ``sync`` maps to canonical ``AcquireDefinitionData.sync``.
    """

    for device_id, payload in _iter_indexed_quantum_devices(dto):
        acquire_payload = payload.get("measure_acquire")
        if not isinstance(acquire_payload, dict):
            continue

        sync = acquire_payload.get("sync")
        if sync is not None and not isinstance(sync, bool):
            _raise_validation_error(
                "Acquire sync must be a boolean when provided.",
                path=f"$.quantum_devices.{device_id}.measure_acquire.sync",
                details={"value": sync, "value_type": type(sync).__name__},
            )
