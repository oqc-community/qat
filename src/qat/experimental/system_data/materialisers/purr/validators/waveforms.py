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


def _validate_waveform_field_bounds(
    *,
    device_id: str,
    field_name: str,
    waveform: dict[str, Any],
) -> None:
    """Validate width and rise bounds for a single drive/readout waveform."""

    width = waveform.get("width")
    if width is not None and (
        not isinstance(width, int | float) or not math.isfinite(width) or width < 0
    ):
        _raise_validation_error(
            "Waveform width must be a finite non-negative number when provided.",
            path=f"$.quantum_devices.{device_id}.{field_name}.width",
            details={"value": width},
        )

    rise = waveform.get("rise")
    if isinstance(rise, int | float) and (not math.isfinite(rise) or rise < 0):
        _raise_validation_error(
            "Waveform rise must be a finite non-negative number when numeric.",
            path=f"$.quantum_devices.{device_id}.{field_name}.rise",
            details={"value": rise},
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
