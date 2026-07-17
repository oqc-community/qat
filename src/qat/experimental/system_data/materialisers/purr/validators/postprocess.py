# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Post-processing and readout-mitigation validation for PuRR ingress payloads."""

from __future__ import annotations

from typing import Any

from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.system_data.materialisers.purr.validators.common import (
    POST_PROCESS_METHOD_LINEAR,
    POST_PROCESS_METHOD_MAX_LIKELIHOOD,
    PROBABILITY_TOLERANCE,
    _is_numeric,
    _is_real,
    _iter_indexed_quantum_devices,
    _raise_validation_error,
)
from qat.experimental.utils.logging import get_logger

logger = get_logger(__name__)


def _validate_readout_mitigation_qubit_map(
    *,
    qubit_id: Any,
    qubit_map: Any,
    qubit_indices: set[int],
) -> None:
    """Validate one per-qubit readout-mitigation matrix and its probability sums."""

    if not isinstance(qubit_map, dict):
        _raise_validation_error(
            "Readout mitigation qubit map must be a mapping.",
            path=f"$.error_mitigation.readout_mitigation.linear.{qubit_id}",
            details={"value_type": type(qubit_map).__name__},
        )

    try:
        qubit_index = int(qubit_id)
    except (TypeError, ValueError):
        _raise_validation_error(
            "Readout mitigation qubit key must be coercible to an integer.",
            path=f"$.error_mitigation.readout_mitigation.linear.{qubit_id}",
            details={"qubit_key": qubit_id},
        )

    if qubit_index not in qubit_indices:
        return

    column_sums = {"0": 0.0, "1": 0.0}
    for key, probability in qubit_map.items():
        if not isinstance(key, str) or "|" not in key:
            _raise_validation_error(
                "Readout mitigation key must have 'measured|prepared' form.",
                path=f"$.error_mitigation.readout_mitigation.linear.{qubit_id}",
                details={"key": key},
            )
        if not _is_real(probability):
            _raise_validation_error(
                "Readout mitigation probability must be numeric.",
                path=(f"$.error_mitigation.readout_mitigation.linear.{qubit_id}.{key}"),
                details={"value_type": type(probability).__name__},
            )
        if probability < 0.0 or probability > 1.0:
            _raise_validation_error(
                "Readout mitigation probability must lie in [0, 1].",
                path=(f"$.error_mitigation.readout_mitigation.linear.{qubit_id}.{key}"),
                details={"value": probability},
            )

        _measured, prepared = key.split("|", maxsplit=1)
        if prepared in column_sums:
            column_sums[prepared] += float(probability)

    for prepared_state, total in column_sums.items():
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            _raise_validation_error(
                "Readout mitigation probabilities must sum to 1 for each prepared state.",
                path=f"$.error_mitigation.readout_mitigation.linear.{qubit_id}",
                details={"prepared_state": prepared_state, "sum": total},
            )


def _validate_linear_post_process_method_payload(
    *,
    method_payload: dict[str, Any],
    path_root: str,
) -> None:
    """Validate the legacy linear post-processing payload shape."""

    mean_z_map_args = method_payload.get("mean_z_map_args")
    if not isinstance(mean_z_map_args, list) or len(mean_z_map_args) != 2:
        _raise_validation_error(
            "linear_map_complex_to_real requires mean_z_map_args with exactly two entries.",
            path=f"{path_root}.mean_z_map_args",
            details={"value": mean_z_map_args},
        )
    if not all(_is_numeric(value) for value in mean_z_map_args):
        _raise_validation_error(
            "mean_z_map_args entries must be int, float, or complex.",
            path=f"{path_root}.mean_z_map_args",
            details={"value_types": [type(value).__name__ for value in mean_z_map_args]},
        )


def _validate_max_likelihood_states(*, states: Any, path_root: str) -> None:
    """Validate the state map used by max_likelihood post-processing."""

    if not isinstance(states, dict) or not states:
        _raise_validation_error(
            "max_likelihood requires a non-empty states mapping.",
            path=f"{path_root}.states",
            details={"value_type": type(states).__name__},
        )

    for key, state_payload in states.items():
        try:
            int(key)
        except (TypeError, ValueError):
            _raise_validation_error(
                "max_likelihood state keys must be coercible to integers.",
                path=f"{path_root}.states",
                details={"invalid_key": key},
            )

        if not isinstance(state_payload, dict):
            _raise_validation_error(
                "max_likelihood state entry must be a mapping.",
                path=f"{path_root}.states.{key}",
                details={"value_type": type(state_payload).__name__},
            )

        location = state_payload.get("location")
        if not _is_numeric(location):
            _raise_validation_error(
                "max_likelihood state location must be int, float, or complex.",
                path=f"{path_root}.states.{key}.location",
                details={"value": location, "value_type": type(location).__name__},
            )

        label = state_payload.get("label")
        if label is not None and not isinstance(label, str):
            _raise_validation_error(
                "max_likelihood state label must be a string when provided.",
                path=f"{path_root}.states.{key}.label",
                details={"value": label, "value_type": type(label).__name__},
            )


def _validate_max_likelihood_post_process_method_payload(
    *,
    method_payload: dict[str, Any],
    path_root: str,
) -> None:
    """Validate the full max_likelihood post-processing payload."""

    _validate_max_likelihood_states(
        states=method_payload.get("states"), path_root=path_root
    )

    noise_est = method_payload.get("noise_est", 1.0)
    if not _is_real(noise_est):
        _raise_validation_error(
            "max_likelihood noise_est must be numeric when provided.",
            path=f"{path_root}.noise_est",
            details={"value": noise_est, "value_type": type(noise_est).__name__},
        )

    p_min = method_payload.get("p_min", 0.0)
    if not _is_real(p_min):
        _raise_validation_error(
            "max_likelihood p_min must be numeric when provided.",
            path=f"{path_root}.p_min",
            details={"value": p_min, "value_type": type(p_min).__name__},
        )
    if p_min < 0.0 or p_min > 1.0:
        _raise_validation_error(
            "max_likelihood p_min must lie in [0, 1] when provided.",
            path=f"{path_root}.p_min",
            details={"value": p_min},
        )

    transform = method_payload.get("transform")
    if transform is not None:
        if (
            not isinstance(transform, list | tuple)
            or len(transform) != 2
            or not all(isinstance(row, list | tuple) and len(row) == 2 for row in transform)
            or not all(_is_real(value) for row in transform for value in row)
        ):
            _raise_validation_error(
                "max_likelihood transform must be a 2x2 numeric matrix when provided.",
                path=f"{path_root}.transform",
                details={"value": transform},
            )

    offset = method_payload.get("offset")
    if offset is not None:
        if (
            not isinstance(offset, list | tuple)
            or len(offset) != 2
            or not all(_is_real(value) for value in offset)
        ):
            _raise_validation_error(
                "max_likelihood offset must be a 2-element numeric vector when provided.",
                path=f"{path_root}.offset",
                details={"value": offset},
            )


def _validate_mean_z_map_args(dto: PurrIngressV010) -> None:
    """Validate the legacy linear discriminator arguments used by readout mapping."""

    for device_id, payload in _iter_indexed_quantum_devices(dto):
        mean_z_map_args = payload.get("mean_z_map_args")
        if mean_z_map_args is None:
            continue

        if not isinstance(mean_z_map_args, list) or len(mean_z_map_args) != 2:
            _raise_validation_error(
                "mean_z_map_args must contain exactly two entries.",
                path=f"$.quantum_devices.{device_id}.mean_z_map_args",
                details={"value": mean_z_map_args},
            )
        if not all(_is_numeric(value) for value in mean_z_map_args):
            _raise_validation_error(
                "mean_z_map_args entries must be int, float, or complex.",
                path=f"$.quantum_devices.{device_id}.mean_z_map_args",
                details={
                    "value_types": [type(value).__name__ for value in mean_z_map_args]
                },
            )


def _validate_post_process_method(dto: PurrIngressV010) -> None:
    """Validate optional modern post-processing payloads before materialisation.

    This mirrors parser acceptance rules in the materialiser so malformed payloads fail at
    ingress validation rather than being silently dropped later.
    """

    for device_id, payload in _iter_indexed_quantum_devices(dto):
        method_payload = payload.get("post_process_method")
        if method_payload is None:
            continue

        path_root = f"$.quantum_devices.{device_id}.post_process_method"
        if not isinstance(method_payload, dict):
            _raise_validation_error(
                "post_process_method must be a mapping when provided.",
                path=path_root,
                details={"value_type": type(method_payload).__name__},
            )

        method = method_payload.get("method")
        if method not in (
            POST_PROCESS_METHOD_LINEAR,
            POST_PROCESS_METHOD_MAX_LIKELIHOOD,
        ):
            _raise_validation_error(
                "post_process_method.method must be one of the supported method names.",
                path=f"{path_root}.method",
                details={
                    "value": method,
                    "supported_methods": (
                        POST_PROCESS_METHOD_LINEAR,
                        POST_PROCESS_METHOD_MAX_LIKELIHOOD,
                    ),
                },
            )

        if method == POST_PROCESS_METHOD_LINEAR:
            _validate_linear_post_process_method_payload(
                method_payload=method_payload,
                path_root=path_root,
            )
            continue

        _validate_max_likelihood_post_process_method_payload(
            method_payload=method_payload,
            path_root=path_root,
        )


def _validate_readout_mitigation(
    error_mitigation: Any,
    qubit_indices: set[int],
) -> None:
    """Validate readout-mitigation probabilities used for canonical readout data."""

    linear = (error_mitigation or {}).get("readout_mitigation", {}).get("linear", {})
    if linear in (None, {}):
        return
    if not isinstance(linear, dict):
        _raise_validation_error(
            "Readout mitigation linear map must be a mapping when provided.",
            path="$.error_mitigation.readout_mitigation.linear",
            details={"value_type": type(linear).__name__},
        )

    for qubit_id, qubit_map in linear.items():
        _validate_readout_mitigation_qubit_map(
            qubit_id=qubit_id,
            qubit_map=qubit_map,
            qubit_indices=qubit_indices,
        )


def _warn_extra_mitigation_entries(error_mitigation: Any, qubit_indices: set[int]) -> None:
    """Warn when readout-mitigation entries exist for qubits not in the device set.

    Mitigation entries for unmaterialised qubits are silently skipped during canonical
    mapping. Warn the user so they can investigate whether the source payload is complete.
    """

    linear = (error_mitigation or {}).get("readout_mitigation", {}).get("linear", {})
    if linear in (None, {}):
        return
    if not isinstance(linear, dict):
        return

    extra_entries = []
    for qubit_id in linear.keys():
        try:
            qubit_index = int(qubit_id)
        except (TypeError, ValueError):
            continue

        if qubit_index not in qubit_indices:
            extra_entries.append(qubit_id)

    if extra_entries:
        logger.warning(
            "Readout mitigation entries exist for qubits not in device set. "
            "These will be silently skipped during canonicalisation. "
            "Qubit IDs: %s. Device qubit indices: %s",
            extra_entries,
            sorted(qubit_indices),
        )
