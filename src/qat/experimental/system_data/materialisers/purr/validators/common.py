# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Shared constants and helper utilities for PuRR ingress validation modules."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from qat.experimental.system_data.materialisers.errors import SourceValidationError
from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.utils.logging import get_logger

logger = get_logger(__name__)

PROBABILITY_TOLERANCE = 1e-6
POST_PROCESS_METHOD_LINEAR = "linear_map_complex_to_real"
POST_PROCESS_METHOD_MAX_LIKELIHOOD = "max_likelihood"
CRCRC_KEY_PATTERN = re.compile(r"^(Q\d+)\.(cross_resonance|cross_resonance_cancellation)$")
CRCRC_CHANNEL_ID_PATTERN = re.compile(
    r"^(Q\d+)\.(Q\d+)\.(cross_resonance|cross_resonance_cancellation)$"
)


def _raise_validation_error(message: str, *, path: str, details: dict[str, Any]) -> None:
    raise SourceValidationError(
        message,
        source_type="purr",
        path=path,
        details=details,
    )


def _iter_device_pulse_views(dto: PurrIngressV010):
    """Yield per-device pulse-channel views nested under quantum_devices."""

    for device_id, device_payload in dto.quantum_devices.items():
        if not isinstance(device_payload, dict):
            continue

        pulse_channels = device_payload.get("pulse_channels")
        if not isinstance(pulse_channels, dict):
            continue

        for pulse_key, pulse_view in pulse_channels.items():
            if not isinstance(pulse_view, dict):
                continue
            yield device_id, device_payload, pulse_key, pulse_view


def _iter_indexed_quantum_devices(dto: PurrIngressV010):
    """Yield quantum-device payloads that already have a stable integer index."""

    for device_id, payload in dto.quantum_devices.items():
        if isinstance(payload, dict) and isinstance(payload.get("index"), int):
            yield device_id, payload


def _is_numeric(value: Any) -> bool:
    """Return True for numeric scalars accepted by IQ and centroid fields."""

    return isinstance(value, int | float | complex)


def _is_real(value: Any) -> bool:
    """Return True for real numeric scalars used by bounds and thresholds."""

    return isinstance(value, int | float)


def _collect_qubit_indices(dto: PurrIngressV010) -> set[int]:
    """Collect the integer qubit indices present in ``quantum_devices``."""

    return {
        payload.get("index")
        for payload in dto.quantum_devices.values()
        if isinstance(payload, dict) and isinstance(payload.get("index"), int)
    }


def _collect_qubit_index_by_id(dto: PurrIngressV010) -> dict[str, int]:
    """Collect a mapping from qubit id strings to integer qubit indices."""

    qubit_index_by_id: dict[str, int] = {}
    for device_id, payload in dto.quantum_devices.items():
        if not isinstance(payload, dict):
            logger.warning(
                "Skipping quantum device '%s' when collecting qubit indices: "
                "payload is not a mapping.",
                device_id,
            )
            continue

        qubit_id = payload.get("id")
        qubit_index = payload.get("index")
        if not isinstance(qubit_id, str) or not isinstance(qubit_index, int):
            logger.warning(
                "Skipping quantum device '%s' when collecting qubit indices: "
                "id/index are not (str, int).",
                device_id,
            )
            continue

        existing_index = qubit_index_by_id.get(qubit_id)
        if existing_index is not None and existing_index != qubit_index:
            logger.warning(
                "Duplicate qubit id '%s' encountered while collecting qubit indices; "
                "overwriting index %s with %s.",
                qubit_id,
                existing_index,
                qubit_index,
            )

        qubit_index_by_id[qubit_id] = qubit_index

    return qubit_index_by_id


def _iter_parsed_coupling_directions(qubit_direction_couplings: Iterable[Any]):
    """Yield coupling direction tuples from entries that already parse cleanly."""

    for entry in qubit_direction_couplings:
        if not isinstance(entry, dict):
            continue
        direction = entry.get("direction")
        if (
            isinstance(direction, list)
            and len(direction) == 2
            and all(isinstance(value, int) for value in direction)
        ):
            yield tuple(direction)
