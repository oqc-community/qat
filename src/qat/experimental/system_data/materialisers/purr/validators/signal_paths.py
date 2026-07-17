# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Channel and baseband validation rules for PuRR ingress payloads."""

from __future__ import annotations

import math
from typing import Any

from qat.experimental.system_data.materialisers.errors import SourceConsistencyError
from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.system_data.materialisers.purr.validators.common import (
    _iter_device_pulse_views,
    _raise_validation_error,
)
from qat.experimental.utils.logging import get_logger

logger = get_logger(__name__)


def _validate_top_level_collections(
    *,
    quantum_devices: dict[str, Any],
    physical_channels: dict[str, Any],
    basebands: dict[str, Any],
) -> None:
    """Require the top-level collections needed by canonical materialisation."""

    if not quantum_devices:
        raise SourceConsistencyError(
            "PuRR ingress payload contains no quantum devices.",
            source_type="purr",
            path="$.quantum_devices",
        )

    if not physical_channels:
        raise SourceConsistencyError(
            "PuRR ingress payload contains no physical channels.",
            source_type="purr",
            path="$.physical_channels",
        )

    if not basebands:
        raise SourceConsistencyError(
            "PuRR ingress payload contains no basebands.",
            source_type="purr",
            path="$.basebands",
        )


def _validate_repeat_limit(repeat_limit: int | None) -> None:
    """Validate the legacy repeat-limit value mapped to canonical acquire limits."""

    if repeat_limit is not None and repeat_limit <= 0:
        _raise_validation_error(
            "repeat_limit must be strictly positive when provided.",
            path="$.repeat_limit",
            details={"value": repeat_limit},
        )


def _validate_passive_reset_time(passive_reset_time: float | None) -> None:
    """Validate passive reset time used when deriving canonical reset metadata."""

    if passive_reset_time is not None and passive_reset_time < 0:
        _raise_validation_error(
            "passive_reset_time must be non-negative when provided.",
            path="$.passive_reset_time",
            details={"value": passive_reset_time},
        )


def _validate_physical_channel_payload(port_id: str, payload: dict[str, Any]) -> None:
    """Validate one physical-channel payload and its hardware timing limits."""

    sample_time = payload.get("sample_time")
    if (
        not isinstance(sample_time, int | float)
        or not math.isfinite(sample_time)
        or sample_time <= 0
    ):
        _raise_validation_error(
            "Physical channel sample_time must be a finite strictly positive number.",
            path=f"$.physical_channels.{port_id}.sample_time",
            details={"value": sample_time},
        )

    block_size = payload.get("block_size", 1)
    if not isinstance(block_size, int) or block_size < 1:
        _raise_validation_error(
            "Physical channel block_size must be an integer >= 1.",
            path=f"$.physical_channels.{port_id}.block_size",
            details={"value": block_size},
        )

    min_blocks = payload.get("min_blocks")
    if min_blocks is not None and (not isinstance(min_blocks, int) or min_blocks < 1):
        _raise_validation_error(
            "Physical channel min_blocks must be an integer >= 1 when provided.",
            path=f"$.physical_channels.{port_id}.min_blocks",
            details={"value": min_blocks},
        )

    max_blocks = payload.get("max_blocks")
    if max_blocks is not None and (not isinstance(max_blocks, int) or max_blocks < 1):
        _raise_validation_error(
            "Physical channel max_blocks must be an integer >= 1 when provided.",
            path=f"$.physical_channels.{port_id}.max_blocks",
            details={"value": max_blocks},
        )

    if (
        isinstance(min_blocks, int)
        and isinstance(max_blocks, int)
        and min_blocks > max_blocks
    ):
        _raise_validation_error(
            "Physical channel min_blocks must be <= max_blocks.",
            path=f"$.physical_channels.{port_id}",
            details={"min_blocks": min_blocks, "max_blocks": max_blocks},
        )

    acquire_allowed = payload.get("acquire_allowed")
    if acquire_allowed is not None and not isinstance(acquire_allowed, bool):
        _raise_validation_error(
            "Physical channel acquire_allowed must be a boolean when provided.",
            path=f"$.physical_channels.{port_id}.acquire_allowed",
            details={"value": acquire_allowed},
        )

    min_duration = payload.get("pulse_duration_min")
    max_duration = payload.get("pulse_duration_max")

    if min_duration is not None and (
        not isinstance(min_duration, int | float)
        or not math.isfinite(min_duration)
        or min_duration <= 0
    ):
        _raise_validation_error(
            "Physical channel pulse_duration_min must be a finite strictly positive number when provided.",
            path=f"$.physical_channels.{port_id}.pulse_duration_min",
            details={"value": min_duration},
        )

    if max_duration is not None and (
        not isinstance(max_duration, int | float)
        or not math.isfinite(max_duration)
        or max_duration <= 0
    ):
        _raise_validation_error(
            "Physical channel pulse_duration_max must be a finite strictly positive number when provided.",
            path=f"$.physical_channels.{port_id}.pulse_duration_max",
            details={"value": max_duration},
        )

    if (
        isinstance(min_duration, int | float)
        and isinstance(max_duration, int | float)
        and min_duration > max_duration
    ):
        _raise_validation_error(
            "Physical channel pulse_duration_min must be <= pulse_duration_max.",
            path=f"$.physical_channels.{port_id}",
            details={"min": min_duration, "max": max_duration},
        )


def _validate_physical_channels(physical_channels: dict[str, Any]) -> None:
    """Validate every physical channel before it is materialised into a PortData record."""

    for port_id, payload in physical_channels.items():
        if not isinstance(payload, dict):
            _raise_validation_error(
                "Physical channel entry must be a mapping.",
                path=f"$.physical_channels.{port_id}",
                details={"entry_type": type(payload).__name__},
            )
        _validate_physical_channel_payload(port_id, payload)


def _validate_baseband_payload(baseband_id: str, payload: dict[str, Any]) -> None:
    """Validate one baseband payload and its primary oscillator frequency."""

    frequency = payload.get("frequency", payload.get("_frequency"))
    if frequency is not None and (
        not isinstance(frequency, int | float)
        or not math.isfinite(frequency)
        or frequency <= 0
    ):
        _raise_validation_error(
            "Baseband frequency must be a finite strictly positive real number when provided.",
            path=f"$.basebands.{baseband_id}.frequency",
            details={"value": frequency},
        )


def _validate_basebands(basebands: dict[str, Any]) -> None:
    """Validate baseband frequency values before they are promoted to oscillators.

    Only the primary frequency is validated here because ``if_frequency`` is not
    currently mapped into the canonical representation.
    """

    for baseband_id, payload in basebands.items():
        if not isinstance(payload, dict):
            _raise_validation_error(
                "Baseband entry must be a mapping.",
                path=f"$.basebands.{baseband_id}",
                details={"entry_type": type(payload).__name__},
            )
        _validate_baseband_payload(baseband_id, payload)


def _validate_pulse_channel_reference(
    pulse_id: str,
    pulse_payload: dict[str, Any],
    physical_channel_ids: frozenset[str],
) -> None:
    """Validate one top-level pulse-channel reference to a known physical channel."""

    physical_ref = pulse_payload.get("physical_channel")
    if isinstance(physical_ref, dict):
        physical_ref = physical_ref.get("id")

    if isinstance(physical_ref, str) and physical_ref not in physical_channel_ids:
        raise SourceConsistencyError(
            "Pulse channel references unknown physical channel.",
            source_type="purr",
            path=f"$.pulse_channels.{pulse_id}.physical_channel",
            details={"physical_channel_id": physical_ref},
        )


def _validate_pulse_channel_references(
    *, pulse_channels: dict[str, Any], physical_channel_ids: frozenset[str]
) -> None:
    """Validate that each pulse-channel reference points at a known physical channel."""

    for pulse_id, pulse_payload in pulse_channels.items():
        if not isinstance(pulse_payload, dict):
            _raise_validation_error(
                "Pulse channel entry must be a mapping.",
                path=f"$.pulse_channels.{pulse_id}",
                details={"entry_type": type(pulse_payload).__name__},
            )
        _validate_pulse_channel_reference(pulse_id, pulse_payload, physical_channel_ids)


def _validate_pulse_channel_frequencies(dto: PurrIngressV010) -> None:
    """Validate pulse frequencies and hardware bounds exposed through pulse views."""

    for device_id, device_payload, pulse_key, pulse_view in _iter_device_pulse_views(dto):
        pulse_channel = pulse_view.get("pulse_channel")
        if not isinstance(pulse_channel, dict):
            continue

        frequency = pulse_channel.get("frequency")
        path = f"$.quantum_devices.{device_id}.pulse_channels.{pulse_key}.pulse_channel.frequency"
        if frequency is not None and (
            not isinstance(frequency, int | float)
            or not math.isfinite(frequency)
            or frequency < 0
        ):
            _raise_validation_error(
                "Pulse channel frequency must be a finite non-negative number when provided.",
                path=path,
                details={"value": frequency},
            )

        physical_channel = pulse_channel.get("physical_channel")
        if not isinstance(physical_channel, dict):
            physical_channel = device_payload.get("physical_channel")
        if not isinstance(physical_channel, dict) or not isinstance(frequency, int | float):
            continue

        min_frequency = physical_channel.get("pulse_channel_min_frequency")
        max_frequency = physical_channel.get("pulse_channel_max_frequency")
        if isinstance(min_frequency, int | float) and frequency < min_frequency:
            _raise_validation_error(
                "Pulse channel frequency is below the declared minimum.",
                path=path,
                details={"value": frequency, "min_frequency": min_frequency},
            )
        if isinstance(max_frequency, int | float) and frequency > max_frequency:
            _raise_validation_error(
                "Pulse channel frequency is above the declared maximum.",
                path=path,
                details={"value": frequency, "max_frequency": max_frequency},
            )


def _validate_pulse_channel_scales(dto: PurrIngressV010) -> None:
    """Validate pulse-channel scale values that map into canonical channel scale."""

    for device_id, _device_payload, pulse_key, pulse_view in _iter_device_pulse_views(dto):
        pulse_channel = pulse_view.get("pulse_channel")
        if not isinstance(pulse_channel, dict):
            continue

        scale = pulse_channel.get("scale")
        if scale is not None and not isinstance(scale, int | float | complex):
            _raise_validation_error(
                "Pulse channel scale must be numeric or complex when provided.",
                path=(
                    f"$.quantum_devices.{device_id}"
                    f".pulse_channels.{pulse_key}.pulse_channel.scale"
                ),
                details={"value_type": type(scale).__name__},
            )


def _warn_sample_time_consistency(dto: PurrIngressV010) -> None:
    """Warn when qubit or resonator channels have inconsistent sample times within a type.

    Qubits may have different sample times than resonators (this is intentional), but all
    qubits should have the same sample time, and all resonators should have the same sample
    time. Inconsistency within a type may indicate source payload errors.
    """

    qubit_times = {}
    resonator_times = {}

    for _device_id, device_data in dto.quantum_devices.items():
        if not isinstance(device_data, dict):
            continue

        pulse_channels = device_data.get("pulse_channels")
        if not isinstance(pulse_channels, dict):
            continue

        for _pulse_key, pulse_view in pulse_channels.items():
            if not isinstance(pulse_view, dict):
                continue

            pulse_channel = pulse_view.get("pulse_channel")
            if not isinstance(pulse_channel, dict):
                continue

            phys_ref = pulse_channel.get("physical_channel")
            if isinstance(phys_ref, dict):
                phys_ref = phys_ref.get("id")

            if not isinstance(phys_ref, str) or phys_ref not in dto.physical_channels:
                continue

            port_data = dto.physical_channels[phys_ref]
            if not isinstance(port_data, dict):
                continue

            sample_time = port_data.get("sample_time")
            if (
                not isinstance(sample_time, int | float)
                or not math.isfinite(sample_time)
                or sample_time <= 0
            ):
                continue

            if isinstance(device_data.get("index"), int):
                qubit_times[phys_ref] = sample_time
            else:
                resonator_times[phys_ref] = sample_time

    qubit_time_values = set(qubit_times.values())
    resonator_time_values = set(resonator_times.values())

    if len(qubit_time_values) > 1:
        logger.warning(
            "Qubit channels have inconsistent sample_time values. "
            "All qubits should share the same sample time. "
            "Sample times observed: %s",
            sorted(qubit_time_values),
        )

    if len(resonator_time_values) > 1:
        logger.warning(
            "Resonator channels have inconsistent sample_time values. "
            "All resonators should share the same sample time. "
            "Sample times observed: %s",
            sorted(resonator_time_values),
        )
