# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Port, oscillator, and logical-channel builders for PuRR materialisation."""

import math
from typing import Any

from qat.experimental.system_data.canonical.schema import (
    ChannelData,
    OscillatorData,
    PortData,
)
from qat.experimental.system_data.materialisers.purr.materialisers.common import (
    _as_complex,
    _as_float,
    _hz_to_int,
    _seconds_to_picoseconds,
)
from qat.experimental.system_data.materialisers.purr.materialisers.external_resources import (
    ExternalResourceRegistry,
)
from qat.experimental.utils.logging import get_logger

logger = get_logger(__name__)


def _register_external_resource_from_payload(
    *,
    payload: dict[str, Any],
    registry: ExternalResourceRegistry,
    fallback_id: str | None = None,
    fallback_type: str | None = None,
) -> str | None:
    """Register a resource referenced in payload and return its canonical id.

    Returns ``None`` when no resource id can be resolved from payload or fallback,
    and logs a warning for that intentional skip path.
    """

    resource_id = payload.get("id")
    if not isinstance(resource_id, str) or not resource_id:
        resource_id = fallback_id
    if resource_id is None:
        logger.warning(
            "Skipping external resource registration because no resource id was found."
        )
        return None
    object_type = payload.get("instrument_type")
    if not isinstance(object_type, str) or not object_type:
        object_type = fallback_type

    attributes = {
        key: value
        for key, value in payload.items()
        if key not in {"id", "instrument_id", "instrument_type"}
    }
    return registry.register(
        resource_id=resource_id,
        object_type=object_type,
        attributes=attributes,
    )


def _iter_device_pulse_views(quantum_devices: dict[str, Any]):
    """Iterate pulse-channel view payloads attached to decoded device records."""

    for device_id, device_payload in quantum_devices.items():
        if not isinstance(device_payload, dict):
            continue

        pulse_channels = device_payload.get("pulse_channels")
        if not isinstance(pulse_channels, dict):
            continue

        for pulse_key, pulse_view in pulse_channels.items():
            if not isinstance(pulse_view, dict):
                continue
            yield device_id, device_payload, pulse_key, pulse_view


def _build_port_block_bounds(
    *,
    payload: dict[str, Any],
    sample_time: int,
    block_size: int,
) -> tuple[int, int]:
    """Compute canonical min/max block bounds from direct values or duration constraints."""

    min_blocks = payload.get("min_blocks")
    max_blocks = payload.get("max_blocks")
    if isinstance(min_blocks, int) and min_blocks >= 1:
        resolved_min_blocks = min_blocks
    else:
        resolved_min_blocks = 1

    if isinstance(max_blocks, int) and max_blocks >= resolved_min_blocks:
        resolved_max_blocks = max_blocks
    else:
        resolved_max_blocks = -1

    if resolved_min_blocks != 1 or resolved_max_blocks != -1:
        return resolved_min_blocks, resolved_max_blocks

    min_blocks = 1
    max_blocks = -1

    min_duration = payload.get("pulse_duration_min")
    if isinstance(min_duration, int | float) and min_duration > 0:
        min_duration_ps = _seconds_to_picoseconds(min_duration)
        if min_duration_ps is not None:
            min_blocks = max(1, math.ceil(min_duration_ps / (sample_time * block_size)))

    max_duration = payload.get("pulse_duration_max")
    if isinstance(max_duration, int | float) and max_duration > 0:
        max_duration_ps = _seconds_to_picoseconds(max_duration)
        if max_duration_ps is not None:
            max_blocks = math.ceil(max_duration_ps / (sample_time * block_size))

    return min_blocks, max_blocks


def _build_ports(
    physical_channels: dict[str, Any],
    external_resources: ExternalResourceRegistry,
) -> tuple[PortData, ...]:
    """Build canonical port records from PuRR physical-channel payloads."""

    ports = []
    for port_id, payload in physical_channels.items():
        if not isinstance(payload, dict):
            continue

        sample_time = _seconds_to_picoseconds(payload.get("sample_time"))
        if sample_time is None:
            continue

        block_size = int(payload.get("block_size", 1) or 1)
        acquire_allowed = bool(payload.get("acquire_allowed", False))
        native_waveform_shapes = tuple(
            shape
            for shape in payload.get("native_waveform_shapes", ())
            if isinstance(shape, str)
        )
        min_blocks, max_blocks = _build_port_block_bounds(
            payload=payload,
            sample_time=sample_time,
            block_size=block_size,
        )

        ports.append(
            PortData(
                id=port_id,
                sample_time=sample_time,
                block_size=block_size,
                min_blocks=min_blocks,
                max_blocks=max_blocks,
                acquire_allowed=acquire_allowed,
                native_waveform_shapes=native_waveform_shapes,
                external_resource_id=_register_external_resource_from_payload(
                    payload=payload,
                    registry=external_resources,
                    fallback_type="port",
                ),
            )
        )

    return tuple(ports)


def _build_oscillators(
    basebands: dict[str, Any],
    external_resources: ExternalResourceRegistry,
) -> tuple[OscillatorData, ...]:
    """Build canonical oscillator records from PuRR baseband payloads."""

    oscillators = []
    for oscillator_id, payload in basebands.items():
        if not isinstance(payload, dict):
            continue

        frequency = payload.get("frequency", payload.get("_frequency"))
        canonical_frequency = _hz_to_int(frequency)
        if canonical_frequency is None:
            continue

        external_resource_id = _register_external_resource_from_payload(
            payload=payload,
            registry=external_resources,
            fallback_type="oscillator",
        )

        oscillators.append(
            OscillatorData(
                id=oscillator_id,
                frequency=canonical_frequency,
                external_resource_id=external_resource_id,
            )
        )

    return tuple(oscillators)


def _resolve_physical_channel(
    *,
    pulse_channel: dict[str, Any],
    device_payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve the physical-channel payload for a logical pulse view."""

    physical_channel = pulse_channel.get("physical_channel")
    if not isinstance(physical_channel, dict):
        physical_channel = device_payload.get("physical_channel")
    if not isinstance(physical_channel, dict):
        return None
    return physical_channel


def _resolve_oscillator_reference(
    *,
    port_id: str,
    physical_channels: dict[str, Any],
) -> str | None:
    """Resolve the oscillator reference id for a channel via its physical port."""

    top_level_port = physical_channels.get(port_id)
    if not isinstance(top_level_port, dict):
        return None

    baseband = top_level_port.get("baseband")
    if isinstance(baseband, dict):
        return baseband.get("id")
    return None


def _build_channels(
    *,
    quantum_devices: dict[str, Any],
    physical_channels: dict[str, Any],
) -> tuple[ChannelData, ...]:
    """Build canonical logical-channel records from PuRR pulse-channel views."""

    channels: dict[str, ChannelData] = {}

    for _device_id, device_payload, _pulse_key, pulse_view in _iter_device_pulse_views(
        quantum_devices
    ):
        pulse_channel = pulse_view.get("pulse_channel")
        if not isinstance(pulse_channel, dict):
            continue

        channel_id = pulse_channel.get("id")
        if not isinstance(channel_id, str) or channel_id in channels:
            continue

        physical_channel = _resolve_physical_channel(
            pulse_channel=pulse_channel,
            device_payload=device_payload,
        )
        if physical_channel is None:
            continue

        port_id = physical_channel.get("id")
        if not isinstance(port_id, str):
            continue

        channels[channel_id] = ChannelData(
            id=channel_id,
            port_id=port_id,
            frequency=_hz_to_int(pulse_channel.get("frequency")) or 0,
            oscillator_reference=_resolve_oscillator_reference(
                port_id=port_id,
                physical_channels=physical_channels,
            ),
            scale=_as_complex(pulse_channel.get("scale")),
            imbalance=_as_float(
                pulse_channel.get("imbalance", physical_channel.get("imbalance", 1.0)),
                default=1.0,
            ),
            phase_offset=_as_float(
                pulse_channel.get("phase_offset", physical_channel.get("phase_offset", 0.0))
            ),
        )

    return tuple(channels.values())
