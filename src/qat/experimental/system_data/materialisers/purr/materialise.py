# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""PuRR-to-canonical materialisation orchestration for the experimental boundary.

This module keeps the public materialisation entrypoint and coordinates adaptation,
ingress validation, and canonical assembly via domain-specific builders.

Stage architecture
==================

PuRR materialisation is intentionally staged so source-boundary concerns remain
separate from compiler-owned enrichment and canonical assembly:

1. Source version compatibility check.
2. Source payload adaptation into boundary-normalised plain data.
3. Source ingress DTO validation and graph consistency validation.
4. Compiler-owned enrichment required for canonical assembly.
5. Canonical system data construction from validated/enriched ingress DTO.

This separation allows validation responsibility to move upstream over time
without changing canonical assembly responsibilities.
"""

import math
from typing import Any

from pydantic import ValidationError

from qat.experimental.system_data.canonical.schema import (
    AttributeEntry,
    CanonicalSystemData,
)
from qat.experimental.system_data.materialisers.errors import (
    SourceValidationError,
    UnsupportedSourceVersionError,
)
from qat.experimental.system_data.materialisers.purr.adapter import adapt_purr_payload
from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.system_data.materialisers.purr.materialisers.capabilities import (
    _build_acquire_limit,
    _build_acquire_modes,
    _build_reset_methods,
)
from qat.experimental.system_data.materialisers.purr.materialisers.couplings import (
    _build_couplings,
)
from qat.experimental.system_data.materialisers.purr.materialisers.external_resources import (
    ExternalResourceRegistry,
)
from qat.experimental.system_data.materialisers.purr.materialisers.qubits import (
    _build_qubits,
)
from qat.experimental.system_data.materialisers.purr.materialisers.signal_paths import (
    _build_channels,
    _build_oscillators,
    _build_ports,
)
from qat.experimental.system_data.materialisers.purr.validate import (
    validate_purr_ingress_graph,
)
from qat.model.target_data import TargetData

_SUPPORTED_PURR_SOURCE_VERSIONS = ("0.1.0",)

_RESET_DEFAULT_ORDER = ("passive", "active", "ddrop")


def _is_qubit_device_payload(device_payload: dict[str, Any]) -> bool:
    """Return True when a device payload structurally represents a qubit.

    PuRR payload naming conventions (for example ``Q*`` IDs) are not relied on
    here because IDs are source-specific and may evolve. The qubit ``index``
    field is the stable structural discriminator in supported payloads.
    """

    if isinstance(device_payload.get("index"), int):
        return True

    return isinstance(device_payload.get("measure_device"), dict)


def _detect_supported_reset_methods(payload: dict[str, Any]) -> list[str]:
    """Collect supported reset method types from qubit payload records."""

    found_methods: set[str] = set()
    quantum_devices = payload.get("quantum_devices")
    if isinstance(quantum_devices, dict):
        for device_payload in quantum_devices.values():
            if not isinstance(device_payload, dict):
                continue
            if not _is_qubit_device_payload(device_payload):
                continue

            pulse_channels = device_payload.get("pulse_channels")
            if isinstance(pulse_channels, dict):
                if "reset" in pulse_channels or isinstance(
                    device_payload.get("ddrop_reset"), dict
                ):
                    found_methods.add("ddrop")
                if "active_reset" in pulse_channels:
                    found_methods.add("active")

            active_reset = device_payload.get("active_reset")
            if isinstance(active_reset, dict):
                found_methods.add("active")

    return [method for method in _RESET_DEFAULT_ORDER if method in found_methods]


def _inject_supported_reset_methods(payload: dict[str, Any]) -> dict[str, Any]:
    """Inject top-level reset capability fields into the ingress payload."""

    updated = dict(payload)
    supported_reset_methods = _detect_supported_reset_methods(updated)

    passive_reset_time = updated.get("passive_reset_time")
    if isinstance(passive_reset_time, int | float) and passive_reset_time >= 0:
        if "passive" not in supported_reset_methods:
            supported_reset_methods.insert(0, "passive")

    existing_supported = updated.get("supported_reset_methods")
    if isinstance(existing_supported, list):
        for reset_type in existing_supported:
            if isinstance(reset_type, str) and reset_type not in supported_reset_methods:
                supported_reset_methods.append(reset_type)

    updated["supported_reset_methods"] = supported_reset_methods

    existing_default = updated.get("default_reset_method")
    if isinstance(existing_default, str) and existing_default in supported_reset_methods:
        updated["default_reset_method"] = existing_default
    else:
        updated["default_reset_method"] = (
            supported_reset_methods[0] if supported_reset_methods else None
        )

    return updated


def _inject_target_data_fields(
    adapted_payload: dict[str, Any],
    target_data: TargetData,
) -> dict[str, Any]:
    """Inject compiler-owned target data fields required by ingress DTO validation."""

    payload = dict(adapted_payload)
    payload["passive_reset_time"] = target_data.QUBIT_DATA.passive_reset_time

    physical_channels = payload.get("physical_channels")
    if isinstance(physical_channels, dict):
        updated_physical_channels = {}
        qubit_block_size = target_data.QUBIT_DATA.samples_per_clock_cycle
        resonator_block_size = target_data.RESONATOR_DATA.samples_per_clock_cycle
        qubit_min_blocks = max(
            1,
            math.ceil(
                target_data.QUBIT_DATA.pulse_duration_min
                / (
                    target_data.QUBIT_DATA.sample_time
                    * target_data.QUBIT_DATA.samples_per_clock_cycle
                )
            ),
        )
        resonator_min_blocks = max(
            1,
            math.ceil(
                target_data.RESONATOR_DATA.pulse_duration_min
                / (
                    target_data.RESONATOR_DATA.sample_time
                    * target_data.RESONATOR_DATA.samples_per_clock_cycle
                )
            ),
        )
        qubit_max_blocks = max(1, target_data.QUBIT_DATA.waveform_memory_size - 1)
        resonator_max_blocks = max(1, target_data.RESONATOR_DATA.waveform_memory_size - 1)

        for channel_id, channel_payload in physical_channels.items():
            if not isinstance(channel_payload, dict):
                updated_physical_channels[channel_id] = channel_payload
                continue

            updated_channel_payload = dict(channel_payload)
            acquire_allowed = bool(updated_channel_payload.get("acquire_allowed", False))
            if acquire_allowed:
                updated_channel_payload["block_size"] = resonator_block_size
                updated_channel_payload["min_blocks"] = resonator_min_blocks
                updated_channel_payload["max_blocks"] = resonator_max_blocks
            else:
                updated_channel_payload["block_size"] = qubit_block_size
                updated_channel_payload["min_blocks"] = qubit_min_blocks
                updated_channel_payload["max_blocks"] = qubit_max_blocks
            updated_physical_channels[channel_id] = updated_channel_payload

        payload["physical_channels"] = updated_physical_channels

    return payload


def _inject_native_waveform_shapes(
    adapted_payload: dict[str, Any],
    native_waveform_shapes: list[str],
) -> dict[str, Any]:
    """Inject compiler-owned native waveform shape fields required by ingress DTO
    validation."""

    payload = dict(adapted_payload)
    physical_channels = payload.get("physical_channels")
    if isinstance(physical_channels, dict):
        updated_physical_channels = {}
        for channel_id, channel_payload in physical_channels.items():
            if not isinstance(channel_payload, dict):
                updated_physical_channels[channel_id] = channel_payload
                continue

            updated_channel_payload = dict(channel_payload)
            updated_channel_payload.setdefault(
                "native_waveform_shapes", tuple(native_waveform_shapes)
            )
            updated_physical_channels[channel_id] = updated_channel_payload

        payload["physical_channels"] = updated_physical_channels

    return payload


def _materialise_canonical_top_level(
    *,
    dto: PurrIngressV010,
    source_version: str,
) -> CanonicalSystemData:
    """Assemble canonical system data from validated PuRR ingress payloads."""

    external_resources = ExternalResourceRegistry()

    acquire_modes, default_acquire_mode = _build_acquire_modes(
        dto.supported_acquire_modes,
        dto.default_acquire_mode,
    )
    reset_methods, default_reset_method = _build_reset_methods(
        dto.supported_reset_methods,
        dto.default_reset_method,
        dto.passive_reset_time,
    )

    return CanonicalSystemData(
        calibration_id=dto.calibration_id,
        acquire_limit=_build_acquire_limit(dto.repeat_limit),
        acquire_modes=acquire_modes,
        default_acquire_mode=default_acquire_mode,
        reset_methods=reset_methods,
        default_reset_method=default_reset_method,
        oscillators=_build_oscillators(dto.basebands, external_resources),
        ports=_build_ports(dto.physical_channels, external_resources),
        channels=_build_channels(
            quantum_devices=dto.quantum_devices,
            physical_channels=dto.physical_channels,
        ),
        qubits=_build_qubits(
            quantum_devices=dto.quantum_devices,
            error_mitigation=dto.error_mitigation,
        ),
        couplings=_build_couplings(
            qubit_direction_couplings=dto.qubit_direction_couplings,
            quantum_devices=dto.quantum_devices,
        ),
        external_resources=external_resources.to_tuple(),
        metadata=(
            AttributeEntry(key="materialiser_source_type", value="purr"),
            AttributeEntry(key="materialiser_source_version", value=source_version),
            AttributeEntry(
                key="materialiser_status",
                value="experimental_partial_mapping",
            ),
        ),
    )


def materialise_purr_v0_1_0(
    *,
    source_payload: dict[str, Any],
    source_version: str,
    target_data: TargetData | None = None,
    supported_acquire_modes: list[str] | None = None,
    native_waveform_shapes: list[str] | None = None,
    decoder_extra_reduce_target_types: list[str] | None = None,
    decoder_extra_reduce_target_suffixes: list[str] | None = None,
) -> CanonicalSystemData:
    """Materialise canonical system data from a PuRR v0.1.0 source payload.

    This runs the PuRR boundary flow:

    1. Source-version compatibility check.
    2. Payload adaptation/decoding into boundary-normalised plain data.
    3. Ingress DTO validation.
    4. Graph-level consistency validation.
    5. Canonical materialisation.

    :param source_payload: Raw parsed PuRR payload.
    :param source_version: Source contract version.
    :param target_data: Compiler target data required by downstream mappings.
    :param supported_acquire_modes: Fallback acquire modes applied when the
        source payload does not provide explicit supported modes.
    :param native_waveform_shapes: Compiler-owned waveform shape defaults
        injected per physical channel when absent from the source payload.
    :param decoder_extra_reduce_target_types: Optional extra fully-qualified
        ``py/reduce`` targets allowed by the source decoder at runtime.
    :param decoder_extra_reduce_target_suffixes: Optional extra terminal type-name
        suffixes allowed by the source decoder at runtime.
    :returns: Materialised canonical system data.
    :raises UnsupportedSourceVersionError: If source version is unsupported.
    :raises SourceValidationError: If DTO validation fails.

    This function is version-specific by design. New PuRR source versions should
    be implemented as new entrypoints and registered in the boundary registry.
    """
    if supported_acquire_modes is None:
        supported_acquire_modes = ["integrator", "raw", "scope"]
    if native_waveform_shapes is None:
        native_waveform_shapes = ["square"]
    if source_version not in _SUPPORTED_PURR_SOURCE_VERSIONS:
        raise UnsupportedSourceVersionError.for_version(
            source_type="purr",
            source_version=source_version,
            supported_versions=_SUPPORTED_PURR_SOURCE_VERSIONS,
        )

    adapted_payload = adapt_purr_payload(
        source_payload,
        extra_reduce_target_types=(
            set(decoder_extra_reduce_target_types)
            if decoder_extra_reduce_target_types is not None
            else None
        ),
        extra_reduce_target_suffixes=(
            set(decoder_extra_reduce_target_suffixes)
            if decoder_extra_reduce_target_suffixes is not None
            else None
        ),
    )

    try:
        # Stage 1: validate the external source payload at the ingress boundary.
        source_ingress_dto = PurrIngressV010.model_validate(adapted_payload)
    except ValidationError as exc:
        raise SourceValidationError(
            "PuRR ingress DTO validation failed.",
            source_type="purr",
            source_version=source_version,
            details={"errors": exc.errors(include_url=False)},
            cause=exc,
        ) from exc

    validate_purr_ingress_graph(source_ingress_dto)

    # Stage 2: apply compiler-owned enrichment used for canonical assembly.
    if target_data is None:
        target_data = TargetData()
    ingress_payload = _inject_target_data_fields(adapted_payload, target_data)
    ingress_payload = _inject_supported_reset_methods(ingress_payload)
    ingress_payload = _inject_native_waveform_shapes(
        ingress_payload, native_waveform_shapes
    )
    ingress_payload.setdefault(
        "supported_acquire_modes",
        source_ingress_dto.supported_acquire_modes or supported_acquire_modes,
    )

    try:
        enriched_ingress_dto = PurrIngressV010.model_validate(ingress_payload)
    except ValidationError as exc:
        raise SourceValidationError(
            "Compiler enrichment produced an invalid PuRR ingress payload.",
            source_type="purr",
            source_version=source_version,
            details={"errors": exc.errors(include_url=False)},
            cause=exc,
        ) from exc

    return _materialise_canonical_top_level(
        dto=enriched_ingress_dto,
        source_version=source_version,
    )
