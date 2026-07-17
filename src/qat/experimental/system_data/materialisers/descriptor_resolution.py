# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Descriptor resolution helpers for materialiser boundary dispatch.

These helpers resolve source type/version from payload metadata and plugin detector hooks,
while preserving structured boundary error behaviour.
"""

from typing import Any

from qat.experimental.system_data.materialisers.errors import (
    SourceValidationError,
    UnsupportedSourceError,
    UnsupportedSourceVersionError,
)
from qat.experimental.system_data.materialisers.plugins import (
    get_registered_materialiser_plugins,
    get_registered_source_versions,
)
from qat.experimental.system_data.materialisers.types import SourceType
from qat.experimental.utils.logging import get_logger

logger = get_logger(__name__)


def _supported_source_values() -> tuple[str, ...]:
    """Return supported source identifiers for structured error payloads."""

    return tuple(member.value for member in SourceType)


def _extract_payload_metadata(
    source_payload: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Extract optional source descriptor hints from payload metadata.

    Descriptor fields are read only from ``payload["metadata"]``.
    If ``metadata`` is present but malformed, the resolver logs a warning and falls
    through to pattern-based source detection.
    """

    metadata = source_payload.get("metadata")
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        logger.warning(
            "Ignoring non-mapping payload metadata while resolving source descriptor. "
            "Falling back to pattern-based source detection."
        )
        metadata = {}

    payload_source_type = metadata.get("source_type")
    payload_source_version = metadata.get("source_version")

    resolved_type = payload_source_type if isinstance(payload_source_type, str) else None
    resolved_version = (
        payload_source_version if isinstance(payload_source_version, str) else None
    )
    return resolved_type, resolved_version


def _resolve_source_type_hint(
    source_type_hint: str,
    source_version_hint: str | None,
) -> SourceType:
    """Resolve a metadata source-type hint to ``SourceType`` or raise unsupported."""

    try:
        return SourceType(source_type_hint)
    except ValueError as exc:
        raise UnsupportedSourceError.for_source(
            source_type=source_type_hint,
            source_version=source_version_hint,
            supported_sources=_supported_source_values(),
        ) from exc


def _detect_source_descriptor_via_plugins(
    source_payload: dict[str, Any],
    *,
    candidate_source_type: SourceType | None = None,
) -> tuple[SourceType, str] | None:
    """Detect source descriptor by delegating to registered plugin detectors."""

    matches: set[tuple[SourceType, str]] = set()
    for plugin in get_registered_materialiser_plugins(candidate_source_type):
        detected = plugin.resolve_type_and_version(source_payload)
        if detected is None:
            continue

        if detected != (plugin.source_type, plugin.source_version):
            raise SourceValidationError(
                "Plugin detector returned descriptor mismatch.",
                source_type=plugin.source_type.value,
                source_version=plugin.source_version,
                path="$.metadata",
                details={
                    "detected_source_type": detected[0].value,
                    "detected_source_version": detected[1],
                },
            )

        matches.add(detected)

    if not matches:
        return None

    if len(matches) > 1:
        raise SourceValidationError(
            "Ambiguous source descriptor detected from payload pattern matching.",
            path="$.metadata",
            details={
                "candidates": tuple(
                    sorted((source.value, version) for source, version in matches)
                )
            },
        )

    return next(iter(matches))


def _resolve_source_version(
    *,
    source_type: SourceType,
    requested_version: str | None,
) -> str:
    """Resolve source version against registered plugin versions."""

    source_versions = get_registered_source_versions(source_type)
    if not source_versions:
        raise UnsupportedSourceError.for_source(
            source_type=source_type.value,
            source_version=requested_version,
            supported_sources=_supported_source_values(),
        )

    if requested_version is None:
        if len(source_versions) == 1:
            return source_versions[0]

        raise SourceValidationError(
            "Could not infer source_version from payload metadata.",
            source_type=source_type.value,
            path="$.source_version",
            details={"supported_versions": source_versions},
        )

    if requested_version not in source_versions:
        raise UnsupportedSourceVersionError.for_version(
            source_type=source_type.value,
            source_version=requested_version,
            supported_versions=source_versions,
        )
    return requested_version


def resolve_source_descriptor(source_payload: dict[str, Any]) -> tuple[SourceType, str]:
    """Resolve source descriptor from payload metadata or pattern detection."""

    source_type_hint, source_version_hint = _extract_payload_metadata(source_payload)

    if source_type_hint is None:
        detected_descriptor = _detect_source_descriptor_via_plugins(source_payload)
        if detected_descriptor is not None:
            return detected_descriptor

        raise UnsupportedSourceError.for_source(
            source_type="unknown",
            source_version=None,
            supported_sources=_supported_source_values(),
        )

    resolved_type = _resolve_source_type_hint(source_type_hint, source_version_hint)

    if source_version_hint is not None:
        resolved_version = _resolve_source_version(
            source_type=resolved_type,
            requested_version=source_version_hint,
        )
        return resolved_type, resolved_version

    detected_descriptor = _detect_source_descriptor_via_plugins(
        source_payload,
        candidate_source_type=resolved_type,
    )
    if detected_descriptor is not None:
        return detected_descriptor

    resolved_version = _resolve_source_version(
        source_type=resolved_type,
        requested_version=None,
    )
    return resolved_type, resolved_version
