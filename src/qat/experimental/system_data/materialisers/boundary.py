# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Experimental compiler materialisation boundary orchestrator.

Architecture overview
=====================

This module is the single public boundary entrypoint where external source data
crosses into compiler-owned canonical system data.

Boundary stages
---------------

1. Resolve source type/version from payload metadata or payload structure
    via descriptor-resolution helpers.
2. Resolve source version into a registered source-specific plugin.
3. Run plugin trust/integrity verification for the source/version.
4. Validate source additional data using the plugin-defined schema.
5. Dispatch to the plugin materialiser to construct canonical data.

Extension points
----------------

To add a new source or source version:

1. Implement a source-specific plugin conforming to ``SourceMaterialiserPlugin``.
2. Register it with ``register_materialiser_plugin`` from
    ``qat.experimental.system_data.materialisers.plugins``.

Each plugin owns source-version dispatch, additional-data validation schema,
integrity verification, and materialisation behavior.
"""

from typing import Any

from pydantic import ValidationError

from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.materialisers.descriptor_resolution import (
    resolve_source_descriptor,
)
from qat.experimental.system_data.materialisers.errors import (
    MaterialisationError,
    SourceIntegrityError,
    SourceValidationError,
    UnsupportedSourceVersionError,
)
from qat.experimental.system_data.materialisers.plugin_loader import load_builtin_plugins
from qat.experimental.system_data.materialisers.plugins import (
    get_materialiser_plugin,
    get_registered_source_versions,
)


def _bootstrap_builtin_plugins() -> None:
    """Load built-in source plugin modules for import-side registration.

    Built-in module paths are defined in ``plugin_loader`` so this boundary module keeps
    orchestration concerns only.
    """

    load_builtin_plugins()


_bootstrap_builtin_plugins()


def _normalise_source_additional_data(
    source_additional_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalise optional source additional data to a concrete mapping."""

    if source_additional_data is None:
        return {}

    if not isinstance(source_additional_data, dict):
        raise SourceValidationError(
            "source_additional_data must be a mapping when provided.",
            path="$.source_additional_data",
            details={"value_type": type(source_additional_data).__name__},
        )

    return source_additional_data


def materialise(
    *,
    source_payload: dict[str, Any],
    source_additional_data: dict[str, Any] | None = None,
) -> CanonicalSystemData:
    """Materialise canonical system data from a supported external source.

    :param source_payload: Raw source payload as a parsed JSON-like dict.
    :param source_additional_data: Strict source/version-specific additional datasets needed
        by the selected source materialiser.
    :returns: Canonical system data.
    :raises UnsupportedSourceError: If source type is unsupported.
    :raises UnsupportedSourceVersionError: If source version is unsupported.
    :raises SourceValidationError: If source metadata/additional-data validation fails.
    :raises MaterialisationError: Passthrough for structured plugin-raised failures.
    :raises SourceIntegrityError: If boundary trust/integrity verification fails. The
        dispatch route is source-type and source-version specific. Unsupported combinations
        fail early with structured boundary errors.
    """
    resolved_source, source_version = resolve_source_descriptor(source_payload)

    source_versions = get_registered_source_versions(resolved_source)

    plugin = get_materialiser_plugin(
        source_type=resolved_source,
        source_version=source_version,
    )
    if plugin is None:
        raise UnsupportedSourceVersionError.for_version(
            source_type=resolved_source.value,
            source_version=source_version,
            supported_versions=source_versions,
        )

    try:
        plugin.verify_integrity(source_payload)
    except MaterialisationError:
        raise
    except Exception as exc:
        raise SourceIntegrityError.for_check_failure(
            source_type=resolved_source.value,
            source_version=source_version,
            check="plugin_integrity_execution",
            cause=exc,
        ) from exc

    normalised_additional_data = _normalise_source_additional_data(source_additional_data)

    try:
        validated_additional_data = plugin.additional_data_model.model_validate(
            normalised_additional_data
        )
    except ValidationError as exc:
        raise SourceValidationError(
            "source_additional_data validation failed.",
            source_type=resolved_source.value,
            source_version=source_version,
            path="$.source_additional_data",
            details={"errors": exc.errors(include_url=False)},
            cause=exc,
        ) from exc

    return plugin.materialise(
        source_payload=source_payload,
        source_version=source_version,
        additional_data=validated_additional_data,
    )
