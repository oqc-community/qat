# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Shared external-resource registry for PuRR materialisation.

This module provides a small deduplicating registry so multiple builder components can
register references to the same external hardware resource without rebuilding duplicate
canonical entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qat.experimental.system_data.canonical.schema import (
    AttributeEntry,
    ExternalResourceData,
)


@dataclass(slots=True)
class _ExternalResourceAccumulator:
    """Mutable accumulation state for one external resource id."""

    resource_id: str
    object_type: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


class ExternalResourceRegistry:
    """Deduplicating registry keyed by external resource id."""

    def __init__(self) -> None:
        self._resources: dict[str, _ExternalResourceAccumulator] = {}

    def register(
        self,
        *,
        resource_id: str | None,
        object_type: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str | None:
        """Register or merge an external resource definition by id.

        If the id is already known, fields are merged deterministically:

        - ``object_type`` keeps the first non-empty value.
        - ``attributes`` keep first-write per key.
        """

        if not isinstance(resource_id, str) or not resource_id:
            return None

        resource = self._resources.get(resource_id)
        if resource is None:
            self._resources[resource_id] = _ExternalResourceAccumulator(
                resource_id=resource_id,
                object_type=object_type,
                attributes=dict(attributes or {}),
            )
            return resource_id

        if resource.object_type is None and object_type is not None:
            resource.object_type = object_type

        for key, value in (attributes or {}).items():
            resource.attributes.setdefault(key, value)

        return resource_id

    def to_tuple(self) -> tuple[ExternalResourceData, ...]:
        """Return canonical immutable external resource records."""

        canonical_resources: list[ExternalResourceData] = []
        for resource_id in sorted(self._resources):
            resource = self._resources[resource_id]
            canonical_resources.append(
                ExternalResourceData(
                    id=resource.resource_id,
                    object_type=resource.object_type,
                    attributes=tuple(
                        AttributeEntry(key=key, value=value)
                        for key, value in resource.attributes.items()
                    ),
                )
            )

        return tuple(canonical_resources)
