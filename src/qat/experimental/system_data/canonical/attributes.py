# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Helpers for canonical system data attribute extraction and conversion."""

from qat.experimental.system_data.canonical.schema import AttributeEntry


def get_attribute_value(
    attributes: tuple[AttributeEntry, ...],
    key: str,
) -> AttributeEntry | None:
    """Return the value for ``key`` from ``attributes`` if present."""
    for attribute in attributes:
        if attribute.key == key:
            return attribute
    return None
