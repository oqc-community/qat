# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Name allocation."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Sequence
from typing import TypeVar

_Key = TypeVar("_Key", bound=Hashable)


def assign_unique_names(
    keys: Sequence[_Key], reserved: Collection[str], prefix: str
) -> dict[_Key, str]:
    """Assign each key a ``{prefix}{n}`` name kept distinct from ``reserved``.

    Names are handed out in order with a monotonic counter, skipping any already
    held in ``reserved`` or issued earlier in the same call. The ``{prefix}{n}``
    scheme is preserved wherever it is free and merely deconflicted otherwise.
    """
    held = set(reserved)
    names: dict[_Key, str] = {}
    counter = 0
    for key in keys:
        while f"{prefix}{counter}" in held:
            counter += 1
        name = f"{prefix}{counter}"
        held.add(name)
        names[key] = name
        counter += 1
    return names


def assign_unique_name(
    reserved: Collection[str], prefix: str, preferred_name: str | None = None
) -> str:
    """Assign one name distinct from ``reserved``.

    If ``preferred_name`` is provided and free, it is returned unchanged.
    Otherwise a deconflicted ``{prefix}{n}`` name is assigned.
    """
    if preferred_name is not None and preferred_name not in reserved:
        return preferred_name
    return assign_unique_names([0], reserved, prefix)[0]
