# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""PuRR v0.1.0 ingress DTO definitions.

The ingress DTO captures the compiler-owned view of the supported PuRR source contract at
the materialisation boundary. It is intentionally permissive for nested legacy payloads so
that structural decoding and top-level compatibility checks can evolve before the full
boundary schema is fully typed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PurrIngressV010(BaseModel):
    """Boundary DTO for PuRR source version ``0.1.0`` payloads.

    Only the top-level collections and scalar compatibility fields used by the current
    materialiser are modelled explicitly.

    Collection field structure
    --------------------------

    The following fields use a two-level mapping shape:

    - ``quantum_devices: dict[str, dict[str, Any]]``
    - ``pulse_channels: dict[str, dict[str, Any]]``
    - ``physical_channels: dict[str, dict[str, Any]]``
    - ``basebands: dict[str, dict[str, Any]]``

    For each field:

    - Outer ``dict[str, ...]`` key: source-defined entity identifier.
        Examples include a device id, pulse-channel id, physical-channel id, or baseband id.
    - Inner ``dict[str, Any]`` value: untyped payload blob for that entity.
        This preserves legacy nested structures while boundary validators and mapping helpers
        enforce shape and semantic constraints.

    The inner payload remains ``Any``-typed by design at this stage so the boundary can
    accept source variants and perform incremental validation/mapping without requiring a
    fully rigid nested DTO schema.
    """

    model_config = ConfigDict(extra="allow")

    calibration_id: str = ""
    supported_acquire_modes: list[str] = Field(default_factory=list)
    supported_reset_methods: list[str] = Field(default_factory=list)
    default_acquire_mode: str | None = None
    default_reset_method: str | None = None
    default_repeat_count: int | None = None
    passive_reset_time: float | None = None
    repeat_limit: int | None = None

    quantum_devices: dict[str, dict[str, Any]]
    pulse_channels: dict[str, dict[str, Any]]
    physical_channels: dict[str, dict[str, Any]]
    basebands: dict[str, dict[str, Any]]
    qubit_direction_couplings: list[dict[str, Any]] = Field(default_factory=list)
    error_mitigation: dict[str, Any] | None = None
