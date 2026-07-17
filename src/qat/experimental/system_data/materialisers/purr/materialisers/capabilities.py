# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Top-level capability builders for PuRR canonical materialisation."""

from qat.experimental.system_data.canonical.schema import (
    AcquireModeData,
    AttributeEntry,
    ResetData,
)
from qat.experimental.system_data.materialisers.purr.materialisers.common import (
    _seconds_to_picoseconds,
)


def _build_acquire_limit(repeat_limit: int | None) -> int:
    """Map the legacy PuRR repeat limit into canonical acquire-limit semantics."""

    if repeat_limit is None:
        return -1
    return repeat_limit


def _build_acquire_modes(
    supported_modes: list[str],
    default_mode: str | None,
) -> tuple[tuple[AcquireModeData, ...], str | None]:
    """Materialise canonical acquire-mode metadata from the PuRR default mode."""

    if not supported_modes:
        return (), None
    if default_mode is None:
        default_mode = supported_modes[0]
    elif default_mode not in supported_modes:
        supported_modes.append(default_mode)
    return tuple(AcquireModeData(type=mode) for mode in supported_modes), default_mode


def _build_reset_methods(
    supported_reset_methods: list[str],
    default_reset_method: str | None,
    passive_reset_time: float | None,
) -> tuple[tuple[ResetData, ...], str | None]:
    """Translate supported reset methods into canonical reset metadata."""

    reset_duration = _seconds_to_picoseconds(passive_reset_time)
    if not supported_reset_methods:
        return (), None

    methods: list[ResetData] = []
    for reset_type in supported_reset_methods:
        if reset_type == "passive" and reset_duration is not None:
            methods.append(
                ResetData(
                    type="passive",
                    attributes=(AttributeEntry(key="duration", value=reset_duration),),
                )
            )
        else:
            methods.append(
                ResetData(
                    type=reset_type,
                    attributes=(
                        AttributeEntry(key="operation_name", value=f"{reset_type}_reset"),
                    ),
                )
            )

    resolved_default = default_reset_method
    supported_set = {method.type for method in methods}
    if resolved_default not in supported_set:
        resolved_default = "passive" if "passive" in supported_set else methods[0].type

    return (
        tuple(methods),
        resolved_default,
    )
