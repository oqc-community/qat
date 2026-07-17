# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Shared conversion helpers for PuRR canonical materialisation."""

import math
from numbers import Number

from qat.experimental.utils.logging import get_logger

logger = get_logger(__name__)

_PICOSECOND_ROUNDING_ABS_TOLERANCE = 1e-9


def _seconds_to_picoseconds(value: float | None) -> int | None:
    """Convert a second-based duration into canonical integer picoseconds."""

    if value is None:
        return None

    scaled_value = value * 1e12
    rounded_value = int(round(scaled_value))
    if not math.isclose(
        scaled_value,
        float(rounded_value),
        rel_tol=0.0,
        abs_tol=_PICOSECOND_ROUNDING_ABS_TOLERANCE,
    ):
        logger.warning(
            "Rounded duration from scaled picoseconds value %s to integer %s.",
            scaled_value,
            rounded_value,
        )
    return rounded_value


def _hz_to_int(value: float | int | None) -> int | None:
    """Convert a source frequency value into canonical integer Hz."""

    if value is None:
        return None
    return int(round(value))


def _as_complex(value: Number, default: complex | None = 1.0 + 0.0j) -> complex | None:
    """Coerce a scalar or complex-like source value into a canonical complex value."""

    if isinstance(value, complex):
        return value
    if isinstance(value, int | float):
        return complex(value, 0.0)
    return default


def _as_float(value: Number, default: float | None = 0.0) -> float | None:
    """Coerce a numeric source value to ``float`` with a deterministic fallback."""

    if isinstance(value, int | float):
        return float(value)
    return default
