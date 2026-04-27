# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""This module provides utilities for representing units of numeric values in the pulse IR,
and converting them into the expected literal values."""

from enum import Enum


class _StrEnum(str, Enum):
    """Since we support Python 3.10, we can't use StrEnum directly.

    This acts as a base class to give that support, so the enum values also behave as
    strings.
    """

    pass


class TimeUnits(_StrEnum):
    """Enumeration of time units in the pulse dialect."""

    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"


class FrequencyUnits(_StrEnum):
    """Enumeration of frequency units in the pulse dialect."""

    HERTZ = "Hz"
    KILOHERTZ = "kHz"
    MEGAHERTZ = "MHz"
    GIGAHERTZ = "GHz"


TIME_UNIT_EXPONENTS: dict[TimeUnits, int] = {
    TimeUnits.NANOSECOND: -9,
    TimeUnits.MICROSECOND: -6,
    TimeUnits.MILLISECOND: -3,
    TimeUnits.SECOND: 0,
}

FREQUENCY_UNIT_EXPONENTS: dict[FrequencyUnits, int] = {
    FrequencyUnits.HERTZ: 0,
    FrequencyUnits.KILOHERTZ: 3,
    FrequencyUnits.MEGAHERTZ: 6,
    FrequencyUnits.GIGAHERTZ: 9,
}
