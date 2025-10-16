# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from enum import Enum

from qat.utils.pydantic import NoExtraFieldsFrozenModel


class Scale(Enum):
    """
    SI unit scales.
    """

    NANO = "n"
    MICRO = "u"
    MILLI = "m"
    DEFAULT = ""
    KILO = "k"
    MEGA = "M"
    GIGA = "G"
    TERA = "T"


class Quantity(NoExtraFieldsFrozenModel):
    amount: float
    scale: Scale

    def __str__(self):
        return f"{self.amount} {self.scale.value}"


class Frequency(Quantity):
    def __str__(self):
        return super().__str__() + "Hz"


class Time(Quantity):
    def __str__(self):
        return super().__str__() + "s"
