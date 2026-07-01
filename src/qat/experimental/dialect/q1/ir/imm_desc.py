# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Semantic immediate attribute types for the QBlox Q1 ISA dialect.

Each subclass of :class:`Q1Imm` is a :class:`~xdsl.ir.Data` ``[int]`` attribute that
enforces hardware-specific value constraints at construction time. Invalid values
raise :class:`~xdsl.utils.exceptions.VerifyException` immediately and cannot exist
in the IR.

The types cover all distinct operand semantics in the Q1 instruction set: boolean
flags, small unsigned fields, durations, gains/offsets, NCO phase, waveform /
acquisition indices, jump addresses, and generic 32-bit values.
"""

from typing import ClassVar, TypeVar

from xdsl.ir import Data
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


class Q1Imm(Data[int]):
    """Base class for all Q1 immediate value attributes.

    Subclasses set :attr:`_MIN` and :attr:`_MAX` (inclusive) and inherit the
    construction-time and IR-level validation. They may also override
    :meth:`_validate` to add extra constraints (e.g. alignment, enum membership).
    """

    _MIN: ClassVar[int]
    _MAX: ClassVar[int]

    def __init__(self, value: int):
        self._validate(value)
        super().__init__(value)

    @classmethod
    def _validate(cls, value: int) -> None:
        if not cls._MIN <= value <= cls._MAX:
            raise VerifyException(
                f"{cls.__name__} value must be in [{cls._MIN}, {cls._MAX}], got {value}"
            )

    def verify(self) -> None:
        self._validate(self.data)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> int:
        with parser.in_angle_brackets():
            return parser.parse_integer(allow_negative=True)

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"<{self.data}>")


@irdl_attr_definition
class BoolImm(Q1Imm):
    """Boolean immediate: ``{0, 1}``."""

    name = "q1.bool_imm"
    _MIN = 0
    _MAX = 1


@irdl_attr_definition
class UI2Imm(Q1Imm):
    """2-bit unsigned immediate: ``[0, 3]``."""

    name = "q1.ui2_imm"
    _MIN = 0
    _MAX = 3


@irdl_attr_definition
class UI3Imm(Q1Imm):
    """3-bit unsigned immediate: ``[0, 7]``."""

    name = "q1.ui3_imm"
    _MIN = 0
    _MAX = 7


@irdl_attr_definition
class UI4Imm(Q1Imm):
    """4-bit unsigned immediate: ``[0, 15]``."""

    name = "q1.ui4_imm"
    _MIN = 0
    _MAX = 15


@irdl_attr_definition
class UI5Imm(Q1Imm):
    """5-bit unsigned immediate: ``[0, 31]``."""

    name = "q1.ui5_imm"
    _MIN = 0
    _MAX = 31


@irdl_attr_definition
class UI6Imm(Q1Imm):
    """6-bit unsigned immediate: ``[0, 63]``."""

    name = "q1.ui6_imm"
    _MIN = 0
    _MAX = 63


@irdl_attr_definition
class UI7Imm(Q1Imm):
    """7-bit unsigned immediate: ``[0, 127]``."""

    name = "q1.ui7_imm"
    _MIN = 0
    _MAX = 127


@irdl_attr_definition
class UI8Imm(Q1Imm):
    """8-bit unsigned immediate: ``[0, 255]``."""

    name = "q1.ui8_imm"
    _MIN = 0
    _MAX = 255


@irdl_attr_definition
class UI10Imm(Q1Imm):
    """10-bit unsigned immediate: ``[0, 1023]``."""

    name = "q1.ui10_imm"
    _MIN = 0
    _MAX = 1023


@irdl_attr_definition
class UI14Imm(Q1Imm):
    """14-bit unsigned immediate: ``[0, 16383]``."""

    name = "q1.ui14_imm"
    _MIN = 0
    _MAX = 16383


# Semantic alias: every jump/loop target address uses the same 14-bit encoding.
AddressImm = UI14Imm


@irdl_attr_definition
class UI16Imm(Q1Imm):
    """16-bit unsigned immediate: ``[0, 65535]``."""

    name = "q1.ui16_imm"
    _MIN = 0
    _MAX = 65535


@irdl_attr_definition
class SI16Imm(Q1Imm):
    """16-bit signed immediate: ``[-32768, 32767]``."""

    name = "q1.si16_imm"
    _MIN = -32768
    _MAX = 32767


@irdl_attr_definition
class DurationImm(Q1Imm):
    """RT-instruction duration in ns: ``[4, 65535]``.

    The minimum value of 4 ns matches the Q1 Real-Time core clock cycle.
    """

    name = "q1.duration_imm"
    _MIN = 4
    _MAX = 65535


@irdl_attr_definition
class UI24Imm(Q1Imm):
    """24-bit unsigned immediate: ``[0, 16_777_215]``."""

    name = "q1.ui24_imm"
    _MIN = 0
    _MAX = 16_777_215


@irdl_attr_definition
class UI32Imm(Q1Imm):
    """32-bit unsigned immediate: ``[0, 2^32 - 1]``."""

    name = "q1.ui32_imm"
    _MIN = 0
    _MAX = 2**32 - 1


@irdl_attr_definition
class SI32Imm(Q1Imm):
    """32-bit signed immediate: ``[-2^31, 2^31 - 1]``."""

    name = "q1.si32_imm"
    _MIN = -(2**31)
    _MAX = 2**31 - 1


@irdl_attr_definition
class SU32Imm(Q1Imm):
    """32-bit signed-or-unsigned immediate: ``[-2^31, 2^32 - 1]``.

    Used by Q1 instructions whose immediate operand accepts either signed
    or unsigned 32-bit representation (e.g. ``move``, ``not``, generic ALU).
    """

    name = "q1.su32_imm"
    _MIN = -(2**31)
    _MAX = 2**32 - 1


@irdl_attr_definition
class NcoPhaseImm(Q1Imm):
    """NCO phase / phase-offset immediate: ``[0, 1_000_000_000]`` (inclusive)."""

    name = "q1.nco_phase_imm"
    _MIN = 0
    _MAX = 1_000_000_000


ImmT = TypeVar("ImmT", bound=Q1Imm)
ImmT1 = TypeVar("ImmT1", bound=Q1Imm)
ImmT2 = TypeVar("ImmT2", bound=Q1Imm)
ImmT3 = TypeVar("ImmT3", bound=Q1Imm)
ImmT4 = TypeVar("ImmT4", bound=Q1Imm)
ImmT5 = TypeVar("ImmT5", bound=Q1Imm)
