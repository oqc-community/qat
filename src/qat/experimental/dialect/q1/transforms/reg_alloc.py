# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Q1 support for xDSL register allocation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from xdsl.backend.register_stack import RegisterStack

from qat.experimental.dialect.q1.ir.reg_desc import (
    IntRegisterType,
    Q1RegisterType,
    Registers,
)


@dataclass
class Q1RegisterStack(RegisterStack):
    """Register stack configured with the Q1 physical register set."""

    _DEFAULT_RESERVED_REGISTERS: ClassVar[tuple[Q1RegisterType, ...]] = (Registers.R0,)
    _DEFAULT_ALLOCATABLE_REGISTERS: ClassVar[tuple[Q1RegisterType, ...]] = tuple(
        reversed(Registers.GPR[1:])
    )

    @classmethod
    def allocatable_registers(
        cls,
        reserved_registers: Iterable[Q1RegisterType] | None = None,
    ) -> tuple[Q1RegisterType, ...]:
        """Return Q1 physical registers available after excluding reserved registers."""

        if reserved_registers is None:
            reserved_registers = cls._DEFAULT_RESERVED_REGISTERS
        reserved = frozenset(reserved_registers)
        return tuple(
            reg for reg in IntRegisterType.allocatable_registers() if reg not in reserved
        )

    @classmethod
    def default_allocatable_registers(cls) -> tuple[Q1RegisterType, ...]:
        return cls._DEFAULT_ALLOCATABLE_REGISTERS

    @classmethod
    def from_reserved_registers(
        cls,
        reserved_registers: Iterable[Q1RegisterType] | None = None,
        allow_infinite: bool = False,
    ) -> Q1RegisterStack:
        """Create a stack with the requested Q1 registers excluded from allocation."""

        return cls.get(
            allocatable_registers=reversed(cls.allocatable_registers(reserved_registers)),
            allow_infinite=allow_infinite,
        )
