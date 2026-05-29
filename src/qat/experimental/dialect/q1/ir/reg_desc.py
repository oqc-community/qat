# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from abc import ABC
from typing import TypeVar

from xdsl.backend.register_type import RegisterType
from xdsl.irdl import irdl_attr_definition


class Q1RegisterType(RegisterType, ABC):
    """A register type in the QBlox Q1 ISA."""

    @classmethod
    def gpr_register(cls, index: int) -> "Q1RegisterType":
        """Get the general-purpose register at the given index."""

        return Registers.GPR[index]


Q1_REGISTER_INDEX_BY_NAME = {f"R{i}": i for i in range(64)}


@irdl_attr_definition
class IntRegisterType(Q1RegisterType, ABC):
    """Integer register type for QBlox Q1 ISA."""

    name = "q1.reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        """Returns a dictionary mapping register name to its index."""

        return Q1_REGISTER_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls) -> str:
        """Provide the prefix for the name for a register at the given index in the
        "infinite" register set.

        For example, if the prefix is `oqc_qat_q1_`, the name of the first infinite register
        will be `oqc_qat_q1_0`.
        """

        return "inf_gpr_"

    @classmethod
    def allocatable_registers(cls):
        """Return the finite set of allocatable physical integer registers."""

        return Registers.GPR


class Registers:
    """Named register constants and classes."""

    UNALLOCATED_INT = IntRegisterType.unallocated()


for i in range(64):
    setattr(Registers, f"R{i}", IntRegisterType.from_name(f"R{i}"))

Registers.GPR = tuple(getattr(Registers, f"R{i}") for i in range(64))


RInvT = TypeVar("RInvT", bound=Q1RegisterType)
