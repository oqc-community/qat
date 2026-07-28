# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Q1 support for xDSL register allocation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from xdsl.backend.block_naive_allocator import BlockNaiveAllocator
from xdsl.backend.register_allocator import live_ins_per_block
from xdsl.backend.register_stack import RegisterStack
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import DiagnosticException

from qat.experimental.dialect.q1.ir.reg_desc import (
    IntRegisterType,
    Q1RegisterType,
    Registers,
)
from qat.experimental.dialect.q1_sequence import SequenceOp


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


class Q1LinearScanAllocator(BlockNaiveAllocator):
    """A linear scan register allocator for Q1 physical registers.

    Implements a register allocator strategy that traverses the use-def SSA chain backwards
    (i.e., from uses to defs) and allocates registers for operands and frees registers for
    results. This operates at the structured control flow level (q1_scf), and allows for
    simple and efficient allocation without resorting to more complex graph colouring
    algorithms and fixed-point iteration.

    The assumption that this operates at the structured control flow level means that it can
    only handle sequences with a single block, and will raise an exception if multiple
    blocks are present.
    """

    def __init__(self, available_registers: Q1RegisterStack):
        """:param available_registers: The Q1 physical registers available for allocation."""
        super().__init__(available_registers, Q1RegisterType)

    def allocate_sequence(self, sequence: SequenceOp) -> None:
        """Allocate registers for the given sequence operation."""

        if len(sequence.body.blocks) == 0:
            return
        if len(sequence.body.blocks) > 1:
            raise DiagnosticException(
                "Q1LinearScanAllocator does not support SequenceOps with more than one "
                "block."
            )

        self.live_ins_per_block = live_ins_per_block(sequence.body.blocks[0])
        self.allocate_block(sequence.body.blocks[0])


class LinearScanRegisterAllocationPass(ModulePass):
    """A pass that applies linear scan register allocation to all sequences in a module.

    It applies register allocation individually to each :class:`SequenceOp` defined
    throughout the module. The allocator works by walking the block within the sequence
    backwards, allocating registers for operands and freeing registers for results. This is
    a simple and efficient allocation strategy that avoids the complexity of graph colouring
    and fixed-point iteration.

    It is aimed at the structured control flow level (q1_scf); if a sequence contains
    multiple blocks, an exception will be raised. There are also certain requirements on
    for loops, see the documentation there.
    """

    name = "q1-lin-scan-reg-alloc"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for sequence in (s for s in op.walk() if isinstance(s, SequenceOp)):
            allocator = Q1LinearScanAllocator(Q1RegisterStack.from_reserved_registers())
            allocator.allocate_sequence(sequence)
