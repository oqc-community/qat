# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from __future__ import annotations

import pytest
from xdsl.backend.block_naive_allocator import BlockNaiveAllocator
from xdsl.backend.register_stack import OutOfRegisters

from qat.experimental.dialect.q1 import IntRegisterType, Registers
from qat.experimental.dialect.q1.transforms.reg_alloc import Q1RegisterStack


def test_q1_register_stack_uses_q1_physical_registers():
    stack = Q1RegisterStack.get()

    assert stack.pop(IntRegisterType) == Registers.R1
    assert stack.pop(IntRegisterType) == Registers.R2


def test_q1_register_stack_reserves_r0_by_default():
    stack = Q1RegisterStack.get()

    assert (
        tuple(stack.pop(IntRegisterType) for _ in range(len(Registers.GPR) - 1))
        == Registers.GPR[1:]
    )
    with pytest.raises(OutOfRegisters):
        stack.pop(IntRegisterType)


def test_q1_register_stack_from_reserved_registers_uses_default_r0_reservation():
    stack = Q1RegisterStack.from_reserved_registers()

    assert stack.pop(IntRegisterType) == Registers.R1


def test_q1_register_stack_can_disable_default_reserved_registers():
    stack = Q1RegisterStack.from_reserved_registers(())

    assert tuple(stack.pop(IntRegisterType) for _ in Registers.GPR) == Registers.GPR
    with pytest.raises(OutOfRegisters):
        stack.pop(IntRegisterType)


def test_q1_register_stack_can_reserve_additional_registers():
    stack = Q1RegisterStack.from_reserved_registers((Registers.R0, Registers.R1))

    assert stack.pop(IntRegisterType) == Registers.R2


def test_q1_register_stack_can_back_xdsl_allocators():
    allocator = BlockNaiveAllocator(Q1RegisterStack.get(), IntRegisterType)

    assert allocator.register_base_class is IntRegisterType
    assert allocator.available_registers.pop(IntRegisterType) == Registers.R1
