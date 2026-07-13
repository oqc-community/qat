# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects.builtin import IntAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1 import IntRegisterType, Registers
from qat.experimental.dialect.q1.ir.reg_desc import Q1_REGISTER_INDEX_BY_NAME


def test_register_factory():
    with pytest.raises(VerifyException):
        IntRegisterType.from_name("x0")

    with pytest.raises(VerifyException):
        IntRegisterType.from_name("R64")

    r0 = IntRegisterType.from_name("R0")
    r1 = IntRegisterType.from_name(StringAttr("R1"))
    r2 = IntRegisterType.from_name(StringAttr("R0"))
    assert r0 == r2
    assert r0 != r1


def test_register_count():
    assert len(Q1_REGISTER_INDEX_BY_NAME) == 64
    assert set(Q1_REGISTER_INDEX_BY_NAME.values()) == set(range(64))


def test_gpr_register_lookup_matches_named_constant():
    assert IntRegisterType.gpr_register(1) == Registers.R1
    assert IntRegisterType.gpr_register(63) == Registers.R63


def test_infinite_registers():
    neg_idx = -5
    with pytest.raises(
        AssertionError, match=f"Infinite index must be positive, got {neg_idx}."
    ):
        IntRegisterType.infinite_register(neg_idx)

    pos_idx = 15
    tmp = IntRegisterType.infinite_register(pos_idx)
    assert tmp.register_name == StringAttr(f"inf_gpr_{pos_idx}")

    # Non-predefined regs should not have an index
    assert tmp.index == IntAttr(~pos_idx)


def test_unallocated_register_constant():
    assert IntRegisterType.unallocated() == Registers.UNALLOCATED_INT


def test_register_descriptor_helpers():
    assert IntRegisterType.infinite_register_prefix() == "inf_gpr_"
    assert IntRegisterType.allocatable_registers() == Registers.GPR
