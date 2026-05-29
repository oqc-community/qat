# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from xdsl.dialects.builtin import IntegerAttr, Signedness

from qat.experimental.dialect.q1.ir.imm_desc import si16, si32, ui16, ui32


def test_immediate_type_widths_and_signedness():
    assert ui32.width.data == 32
    assert ui32.signedness.data == Signedness.UNSIGNED

    assert ui16.width.data == 16
    assert ui16.signedness.data == Signedness.UNSIGNED

    assert si32.width.data == 32
    assert si32.signedness.data == Signedness.SIGNED

    assert si16.width.data == 16
    assert si16.signedness.data == Signedness.SIGNED


def test_immediate_attrs_validate_values_against_signedness():
    assert IntegerAttr(10, ui32) == IntegerAttr(10, ui32)
    assert IntegerAttr(-1, si32) == IntegerAttr(-1, si32)
