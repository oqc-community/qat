# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for the gate dialect qubit types."""

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import IntegerAttr, i32, i64
from xdsl.ir import TypeAttribute
from xdsl.parser import Parser
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.gate.ir import (
    AlgorithmicQubitType,
    Gate,
    PhysicalQubitType,
    QubitType,
)


class TestQubitTypes:
    @pytest.mark.parametrize("qubit_type", [AlgorithmicQubitType(), PhysicalQubitType(0)])
    def test_subtypes_are_qubit_types(self, qubit_type):
        assert isinstance(qubit_type, QubitType)
        assert isinstance(qubit_type, TypeAttribute)

    def test_algorithmic_and_physical_are_distinct(self):
        assert AlgorithmicQubitType() != PhysicalQubitType(0)
        assert AlgorithmicQubitType() == AlgorithmicQubitType()

    @pytest.mark.parametrize(
        "type_, name",
        [
            (AlgorithmicQubitType, "gate.qubit.algorithmic"),
            (PhysicalQubitType, "gate.qubit.physical"),
        ],
    )
    def test_names(self, type_, name):
        assert type_.name == name


class TestPhysicalQubitType:
    @pytest.mark.parametrize("index", [0, 1, 127])
    def test_int_index_is_converted_to_integer_attr(self, index):
        qubit = PhysicalQubitType(index)
        assert qubit.index == IntegerAttr(index, i64)

    def test_integer_attr_index_is_kept(self):
        index = IntegerAttr(3, i32)
        assert PhysicalQubitType(index).index == index

    def test_equality_depends_on_index(self):
        assert PhysicalQubitType(1) == PhysicalQubitType(1)
        assert PhysicalQubitType(1) != PhysicalQubitType(2)

    def test_non_integer_index_fails_verification(self):
        with pytest.raises(VerifyException):
            PhysicalQubitType("zero")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "index", [-1, IntegerAttr(-1, i32)], ids=["int", "integer_attr"]
    )
    def test_negative_index_fails_verification(self, index):
        with pytest.raises(VerifyException, match="cannot be negative"):
            PhysicalQubitType(index)

    def test_negative_index_fails_verification_when_parsed(self):
        ctx = Context()
        ctx.load_dialect(Gate)
        with pytest.raises(VerifyException, match="cannot be negative"):
            Parser(ctx, "!gate.qubit.physical<-1 : i64>").parse_type()
