# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests that xDSL's ``CommonSubexpressionElimination`` merges duplicates of any pure pulse
operation, not only analytical waveform ops.

Operands are produced by a non-pure test op so that constant folding cannot remove the
operations under test, which leaves CSE as the only reason a duplicate disappears.
"""

import pytest
from xdsl.dialects.builtin import ModuleOp, f64
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination

from qat.experimental.dialect.pulse.ir import (
    AddOp,
    AmplitudeAttr,
    AmplitudeType,
    ConstantOp,
    FrequencyType,
    MixOp,
    Pulse,
    ScaleOp,
    SubOp,
    TimeAttr,
    TimeType,
    WaveformType,
)

from tests.unit.utils.ir import (
    build_module_from_ops,
    create_context,
    get_operations_with_type,
)

_CONTEXT = create_context(Pulse)


@irdl_op_definition
class _ProducerOp(IRDLOperation):
    """A non-pure op producing an opaque SSA value, so it is never merged or folded."""

    name = "test.producer"
    result = result_def()

    def __init__(self, result_type: Attribute):
        super().__init__(result_types=[result_type])


@irdl_op_definition
class _ConsumerOp(IRDLOperation):
    """A non-pure op using a value, to check which producer the use points at."""

    name = "test.consumer"
    value = operand_def()

    def __init__(self, value: SSAValue):
        super().__init__(operands=[value])


def _make_binary(op_cls, lhs, rhs, result_type):
    # ``MixOp`` always yields a waveform, so its constructor takes no result type.
    if op_cls is MixOp:
        return MixOp(lhs, rhs)
    return op_cls(lhs, rhs, result_type)


def _run_cse(ops: list) -> ModuleOp:
    """Run CSE over *ops*, with every otherwise unused result consumed so that CSE's removal
    of dead pure ops does not hide whether duplicates were merged."""
    consumers = [
        _ConsumerOp(result) for op in ops for result in op.results if not result.uses
    ]
    module = build_module_from_ops([*ops, *consumers])
    CommonSubexpressionElimination().apply(_CONTEXT, module)
    return module


# Pure binary ops over each type they accept, covering non-waveform and waveform operands.
_BINARY_CASES = [
    pytest.param(AddOp, TimeType(), id="add-time"),
    pytest.param(AddOp, FrequencyType(), id="add-frequency"),
    pytest.param(AddOp, WaveformType(), id="add-waveform"),
    pytest.param(SubOp, TimeType(), id="sub-time"),
    pytest.param(SubOp, AmplitudeType(), id="sub-amplitude"),
    pytest.param(SubOp, WaveformType(), id="sub-waveform"),
    pytest.param(MixOp, WaveformType(), id="mix-waveform"),
]


class TestPureBinaryOps:
    """Duplicate arithmetic on pulse types is merged."""

    @pytest.mark.parametrize(("op_cls", "operand_type"), _BINARY_CASES)
    def test_duplicate_ops_are_merged(self, op_cls, operand_type):
        lhs, rhs = _ProducerOp(operand_type), _ProducerOp(operand_type)
        first = _make_binary(op_cls, lhs, rhs, operand_type)
        second = _make_binary(op_cls, lhs, rhs, operand_type)
        first_user, second_user = _ConsumerOp(first.result), _ConsumerOp(second.result)

        module = _run_cse([lhs, rhs, first, second, first_user, second_user])

        assert len(get_operations_with_type(module, op_cls)) == 1
        assert first_user.value is second_user.value

    @pytest.mark.parametrize(("op_cls", "operand_type"), _BINARY_CASES)
    def test_ops_with_different_operands_are_not_merged(self, op_cls, operand_type):
        lhs, rhs, other = (_ProducerOp(operand_type) for _ in range(3))
        first = _make_binary(op_cls, lhs, rhs, operand_type)
        second = _make_binary(op_cls, lhs, other, operand_type)

        module = _run_cse([lhs, rhs, other, first, second])

        assert len(get_operations_with_type(module, op_cls)) == 2

    @pytest.mark.parametrize(
        "op_cls", [pytest.param(SubOp, id="sub"), pytest.param(MixOp, id="mix")]
    )
    def test_swapped_operands_of_non_commutative_ops_are_not_merged(self, op_cls):
        operand_type = WaveformType()
        lhs, rhs = _ProducerOp(operand_type), _ProducerOp(operand_type)
        first = _make_binary(op_cls, lhs, rhs, operand_type)
        second = _make_binary(op_cls, rhs, lhs, operand_type)

        module = _run_cse([lhs, rhs, first, second])

        # Mix is element-wise multiplication but does not declare ``Commutative``, so
        # CSE conservatively keeps both. Sub is genuinely order dependent.
        assert len(get_operations_with_type(module, op_cls)) == 2

    def test_different_op_types_with_same_operands_are_not_merged(self):
        operand_type = TimeType()
        lhs, rhs = _ProducerOp(operand_type), _ProducerOp(operand_type)
        add = AddOp(lhs, rhs, operand_type)
        sub = SubOp(lhs, rhs, operand_type)

        module = _run_cse([lhs, rhs, add, sub])

        assert len(get_operations_with_type(module, AddOp)) == 1
        assert len(get_operations_with_type(module, SubOp)) == 1

    def test_chained_duplicates_are_merged_transitively(self):
        """Once the inner duplicates merge, the outer ones have identical operands."""
        operand_type = TimeType()
        lhs, rhs = _ProducerOp(operand_type), _ProducerOp(operand_type)
        inner_1, inner_2 = AddOp(lhs, rhs, operand_type), AddOp(lhs, rhs, operand_type)
        outer_1 = SubOp(inner_1, rhs, operand_type)
        outer_2 = SubOp(inner_2, rhs, operand_type)

        module = _run_cse([lhs, rhs, inner_1, inner_2, outer_1, outer_2])

        assert len(get_operations_with_type(module, AddOp)) == 1
        assert len(get_operations_with_type(module, SubOp)) == 1


class TestScaleOp:
    """Duplicate scaling is merged only when both the factor and operand match."""

    @pytest.mark.parametrize("operand_type", [AmplitudeType(), WaveformType()])
    def test_duplicate_scale_ops_are_merged(self, operand_type):
        factor, operand = _ProducerOp(f64), _ProducerOp(operand_type)
        first = ScaleOp(factor, operand, operand_type)
        second = ScaleOp(factor, operand, operand_type)
        first_user, second_user = _ConsumerOp(first.result), _ConsumerOp(second.result)

        module = _run_cse([factor, operand, first, second, first_user, second_user])

        assert len(get_operations_with_type(module, ScaleOp)) == 1
        assert first_user.value is second_user.value

    def test_different_factors_are_not_merged(self):
        factor, other_factor = _ProducerOp(f64), _ProducerOp(f64)
        operand = _ProducerOp(WaveformType())
        first = ScaleOp(factor, operand, WaveformType())
        second = ScaleOp(other_factor, operand, WaveformType())

        module = _run_cse([factor, other_factor, operand, first, second])

        assert len(get_operations_with_type(module, ScaleOp)) == 2

    def test_different_operands_are_not_merged(self):
        factor = _ProducerOp(f64)
        operand, other_operand = _ProducerOp(WaveformType()), _ProducerOp(WaveformType())
        first = ScaleOp(factor, operand, WaveformType())
        second = ScaleOp(factor, other_operand, WaveformType())

        module = _run_cse([factor, operand, other_operand, first, second])

        assert len(get_operations_with_type(module, ScaleOp)) == 2

    def test_add_and_mix_of_the_same_operands_are_not_merged(self):
        """Different ops on identical operands stay distinct."""
        operand = _ProducerOp(WaveformType())
        add = AddOp(operand, operand, WaveformType())
        mix = MixOp(operand, operand)

        module = _run_cse([operand, add, mix])

        assert len(get_operations_with_type(module, AddOp)) == 1
        assert len(get_operations_with_type(module, MixOp)) == 1


class TestConstantOp:
    """Duplicate constants are merged, which is what lets dependent ops match."""

    def test_equal_constants_are_merged(self):
        first, second = ConstantOp(TimeAttr(80_000)), ConstantOp(TimeAttr(80_000))
        first_user, second_user = _ConsumerOp(first.result), _ConsumerOp(second.result)

        module = _run_cse([first, second, first_user, second_user])

        assert len(get_operations_with_type(module, ConstantOp)) == 1
        assert first_user.value is second_user.value

    def test_constants_with_different_values_are_not_merged(self):
        first, second = ConstantOp(TimeAttr(80_000)), ConstantOp(TimeAttr(160_000))

        module = _run_cse([first, second])

        assert len(get_operations_with_type(module, ConstantOp)) == 2

    def test_constants_with_different_types_are_not_merged(self):
        time, amplitude = ConstantOp(TimeAttr(1)), ConstantOp(AmplitudeAttr(1))

        module = _run_cse([time, amplitude])

        assert len(get_operations_with_type(module, ConstantOp)) == 2

    def test_ops_on_merged_constants_are_merged(self):
        """Ops whose operands were separate but equal constants become duplicates."""
        lhs_1, rhs_1 = ConstantOp(TimeAttr(80_000)), ConstantOp(TimeAttr(40_000))
        lhs_2, rhs_2 = ConstantOp(TimeAttr(80_000)), ConstantOp(TimeAttr(40_000))
        first = AddOp(lhs_1, rhs_1, TimeType())
        second = AddOp(lhs_2, rhs_2, TimeType())

        module = _run_cse([lhs_1, rhs_1, lhs_2, rhs_2, first, second])

        assert len(get_operations_with_type(module, ConstantOp)) == 2
        assert len(get_operations_with_type(module, AddOp)) == 1


class TestNonPureOps:
    """Ops without the ``Pure`` trait are never merged."""

    def test_duplicate_non_pure_ops_are_kept(self):
        first, second = _ProducerOp(TimeType()), _ProducerOp(TimeType())

        module = _run_cse([first, second])

        assert len(get_operations_with_type(module, _ProducerOp)) == 2
