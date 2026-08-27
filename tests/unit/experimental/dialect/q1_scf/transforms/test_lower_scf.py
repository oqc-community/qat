# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for
:class:`~qat.experimental.dialect.q1_scf.transforms.lower_scf.LowerScfToQ1ScfPass`.

Covers ``scf.for`` lowering, rejection of unsupported configurations, and stub
behaviour for ``scf.if`` and ``scf.while``.
"""

from __future__ import annotations

import pytest
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp
from xdsl.dialects.scf import (
    ForOp as ScfForOp,
    IfOp as ScfIfOp,
    WhileOp as ScfWhileOp,
    YieldOp as ScfYieldOp,
)
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.q1 import (
    AddRsImmRdOp,
    MoveImmRdOp,
    MoveRsRdOp,
    NotRsRdOp,
    Registers,
    StopOp,
)
from qat.experimental.dialect.q1.ir.imm_desc import SU32Imm
from qat.experimental.dialect.q1.ir.reg_desc import IntRegisterType
from qat.experimental.dialect.q1_scf import ForOp, YieldOp
from qat.experimental.dialect.q1_scf.transforms.lower_scf import (
    LowerScfToQ1ScfPass,
    _get_static_integer,
)
from qat.experimental.dialect.q1_sequence import SequenceOp

R = Registers.UNALLOCATED_INT


def _make_shots_loop(lb: int, ub: int, step: int) -> SequenceOp:
    """Return a minimal ``scf.for`` with no iter_args inside a :class:`SequenceOp`."""
    lb_op = ArithConstantOp.from_int_and_width(lb, IndexType())
    ub_op = ArithConstantOp.from_int_and_width(ub, IndexType())
    step_op = ArithConstantOp.from_int_and_width(step, IndexType())

    body_block = Block(arg_types=[IndexType()])
    body_block.add_op(ScfYieldOp())

    for_op = ScfForOp(
        lb=lb_op.result, ub=ub_op.result, step=step_op.result, iter_args=[], body=body_block
    )

    entry = Block()
    entry.add_ops([lb_op, ub_op, step_op, for_op, StopOp()])
    return SequenceOp("Q0", Region([entry]))


def _lower(seq: SequenceOp) -> Block:
    """Run :class:`LowerScfToQ1ScfPass` on *seq*, verify the result, and return the entry
    block."""
    module = ModuleOp([seq])
    LowerScfToQ1ScfPass().apply(None, module)
    module.verify()
    return list(seq.body.blocks)[0]


def _ops_of_type(block: Block, op_type: type) -> list:
    """Collect all top-level ops of *op_type* in *block*."""
    return [op for op in block.ops if isinstance(op, op_type)]


class TestGetStaticInteger:
    def test_returns_integer_for_arith_constant(self):
        const = ArithConstantOp.from_int_and_width(42, IndexType())
        assert _get_static_integer(const.result) == 42

    def test_returns_none_for_block_arg(self):
        block = Block(arg_types=[IndexType()])
        assert _get_static_integer(block.args[0]) is None

    def test_returns_none_for_non_integer_constant(self):
        from xdsl.dialects.arith import ConstantOp
        from xdsl.dialects.builtin import FloatAttr, f32

        const = ConstantOp(FloatAttr(1.0, f32))
        assert _get_static_integer(const.result) is None


class TestForLowering:
    def test_shots_loop_produces_q1_scf_for(self):
        """A shots-style loop lowers to a ``q1_scf.for`` in the entry block."""
        seq = _make_shots_loop(0, 5, 1)
        entry = _lower(seq)

        for_ops = _ops_of_type(entry, ForOp)
        assert len(for_ops) == 1, "expected exactly one q1_scf.ForOp"

    def test_shots_loop_removes_scf_for(self):
        """No ``scf.for`` survives after lowering."""
        seq = _make_shots_loop(0, 5, 1)
        entry = _lower(seq)

        scf_for_ops = _ops_of_type(entry, ScfForOp)
        assert len(scf_for_ops) == 0, "scf.ForOp should be gone after lowering"

    def test_move_imm_carries_correct_count(self):
        """The ``q1.ir.move`` loaded before the loop carries the right iteration count."""
        seq = _make_shots_loop(0, 10, 1)
        entry = _lower(seq)

        move_ops = _ops_of_type(entry, MoveImmRdOp)
        assert len(move_ops) == 1
        assert move_ops[0].imm.data == 10

    def test_count_computed_from_step(self):
        """``(ub - lb) // step`` is used as the iteration count."""
        seq = _make_shots_loop(2, 12, 2)  # (12-2)//2 = 5
        entry = _lower(seq)

        move_ops = _ops_of_type(entry, MoveImmRdOp)
        assert move_ops[0].imm.data == 5

    def test_body_ops_are_preserved(self):
        """Ops inside the loop body survive lowering unchanged."""
        lb_op = ArithConstantOp.from_int_and_width(0, IndexType())
        ub_op = ArithConstantOp.from_int_and_width(4, IndexType())
        step_op = ArithConstantOp.from_int_and_width(1, IndexType())

        body_block = Block(arg_types=[IndexType()])
        body_move = MoveImmRdOp(SU32Imm(3), IntRegisterType.unallocated())
        body_block.add_ops([body_move, ScfYieldOp()])

        for_op = ScfForOp(
            lb=lb_op.result,
            ub=ub_op.result,
            step=step_op.result,
            iter_args=[],
            body=body_block,
        )
        entry = Block()
        entry.add_ops([lb_op, ub_op, step_op, for_op, StopOp()])
        lower_entry = _lower(SequenceOp("Q0", Region([entry])))

        (for_op,) = _ops_of_type(lower_entry, ForOp)
        body_ops = list(list(for_op.body.blocks)[0].ops)
        assert len(body_ops) == 2
        assert isinstance(body_ops[0], MoveImmRdOp)
        assert body_ops[0].imm.data == 3
        assert isinstance(body_ops[1], YieldOp)

    def test_body_induction_arg_is_int_register(self):
        """The body's first block argument becomes an unallocated ``q1.reg``."""
        seq = _make_shots_loop(0, 5, 1)
        entry = _lower(seq)

        (for_op,) = _ops_of_type(entry, ForOp)
        induction = list(for_op.body.blocks)[0].args[0]
        assert isinstance(induction.type, IntRegisterType)

    def test_body_terminator_is_q1_scf_yield(self):
        """``scf.yield`` inside the body is replaced by ``q1_scf.yield``."""
        seq = _make_shots_loop(0, 5, 1)
        entry = _lower(seq)

        (for_op,) = _ops_of_type(entry, ForOp)
        terminator = list(for_op.body.blocks)[0].last_op
        assert isinstance(terminator, YieldOp)

    def test_no_scf_yield_survives(self):
        """No ``scf.yield`` survives in the entire sequence body."""
        seq = _make_shots_loop(0, 5, 1)
        entry = _lower(seq)

        (for_op,) = _ops_of_type(entry, ForOp)
        for op in for_op.body.walk():
            assert not isinstance(op, ScfYieldOp)

    def test_iter_count_is_move_op_result(self):
        """The ``q1_scf.for`` iter_count is the result of the ``q1.ir.move``."""
        seq = _make_shots_loop(0, 7, 1)
        entry = _lower(seq)

        move_op = _ops_of_type(entry, MoveImmRdOp)[0]
        for_op = _ops_of_type(entry, ForOp)[0]
        assert for_op.iter_count is move_op.rd

    def test_iter_args_pass_through(self):
        """A ``q1.reg``-typed iter_arg is preserved as a loop-carried value."""
        lb_op = ArithConstantOp.from_int_and_width(0, IndexType())
        ub_op = ArithConstantOp.from_int_and_width(4, IndexType())
        step_op = ArithConstantOp.from_int_and_width(1, IndexType())

        entry = Block(arg_types=[R])
        (carried,) = entry.args

        body_block = Block(arg_types=[IndexType(), R])
        body_block.add_op(ScfYieldOp(body_block.args[1]))

        for_op = ScfForOp(
            lb=lb_op.result,
            ub=ub_op.result,
            step=step_op.result,
            iter_args=[carried],
            body=body_block,
        )
        entry.add_ops([lb_op, ub_op, step_op, for_op, StopOp()])
        seq = SequenceOp("Q0", Region([entry]))

        lower_entry = _lower(seq)

        q1_for_ops = _ops_of_type(lower_entry, ForOp)
        assert len(q1_for_ops) == 1
        (q1_for,) = q1_for_ops
        assert len(list(q1_for.iter_args)) == 1
        assert q1_for.iter_args[0] is carried

    def test_rejects_non_constant_lower_bound(self):
        """A non-constant lower bound raises :class:`PassFailedException`."""
        entry = Block(arg_types=[IndexType()])
        (lb_val,) = entry.args  # dynamic
        ub_op = ArithConstantOp.from_int_and_width(10, IndexType())
        step_op = ArithConstantOp.from_int_and_width(1, IndexType())
        body_block = Block(arg_types=[IndexType()])
        body_block.add_op(ScfYieldOp())
        for_op = ScfForOp(
            lb=lb_val, ub=ub_op.result, step=step_op.result, iter_args=[], body=body_block
        )
        entry.add_ops([ub_op, step_op, for_op, StopOp()])
        module = ModuleOp([SequenceOp("Q0", Region([entry]))])

        with pytest.raises(PassFailedException, match="lower bound"):
            LowerScfToQ1ScfPass().apply(None, module)

    def test_rejects_non_constant_upper_bound(self):
        """A non-constant upper bound raises :class:`PassFailedException`."""
        entry = Block(arg_types=[IndexType()])
        (ub_val,) = entry.args  # dynamic
        lb_op = ArithConstantOp.from_int_and_width(0, IndexType())
        step_op = ArithConstantOp.from_int_and_width(1, IndexType())
        body_block = Block(arg_types=[IndexType()])
        body_block.add_op(ScfYieldOp())
        for_op = ScfForOp(
            lb=lb_op.result, ub=ub_val, step=step_op.result, iter_args=[], body=body_block
        )
        entry.add_ops([lb_op, step_op, for_op, StopOp()])
        module = ModuleOp([SequenceOp("Q0", Region([entry]))])

        with pytest.raises(PassFailedException, match="upper bound"):
            LowerScfToQ1ScfPass().apply(None, module)

    def test_rejects_non_constant_step(self):
        """A non-constant step raises :class:`PassFailedException`."""
        entry = Block(arg_types=[IndexType()])
        (step_val,) = entry.args  # dynamic
        lb_op = ArithConstantOp.from_int_and_width(0, IndexType())
        ub_op = ArithConstantOp.from_int_and_width(10, IndexType())
        body_block = Block(arg_types=[IndexType()])
        body_block.add_op(ScfYieldOp())
        for_op = ScfForOp(
            lb=lb_op.result, ub=ub_op.result, step=step_val, iter_args=[], body=body_block
        )
        entry.add_ops([lb_op, ub_op, for_op, StopOp()])
        module = ModuleOp([SequenceOp("Q0", Region([entry]))])

        with pytest.raises(PassFailedException, match="step"):
            LowerScfToQ1ScfPass().apply(None, module)

    def test_rejects_zero_trip_count_equal_bounds(self):
        """A loop with ``lb == ub`` has zero iterations and is rejected."""
        seq = _make_shots_loop(5, 5, 1)
        module = ModuleOp([seq])

        with pytest.raises(PassFailedException, match="iteration count"):
            LowerScfToQ1ScfPass().apply(None, module)

    def test_rejects_negative_trip_count(self):
        """A loop where ``lb > ub`` with positive step is rejected."""
        seq = _make_shots_loop(10, 5, 1)
        module = ModuleOp([seq])

        with pytest.raises(PassFailedException, match="iteration count"):
            LowerScfToQ1ScfPass().apply(None, module)

    def test_body_uses_induction_var_remapped_to_ascending_index(self):
        """Body uses of the induction var are remapped to lb + (count - counter) * step."""
        lb_op = ArithConstantOp.from_int_and_width(0, IndexType())
        ub_op = ArithConstantOp.from_int_and_width(5, IndexType())
        step_op = ArithConstantOp.from_int_and_width(1, IndexType())

        body_block = Block(arg_types=[IndexType()])
        body_move = MoveRsRdOp(body_block.args[0], IntRegisterType.unallocated())
        body_block.add_ops([body_move, ScfYieldOp()])

        for_op = ScfForOp(
            lb=lb_op.result,
            ub=ub_op.result,
            step=step_op.result,
            iter_args=[],
            body=body_block,
        )
        entry = Block()
        entry.add_ops([lb_op, ub_op, step_op, for_op, StopOp()])
        lower_entry = _lower(SequenceOp("Q0", Region([entry])))

        (for_op,) = _ops_of_type(lower_entry, ForOp)
        body_block = list(for_op.body.blocks)[0]
        countdown = body_block.args[0]
        assert isinstance(countdown.type, IntRegisterType)

        # lb=0, step=1, count=5: ascending = ~countdown + (0 + 5 + 1)
        not_op, add_op, move_op, _yield = list(body_block.ops)
        assert isinstance(not_op, NotRsRdOp)
        assert not_op.rs is countdown
        assert isinstance(add_op, AddRsImmRdOp)
        assert add_op.rs is not_op.rd
        assert isinstance(move_op, MoveRsRdOp)
        assert move_op.rs is add_op.rd

    def test_rejects_incompatible_iter_arg_type(self):
        """An iter_arg that is not ``q1.reg`` is rejected."""
        lb_op = ArithConstantOp.from_int_and_width(0, IndexType())
        ub_op = ArithConstantOp.from_int_and_width(5, IndexType())
        step_op = ArithConstantOp.from_int_and_width(1, IndexType())

        entry = Block(arg_types=[IndexType()])
        (idx_val,) = entry.args  # IndexType, not q1.reg

        body_block = Block(arg_types=[IndexType(), IndexType()])
        body_block.add_op(ScfYieldOp(body_block.args[1]))

        for_op = ScfForOp(
            lb=lb_op.result,
            ub=ub_op.result,
            step=step_op.result,
            iter_args=[idx_val],
            body=body_block,
        )
        entry.add_ops([lb_op, ub_op, step_op, for_op, StopOp()])
        module = ModuleOp([SequenceOp("Q0", Region([entry]))])

        with pytest.raises(PassFailedException, match="iter_arg"):
            LowerScfToQ1ScfPass().apply(None, module)

    @pytest.mark.parametrize(
        "lb,ub,step,expected_count",
        [
            (0, 1000, 1, 1000),
            (0, 100, 1, 100),
            (5, 15, 2, 5),
            (0, 9, 3, 3),
            (0, 5, 2, 3),  # ceil(5/2)=3, floor gives 2
            (0, 1, 2, 1),  # 0 < (ub-lb) < step: runs once
            (0, 3, 5, 1),  # 0 < (ub-lb) < step with larger step
        ],
    )
    def test_count_parametrized(self, lb: int, ub: int, step: int, expected_count: int):
        """``MoveImmRdOp`` carries ``ceil((ub - lb) / step)`` for various configurations."""
        seq = _make_shots_loop(lb, ub, step)
        entry = _lower(seq)

        move_ops = _ops_of_type(entry, MoveImmRdOp)
        assert move_ops[0].imm.data == expected_count

    def test_step_one_emits_not_add_without_multiply(self):
        """Step=1 emits only NOT + ADD (lb+count+1 as immediate); no multiply op."""
        lb_op = ArithConstantOp.from_int_and_width(3, IndexType())
        ub_op = ArithConstantOp.from_int_and_width(8, IndexType())
        step_op = ArithConstantOp.from_int_and_width(1, IndexType())
        body_block = Block(arg_types=[IndexType()])
        body_block.add_ops(
            [MoveRsRdOp(body_block.args[0], IntRegisterType.unallocated()), ScfYieldOp()]
        )
        for_op = ScfForOp(
            lb=lb_op.result,
            ub=ub_op.result,
            step=step_op.result,
            iter_args=[],
            body=body_block,
        )
        entry = Block()
        entry.add_ops([lb_op, ub_op, step_op, for_op, StopOp()])
        lower_entry = _lower(SequenceOp("Q0", Region([entry])))

        (for_op,) = _ops_of_type(lower_entry, ForOp)
        not_op, add_op, _move, _yield = list(list(for_op.body.blocks)[0].ops)
        assert isinstance(not_op, NotRsRdOp)
        assert isinstance(add_op, AddRsImmRdOp)
        assert add_op.imm.data == 3 + 5 + 1  # lb + count + 1

    def test_single_iteration_correct_add_immediate(self):
        """A one-trip loop (lb=0, ub=1, step=1) uses ADD immediate lb+count+1 = 2."""
        lb_op = ArithConstantOp.from_int_and_width(0, IndexType())
        ub_op = ArithConstantOp.from_int_and_width(1, IndexType())
        step_op = ArithConstantOp.from_int_and_width(1, IndexType())
        body_block = Block(arg_types=[IndexType()])
        body_block.add_ops(
            [MoveRsRdOp(body_block.args[0], IntRegisterType.unallocated()), ScfYieldOp()]
        )
        for_op = ScfForOp(
            lb=lb_op.result,
            ub=ub_op.result,
            step=step_op.result,
            iter_args=[],
            body=body_block,
        )
        entry = Block()
        entry.add_ops([lb_op, ub_op, step_op, for_op, StopOp()])
        lower_entry = _lower(SequenceOp("Q0", Region([entry])))

        (for_op,) = _ops_of_type(lower_entry, ForOp)
        not_op, add_op, _move, _yield = list(list(for_op.body.blocks)[0].ops)
        assert isinstance(not_op, NotRsRdOp)
        assert isinstance(add_op, AddRsImmRdOp)
        assert add_op.imm.data == 2  # lb + count + 1 = 0 + 1 + 1

    def test_while_op_raises_not_supported(self):
        """``scf.while`` anywhere in the module raises immediately."""
        # Build a minimal scf.while; entry block arg provides the SSAValue argument.
        from xdsl.dialects.scf import ConditionOp

        entry = Block(arg_types=[R])

        before_block = Block(arg_types=[R])
        before_block.add_op(ConditionOp(before_block.args[0]))

        after_block = Block(arg_types=[R])
        after_block.add_op(ScfYieldOp(after_block.args[0]))

        while_op = ScfWhileOp(
            arguments=[entry.args[0]],
            result_types=[R],
            before_region=Region([before_block]),
            after_region=Region([after_block]),
        )
        entry.add_ops([while_op, StopOp()])
        module = ModuleOp([SequenceOp("Q0", Region([entry]))])

        with pytest.raises(PassFailedException, match="scf.while"):
            LowerScfToQ1ScfPass().apply(None, module)


class TestIfLowering:
    def test_if_op_raises_not_supported(self):
        """``scf.if`` anywhere in the module raises immediately."""
        from xdsl.dialects.builtin import i1

        cond_op = ArithConstantOp(IntegerAttr(1, i1))
        then_block = Block()
        then_block.add_op(ScfYieldOp())
        if_op = ScfIfOp(
            cond=cond_op.result,
            return_types=[],
            true_region=Region([then_block]),
        )
        entry = Block()
        entry.add_ops([cond_op, if_op, StopOp()])
        module = ModuleOp([SequenceOp("Q0", Region([entry]))])

        with pytest.raises(PassFailedException, match="scf.if"):
            LowerScfToQ1ScfPass().apply(None, module)
