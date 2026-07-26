# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for the q1_scf to q1_cf lowering pass.

The pass expands the structured containers (:class:`IfOp`, :class:`WhileOp`,
:class:`ForOp`) inside a ``SequenceOp`` into a ``q1_cf`` block CFG. Predicates
carry through unchanged, no register is introduced, and live values thread as
successor block arguments.

Coverage:
* Result shape: every container dissolves, the body verifies, and no q1_scf op
  or region remains.
* Terminator selection: a flag predicate lowers to ``flag_branch`` and a
  comparison predicate to ``comparison_branch``, each carrying the source
  predicate unchanged across signed and unsigned kinds.
* Fall-through: the ``else`` successor of every conditional branch is the branch
  block's layout neighbour, the invariant the linearisation pass relies on.
* Result threading: container results become continuation block arguments fed by
  each region's yield.
* The counted loop maps onto a self-looping ``loop_branch`` whose counter,
  induction argument, and back-edge argument coincide.
* Register neutrality: the pass adds only ``q1_cf`` branches and invents no
  register.
* Nesting dissolves through the greedy worklist, and every sequence in a module
  is rewritten.
* End to end: a structured program lowers to q1_cf and then, through the
  linearisation pass, to flat q1 with equivalent control flow.
"""

from __future__ import annotations

from io import StringIO

import pytest
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Operation, Region
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.q1 import MoveImmRdOp, NotRsRdOp, Registers, StopOp
from qat.experimental.dialect.q1.ir.imm_desc import SU32Imm
from qat.experimental.dialect.q1.ir.reg_desc import IntRegisterType
from qat.experimental.dialect.q1.target import emit_program
from qat.experimental.dialect.q1_cf import (
    BinaryPredicateBranchOp,
    JmpBranchOp,
    LoopBranchOp,
    UnaryPredicateBranchOp,
)
from qat.experimental.dialect.q1_cf.ir.attrs import BinaryPredicateAttr, UnaryPredicateAttr
from qat.experimental.dialect.q1_cf.transforms.linearise_q1_cf import LineariseQ1CfToQ1Pass
from qat.experimental.dialect.q1_scf import (
    BinaryPredicate,
    ConditionOp,
    ForOp,
    IfOp,
    UnaryPredicate,
    WhileOp,
    YieldOp,
)
from qat.experimental.dialect.q1_scf.transforms.lower_to_cf import (
    ForLowering,
    IfLowering,
    LowerQ1ScfToQ1CfPass,
    WhileLowering,
    _split_continuation,
)
from qat.experimental.dialect.q1_sequence import SequenceOp

R = Registers.UNALLOCATED_INT
R0, R1, R2 = Registers.R0, Registers.R1, Registers.R2


def _lower(seq: SequenceOp) -> Block:
    """Run the pass on a lone sequence and return its single body block.

    The result is verified to hold the pass's contract: the body verifies and no q1_scf
    operation survives.
    """
    module = ModuleOp([seq])
    module.verify()
    LowerQ1ScfToQ1CfPass().apply(None, module)
    module.verify()
    for op in seq.body.walk():
        assert op.dialect_name() != "q1_scf"
    assert len(seq.body.blocks) >= 1
    return seq.body.blocks[0]


def _branch(block: Block) -> Operation:
    """Return the terminator of a block."""
    terminator = block.last_op
    assert terminator is not None
    return terminator


def _register_types(seq: SequenceOp) -> set[IntRegisterType]:
    """Collect every register type appearing on a value in the sequence."""
    types: set[IntRegisterType] = set()
    for op in seq.body.walk():
        for value in (*op.results, *op.operands):
            if isinstance(value.type, IntRegisterType):
                types.add(value.type)
    return types


class TestIfLowering:
    def test_no_else_no_results_lowers_to_flag_branch(self):
        then_block = Block([YieldOp()])
        seq = self._flag_if(then_block, results=[], else_region=None)
        head = _lower(seq)

        branch = _branch(head)
        assert isinstance(branch, UnaryPredicateBranchOp)
        # The synthesised empty else stub is the branch's layout neighbour.
        assert head.next_block is branch.else_block

    def test_else_and_results_thread_through_continuation(self):
        seq = self._result_if()
        head = _lower(seq)

        branch = _branch(head)
        assert isinstance(branch, BinaryPredicateBranchOp)
        assert head.next_block is branch.else_block
        # Both arms jump to a continuation block that carries the single result.
        then_exit = _branch(branch.then_block)
        else_exit = _branch(branch.else_block)
        assert isinstance(then_exit, JmpBranchOp)
        assert isinstance(else_exit, JmpBranchOp)
        assert then_exit.successor is else_exit.successor
        assert len(then_exit.successor.args) == 1

    @staticmethod
    def _flag_if(then_block, results, else_region):
        entry = Block(arg_types=[R])
        op = IfOp(
            UnaryPredicate.nez, [entry.args[0]], results, Region([then_block]), else_region
        )
        entry.add_ops([op, StopOp()])
        return SequenceOp("Q0", Region([entry]))

    @staticmethod
    def _result_if():
        entry = Block(arg_types=[R])
        (flag,) = entry.args
        then_block = Block()
        then_block.add_op(YieldOp(flag))
        else_block = Block()
        else_block.add_op(YieldOp(flag))
        op = IfOp(
            BinaryPredicate.slt,
            [flag, flag],
            [R],
            Region([then_block]),
            Region([else_block]),
        )
        entry.add_ops([op, StopOp()])
        return SequenceOp("Q0", Region([entry]))


class TestWhileLowering:
    def test_lowers_to_header_body_and_back_edge(self):
        seq = _while_sequence()
        head = _lower(seq)

        enter = _branch(head)
        assert isinstance(enter, JmpBranchOp)
        header = enter.successor

        condition = _branch(header)
        assert isinstance(condition, UnaryPredicateBranchOp)
        # The exit stub is the header's neighbour and carries the else edge.
        assert header.next_block is condition.else_block
        exit_branch = _branch(condition.else_block)
        assert isinstance(exit_branch, JmpBranchOp)

        # The body ends in a back-edge to the header.
        back_edge = _branch(condition.then_block)
        assert isinstance(back_edge, JmpBranchOp)
        assert back_edge.successor is header


class TestForLowering:
    def test_lowers_to_self_looping_loop_branch(self):
        seq = _for_sequence()
        head = _lower(seq)

        enter = _branch(head)
        assert isinstance(enter, JmpBranchOp)
        body = enter.successor

        loop = _branch(body)
        assert isinstance(loop, LoopBranchOp)
        # Self-loop: the body block is its own loop successor.
        assert loop.body_block is body
        # Counter, induction argument, and back-edge argument coincide.
        assert loop.counter is body.args[0]
        assert loop.body_arguments[0] is body.args[0]

    def test_rejects_statically_zero_trip_count(self):
        entry = Block()
        zero_count = MoveImmRdOp(SU32Imm(0), R2)
        body = Block(arg_types=[R2])
        body.add_op(YieldOp())
        for_op = ForOp(zero_count.rd, [], Region([body]))
        entry.add_ops([zero_count, for_op, StopOp()])

        with pytest.raises(PassFailedException, match="statically zero iter_count"):
            _lower(SequenceOp("Q0", Region([entry])))


class TestMalformedLoweringInputs:
    def test_split_continuation_rejects_detached_operation(self, mocker):
        seed = MoveImmRdOp(SU32Imm(1), R2)
        op = ForOp(seed.rd, [], Region([Block(arg_types=[R2])]))
        with pytest.raises(PassFailedException, match="detached from its parent block"):
            _split_continuation(op, mocker.MagicMock())

    def test_if_lowering_rejects_empty_then_region(self, mocker):
        entry = Block(arg_types=[R])
        (flag,) = entry.args
        op = IfOp(UnaryPredicate.nez, [flag], [], Region([]))
        entry.add_op(op)
        Region([entry])
        with pytest.raises(PassFailedException, match="then region must contain"):
            IfLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_if_lowering_rejects_non_terminating_then_region(self, mocker):
        entry = Block(arg_types=[R])
        (flag,) = entry.args
        op = IfOp(UnaryPredicate.nez, [flag], [], Region([Block()]))
        entry.add_op(op)
        Region([entry])
        with pytest.raises(PassFailedException, match="then region must terminate"):
            IfLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_if_lowering_rejects_wrong_then_terminator_type(self, mocker):
        entry = Block(arg_types=[R])
        (flag,) = entry.args
        op = IfOp(UnaryPredicate.nez, [flag], [], Region([Block([StopOp()])]))
        entry.add_op(op)
        Region([entry])
        with pytest.raises(PassFailedException, match="then region terminator must be"):
            IfLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_if_lowering_rejects_non_terminating_else_region(self, mocker):
        entry = Block(arg_types=[R])
        (flag,) = entry.args
        op = IfOp(
            UnaryPredicate.nez,
            [flag],
            [],
            Region([Block([YieldOp()])]),
            Region([Block()]),
        )
        entry.add_op(op)
        Region([entry])
        with pytest.raises(PassFailedException, match="else region must terminate"):
            IfLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_if_lowering_rejects_wrong_else_terminator_type(self, mocker):
        entry = Block(arg_types=[R])
        (flag,) = entry.args
        op = IfOp(
            UnaryPredicate.nez,
            [flag],
            [],
            Region([Block([YieldOp()])]),
            Region([Block([StopOp()])]),
        )
        entry.add_op(op)
        Region([entry])
        with pytest.raises(PassFailedException, match="else region terminator must be"):
            IfLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_while_lowering_rejects_empty_before_region(self, mocker):
        op = WhileOp([], [], Region([]), Region([Block([YieldOp()])]))
        head = Block([op])
        Region([head])
        with pytest.raises(PassFailedException, match="before region must contain"):
            WhileLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_while_lowering_rejects_missing_condition(self, mocker):
        before = Block()
        op = WhileOp([], [], Region([before]), Region([Block([YieldOp()])]))
        head = Block([op])
        Region([head])
        with pytest.raises(PassFailedException, match="before region must terminate"):
            WhileLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_while_lowering_rejects_wrong_condition_terminator_type(self, mocker):
        before = Block([StopOp()])
        op = WhileOp([], [], Region([before]), Region([Block([YieldOp()])]))
        head = Block([op])
        Region([head])
        with pytest.raises(PassFailedException, match="before region terminator must be"):
            WhileLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_while_lowering_rejects_empty_after_region(self, mocker):
        before = Block(arg_types=[R])
        (flag,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [flag], []))
        op = WhileOp([], [], Region([before]), Region([]))
        head = Block([op])
        Region([head])
        with pytest.raises(PassFailedException, match="after region must contain"):
            WhileLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_while_lowering_rejects_non_terminating_after_region(self, mocker):
        before = Block(arg_types=[R])
        (flag,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [flag], []))
        op = WhileOp([], [], Region([before]), Region([Block()]))
        head = Block([op])
        Region([head])
        with pytest.raises(PassFailedException, match="after region must terminate"):
            WhileLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_while_lowering_rejects_wrong_after_terminator_type(self, mocker):
        before = Block(arg_types=[R])
        (flag,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [flag], []))
        op = WhileOp([], [], Region([before]), Region([Block([StopOp()])]))
        head = Block([op])
        Region([head])
        with pytest.raises(PassFailedException, match="after region terminator must be"):
            WhileLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_for_lowering_rejects_empty_body(self, mocker):
        seed = MoveImmRdOp(SU32Imm(1), R2)
        op = ForOp(seed.rd, [], Region([]))
        head = Block([seed, op])
        Region([head])
        with pytest.raises(PassFailedException, match="body region must contain"):
            ForLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_for_lowering_rejects_non_terminating_body(self, mocker):
        seed = MoveImmRdOp(SU32Imm(1), R2)
        op = ForOp(seed.rd, [], Region([Block(arg_types=[R2])]))
        head = Block([seed, op])
        Region([head])
        with pytest.raises(PassFailedException, match="body region must terminate"):
            ForLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_for_lowering_rejects_wrong_body_terminator_type(self, mocker):
        seed = MoveImmRdOp(SU32Imm(1), R2)
        op = ForOp(seed.rd, [], Region([Block(arg_types=[R2], ops=[StopOp()])]))
        head = Block([seed, op])
        Region([head])
        with pytest.raises(PassFailedException, match="body region terminator must be"):
            ForLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_split_continuation_rejects_missing_parent_region(self, mocker):
        entry = Block(arg_types=[R])
        (flag,) = entry.args
        op = IfOp(UnaryPredicate.nez, [flag], [R], Region([Block([YieldOp(flag)])]))
        entry.add_op(op)
        Region([entry])
        mocker.patch.object(Block, "parent_region", return_value=None)
        with pytest.raises(PassFailedException, match="detached from its region"):
            _split_continuation(op, mocker.MagicMock())

    def test_if_lowering_rejects_missing_parent_region_for_else_stub(self, mocker):
        entry = Block(arg_types=[R])
        (flag,) = entry.args
        op = IfOp(UnaryPredicate.nez, [flag], [], Region([Block([YieldOp()])]))
        entry.add_op(op)
        Region([entry])
        mocker.patch.object(Block, "parent_region", return_value=None)
        with pytest.raises(PassFailedException, match="detached from its region"):
            IfLowering().match_and_rewrite(op, mocker.MagicMock())

    def test_while_lowering_rejects_missing_parent_region_for_exit_stub(self, mocker):
        before = Block()
        before.add_op(ConditionOp(UnaryPredicate.nez, [before.insert_arg(R, 0)], []))
        after = Block()
        after.add_op(YieldOp())
        op = WhileOp([], [], Region([before]), Region([after]))
        head = Block([op])
        Region([head])
        mocker.patch.object(Block, "parent_region", return_value=None)
        with pytest.raises(PassFailedException, match="detached from its region"):
            WhileLowering().match_and_rewrite(op, mocker.MagicMock())


class TestPredicateSelection:
    @pytest.mark.parametrize(
        "predicate",
        [BinaryPredicate.slt, BinaryPredicate.sge, BinaryPredicate.ult],
    )
    def test_comparison_predicate_selects_comparison_branch(self, predicate):
        entry = Block(arg_types=[R])
        (flag,) = entry.args
        op = IfOp(predicate, [flag, flag], [], Region([Block([YieldOp()])]))
        entry.add_ops([op, StopOp()])
        head = _lower(SequenceOp("Q0", Region([entry])))

        branch = _branch(head)
        assert isinstance(branch, BinaryPredicateBranchOp)
        assert branch.predicate == BinaryPredicateAttr(predicate)

    @pytest.mark.parametrize("predicate", [UnaryPredicate.nez, UnaryPredicate.eqz])
    def test_flag_predicate_selects_flag_branch(self, predicate):
        entry = Block(arg_types=[R])
        (flag,) = entry.args
        op = IfOp(predicate, [flag], [], Region([Block([YieldOp()])]))
        entry.add_ops([op, StopOp()])
        head = _lower(SequenceOp("Q0", Region([entry])))

        branch = _branch(head)
        assert isinstance(branch, UnaryPredicateBranchOp)
        assert branch.predicate == UnaryPredicateAttr(predicate)


class TestRegisterNeutrality:
    def test_pass_adds_only_branches_and_no_register(self):
        entry = Block(arg_types=[R0])
        (flag,) = entry.args
        then_block = Block()
        then_block.add_op(YieldOp(flag))
        else_block = Block()
        else_block.add_op(YieldOp(flag))
        op = IfOp(
            BinaryPredicate.slt,
            [flag, flag],
            [R0],
            Region([then_block]),
            Region([else_block]),
        )
        entry.add_ops([op, StopOp()])
        seq = SequenceOp("Q0", Region([entry]))

        before = _register_types(seq)
        _lower(seq)
        after = _register_types(seq)

        assert after == before
        for op in seq.body.walk():
            assert op.dialect_name() in {"q1_cf", "q1"}
            if op.dialect_name() == "q1":
                assert isinstance(op, StopOp)


class TestPassDriver:
    def test_dissolves_nesting(self):
        # An if nested in a while body must fully dissolve through the worklist.
        inner_then = Block([YieldOp()])
        body = Block(arg_types=[R])
        (acc,) = body.args
        inner = IfOp(UnaryPredicate.nez, [acc], [], Region([inner_then]))
        body.add_ops([inner, YieldOp(acc)])

        before = Block(arg_types=[R])
        (init,) = before.args
        before.add_op(ConditionOp(UnaryPredicate.nez, [init], [init]))
        entry = Block(arg_types=[R])
        (seed,) = entry.args
        loop = WhileOp([seed], [R], Region([before]), Region([body]))
        entry.add_ops([loop, StopOp()])
        seq = SequenceOp("Q0", Region([entry]))

        _lower(seq)  # asserts no q1_scf residue

    def test_dissolves_second_nested_shape(self):
        # A for nested in a while nested in an if also dissolves fully.
        for_body = Block(arg_types=[R])
        for_body.add_op(YieldOp())

        while_body = Block(arg_types=[R])
        (acc,) = while_body.args
        while_body.add_ops([ForOp(acc, [], Region([for_body])), YieldOp(acc)])

        entry = Block(arg_types=[R])
        (seed,) = entry.args
        while_before = Block(arg_types=[R])
        (init,) = while_before.args
        while_before.add_op(ConditionOp(UnaryPredicate.nez, [init], [init]))
        nested_while = WhileOp([seed], [R], Region([while_before]), Region([while_body]))

        then_block = Block()
        then_block.add_ops([nested_while, YieldOp()])

        entry.add_ops(
            [IfOp(UnaryPredicate.nez, [seed], [], Region([then_block])), StopOp()]
        )
        seq = SequenceOp("Q0", Region([entry]))

        _lower(seq)  # asserts no q1_scf residue

    def test_rewrites_every_sequence_in_module(self):
        seqs = [
            self._trivial_if("Q0"),
            self._trivial_if("Q1"),
        ]
        module = ModuleOp(seqs)
        module.verify()
        LowerQ1ScfToQ1CfPass().apply(None, module)
        module.verify()
        for op in module.walk():
            assert op.dialect_name() != "q1_scf"

    @staticmethod
    def _trivial_if(channel):
        entry = Block(arg_types=[R])
        (flag,) = entry.args
        op = IfOp(UnaryPredicate.nez, [flag], [], Region([Block([YieldOp()])]))
        entry.add_ops([op, StopOp()])
        return SequenceOp(channel, Region([entry]))


class TestEndToEnd:
    def test_for_loop_lowers_to_flat_counted_loop(self):
        entry = Block()
        init = MoveImmRdOp(SU32Imm(10), R2)
        body = Block(arg_types=[R2])
        body.add_op(YieldOp())
        for_op = ForOp(init.rd, [], Region([body]))
        entry.add_ops([init, for_op, StopOp()])
        seq = SequenceOp("Q0", Region([entry]))

        _run_full_lowering(seq)

        assert _assembly(seq) == [
            "move 10, R2",
            "bb1:",
            "loop R2, @bb1",
            "stop",
        ]

    def test_if_lowers_to_flat_conditional_jump(self):
        entry = Block()
        lhs_seed = MoveImmRdOp(SU32Imm(1), R0)
        lhs = NotRsRdOp(lhs_seed.rd, R0)
        rhs_seed = MoveImmRdOp(SU32Imm(2), R1)
        rhs = NotRsRdOp(rhs_seed.rd, R1)
        then_block = Block([YieldOp()])
        if_op = IfOp(BinaryPredicate.slt, [lhs.rd, rhs.rd], [], Region([then_block]))
        entry.add_ops([lhs_seed, lhs, rhs_seed, rhs, if_op, StopOp()])
        seq = SequenceOp("Q0", Region([entry]))

        _run_full_lowering(seq)

        assert _assembly(seq) == [
            "move 1, R0",
            "not R0, R0",
            "move 2, R1",
            "not R1, R1",
            "cmp R0, R1",
            "jl @bb2",
            "jmp @bb3",
            "bb2:",
            "bb3:",
            "stop",
        ]


def _while_sequence() -> SequenceOp:
    entry = Block(arg_types=[R])
    (init,) = entry.args
    before = Block(arg_types=[R])
    (acc,) = before.args
    before.add_op(ConditionOp(UnaryPredicate.nez, [acc], [acc]))
    after = Block(arg_types=[R])
    (next_acc,) = after.args
    after.add_op(YieldOp(next_acc))
    op = WhileOp([init], [R], Region([before]), Region([after]))
    entry.add_ops([op, StopOp()])
    return SequenceOp("Q0", Region([entry]))


def _for_sequence() -> SequenceOp:
    entry = Block(arg_types=[R, R])
    count, seed = entry.args
    body = Block(arg_types=[R, R])
    (_, carried) = body.args
    body.add_op(YieldOp(carried))
    op = ForOp(count, [seed], Region([body]))
    entry.add_ops([op, StopOp()])
    return SequenceOp("Q0", Region([entry]))


def _run_full_lowering(seq: SequenceOp) -> None:
    module = ModuleOp([seq])
    module.verify()
    LowerQ1ScfToQ1CfPass().apply(None, module)
    module.verify()
    LineariseQ1CfToQ1Pass().apply(None, module)
    module.verify()


def _assembly(seq: SequenceOp) -> list[str]:
    output = StringIO()
    emit_program(seq.body, output)
    return [line.strip() for line in output.getvalue().splitlines() if line.strip()]
