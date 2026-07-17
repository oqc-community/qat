# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for the q1_scf operations.

Covers the three structured containers (:class:`IfOp`, :class:`WhileOp`,
:class:`ForOp`) and their two terminators (:class:`ConditionOp`,
:class:`YieldOp`): trait invariants, verifier constraints, region separation
from q1_cf, and parse-verify-print round-trips.
"""

from __future__ import annotations

import re
from io import StringIO

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp, i32
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import HasAncestor, IsTerminator
from xdsl.utils.exceptions import ParseError, VerifyException

from qat.experimental.dialect.q1 import Q1, Registers, StopOp
from qat.experimental.dialect.q1_cf import JmpBranchOp, Q1_cf
from qat.experimental.dialect.q1_scf import (
    ComparisonPredicate,
    ConditionOp,
    FlagPredicate,
    ForOp,
    IfOp,
    IterDomainAttr,
    IterParameter,
    Q1_scf,
    WhileOp,
    YieldOp,
)
from qat.experimental.dialect.q1_scf.ir.ops import _yield_terminator
from qat.experimental.dialect.q1_sequence import Q1_sequence, SequenceOp

_REG = Registers.UNALLOCATED_INT

_CONTAINER_OPS = [IfOp, WhileOp, ForOp]
_CONTAINER_IDS = [op.name for op in _CONTAINER_OPS]


def _build_print_reparse(seq_op: SequenceOp) -> str:
    """Print a SequenceOp inside a module, reparse and re-verify it.

    :returns: The final printed IR string, which must be identical across the
        print/parse/print cycle.
    """
    module = ModuleOp([seq_op])
    module.verify()
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    ir_text = stream.getvalue()

    ctx = Context()
    for dialect in (Builtin, Q1, Q1_cf, Q1_sequence, Q1_scf):
        ctx.load_dialect(dialect)

    reparsed = Parser(ctx, ir_text).parse_op()
    reparsed.verify()

    stream2 = StringIO()
    Printer(stream=stream2).print_op(reparsed)
    printed = stream2.getvalue()
    assert printed == ir_text
    return printed


def _flag_if(predicate: FlagPredicate = FlagPredicate.nez) -> tuple[Block, IfOp]:
    """Build a result-less if with a single flag operand and an empty then."""
    entry = Block(arg_types=[_REG])
    (flag,) = entry.args
    then_region = Region([Block([YieldOp()])])
    op = IfOp(predicate, [flag], [], then_region)
    entry.add_ops([op, StopOp()])
    return entry, op


class TestOpMetadata:
    @pytest.mark.parametrize(
        "op_type,mnemonic",
        [
            (IfOp, "q1_scf.if"),
            (WhileOp, "q1_scf.while"),
            (ForOp, "q1_scf.for"),
            (ConditionOp, "q1_scf.condition"),
            (YieldOp, "q1_scf.yield"),
        ],
        ids=lambda value: value if isinstance(value, str) else value.__name__,
    )
    def test_name(self, op_type, mnemonic):
        assert op_type.name == mnemonic

    def test_yield_is_terminator(self):
        assert YieldOp.has_trait(IsTerminator)

    def test_condition_is_terminator(self):
        assert ConditionOp.has_trait(IsTerminator)

    @pytest.mark.parametrize("op_type", _CONTAINER_OPS, ids=_CONTAINER_IDS)
    def test_container_has_sequence_ancestor(self, op_type):
        assert op_type.has_trait(HasAncestor(SequenceOp))


class TestYieldOp:
    def test_round_trip_empty(self):
        entry, _ = _flag_if()
        assert "q1_scf.yield" in _build_print_reparse(SequenceOp("ch0", Region([entry])))

    def test_round_trip_with_operands(self):
        entry = Block(arg_types=[_REG])
        (flag,) = entry.args
        then_region = Region([Block([YieldOp(flag)])])
        else_region = Region([Block([YieldOp(flag)])])
        op = IfOp(FlagPredicate.nez, [flag], [_REG], then_region, else_region)
        entry.add_ops([op, StopOp()])
        assert "q1_scf.yield" in _build_print_reparse(SequenceOp("ch0", Region([entry])))

    def test_rejected_outside_container(self):
        module = ModuleOp(Region([Block([YieldOp()])]))
        with pytest.raises(VerifyException, match="expects parent op to be one of"):
            module.verify()


class TestConditionArity:
    def _while_with_condition(self, predicate, predicate_args_count):
        entry = Block(arg_types=[_REG])
        (init,) = entry.args
        before = Block(arg_types=[_REG])
        (acc,) = before.args
        args = [acc] * predicate_args_count
        before.add_op(ConditionOp(predicate, args, [acc]))
        after = Block(arg_types=[_REG])
        (acc2,) = after.args
        after.add_op(YieldOp(acc2))
        op = WhileOp([init], [_REG], Region([before]), Region([after]))
        entry.add_ops([op, StopOp()])
        return SequenceOp("ch0", Region([entry]))

    def test_flag_condition_round_trips(self):
        seq = self._while_with_condition(FlagPredicate.nez, 1)
        assert "q1_scf.condition" in _build_print_reparse(seq)

    def test_comparison_condition_round_trips(self):
        seq = self._while_with_condition(ComparisonPredicate.slt, 2)
        assert "q1_scf.condition" in _build_print_reparse(seq)

    def test_flag_condition_rejects_two_operands(self):
        seq = self._while_with_condition(FlagPredicate.nez, 2)
        with pytest.raises(VerifyException, match="predicate expects 1 operand"):
            seq.verify()

    def test_comparison_condition_rejects_one_operand(self):
        seq = self._while_with_condition(ComparisonPredicate.slt, 1)
        with pytest.raises(VerifyException, match="predicate expects 2 operand"):
            seq.verify()

    def test_rejects_non_reg_predicate_operand(self):
        entry = Block(arg_types=[i32])
        (bad,) = entry.args
        cond = ConditionOp(FlagPredicate.nez, [bad], [])
        entry.add_op(cond)
        with pytest.raises(VerifyException, match="should be of base attribute q1.reg"):
            cond.verify()

    def test_rejected_outside_while(self):
        entry = Block(arg_types=[_REG])
        (acc,) = entry.args
        entry.add_op(ConditionOp(FlagPredicate.nez, [acc], [acc]))
        module = ModuleOp(Region([entry]))
        with pytest.raises(VerifyException, match="expects parent op 'q1_scf.while'"):
            module.verify()


class TestWhileOp:
    def _while(
        self,
        before: Block,
        after: Block,
        init_args=(_REG,),
        result_types=(_REG,),
    ) -> SequenceOp:
        entry = Block(arg_types=list(init_args))
        op = WhileOp(
            list(entry.args), list(result_types), Region([before]), Region([after])
        )
        entry.add_ops([op, StopOp()])
        return SequenceOp("ch0", Region([entry]))

    def test_round_trips(self):
        before = Block(arg_types=[_REG])
        (acc,) = before.args
        before.add_op(ConditionOp(ComparisonPredicate.slt, [acc, acc], [acc]))
        after = Block(arg_types=[_REG])
        (acc2,) = after.args
        after.add_op(YieldOp(acc2))
        assert "q1_scf.while" in _build_print_reparse(self._while(before, after))

    def test_no_result_round_trips(self):
        # A while carrying no values exercises the result-less print/parse branches
        # (no "-> (...)" clause). The condition predicate references a register from
        # the enclosing block.
        entry = Block(arg_types=[_REG])
        (flag,) = entry.args
        before = Block()
        before.add_op(ConditionOp(FlagPredicate.nez, [flag], []))
        after = Block()
        after.add_op(YieldOp())
        op = WhileOp([], [], Region([before]), Region([after]))
        entry.add_ops([op, StopOp()])
        printed = _build_print_reparse(SequenceOp("ch0", Region([entry])))
        assert "q1_scf.while" in printed

    def test_before_arg_type_mismatch(self):
        before = Block(arg_types=[Registers.R0])  # differs from the _REG init type
        (acc,) = before.args
        before.add_op(ConditionOp(FlagPredicate.nez, [acc], [acc]))
        after = Block(arg_types=[_REG])
        after.add_op(YieldOp(after.args[0]))
        with pytest.raises(VerifyException, match="does not match init operand type"):
            self._while(before, after).verify()

    def test_before_arg_count_mismatch(self):
        before = Block(arg_types=[_REG, _REG])  # one init, two before args
        acc = before.args[0]
        before.add_op(ConditionOp(FlagPredicate.nez, [acc], [acc]))
        after = Block(arg_types=[_REG])
        after.add_op(YieldOp(after.args[0]))
        with pytest.raises(VerifyException, match="before region expects 1 block"):
            self._while(before, after).verify()

    def test_missing_condition_terminator(self):
        before = Block(arg_types=[_REG])
        before.add_op(YieldOp(before.args[0]))  # wrong terminator
        after = Block(arg_types=[_REG])
        after.add_op(YieldOp(after.args[0]))
        with pytest.raises(VerifyException, match="terminated by q1_scf.condition"):
            self._while(before, after).verify()

    def test_results_must_match_forwarded_args(self):
        before = Block(arg_types=[_REG])
        (acc,) = before.args
        before.add_op(ConditionOp(FlagPredicate.nez, [acc], []))  # forwards nothing
        after = Block(arg_types=[_REG])
        after.add_op(YieldOp(after.args[0]))
        with pytest.raises(VerifyException, match="forwarded"):
            self._while(before, after).verify()

    def test_after_feedback_edge_mismatch(self):
        before = Block(arg_types=[_REG])
        (acc,) = before.args
        before.add_op(ConditionOp(FlagPredicate.nez, [acc], [acc]))
        after = Block(arg_types=[_REG])
        after.add_op(YieldOp())  # yields nothing back to the before block
        with pytest.raises(VerifyException, match="feedback edge"):
            self._while(before, after).verify()

    def test_after_args_must_match_condition_forwarded_values(self):
        before = Block(arg_types=[_REG])
        (acc,) = before.args
        before.add_op(ConditionOp(FlagPredicate.nez, [acc], []))
        after = Block(arg_types=[_REG])
        (next_acc,) = after.args
        after.add_op(YieldOp(next_acc))
        with pytest.raises(VerifyException, match="after block arguments must match"):
            self._while(before, after, result_types=()).verify()

    def test_missing_after_terminator(self):
        before = Block(arg_types=[_REG])
        (acc,) = before.args
        before.add_op(ConditionOp(FlagPredicate.nez, [acc], [acc]))
        after = Block(arg_types=[_REG])
        after.add_op(StopOp())  # not a yield
        with pytest.raises(VerifyException, match="terminated by q1_scf.yield"):
            self._while(before, after).verify()

    def test_rejects_q1_cf_in_before(self):
        # The q1_cf branch must terminate its own block, so it lives in a first
        # block that jumps to the condition block. The layering check still fires.
        entry = Block(arg_types=[_REG])
        (init,) = entry.args
        cond_block = Block(arg_types=[_REG])
        (acc,) = cond_block.args
        cond_block.add_op(ConditionOp(FlagPredicate.nez, [acc], [acc]))
        branch_block = Block(arg_types=[_REG])
        (carried,) = branch_block.args
        branch_block.add_op(JmpBranchOp([carried], cond_block))
        before = Region([branch_block, cond_block])
        after = Block(arg_types=[_REG])
        after.add_op(YieldOp(after.args[0]))
        op = WhileOp([init], [_REG], before, Region([after]))
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="never share a region"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_rejects_q1_cf_in_after(self):
        # The after region is multi-block: a q1_cf branch terminates its own block
        # and jumps to the yielding block. The layering check rejects it regardless.
        entry = Block(arg_types=[_REG])
        (init,) = entry.args
        before = Block(arg_types=[_REG])
        (acc,) = before.args
        before.add_op(ConditionOp(FlagPredicate.nez, [acc], [acc]))
        yield_block = Block(arg_types=[_REG])
        (yielded,) = yield_block.args
        yield_block.add_op(YieldOp(yielded))
        branch_block = Block(arg_types=[_REG])
        (carried,) = branch_block.args
        branch_block.add_op(JmpBranchOp([carried], yield_block))
        after = Region([branch_block, yield_block])
        op = WhileOp([init], [_REG], Region([before]), after)
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="never share a region"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_before_region_must_be_single_block(self):
        # An empty before region carries no q1_cf op, so the single-block guard
        # fires with a clean verifier error rather than raising from Region.block.
        after = Block(arg_types=[_REG])
        after.add_op(YieldOp(after.args[0]))
        entry = Block(arg_types=[_REG])
        (init,) = entry.args
        op = WhileOp([init], [_REG], Region(), Region([after]))
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="before region must be a single block"):
            SequenceOp("ch0", Region([entry])).verify()


class TestForOp:
    def _for(self, body: Block, iter_args=(_REG,), iter_domain=None) -> SequenceOp:
        entry = Block(arg_types=[_REG, *iter_args])
        count, *carried = entry.args
        op = ForOp(count, carried, Region([body]), iter_domain)
        entry.add_ops([op, StopOp()])
        return SequenceOp("ch0", Region([entry]))

    def test_round_trips_with_iter_args(self):
        body = Block(arg_types=[_REG, _REG])
        _induction, carried = body.args
        body.add_op(YieldOp(carried))
        assert "q1_scf.for" in _build_print_reparse(self._for(body))

    def test_accepts_bare_block_body(self):
        # ForOp.__init__ wraps a bare Block into a single-block region.
        body = Block(arg_types=[_REG, _REG])
        _induction, carried = body.args
        body.add_op(YieldOp(carried))
        entry = Block(arg_types=[_REG, _REG])
        count, init = entry.args
        op = ForOp(count, [init], body)  # bare Block, not a Region
        entry.add_ops([op, StopOp()])
        assert "q1_scf.for" in _build_print_reparse(SequenceOp("ch0", Region([entry])))

    def test_round_trips_without_iter_args(self):
        body = Block(arg_types=[_REG])
        body.add_op(YieldOp())
        seq = self._for(body, iter_args=())
        printed = _build_print_reparse(seq)
        assert "iter_args" not in printed

    def test_round_trips_with_iter_domain(self):
        body = Block(arg_types=[_REG, _REG])
        _induction, carried = body.args
        body.add_op(YieldOp(carried))
        iter_domain = IterDomainAttr(0.0, 10.0, 2.0, 5, IterParameter.frequency)
        printed = _build_print_reparse(self._for(body, iter_domain=iter_domain))
        assert "iter_domain" in printed

    def test_parse_rejects_non_iter_domain_attribute(self):
        body = Block(arg_types=[_REG, _REG])
        _induction, carried = body.args
        body.add_op(YieldOp(carried))
        iter_domain = IterDomainAttr(0.0, 10.0, 2.0, 5, IterParameter.frequency)
        seq = self._for(body, iter_domain=iter_domain)
        stream = StringIO()
        Printer(stream=stream).print_op(ModuleOp([seq]))
        ir_text = re.sub(
            r"iter #q1_scf\.iter_domain<[^>]*>", "iter 0 : i32", stream.getvalue()
        )

        ctx = Context()
        for dialect in (Builtin, Q1, Q1_cf, Q1_sequence, Q1_scf):
            ctx.load_dialect(dialect)
        with pytest.raises(ParseError, match="q1_scf.iter_domain"):
            Parser(ctx, ir_text).parse_op()

    def test_missing_induction_argument(self):
        body = Block()  # no block arguments at all
        body.add_op(YieldOp())
        with pytest.raises(VerifyException, match="induction counter"):
            self._for(body, iter_args=()).verify()

    def test_induction_counter_wrong_type(self):
        body = Block(arg_types=[i32, _REG])  # induction counter is not a q1.reg
        _induction, carried = body.args
        body.add_op(YieldOp(carried))
        with pytest.raises(VerifyException, match="induction counter must be a q1.reg"):
            self._for(body).verify()

    def test_induction_counter_must_match_iter_count_type(self):
        body = Block(arg_types=[_REG, _REG])
        _induction, carried = body.args
        body.add_op(YieldOp(carried))
        entry = Block(arg_types=[Registers.R0, _REG])
        count, init = entry.args
        op = ForOp(count, [init], Region([body]))
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="does not match iter_count type"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_carried_arg_count_mismatch(self):
        body = Block(arg_types=[_REG])  # induction only, missing carried arg
        body.add_op(YieldOp())
        with pytest.raises(VerifyException, match="carried"):
            self._for(body).verify()

    def test_carried_arg_type_mismatch(self):
        body = Block(arg_types=[_REG, Registers.R0])  # carried type differs
        _induction, carried = body.args
        body.add_op(YieldOp())
        with pytest.raises(VerifyException, match="does not match iter_arg"):
            self._for(body).verify()

    def test_yield_type_mismatch(self):
        body = Block(arg_types=[_REG, _REG])
        body.add_op(YieldOp())  # yields nothing, one carried value expected
        with pytest.raises(VerifyException, match="carried value types"):
            self._for(body).verify()

    def test_missing_yield_terminator(self):
        # A body with no terminator fails the single-block implicit-terminator trait.
        body = Block(arg_types=[_REG, _REG])
        with pytest.raises(VerifyException, match="at least a terminator"):
            self._for(body).verify()

    def test_allocate_registers_defers_to_compiler_911(self):
        body = Block(arg_types=[_REG, _REG])
        _induction, carried = body.args
        body.add_op(YieldOp(carried))
        entry = Block(arg_types=[_REG, _REG])
        count, init = entry.args
        op = ForOp(count, [init], Region([body]))
        with pytest.raises(NotImplementedError, match="COMPILER-911"):
            op.allocate_registers()


class TestIfOp:
    def test_round_trips_flag_no_else(self):
        entry, _ = _flag_if(FlagPredicate.eqz)
        assert "q1_scf.if" in _build_print_reparse(SequenceOp("ch0", Region([entry])))

    def test_round_trips_comparison(self):
        entry = Block(arg_types=[_REG, _REG])
        left, right = entry.args
        op = IfOp(ComparisonPredicate.sge, [left, right], [], Region([Block([YieldOp()])]))
        entry.add_ops([op, StopOp()])
        assert "q1_scf.if" in _build_print_reparse(SequenceOp("ch0", Region([entry])))

    def test_round_trips_with_results_and_else(self):
        entry = Block(arg_types=[_REG, _REG])
        flag, val = entry.args
        then_region = Region([Block([YieldOp(val)])])
        else_region = Region([Block([YieldOp(val)])])
        op = IfOp(FlagPredicate.nez, [flag], [_REG], then_region, else_region)
        entry.add_ops([op, StopOp()])
        assert "else" in _build_print_reparse(SequenceOp("ch0", Region([entry])))

    def test_predicate_arity_mismatch(self):
        entry = Block(arg_types=[_REG])
        (flag,) = entry.args
        op = IfOp(ComparisonPredicate.eq, [flag], [], Region([Block([YieldOp()])]))
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="predicate expects 2 operand"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_results_require_else_region(self):
        entry = Block(arg_types=[_REG, _REG])
        flag, val = entry.args
        then_region = Region([Block([YieldOp(val)])])
        op = IfOp(FlagPredicate.nez, [flag], [_REG], then_region)  # no else
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="must have an else region"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_then_else_type_mismatch(self):
        entry = Block(arg_types=[_REG, _REG])
        flag, val = entry.args
        then_region = Region([Block([YieldOp(val)])])
        else_region = Region([Block([YieldOp()])])  # yields nothing
        op = IfOp(FlagPredicate.nez, [flag], [_REG], then_region, else_region)
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="matching types"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_missing_then_terminator(self):
        entry = Block(arg_types=[_REG])
        (flag,) = entry.args
        then_region = Region([Block([StopOp()])])  # not a yield
        op = IfOp(FlagPredicate.nez, [flag], [], then_region)
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="then region must be terminated"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_then_result_type_mismatch(self):
        entry = Block(arg_types=[_REG, _REG])
        flag, val = entry.args
        then_region = Region([Block([YieldOp()])])  # yields nothing, one result expected
        else_region = Region([Block([YieldOp(val)])])
        op = IfOp(FlagPredicate.nez, [flag], [_REG], then_region, else_region)
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="do not match the results"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_missing_else_terminator(self):
        entry = Block(arg_types=[_REG])
        (flag,) = entry.args
        then_region = Region([Block([YieldOp()])])
        else_region = Region([Block([StopOp()])])  # present but not a yield
        op = IfOp(FlagPredicate.nez, [flag], [], then_region, else_region)
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="else region must be terminated"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_rejects_q1_cf_in_then(self):
        # The q1_cf branch terminates its own block and jumps to the yielding
        # block. The layering check rejects the region regardless.
        entry = Block(arg_types=[_REG])
        (flag,) = entry.args
        yield_block = Block([YieldOp()])
        branch_block = Block([JmpBranchOp([], yield_block)])
        op = IfOp(FlagPredicate.nez, [flag], [], Region([branch_block, yield_block]))
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="never share a region"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_rejects_q1_cf_in_else(self):
        # The else region is multi-block: a q1_cf branch terminates its own block
        # and jumps to the yielding block. The layering check rejects it regardless.
        entry = Block(arg_types=[_REG])
        (flag,) = entry.args
        then_region = Region([Block([YieldOp()])])
        yield_block = Block([YieldOp()])
        branch_block = Block([JmpBranchOp([], yield_block)])
        else_region = Region([branch_block, yield_block])
        op = IfOp(FlagPredicate.nez, [flag], [], then_region, else_region)
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="never share a region"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_then_region_must_be_single_block(self):
        # An empty then region carries no q1_cf op, so the single-block guard fires.
        entry = Block(arg_types=[_REG])
        (flag,) = entry.args
        op = IfOp(FlagPredicate.nez, [flag], [], Region())
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="then region must be a single block"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_else_region_must_be_single_block(self):
        # A multi-block else region without a q1_cf op still fails the guard.
        entry = Block(arg_types=[_REG])
        (flag,) = entry.args
        then_region = Region([Block([YieldOp()])])
        else_region = Region([Block([YieldOp()]), Block([YieldOp()])])
        op = IfOp(FlagPredicate.nez, [flag], [], then_region, else_region)
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="else region must be a single block"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_rejects_non_reg_predicate_operand(self):
        entry = Block(arg_types=[i32])
        (bad,) = entry.args
        op = IfOp(FlagPredicate.nez, [bad], [], Region([Block([YieldOp()])]))
        entry.add_ops([op, StopOp()])
        with pytest.raises(VerifyException, match="should be of base attribute q1.reg"):
            SequenceOp("ch0", Region([entry])).verify()

    def test_parse_rejects_unknown_predicate(self):
        entry, _ = _flag_if()
        stream = StringIO()
        Printer(stream=stream).print_op(ModuleOp([SequenceOp("ch0", Region([entry]))]))
        ir_text = stream.getvalue().replace("q1_scf.if nez", "q1_scf.if bogus")

        ctx = Context()
        for dialect in (Builtin, Q1, Q1_cf, Q1_sequence, Q1_scf):
            ctx.load_dialect(dialect)
        with pytest.raises(ParseError, match="flag or comparison predicate"):
            Parser(ctx, ir_text).parse_op()

    def test_allocate_registers_defers_to_compiler_911(self):
        _entry, op = _flag_if()
        with pytest.raises(NotImplementedError, match="COMPILER-911"):
            op.allocate_registers()


class TestYieldTerminatorHelper:
    def test_returns_none_for_multi_block_region(self):
        region = Region([Block([YieldOp()]), Block([YieldOp()])])
        assert _yield_terminator(region) is None

    def test_returns_none_for_empty_region(self):
        assert _yield_terminator(Region()) is None

    def test_returns_yield_for_single_block(self):
        yield_op = YieldOp()
        region = Region([Block([yield_op])])
        assert _yield_terminator(region) is yield_op


class TestSequenceAncestor:
    @pytest.mark.parametrize("op_type", _CONTAINER_OPS, ids=_CONTAINER_IDS)
    def test_container_rejected_outside_sequence(self, op_type):
        entry, op = _flag_if()
        if op_type is IfOp:
            container_block = entry
        elif op_type is WhileOp:
            container_block = Block(arg_types=[_REG])
            (init,) = container_block.args
            before = Block(arg_types=[_REG])
            (acc,) = before.args
            before.add_op(ConditionOp(FlagPredicate.nez, [acc], [acc]))
            after = Block(arg_types=[_REG])
            after.add_op(YieldOp(after.args[0]))
            container_block.add_op(
                WhileOp([init], [_REG], Region([before]), Region([after]))
            )
        else:
            container_block = Block(arg_types=[_REG, _REG])
            count, init = container_block.args
            body = Block(arg_types=[_REG, _REG])
            _induction, carried = body.args
            body.add_op(YieldOp(carried))
            container_block.add_op(ForOp(count, [init], Region([body])))
        module = ModuleOp(Region([container_block]))
        with pytest.raises(VerifyException, match="q1_sequence.sequence"):
            module.verify()


def test_nested_containers_round_trip():
    entry = Block(arg_types=[_REG, _REG, _REG])
    flag, count, seed = entry.args

    # if with feedforward results, both regions yielding the seed register.
    if_op = IfOp(
        FlagPredicate.nez,
        [flag],
        [_REG],
        Region([Block([YieldOp(seed)])]),
        Region([Block([YieldOp(seed)])]),
    )

    # while accumulating over the if result.
    before = Block(arg_types=[_REG])
    (acc,) = before.args
    before.add_op(ConditionOp(ComparisonPredicate.slt, [acc, acc], [acc]))
    after = Block(arg_types=[_REG])
    after.add_op(YieldOp(after.args[0]))
    while_op = WhileOp([if_op.output[0]], [_REG], Region([before]), Region([after]))

    # iterating counted loop carrying the while result.
    body = Block(arg_types=[_REG, _REG])
    _induction, carried = body.args
    body.add_op(YieldOp(carried))
    for_op = ForOp(
        count,
        [while_op.res[0]],
        Region([body]),
        IterDomainAttr(0.0, 10.0, 2.0, 5, IterParameter.duration),
    )

    entry.add_ops([if_op, while_op, for_op, StopOp()])
    printed = _build_print_reparse(SequenceOp("ch0", Region([entry])))
    assert "q1_scf.if" in printed
    assert "q1_scf.while" in printed
    assert "q1_scf.for" in printed
    assert "iter_domain" in printed
