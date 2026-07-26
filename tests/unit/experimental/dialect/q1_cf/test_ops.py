# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for the q1_cf operations.

Covers the four concrete ops and the branch machinery they share (verifier
rules, successor plumbing, register constraints and const_evaluate), all of
which now live alongside the ops in ``q1_cf.ir.ops``.

Coverage:
* Name and trait invariants for every operation.
* Verifier: successor-argument arity and per-position type equality.
* Verifier: distinct then/else (and body/exit) successors.
* Verifier: the physical fall-through invariant for conditional branches.
* Register constraints: every operand is reported as an input, none as output.
* const_evaluate: each predicate's constant truth table, including the
  signed/unsigned reinterpretation boundary.
* Parse-verify-print round-trip for each op and every predicate spelling.
* Integration: a multi-block CFG with a back-edge (loop) parses, verifies, prints.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import StringIO

import pytest
from xdsl.backend.register_allocatable import (
    HasRegisterConstraintsTrait,
    RegisterConstraints,
)
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.ir import Block, Region, SSAValue
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.common.cfg import SuccessorOperandsTrait, successor_edges
from qat.experimental.dialect.q1 import (
    INT_REGISTER_BIT_WIDTH,
    INT_REGISTER_VALUE_MASK,
    Q1,
    Registers,
    StopOp,
)
from qat.experimental.dialect.q1_cf import (
    BinaryPredicate,
    BinaryPredicateBranchOp,
    JmpBranchOp,
    LoopBranchOp,
    Q1_cf,
    UnaryPredicate,
    UnaryPredicateBranchOp,
)
from qat.experimental.dialect.q1_sequence import Q1_sequence, SequenceOp

# Signless bit patterns used to exercise const_evaluate. const_evaluate receives
# operands as raw signless bits plus a width, and reinterprets them as signed or
# unsigned per the predicate. _NEG1 and _MAXU are the same all-ones pattern under
# two readings, so they pin that boundary.
_NEG1 = INT_REGISTER_VALUE_MASK  # all-ones bits; signed reading is -1
_MAXU = INT_REGISTER_VALUE_MASK  # same all-ones bits; unsigned reading is the maximum


def _make_jmp(dest_b: Block, dest_args: Sequence[SSAValue] = ()) -> JmpBranchOp:
    return JmpBranchOp(list(dest_args), dest_b)


def _make_unary_predicate_branch_op(
    then_b: Block,
    else_b: Block,
    then_args: Sequence[SSAValue] = (),
    else_args: Sequence[SSAValue] = (),
    predicate: UnaryPredicate = UnaryPredicate.eqz,
) -> UnaryPredicateBranchOp:
    return UnaryPredicateBranchOp(
        predicate,
        create_ssa_value(Registers.UNALLOCATED_INT),
        list(then_args),
        list(else_args),
        then_b,
        else_b,
    )


def _make_binary_predicate_branch_op(
    then_b: Block,
    else_b: Block,
    then_args: Sequence[SSAValue] = (),
    else_args: Sequence[SSAValue] = (),
    predicate: BinaryPredicate = BinaryPredicate.eq,
) -> BinaryPredicateBranchOp:
    return BinaryPredicateBranchOp(
        predicate,
        create_ssa_value(Registers.UNALLOCATED_INT),
        create_ssa_value(Registers.UNALLOCATED_INT),
        list(then_args),
        list(else_args),
        then_b,
        else_b,
    )


def _make_loop_branch_op(
    body_b: Block,
    exit_b: Block,
    body_args: Sequence[SSAValue] = (),
    exit_args: Sequence[SSAValue] = (),
) -> LoopBranchOp:
    return LoopBranchOp(
        create_ssa_value(Registers.UNALLOCATED_INT),
        list(body_args),
        list(exit_args),
        body_b,
        exit_b,
    )


# The two conditional families share the then/else successor contract, so the
# verifier suite parametrises over a branch-op maker for each.
_BRANCH_OP_MAKERS = [
    pytest.param(_make_unary_predicate_branch_op, id="unary_predicate_branch"),
    pytest.param(_make_binary_predicate_branch_op, id="binary_predicate_branch"),
]

# (op_type, mnemonic) for every op; drives name + trait coverage.
_ALL_OPS = [
    (JmpBranchOp, "q1_cf.jmp_branch"),
    (UnaryPredicateBranchOp, "q1_cf.unary_predicate_branch"),
    (BinaryPredicateBranchOp, "q1_cf.binary_predicate_branch"),
    (LoopBranchOp, "q1_cf.loop_branch"),
]

_ALL_OP_IDS = [mnemonic for _, mnemonic in _ALL_OPS]

# Every q1_cf op carries exactly these three traits. Each supplies its own edge
# layout through SuccessorOperandsTrait, exercised in TestSuccessorOperandsTrait.
_EXPECTED_TRAITS = (IsTerminator, HasRegisterConstraintsTrait, SuccessorOperandsTrait)


def _build(op_type, then_b: Block, else_b: Block):
    """Build a canonical instance of any q1_cf op, dispatching on its type.

    The instance is structurally well-formed for trait/name inspection; it is
    not required to pass ``verify_`` (fall-through/back-edge rules are exercised
    by the verifier suites).
    """
    if op_type is JmpBranchOp:
        return _make_jmp(then_b)
    if op_type is UnaryPredicateBranchOp:
        return _make_unary_predicate_branch_op(then_b, else_b)
    if op_type is BinaryPredicateBranchOp:
        return _make_binary_predicate_branch_op(then_b, else_b)
    return _make_loop_branch_op(then_b, else_b)


def _assert_traits(op, expected_traits: tuple) -> None:
    assert len(expected_traits) == len(op.traits.traits)
    for trait in expected_traits:
        assert op.has_trait(trait)


def _assert_verifies(op) -> None:
    """Assert that ``op`` passes verification.

    ``verify_`` signals failure by raising :class:`VerifyException`; catching it
    and calling :func:`pytest.fail` turns "no exception raised" into an explicit,
    self-describing assertion.
    """
    try:
        op.verify_()
    except VerifyException as exc:  # pragma: no cover - only hit on regression
        pytest.fail(f"expected {op.name} to verify, but it raised: {exc}")


def _build_print_reparse(seq_op: SequenceOp) -> str:
    """Print a SequenceOp inside a module, reparse and re-verify it.

    :returns: The final printed IR string.
    """
    module = ModuleOp([seq_op])
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    ir_text = stream.getvalue()

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Q1)
    ctx.load_dialect(Q1_cf)
    ctx.load_dialect(Q1_sequence)

    reparsed = Parser(ctx, ir_text).parse_op()
    reparsed.verify()

    stream2 = StringIO()
    Printer(stream=stream2).print_op(reparsed)
    return stream2.getvalue()


class TestOpMetadata:
    @pytest.mark.parametrize("op_type,mnemonic", _ALL_OPS, ids=_ALL_OP_IDS)
    def test_name(self, op_type, mnemonic):
        assert op_type.name == mnemonic

    @pytest.mark.parametrize("op_type,mnemonic", _ALL_OPS, ids=_ALL_OP_IDS)
    def test_traits(self, op_type, mnemonic):
        op = _build(op_type, Block([StopOp()]), Block([StopOp()]))
        _assert_traits(op, _EXPECTED_TRAITS)


class TestSuccessorOperandsTrait:
    def test_jmp_branch_reports_its_single_edge(self):
        dest = Block()
        arg = create_ssa_value(Registers.UNALLOCATED_INT)
        op = _make_jmp(dest, [arg])
        assert successor_edges(op) == [(dest, [arg])]

    @pytest.mark.parametrize("maker", _BRANCH_OP_MAKERS)
    def test_conditional_branch_reports_then_before_else(self, maker):
        then_b, else_b = Block(), Block()
        then_arg = create_ssa_value(Registers.UNALLOCATED_INT)
        else_arg = create_ssa_value(Registers.UNALLOCATED_INT)
        op = maker(then_b, else_b, then_args=[then_arg], else_args=[else_arg])
        assert successor_edges(op) == [(then_b, [then_arg]), (else_b, [else_arg])]

    def test_loop_branch_reports_body_before_exit(self):
        body_b, exit_b = Block(), Block()
        body_arg = create_ssa_value(Registers.UNALLOCATED_INT)
        exit_arg = create_ssa_value(Registers.UNALLOCATED_INT)
        op = _make_loop_branch_op(body_b, exit_b, [body_arg], [exit_arg])
        assert successor_edges(op) == [(body_b, [body_arg]), (exit_b, [exit_arg])]

    def test_terminator_without_the_trait_has_no_edges(self):
        assert successor_edges(StopOp()) == []

    def test_trait_rejects_a_foreign_operation(self):
        trait = JmpBranchOp.get_trait(SuccessorOperandsTrait)
        assert trait is not None
        with pytest.raises(AssertionError):
            trait.successor_edges(StopOp())

    @pytest.mark.parametrize(
        "op_type",
        [UnaryPredicateBranchOp, BinaryPredicateBranchOp, LoopBranchOp],
    )
    def test_each_branch_trait_rejects_a_foreign_operation(self, op_type):
        trait = op_type.get_trait(SuccessorOperandsTrait)
        assert trait is not None
        with pytest.raises(AssertionError):
            trait.successor_edges(StopOp())


class TestUnconditionalBranch:
    def test_valid_no_args(self):
        dest = Block([StopOp()])
        op = _make_jmp(dest)
        _assert_verifies(op)
        assert op.successor is dest
        assert len(op.successor_arguments) == 0

    def test_valid_with_args(self):
        arg_val = create_ssa_value(Registers.R0)
        dest = Block(arg_types=[Registers.R0], ops=[StopOp()])
        op = _make_jmp(dest, [arg_val])
        _assert_verifies(op)
        assert list(op.successor_arguments) == [arg_val]
        assert op.successor_arguments[0].type == dest.args[0].type

    def test_type_mismatch_rejects(self):
        arg_val = create_ssa_value(Registers.R0)
        dest = Block(arg_types=[Registers.R1], ops=[StopOp()])
        with pytest.raises(VerifyException, match=r"successor operand \d+ type"):
            _make_jmp(dest, [arg_val]).verify_()

    def test_too_few_args_rejects(self):
        dest = Block(arg_types=[Registers.R0, Registers.R1], ops=[StopOp()])
        with pytest.raises(VerifyException, match="successor block expects 2 argument"):
            _make_jmp(dest).verify_()

    def test_too_many_args_rejects(self):
        arg_val = create_ssa_value(Registers.R0)
        dest = Block([StopOp()])
        with pytest.raises(VerifyException, match="successor block expects 0 argument"):
            _make_jmp(dest, [arg_val]).verify_()


class TestConditionalBranch:
    @pytest.mark.parametrize("branch_op_maker", _BRANCH_OP_MAKERS)
    def test_same_then_else_rejects(self, branch_op_maker):
        b = Block([StopOp()])
        with pytest.raises(VerifyException, match="then and else blocks must be different"):
            branch_op_maker(b, b).verify_()

    @pytest.mark.parametrize("branch_op_maker", _BRANCH_OP_MAKERS)
    def test_then_arg_type_mismatch_rejects(self, branch_op_maker):
        arg_val = create_ssa_value(Registers.R0)
        then_b = Block(arg_types=[Registers.R1], ops=[StopOp()])
        else_b = Block([StopOp()])
        with pytest.raises(VerifyException, match=r"then operand \d+ type"):
            branch_op_maker(then_b, else_b, then_args=[arg_val]).verify_()

    @pytest.mark.parametrize("branch_op_maker", _BRANCH_OP_MAKERS)
    def test_else_arg_type_mismatch_rejects(self, branch_op_maker):
        arg_val = create_ssa_value(Registers.R0)
        then_b = Block([StopOp()])
        else_b = Block(arg_types=[Registers.R1], ops=[StopOp()])
        with pytest.raises(VerifyException, match=r"else operand \d+ type"):
            branch_op_maker(then_b, else_b, else_args=[arg_val]).verify_()

    @pytest.mark.parametrize("branch_op_maker", _BRANCH_OP_MAKERS)
    def test_then_arity_mismatch_rejects(self, branch_op_maker):
        then_b = Block(arg_types=[Registers.R0], ops=[StopOp()])
        else_b = Block([StopOp()])
        with pytest.raises(VerifyException, match="then block expects 1 argument"):
            branch_op_maker(then_b, else_b).verify_()

    @pytest.mark.parametrize("branch_op_maker", _BRANCH_OP_MAKERS)
    def test_else_arity_mismatch_rejects(self, branch_op_maker):
        arg_val = create_ssa_value(Registers.R0)
        then_b = Block([StopOp()])
        else_b = Block([StopOp()])
        with pytest.raises(VerifyException, match="else block expects 0 argument"):
            branch_op_maker(then_b, else_b, else_args=[arg_val]).verify_()

    @pytest.mark.parametrize("branch_op_maker", _BRANCH_OP_MAKERS)
    def test_fall_through_invariant_rejects(self, branch_op_maker):
        """else_block must immediately follow the branch's own block."""
        then_b = Block([StopOp()])
        else_b = Block([StopOp()])
        # entry -> then -> else: next_block of entry is then_b, not else_b -> violation
        entry_b = Block([branch_op_maker(then_b, else_b)])
        Region([entry_b, then_b, else_b])
        with pytest.raises(VerifyException, match="fall-through invariant"):
            entry_b.ops.first.verify_()  # type: ignore[union-attr]

    @pytest.mark.parametrize("branch_op_maker", _BRANCH_OP_MAKERS)
    def test_fall_through_satisfied(self, branch_op_maker):
        """else_block immediately following satisfies the invariant."""
        then_b = Block([StopOp()])
        else_b = Block([StopOp()])
        entry_b = Block([branch_op_maker(then_b, else_b)])
        # entry -> else -> then: next_block of entry is else_b -> invariant holds
        Region([entry_b, else_b, then_b])
        op = entry_b.ops.first
        assert op is not None
        assert entry_b.next_block is else_b
        assert op.else_block is else_b
        assert op.then_block is then_b
        _assert_verifies(op)


class TestLoopBranch:
    def test_valid(self):
        body_b = Block([StopOp()])
        exit_b = Block([StopOp()])
        op = _make_loop_branch_op(body_b, exit_b)
        _assert_verifies(op)
        assert op.body_block is body_b
        assert op.exit_block is exit_b
        assert op.body_block is not op.exit_block

    def test_same_body_exit_rejects(self):
        b = Block([StopOp()])
        with pytest.raises(VerifyException, match="body and exit blocks must be different"):
            _make_loop_branch_op(b, b).verify_()

    def test_body_arg_type_mismatch_rejects(self):
        arg_val = create_ssa_value(Registers.R0)
        body_b = Block(arg_types=[Registers.R1], ops=[StopOp()])
        exit_b = Block([StopOp()])
        with pytest.raises(VerifyException, match=r"body operand \d+ type"):
            _make_loop_branch_op(body_b, exit_b, body_args=[arg_val]).verify_()

    def test_exit_arg_type_mismatch_rejects(self):
        arg_val = create_ssa_value(Registers.R0)
        body_b = Block([StopOp()])
        exit_b = Block(arg_types=[Registers.R1], ops=[StopOp()])
        with pytest.raises(VerifyException, match=r"exit operand \d+ type"):
            _make_loop_branch_op(body_b, exit_b, exit_args=[arg_val]).verify_()

    def test_body_arity_mismatch_rejects(self):
        body_b = Block(arg_types=[Registers.R0], ops=[StopOp()])
        exit_b = Block([StopOp()])
        with pytest.raises(VerifyException, match="body block expects 1 argument"):
            _make_loop_branch_op(body_b, exit_b).verify_()

    def test_exit_arity_mismatch_rejects(self):
        arg_val = create_ssa_value(Registers.R0)
        body_b = Block([StopOp()])
        exit_b = Block([StopOp()])
        with pytest.raises(VerifyException, match="exit block expects 0 argument"):
            _make_loop_branch_op(body_b, exit_b, exit_args=[arg_val]).verify_()


class TestRegisterConstraints:
    """Every branch reports all operands as inputs and produces no results."""

    def test_unconditional(self):
        arg_val = create_ssa_value(Registers.R0)
        dest = Block(arg_types=[Registers.R0], ops=[StopOp()])
        self._assert_ins_only(_make_jmp(dest, [arg_val]))

    @pytest.mark.parametrize("branch_op_maker", _BRANCH_OP_MAKERS)
    def test_conditional(self, branch_op_maker):
        then_b = Block([StopOp()])
        else_b = Block([StopOp()])
        self._assert_ins_only(branch_op_maker(then_b, else_b))

    def test_loop(self):
        body_b = Block([StopOp()])
        exit_b = Block([StopOp()])
        self._assert_ins_only(_make_loop_branch_op(body_b, exit_b))

    @staticmethod
    def _assert_ins_only(op) -> None:
        constraints = op.get_register_constraints()
        assert isinstance(constraints, RegisterConstraints)
        assert tuple(constraints.ins) == tuple(op.operands)
        assert len(op.operands) > 0
        assert tuple(constraints.outs) == ()
        assert tuple(constraints.inouts) == ()


# (predicate, rs, taken) truth-table entries for the flag branch.
_UNARY_PREDICATE_CONST_CASES = [
    pytest.param(UnaryPredicate.eqz, 0, True, id="eqz-zero-taken"),
    pytest.param(UnaryPredicate.eqz, 5, False, id="eqz-nonzero-fallthrough"),
    pytest.param(UnaryPredicate.nez, 5, True, id="nez-nonzero-taken"),
    pytest.param(UnaryPredicate.nez, 0, False, id="nez-zero-fallthrough"),
    pytest.param(UnaryPredicate.ltz, _NEG1, True, id="ltz-negative-taken"),
    pytest.param(UnaryPredicate.ltz, 0, False, id="ltz-zero-fallthrough"),
    pytest.param(UnaryPredicate.ltz, 1, False, id="ltz-positive-fallthrough"),
    pytest.param(UnaryPredicate.gez, 0, True, id="gez-zero-taken"),
    pytest.param(UnaryPredicate.gez, 1, True, id="gez-positive-taken"),
    pytest.param(UnaryPredicate.gez, _NEG1, False, id="gez-negative-fallthrough"),
]

# (predicate, lhs, rhs, taken) truth-table entries for the comparison branch.
# The -1/1 and MAXU pairs pin the signed vs unsigned reinterpretation boundary.
_BINARY_PREDICATE_CONST_CASES = [
    pytest.param(BinaryPredicate.eq, 7, 7, True, id="eq-equal-taken"),
    pytest.param(BinaryPredicate.eq, 7, 8, False, id="eq-unequal-fallthrough"),
    pytest.param(BinaryPredicate.ne, 7, 8, True, id="ne-unequal-taken"),
    pytest.param(BinaryPredicate.ne, 7, 7, False, id="ne-equal-fallthrough"),
    pytest.param(BinaryPredicate.slt, _NEG1, 1, True, id="slt-signed-lt-taken"),
    pytest.param(BinaryPredicate.slt, 1, _NEG1, False, id="slt-signed-not-lt"),
    pytest.param(BinaryPredicate.sle, 5, 5, True, id="sle-signed-equal-taken"),
    pytest.param(BinaryPredicate.sle, 1, _NEG1, False, id="sle-signed-not-le"),
    pytest.param(BinaryPredicate.sgt, 1, _NEG1, True, id="sgt-signed-gt-taken"),
    pytest.param(BinaryPredicate.sgt, _NEG1, 1, False, id="sgt-signed-not-gt"),
    pytest.param(BinaryPredicate.sge, 5, 5, True, id="sge-signed-equal-taken"),
    pytest.param(BinaryPredicate.sge, _NEG1, 1, False, id="sge-signed-not-ge"),
    pytest.param(BinaryPredicate.ult, 1, _MAXU, True, id="ult-unsigned-lt-taken"),
    pytest.param(BinaryPredicate.ult, _MAXU, 1, False, id="ult-unsigned-not-lt"),
    pytest.param(BinaryPredicate.ule, 1, _MAXU, True, id="ule-unsigned-le-taken"),
    pytest.param(BinaryPredicate.ule, _MAXU, 1, False, id="ule-unsigned-not-le"),
    pytest.param(BinaryPredicate.ugt, _MAXU, 1, True, id="ugt-unsigned-gt-taken"),
    pytest.param(BinaryPredicate.ugt, 1, _MAXU, False, id="ugt-unsigned-not-gt"),
    pytest.param(BinaryPredicate.uge, _MAXU, 1, True, id="uge-unsigned-ge-taken"),
    pytest.param(BinaryPredicate.uge, 1, _MAXU, False, id="uge-unsigned-not-ge"),
]


class TestConstEvaluate:
    """Each conditional branch folds its predicate over constant operand values."""

    @pytest.mark.parametrize(("predicate", "rs", "taken"), _UNARY_PREDICATE_CONST_CASES)
    def test_unary_predicate(self, predicate: UnaryPredicate, rs: int, taken: bool):
        op = _make_unary_predicate_branch_op(
            Block([StopOp()]), Block([StopOp()]), predicate=predicate
        )
        assert op.const_evaluate(rs, INT_REGISTER_BIT_WIDTH) is taken

    @pytest.mark.parametrize(
        ("predicate", "lhs", "rhs", "taken"), _BINARY_PREDICATE_CONST_CASES
    )
    def test_binary_predicate(
        self, predicate: BinaryPredicate, lhs: int, rhs: int, taken: bool
    ):
        op = _make_binary_predicate_branch_op(
            Block([StopOp()]), Block([StopOp()]), predicate=predicate
        )
        assert op.const_evaluate(lhs, rhs, INT_REGISTER_BIT_WIDTH) is taken

    def test_signed_unsigned_divergence(self):
        """The same operands flip verdict under signed vs unsigned reading."""
        signed = _make_binary_predicate_branch_op(
            Block([StopOp()]), Block([StopOp()]), predicate=BinaryPredicate.sgt
        )
        unsigned = _make_binary_predicate_branch_op(
            Block([StopOp()]), Block([StopOp()]), predicate=BinaryPredicate.uge
        )
        assert signed.const_evaluate(_NEG1, 1, INT_REGISTER_BIT_WIDTH) is False
        assert unsigned.const_evaluate(_NEG1, 1, INT_REGISTER_BIT_WIDTH) is True


class TestRoundTrip:
    def test_jmp_no_args(self):
        """Unconditional jump with no successor arguments round-trips."""
        dest = Block([StopOp()])
        entry = Block([JmpBranchOp([], dest)])
        seq = SequenceOp("ch0", Region([entry, dest]))
        result = _build_print_reparse(seq)
        assert "q1_cf.jmp_branch" in result

    def test_flag_no_args(self):
        """Flag conditional branch (no successor args) round-trips."""
        then_b = Block([StopOp()])
        else_b = Block([StopOp()])
        entry = Block(arg_types=[Registers.R0])
        entry.add_op(
            UnaryPredicateBranchOp(
                UnaryPredicate.eqz, entry.args[0], [], [], then_b, else_b
            )
        )
        seq = SequenceOp("ch0", Region([entry, else_b, then_b]))
        result = _build_print_reparse(seq)
        assert "q1_cf.unary_predicate_branch eqz" in result

    def test_cond_with_successor_args(self):
        """Comparison branch with distinct per-successor block args round-trips.

        Also the canary for correct variadic behaviour: ``then`` forwards an
        ``R0`` value and ``else`` an ``R1`` value, so the operand segments must
        accept differing register instances per operand group.
        """
        then_b = Block(arg_types=[Registers.R0], ops=[StopOp()])
        else_b = Block(arg_types=[Registers.R1], ops=[StopOp()])
        entry = Block(arg_types=[Registers.R0, Registers.R1])
        lhs, rhs = entry.args[0], entry.args[1]
        entry.add_op(
            BinaryPredicateBranchOp(
                BinaryPredicate.slt, lhs, rhs, [lhs], [rhs], then_b, else_b
            )
        )
        seq = SequenceOp("ch0", Region([entry, else_b, then_b]))
        result = _build_print_reparse(seq)
        assert "q1_cf.binary_predicate_branch slt" in result

    def test_loop_with_back_edge(self):
        """Loop op with a back-edge (body == header) round-trips."""
        exit_b = Block([StopOp()])
        header = Block(arg_types=[Registers.R0])
        n = header.args[0]
        header.add_op(LoopBranchOp(n, [n], [], header, exit_b))
        seq = SequenceOp("ch0", Region([header, exit_b]))
        result = _build_print_reparse(seq)
        assert "q1_cf.loop_branch" in result

    @pytest.mark.parametrize("predicate", list(UnaryPredicate), ids=lambda p: p.value)
    def test_flag_predicate_round_trip(self, predicate: UnaryPredicate):
        """Every unary predicate spelling survives print/reparse."""
        then_b = Block([StopOp()])
        else_b = Block([StopOp()])
        entry = Block(arg_types=[Registers.R0])
        entry.add_op(
            UnaryPredicateBranchOp(predicate, entry.args[0], [], [], then_b, else_b)
        )
        seq = SequenceOp("ch0", Region([entry, else_b, then_b]))
        result = _build_print_reparse(seq)
        assert f"q1_cf.unary_predicate_branch {predicate.value}" in result

    @pytest.mark.parametrize("predicate", list(BinaryPredicate), ids=lambda p: p.value)
    def test_comparison_predicate_round_trip(self, predicate: BinaryPredicate):
        """Every binary predicate spelling survives print/reparse."""
        then_b = Block([StopOp()])
        else_b = Block([StopOp()])
        entry = Block(arg_types=[Registers.R0, Registers.R1])
        lhs, rhs = entry.args[0], entry.args[1]
        entry.add_op(BinaryPredicateBranchOp(predicate, lhs, rhs, [], [], then_b, else_b))
        seq = SequenceOp("ch0", Region([entry, else_b, then_b]))
        result = _build_print_reparse(seq)
        assert f"q1_cf.binary_predicate_branch {predicate.value}" in result


class TestIntegration:
    def test_back_edge_cfg_parses_verifies_prints(self):
        """A while-loop CFG (back-edge from body to header) is valid."""
        exit_b = Block([StopOp()])
        header = Block(arg_types=[Registers.R0])
        n = header.args[0]
        header.add_op(LoopBranchOp(n, [n], [], header, exit_b))
        seq = SequenceOp("loop_seq", Region([header, exit_b]))
        result = _build_print_reparse(seq)
        assert "q1_cf.loop_branch" in result
