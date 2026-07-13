# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""q1_cf dialect operations.

Target-specific CFG operations for QBlox Q1. Each operation is a block
terminator that carries one or more named successor blocks, each with its own
group of SSA block arguments. Conditions are expressed as SSA operands
(``q1.reg``) plus a predicate attribute rather than via the physical flag
registers; the flag machinery is reintroduced at linearisation to flat ``q1``.

The dialect occupies the layer above flat ``q1`` (ISA mnemonics). It provides
four terminators:

- ``JmpBranchOp`` — unconditional branch.
- ``FlagBranchOp`` — branch on a condition-code test of one register.
- ``ComparisonBranchOp`` — branch on a comparison of two operands.
- ``LoopBranchOp`` — counted-loop back-edge.

Reference: https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html
"""

from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import Self
from xdsl.backend.register_allocatable import HasRegisterConstraints, RegisterConstraints
from xdsl.ir import Block, Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    Successor,
    irdl_op_definition,
    operand_def,
    prop_def,
    successor_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator
from xdsl.utils.comparisons import to_signed, to_unsigned
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1.ir.reg_desc import IntRegisterType
from qat.experimental.dialect.q1_cf.ir.attrs import (
    ComparisonPredicate,
    ComparisonPredicateAttr,
    FlagPredicate,
    FlagPredicateAttr,
)


def _print_type_pair(printer: Printer, value: SSAValue) -> None:
    """Print a single ``%value : type`` operand pair."""
    printer.print_ssa_value(value)
    printer.print_string(" : ")
    printer.print_attribute(value.type)


def _parse_type_pair(parser: Parser) -> SSAValue:
    """Parse and resolve a single ``%value : type`` operand pair."""
    unresolved = parser.parse_unresolved_operand()
    parser.parse_punctuation(":")
    type_ = parser.parse_type()
    return parser.resolve_operand(unresolved, type_)


def _print_successor(printer: Printer, block: Block, args: Sequence[SSAValue]) -> None:
    """Print a ``^block(%arg : type, ...)`` successor edge."""
    printer.print_block_name(block)
    printer.print_string("(")
    printer.print_list(args, lambda value: _print_type_pair(printer, value))
    printer.print_string(")")


def _parse_successor(parser: Parser) -> tuple[Block, list[SSAValue]]:
    """Parse a ``^block(%arg : type, ...)`` successor edge.

    :returns: The target block and its forwarded operand group.
    """
    block = parser.parse_successor()
    args = parser.parse_comma_separated_list(
        parser.Delimiter.PAREN, lambda: _parse_type_pair(parser)
    )
    return block, args


def _verify_successor_args(
    op_args: Sequence[SSAValue],
    block_args: Sequence[SSAValue],
    op_name: str,
    label: str,
) -> None:
    """Verify a successor operand group against its block's signature.

    Enforces both arity (one operand per block argument) and per-position type
    equality, so a successor edge is well-typed end to end.

    :param op_args: The operand group forwarded along the successor edge.
    :param block_args: The target block's arguments.
    :param op_name: The operation name, for diagnostics.
    :param label: The successor role (``successor``/``then``/``else``/``body``/``exit``).
    :raises VerifyException: If the operand count or any operand type differs
        from the block signature.
    """
    if len(op_args) != len(block_args):
        raise VerifyException(
            f"{op_name}: {label} block expects {len(block_args)} argument(s), but"
            f" {len(op_args)} operand(s) were supplied"
        )
    for position, (op_arg, block_arg) in enumerate(zip(op_args, block_args, strict=True)):
        if op_arg.type != block_arg.type:
            raise VerifyException(
                f"{op_name}: {label} operand {position} type {op_arg.type!r} does"
                f" not match block argument type {block_arg.type!r}"
            )


def _verify_distinct_successors(
    op_name: str, first: Block, second: Block, first_label: str, second_label: str
) -> None:
    """Reject a branch whose two successors are the same block.

    :raises VerifyException: If ``first`` and ``second`` are the same block.
    """
    if first is second:
        raise VerifyException(
            f"{op_name}: {first_label} and {second_label} blocks must be different"
        )


def _verify_fall_through(op, else_block: Block) -> None:
    """Verify that ``else_block`` immediately follows the branch's own block.

    q1_cf conditional branches fall through to their ``else`` successor, so it
    must be laid out directly after the block that holds the branch. The check
    is skipped for a branch not yet placed in a region.

    :param op: The branch operation being verified.
    :param else_block: The fall-through successor.
    :raises VerifyException: If the branch is placed in a region and its block is
        not immediately followed by ``else_block``.
    """
    parent_block = op.parent
    if (
        parent_block is not None
        and parent_block.parent is not None
        and parent_block.next_block is not else_block
    ):
        raise VerifyException(
            f"{op.name}: else block must immediately follow the parent"
            f" block (fall-through invariant)"
        )


@irdl_op_definition
class JmpBranchOp(HasRegisterConstraints, IRDLOperation):
    """Unconditional branch to a successor block.

    Carries optional per-successor block arguments. At linearisation to flat
    ``q1`` this becomes ``q1.i.jmp`` (or is elided when ``successor`` is the
    fall-through block).

    .. code-block:: mlir

        q1_cf.jmp_branch ^dest(%arg0 : q1.reg, ...)
    """

    name = "q1_cf.jmp_branch"

    successor_arguments = var_operand_def(IntRegisterType)
    successor = successor_def()

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        successor_arguments: Sequence[SSAValue | Operation],
        successor: Successor,
    ):
        super().__init__(operands=[successor_arguments], successors=(successor,))

    def verify_(self) -> None:
        _verify_successor_args(
            self.successor_arguments, self.successor.args, self.name, "successor"
        )

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(self.operands, [], [])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        _print_successor(printer, self.successor, self.successor_arguments)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        successor, successor_args = _parse_successor(parser)
        return cls(successor_args, successor)


@irdl_op_definition
class FlagBranchOp(HasRegisterConstraints, IRDLOperation):
    """Branch on a condition-code test of a single register.

    The ``predicate`` selects the zero/sign test applied to ``rs``. The ``then``
    block is taken when the test holds; ``else`` is the fall-through successor
    and must immediately follow the parent block in its region. Lowers to a
    ``test``/``cmp`` against zero followed by the matching flag jump at flat q1.

    .. code-block:: mlir

        q1_cf.flag_branch eqz %rs : q1.reg, ^then(...), ^else(...)
    """

    name = "q1_cf.flag_branch"

    predicate = prop_def(FlagPredicateAttr)
    rs = operand_def(IntRegisterType)
    then_arguments = var_operand_def(IntRegisterType)
    else_arguments = var_operand_def(IntRegisterType)

    irdl_options = (AttrSizedOperandSegments(),)

    then_block = successor_def()
    else_block = successor_def()

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        predicate: FlagPredicate | FlagPredicateAttr,
        rs: SSAValue | Operation,
        then_arguments: Sequence[SSAValue | Operation],
        else_arguments: Sequence[SSAValue | Operation],
        then_block: Successor,
        else_block: Successor,
    ):
        if isinstance(predicate, FlagPredicate):
            predicate = FlagPredicateAttr(predicate)
        super().__init__(
            operands=[rs, then_arguments, else_arguments],
            successors=(then_block, else_block),
            properties={"predicate": predicate},
        )

    def verify_(self) -> None:
        _verify_distinct_successors(
            self.name, self.then_block, self.else_block, "then", "else"
        )
        _verify_successor_args(self.then_arguments, self.then_block.args, self.name, "then")
        _verify_successor_args(self.else_arguments, self.else_block.args, self.name, "else")
        _verify_fall_through(self, self.else_block)

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(self.operands, [], [])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_string(self.predicate.data.value)
        printer.print_string(" ")
        _print_type_pair(printer, self.rs)
        printer.print_string(", ")
        _print_successor(printer, self.then_block, self.then_arguments)
        printer.print_string(", ")
        _print_successor(printer, self.else_block, self.else_arguments)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        predicate = parser.parse_str_enum(FlagPredicate)
        rs = _parse_type_pair(parser)
        parser.parse_punctuation(",")
        then_block, then_args = _parse_successor(parser)
        parser.parse_punctuation(",")
        else_block, else_args = _parse_successor(parser)
        return cls(predicate, rs, then_args, else_args, then_block, else_block)

    def const_evaluate(self, rs: int, bitwidth: int) -> bool:
        """Evaluate the flag predicate on a constant operand value.

        :param rs: Signless bit pattern of the tested register.
        :param bitwidth: Register width, used to reinterpret ``rs`` as signed or
            unsigned according to the predicate.
        :returns: ``True`` if the ``then`` edge is taken, ``False`` if the
            branch falls through to ``else``.
        """
        predicate = self.predicate.data
        if predicate is FlagPredicate.eqz:
            return to_unsigned(rs, bitwidth) == 0
        if predicate is FlagPredicate.nez:
            return to_unsigned(rs, bitwidth) != 0
        if predicate is FlagPredicate.ltz:
            return to_signed(rs, bitwidth) < 0
        return to_signed(rs, bitwidth) >= 0  # gez


@irdl_op_definition
class ComparisonBranchOp(HasRegisterConstraints, IRDLOperation):
    """Branch on a comparison of two operands.

    The ``predicate`` selects the signed/unsigned comparison applied to ``lhs``
    and ``rhs``, both register-typed SSA values. The ``then`` block is taken when
    the comparison holds; ``else`` is the fall-through successor and must
    immediately follow the parent block in its region. Lowers to a ``cmp``
    followed by the matching flag jump at flat q1.

    .. code-block:: mlir

        q1_cf.comparison_branch slt %lhs : q1.reg, %rhs : q1.reg, ^then(...), ^else(...)
    """

    name = "q1_cf.comparison_branch"

    predicate = prop_def(ComparisonPredicateAttr)
    lhs = operand_def(IntRegisterType)
    rhs = operand_def(IntRegisterType)
    then_arguments = var_operand_def(IntRegisterType)
    else_arguments = var_operand_def(IntRegisterType)

    irdl_options = (AttrSizedOperandSegments(),)

    then_block = successor_def()
    else_block = successor_def()

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        predicate: ComparisonPredicate | ComparisonPredicateAttr,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        then_arguments: Sequence[SSAValue | Operation],
        else_arguments: Sequence[SSAValue | Operation],
        then_block: Successor,
        else_block: Successor,
    ):
        if isinstance(predicate, ComparisonPredicate):
            predicate = ComparisonPredicateAttr(predicate)
        super().__init__(
            operands=[lhs, rhs, then_arguments, else_arguments],
            successors=(then_block, else_block),
            properties={"predicate": predicate},
        )

    def verify_(self) -> None:
        _verify_distinct_successors(
            self.name, self.then_block, self.else_block, "then", "else"
        )
        _verify_successor_args(self.then_arguments, self.then_block.args, self.name, "then")
        _verify_successor_args(self.else_arguments, self.else_block.args, self.name, "else")
        _verify_fall_through(self, self.else_block)

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(self.operands, [], [])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_string(self.predicate.data.value)
        printer.print_string(" ")
        _print_type_pair(printer, self.lhs)
        printer.print_string(", ")
        _print_type_pair(printer, self.rhs)
        printer.print_string(", ")
        _print_successor(printer, self.then_block, self.then_arguments)
        printer.print_string(", ")
        _print_successor(printer, self.else_block, self.else_arguments)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        predicate = parser.parse_str_enum(ComparisonPredicate)
        lhs = _parse_type_pair(parser)
        parser.parse_punctuation(",")
        rhs = _parse_type_pair(parser)
        parser.parse_punctuation(",")
        then_block, then_args = _parse_successor(parser)
        parser.parse_punctuation(",")
        else_block, else_args = _parse_successor(parser)
        return cls(predicate, lhs, rhs, then_args, else_args, then_block, else_block)

    def const_evaluate(self, lhs: int, rhs: int, bitwidth: int) -> bool:
        """Evaluate the comparison predicate on constant operand values.

        This is the hook a future CFG-canonicalisation pattern will call when both
        operands fold to constants: the branch is then rewritten to an
        unconditional ``q1_cf.jmp_branch`` to the statically taken edge (the dead
        edge and any blocks it solely reaches become unreachable and are pruned).
        It exists ahead of that pattern so the constant-folding semantics live with
        the op that defines them; the rewrite itself is tracked separately.

        :param lhs: Signless bit pattern of the left comparison operand.
        :param rhs: Signless bit pattern of the right comparison operand.
        :param bitwidth: Register width, used to reinterpret both operands as
            signed or unsigned according to the predicate.
        :returns: ``True`` if the ``then`` edge is taken, ``False`` if the
            branch falls through to ``else``.
        """
        unsigned_lhs, unsigned_rhs = to_unsigned(lhs, bitwidth), to_unsigned(rhs, bitwidth)
        signed_lhs, signed_rhs = to_signed(lhs, bitwidth), to_signed(rhs, bitwidth)
        match self.predicate.data:
            case ComparisonPredicate.eq:
                return unsigned_lhs == unsigned_rhs
            case ComparisonPredicate.ne:
                return unsigned_lhs != unsigned_rhs
            case ComparisonPredicate.slt:
                return signed_lhs < signed_rhs
            case ComparisonPredicate.sle:
                return signed_lhs <= signed_rhs
            case ComparisonPredicate.sgt:
                return signed_lhs > signed_rhs
            case ComparisonPredicate.sge:
                return signed_lhs >= signed_rhs
            case ComparisonPredicate.ult:
                return unsigned_lhs < unsigned_rhs
            case ComparisonPredicate.ule:
                return unsigned_lhs <= unsigned_rhs
            case ComparisonPredicate.ugt:
                return unsigned_lhs > unsigned_rhs
            case ComparisonPredicate.uge:
                return unsigned_lhs >= unsigned_rhs
            case _:
                raise ValueError(f"Unhandled comparison predicate: {self.predicate.data}")


@irdl_op_definition
class LoopBranchOp(HasRegisterConstraints, IRDLOperation):
    """Decrement ``counter`` and branch to ``body`` while non-zero; else ``exit``.

    Lowers to ``q1.ri.loop`` / ``q1.rr.loop`` at flat q1.

    .. code-block:: mlir

        q1_cf.loop_branch %counter : q1.reg, ^body(%a : q1.reg, ...), ^exit(%b : q1.reg, ...)
    """

    name = "q1_cf.loop_branch"

    counter = operand_def(IntRegisterType)
    body_arguments = var_operand_def(IntRegisterType)
    exit_arguments = var_operand_def(IntRegisterType)

    irdl_options = (AttrSizedOperandSegments(),)

    body_block = successor_def()
    exit_block = successor_def()

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        counter: SSAValue | Operation,
        body_arguments: Sequence[SSAValue | Operation],
        exit_arguments: Sequence[SSAValue | Operation],
        body_block: Successor,
        exit_block: Successor,
    ):
        super().__init__(
            operands=[counter, body_arguments, exit_arguments],
            successors=(body_block, exit_block),
        )

    def verify_(self) -> None:
        _verify_distinct_successors(
            self.name, self.body_block, self.exit_block, "body", "exit"
        )
        _verify_successor_args(self.body_arguments, self.body_block.args, self.name, "body")
        _verify_successor_args(self.exit_arguments, self.exit_block.args, self.name, "exit")

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(self.operands, [], [])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        _print_type_pair(printer, self.counter)
        printer.print_string(", ")
        _print_successor(printer, self.body_block, self.body_arguments)
        printer.print_string(", ")
        _print_successor(printer, self.exit_block, self.exit_arguments)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        counter = _parse_type_pair(parser)
        parser.parse_punctuation(",")
        body_block, body_args = _parse_successor(parser)
        parser.parse_punctuation(",")
        exit_block, exit_args = _parse_successor(parser)
        return cls(counter, body_args, exit_args, body_block, exit_block)
