# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""``q1_scf`` dialect operations.

Structured control flow for a single QBlox Q1 sequencer. The dialect sits above
``q1_cf``, the target-specific block CFG, and below the pulse level.

It defines one conditional and two loop containers together with two terminators:

- :class:`IfOp` is the structured conditional used for single-sequencer feedforward.
- :class:`WhileOp` is the general loop with an explicit ``before``/``after`` split.
- :class:`ForOp` is the counted loop whose induction value is the remaining count.
- :class:`ConditionOp` terminates the ``before`` region of a :class:`WhileOp`.
- :class:`YieldOp` terminates :class:`ForOp`, the ``after`` region of a
  :class:`WhileOp`, and each region of an :class:`IfOp`.

A condition carries a predicate and its register operands in the same shape as
``q1_cf``. The Q1 ISA provides no instruction that reduces a comparison to a
boolean, so a relational guard is represented as register operands paired with a
predicate that lowering regenerates adjacent to the branch reading the ALU flags.
Every predicate operand is a ``q1.reg`` and the dialect defines no ``i1`` operands.
A measurement outcome therefore reaches ``q1_scf`` only as a ``q1.reg`` dequeued
through ``q1_linq``, never through ``q1_trigger``.

Reference: https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html
"""

from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import Self
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    base,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasAncestor,
    HasParent,
    IsTerminator,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1.ir.reg_desc import IntRegisterType
from qat.experimental.dialect.q1_cf.ir.attrs import (
    ComparisonPredicate,
    ComparisonPredicateAttr,
    FlagPredicate,
    FlagPredicateAttr,
)
from qat.experimental.dialect.q1_scf.ir.attrs import IterDomainAttr
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp

# A predicate is either a single-operand flag test or a two-operand comparison.
PredicateAttr = FlagPredicateAttr | ComparisonPredicateAttr
_PREDICATE_CONSTRAINT = base(FlagPredicateAttr) | base(ComparisonPredicateAttr)


def _coerce_predicate(
    predicate: PredicateAttr | FlagPredicate | ComparisonPredicate,
) -> PredicateAttr:
    """Wrap a bare predicate enum member in its attribute, passing attrs through."""
    if isinstance(predicate, FlagPredicate):
        return FlagPredicateAttr(predicate)
    if isinstance(predicate, ComparisonPredicate):
        return ComparisonPredicateAttr(predicate)
    return predicate


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


def _print_predicate_operands(
    printer: Printer, predicate: PredicateAttr, operands: Sequence[SSAValue]
) -> None:
    """Print ``<predicate> %a[ , %b]`` for a predicate and its operands."""
    printer.print_string(predicate.data.value)
    printer.print_string(" ")
    printer.print_list(operands, lambda value: _print_type_pair(printer, value))


def _parse_predicate(parser: Parser) -> PredicateAttr:
    """Parse a bare predicate keyword into its attribute wrapper.

    The flag and comparison predicate spellings are disjoint, so the keyword alone
    selects the predicate family.

    :returns: A :class:`FlagPredicateAttr` or :class:`ComparisonPredicateAttr`.
    """
    keyword = parser.parse_optional_identifier()
    if keyword is not None:
        flag = FlagPredicateAttr.enum_type.__members__.get(keyword)
        if flag is not None:
            return FlagPredicateAttr(flag)
        comparison = ComparisonPredicateAttr.enum_type.__members__.get(keyword)
        if comparison is not None:
            return ComparisonPredicateAttr(comparison)
    return parser.raise_error("expected a q1_scf flag or comparison predicate")


def _parse_predicate_operands(parser: Parser, predicate: PredicateAttr) -> list[SSAValue]:
    """Parse the register operands that follow a predicate keyword."""
    operands = [_parse_type_pair(parser)]
    for _ in range(1 if isinstance(predicate, ComparisonPredicateAttr) else 0):
        parser.parse_punctuation(",")
        operands.append(_parse_type_pair(parser))
    return operands


def _reject_q1_cf(op: Operation, region: Region, label: str) -> None:
    """Reject any ``q1_cf`` operation directly inside ``region``.

    ``q1_scf`` and ``q1_cf`` never share a region because structured control flow
    and target-specific CFG are separate layers. Only the direct block-level ops of
    the region are inspected. Nested ``q1_scf`` containers police their own regions,
    and non-``q1_scf`` nested ops such as ``pulse`` form regions of their own.

    :raises VerifyException: If a ``q1_cf`` operation is found in the region.
    """
    for block in region.blocks:
        for child in block.ops:
            if child.name.startswith("q1_cf."):
                raise VerifyException(
                    f"{op.name}: {label} region must not contain q1_cf operations"
                    f" (found '{child.name}'). q1_scf and q1_cf never share a region"
                )


def _yield_terminator(region: Region) -> YieldOp | None:
    """Return the region's :class:`YieldOp` terminator, or ``None``.

    Returns ``None`` unless the region is a single block terminated by a
    :class:`YieldOp`, so a malformed empty or multi-block region fails through the
    caller's verifier rather than raising from :attr:`Region.block`.
    """
    if len(region.blocks) != 1:
        return None
    last_op = region.block.last_op
    return last_op if isinstance(last_op, YieldOp) else None


@irdl_op_definition
class YieldOp(IRDLOperation):
    """Terminate a structured region, forwarding register values to its parent.

    Terminates the body of a :class:`ForOp`, the ``after`` region of a
    :class:`WhileOp`, and each region of an :class:`IfOp`.

    .. code-block:: mlir

        q1_scf.yield %a, %b : q1.reg, q1.reg
    """

    name = "q1_scf.yield"

    arguments = var_operand_def(IntRegisterType)

    assembly_format = "attr-dict ($arguments^ `:` type($arguments))?"

    traits = lazy_traits_def(lambda: (IsTerminator(), HasParent(ForOp, WhileOp, IfOp)))

    def __init__(self, *arguments: SSAValue | Operation):
        super().__init__(operands=[arguments])


@irdl_op_definition
class ConditionOp(IRDLOperation):
    """Terminate the ``before`` region of a :class:`WhileOp`.

    The loop continues while ``predicate`` holds over its register operands, and
    the ``forward_args`` are threaded to the ``after`` region or, on exit, to the
    loop results.

    .. code-block:: mlir

        q1_scf.condition slt %x : q1.reg, %n : q1.reg (%acc : q1.reg)
    """

    name = "q1_scf.condition"

    predicate = prop_def(_PREDICATE_CONSTRAINT)
    predicate_args = var_operand_def(IntRegisterType)
    forward_args = var_operand_def(IntRegisterType)

    irdl_options = (AttrSizedOperandSegments(),)

    traits = lazy_traits_def(lambda: (IsTerminator(), HasParent(WhileOp)))

    def __init__(
        self,
        predicate: PredicateAttr | FlagPredicate | ComparisonPredicate,
        predicate_args: Sequence[SSAValue | Operation],
        forward_args: Sequence[SSAValue | Operation],
    ):
        super().__init__(
            operands=[predicate_args, forward_args],
            properties={"predicate": _coerce_predicate(predicate)},
        )

    def verify_(self) -> None:
        expected = 2 if isinstance(self.predicate, ComparisonPredicateAttr) else 1
        if len(self.predicate_args) != expected:
            raise VerifyException(
                f"{self.name}: {self.predicate.data.value} predicate expects"
                f" {expected} operand(s), but {len(self.predicate_args)} were supplied"
            )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        _print_predicate_operands(printer, self.predicate, self.predicate_args)
        printer.print_string(" (")
        printer.print_list(
            self.forward_args, lambda value: _print_type_pair(printer, value)
        )
        printer.print_string(")")

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        predicate = _parse_predicate(parser)
        predicate_args = _parse_predicate_operands(parser, predicate)
        forward_args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: _parse_type_pair(parser)
        )
        return cls(predicate, predicate_args, forward_args)


@irdl_op_definition
class WhileOp(IRDLOperation):
    """General structured loop with a ``before``/``after`` split.

    The ``before`` region computes and tests the loop condition and is terminated
    by :class:`ConditionOp`. The ``after`` region is the loop body, terminated by
    :class:`YieldOp`, whose yielded values feed back to the ``before`` block
    arguments. Any structured loop is expressible in this form.

    .. code-block:: mlir

        q1_scf.while (%acc : q1.reg) -> (q1.reg) { ... } do { ... }
    """

    name = "q1_scf.while"

    init_args = var_operand_def(IntRegisterType)
    res = var_result_def(IntRegisterType)
    before_region = region_def()
    after_region = region_def()

    traits = traits_def(RecursiveMemoryEffect(), HasAncestor(SequenceOp))

    def __init__(
        self,
        init_args: Sequence[SSAValue | Operation],
        result_types: Sequence[Attribute],
        before_region: Region | Sequence[Operation] | Sequence[Block],
        after_region: Region | Sequence[Operation] | Sequence[Block],
    ):
        super().__init__(
            operands=[init_args],
            result_types=[result_types],
            regions=[before_region, after_region],
        )

    def verify_(self) -> None:
        _reject_q1_cf(self, self.before_region, "before")
        _reject_q1_cf(self, self.after_region, "after")

        for region, label in (
            (self.before_region, "before"),
            (self.after_region, "after"),
        ):
            if len(region.blocks) != 1:
                raise VerifyException(f"{self.name}: {label} region must be a single block")

        before_args = self.before_region.block.args
        if len(before_args) != len(self.init_args):
            raise VerifyException(
                f"{self.name}: before region expects {len(self.init_args)} block"
                f" argument(s), but got {len(before_args)}"
            )
        for position, (block_arg, init) in enumerate(
            zip(before_args, self.init_args, strict=True)
        ):
            if block_arg.type != init.type:
                raise VerifyException(
                    f"{self.name}: before block argument {position} type"
                    f" {block_arg.type!r} does not match init operand type {init.type!r}"
                )

        condition = self.before_region.block.last_op
        if not isinstance(condition, ConditionOp):
            raise VerifyException(
                f"{self.name}: before region must be terminated by q1_scf.condition"
            )
        if tuple(v.type for v in condition.forward_args) != tuple(v.type for v in self.res):
            raise VerifyException(
                f"{self.name}: results must match the q1_scf.condition forwarded arguments"
            )
        if tuple(v.type for v in self.after_region.block.args) != tuple(
            v.type for v in condition.forward_args
        ):
            raise VerifyException(
                f"{self.name}: after block arguments must match the q1_scf.condition"
                " forwarded arguments"
            )

        after = _yield_terminator(self.after_region)
        if after is None:
            raise VerifyException(
                f"{self.name}: after region must be terminated by q1_scf.yield"
            )
        if tuple(v.type for v in after.arguments) != tuple(v.type for v in before_args):
            raise VerifyException(
                f"{self.name}: after region must yield the before block argument"
                f" types (the loop feedback edge)"
            )

    def print(self, printer: Printer) -> None:
        printer.print_string(" (")
        printer.print_list(self.init_args, lambda value: _print_type_pair(printer, value))
        printer.print_string(")")
        if self.res:
            printer.print_string(" -> (")
            printer.print_list(tuple(v.type for v in self.res), printer.print_attribute)
            printer.print_string(")")
        printer.print_string(" ")
        printer.print_region(self.before_region)
        printer.print_string(" do ")
        printer.print_region(self.after_region)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        init_args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: _parse_type_pair(parser)
        )
        result_types: list[Attribute] = []
        if parser.parse_optional_punctuation("->"):
            result_types = parser.parse_comma_separated_list(
                parser.Delimiter.PAREN, parser.parse_type
            )
        before_region = parser.parse_region()
        parser.parse_keyword("do")
        after_region = parser.parse_region()
        return cls(init_args, result_types, before_region, after_region)


@irdl_op_definition
class ForOp(IRDLOperation):
    """Counted loop whose induction value is the remaining count.

    ``iter_count`` counts down to zero, mapping one-to-one onto the native Q1
    decrement-and-branch loop and onto ``q1_cf.loop_branch``. The body is a single
    block terminated by :class:`YieldOp`, and its entry block arguments are the
    induction counter followed by the carried values. A :class:`ForOp` is reducible
    to a :class:`WhileOp`. The optional ``iter_domain`` records the linear iteration
    the counted loop realises.

    .. code-block:: mlir

        q1_scf.for %n : q1.reg iter_args(%acc : q1.reg) { ... }
    """

    name = "q1_scf.for"

    iter_count = operand_def(IntRegisterType)
    iter_args = var_operand_def(IntRegisterType)
    res = var_result_def(IntRegisterType)
    iter_domain = opt_prop_def(IterDomainAttr)

    body = region_def("single_block")

    traits = traits_def(
        SingleBlockImplicitTerminator(YieldOp),
        RecursiveMemoryEffect(),
        HasAncestor(SequenceOp),
    )

    def __init__(
        self,
        iter_count: SSAValue | Operation,
        iter_args: Sequence[SSAValue | Operation],
        body: Region | Sequence[Operation] | Sequence[Block] | Block,
        iter_domain: IterDomainAttr | None = None,
    ):
        if isinstance(body, Block):
            body = [body]
        super().__init__(
            operands=[iter_count, iter_args],
            result_types=[[SSAValue.get(arg).type for arg in iter_args]],
            regions=[body],
            properties=({"iter_domain": iter_domain} if iter_domain is not None else {}),
        )

    def verify_(self) -> None:
        _reject_q1_cf(self, self.body, "body")

        block_args = self.body.block.args
        if not block_args:
            raise VerifyException(
                f"{self.name}: body block must have the induction counter as its"
                f" first argument"
            )
        induction, *carried = block_args
        if not isinstance(induction.type, IntRegisterType):
            raise VerifyException(
                f"{self.name}: body induction counter must be a q1.reg, got"
                f" {induction.type!r}"
            )
        if induction.type != self.iter_count.type:
            raise VerifyException(
                f"{self.name}: body induction counter type {induction.type!r} does not"
                f" match iter_count type {self.iter_count.type!r}"
            )
        if len(carried) != len(self.iter_args):
            raise VerifyException(
                f"{self.name}: body block expects {len(self.iter_args) + 1} argument(s)"
                f" (induction counter plus carried values), but got {len(block_args)}"
            )
        for position, (block_arg, iter_arg) in enumerate(
            zip(carried, self.iter_args, strict=True)
        ):
            if block_arg.type != iter_arg.type:
                raise VerifyException(
                    f"{self.name}: carried block argument {position} type"
                    f" {block_arg.type!r} does not match iter_arg type {iter_arg.type!r}"
                )

        yield_op = _yield_terminator(self.body)
        if yield_op is None:
            raise VerifyException(f"{self.name}: body must be terminated by q1_scf.yield")
        carried_types = tuple(v.type for v in self.iter_args)
        if tuple(v.type for v in yield_op.arguments) != carried_types:
            raise VerifyException(
                f"{self.name}: body terminator operands must match the carried value types"
            )

    def allocate_registers(self) -> None:
        """Entry point for structured register allocation over this loop.

        Register allocation is implemented by COMPILER-911. This dialect exposes only the
        hook.

        :raises NotImplementedError: Always, until COMPILER-911 supplies the pass.
        """
        raise NotImplementedError(
            "q1_scf.for register allocation is provided by COMPILER-911"
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        _print_type_pair(printer, self.iter_count)
        if self.iter_args:
            printer.print_string(" iter_args(")
            printer.print_list(
                self.iter_args, lambda value: _print_type_pair(printer, value)
            )
            printer.print_string(")")
        if self.iter_domain is not None:
            printer.print_string(" iter ")
            printer.print_attribute(self.iter_domain)
        printer.print_string(" ")
        printer.print_region(self.body)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        iter_count = _parse_type_pair(parser)
        iter_args: list[SSAValue] = []
        if parser.parse_optional_keyword("iter_args"):
            iter_args = parser.parse_comma_separated_list(
                parser.Delimiter.PAREN, lambda: _parse_type_pair(parser)
            )
        iter_domain: IterDomainAttr | None = None
        if parser.parse_optional_keyword("iter"):
            attribute = parser.parse_attribute()
            if not isinstance(attribute, IterDomainAttr):
                parser.raise_error(
                    f"expected a {IterDomainAttr.name} attribute after 'iter'"
                )
            iter_domain = attribute
        body = parser.parse_region()
        return cls(iter_count, iter_args, body, iter_domain)


@irdl_op_definition
class IfOp(IRDLOperation):
    """Structured conditional for single-sequencer feedforward.

    Evaluates ``predicate`` over its register operands, runs the ``then`` region
    or the optional ``else`` region, and returns their yielded values as its
    results. Both regions are terminated by :class:`YieldOp`. Feedforward here
    denotes intra-sequencer conditional execution, distinct from a loop or a
    cross-sequencer message.

    .. code-block:: mlir

        q1_scf.if nez %flag : q1.reg -> (q1.reg) { ... } else { ... }
    """

    name = "q1_scf.if"

    predicate = prop_def(_PREDICATE_CONSTRAINT)
    predicate_args = var_operand_def(IntRegisterType)
    output = var_result_def(IntRegisterType)

    then_region = region_def()
    else_region = region_def()

    traits = traits_def(RecursiveMemoryEffect(), HasAncestor(SequenceOp))

    def __init__(
        self,
        predicate: PredicateAttr | FlagPredicate | ComparisonPredicate,
        predicate_args: Sequence[SSAValue | Operation],
        result_types: Sequence[Attribute],
        then_region: Region | Sequence[Operation] | Sequence[Block],
        else_region: Region | Sequence[Operation] | Sequence[Block] | None = None,
    ):
        if else_region is None:
            else_region = Region()
        super().__init__(
            operands=[predicate_args],
            result_types=[result_types],
            regions=[then_region, else_region],
            properties={"predicate": _coerce_predicate(predicate)},
        )

    def verify_(self) -> None:
        _reject_q1_cf(self, self.then_region, "then")
        _reject_q1_cf(self, self.else_region, "else")

        if len(self.else_region.blocks) > 1:
            raise VerifyException(f"{self.name}: else region must be a single block")
        if len(self.then_region.blocks) != 1:
            raise VerifyException(f"{self.name}: then region must be a single block")

        expected = 2 if isinstance(self.predicate, ComparisonPredicateAttr) else 1
        if len(self.predicate_args) != expected:
            raise VerifyException(
                f"{self.name}: {self.predicate.data.value} predicate expects"
                f" {expected} operand(s), but {len(self.predicate_args)} were supplied"
            )

        then_yield = _yield_terminator(self.then_region)
        if then_yield is None:
            raise VerifyException(
                f"{self.name}: then region must be terminated by q1_scf.yield"
            )
        then_types = tuple(v.type for v in then_yield.arguments)
        if then_types != tuple(v.type for v in self.output):
            raise VerifyException(
                f"{self.name}: then region yielded types do not match the results"
            )

        has_else = bool(self.else_region.blocks)
        if self.output and not has_else:
            raise VerifyException(
                f"{self.name}: an if with results must have an else region"
            )
        if has_else:
            else_yield = _yield_terminator(self.else_region)
            if else_yield is None:
                raise VerifyException(
                    f"{self.name}: else region must be terminated by q1_scf.yield"
                )
            if tuple(v.type for v in else_yield.arguments) != then_types:
                raise VerifyException(
                    f"{self.name}: then and else regions must yield matching types"
                )

    def allocate_registers(self) -> None:
        """Entry point for structured register allocation over this conditional.

        Register allocation is implemented by COMPILER-911. This dialect exposes only the
        hook.

        :raises NotImplementedError: Always, until COMPILER-911 supplies the pass.
        """
        raise NotImplementedError(
            "q1_scf.if register allocation is provided by COMPILER-911"
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        _print_predicate_operands(printer, self.predicate, self.predicate_args)
        if self.output:
            printer.print_string(" -> (")
            printer.print_list(tuple(v.type for v in self.output), printer.print_attribute)
            printer.print_string(")")
        printer.print_string(" ")
        printer.print_region(self.then_region)
        if self.else_region.blocks:
            printer.print_string(" else ")
            printer.print_region(self.else_region)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        predicate = _parse_predicate(parser)
        predicate_args = _parse_predicate_operands(parser, predicate)
        result_types: list[Attribute] = []
        if parser.parse_optional_punctuation("->"):
            result_types = parser.parse_comma_separated_list(
                parser.Delimiter.PAREN, parser.parse_type
            )
        then_region = parser.parse_region()
        else_region = Region()
        if parser.parse_optional_keyword("else"):
            else_region = parser.parse_region()
        return cls(predicate, predicate_args, result_types, then_region, else_region)
