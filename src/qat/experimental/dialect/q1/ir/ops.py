# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Concrete QBlox Q1 ISA operations.

This module mirrors the single-sequencer assembly instruction set as MLIR
operations. Operation names use the pattern `q1.<signature>.<mnemonic>`
where the signature encodes operand/result shapes. For example `rir`
encodes the variant register-immediate-register.

Each operation inherits a reusable format class from
`qat.experimental.dialect.q1.ir.abstract_ops`.

Flag mnemonic glossary used in op docstrings:

* `ZF`: zero flag
* `NF`: negative/sign flag
* `CF`: carry flag
* `OF`: overflow flag
"""

from xdsl.backend.assembly_printer import AssemblyPrinter
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import irdl_op_definition, prop_def, traits_def
from xdsl.traits import Commutative, IsTerminator, Pure

from qat.experimental.dialect.q1.ir.abstract_ops import (
    AssemblyInstructionArg,
    IIIIIOperation,
    IIIIOperation,
    IIIOperation,
    IIOperation,
    IOperation,
    IRdOperation,
    IRsIIOperation,
    IRsIOperation,
    IRsIRsIOperation,
    IRsRsRsIOperation,
    NullaryOperation,
    Q1AsmOperation,
    RdIOperation,
    RdRdOperation,
    RdRsOperation,
    RsIIOperation,
    RsIOperation,
    RsIRdOperation,
    RsIRsOperation,
    RsOperation,
    RsRdOperation,
    RsRsIOperation,
    RsRsOperation,
    RsRsRdOperation,
    RsRsRsIOperation,
)
from qat.experimental.dialect.q1.ir.attrs import LabelAttr
from qat.experimental.dialect.q1.ir.imm_desc import UI32
from qat.experimental.dialect.q1.ir.reg_desc import IntRegisterType

# region Core Instructions


@irdl_op_definition
class LabelOp(Q1AsmOperation):
    """Emit a textual label used as a branch or jump target.

    Example:

        func:         # Label reference that jumping operations can refer to
        add R1, R2, R3
    """

    name = "q1.x.label"

    reference = prop_def(LabelAttr)

    def __init__(
        self,
        reference: str | LabelAttr,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(reference, str):
            reference = LabelAttr(reference)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(properties={"reference": reference, "comment": comment})

    def assembly_mnemonic(self) -> str:
        return ""

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return ()

    def assembly_line(self) -> str | None:
        if self.comment is None:
            return f"{self.reference.data}:"

        return f"{self.reference.data}: # {self.comment.data}"


@irdl_op_definition
class DefDirectiveOp(Q1AsmOperation):
    """Define an alias in Q1 assembly via `.DEF <name> <value>`.

    Example:

        .DEF SQG_TIME 100         # Time in ns for single qubit gate
        wait $SQG_TIME
    """

    name = "q1.xx.DEF"

    alias = prop_def(StringAttr)
    value = prop_def(StringAttr)

    def __init__(
        self,
        alias: str | StringAttr,
        value: str | StringAttr,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(alias, str):
            alias = StringAttr(alias)

        if isinstance(value, str):
            value = StringAttr(value)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            properties={
                "alias": alias,
                "value": value,
                "comment": comment,
            }
        )

    def assembly_mnemonic(self) -> str:
        return f".{super().assembly_mnemonic()}"

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.alias, self.value

    def assembly_line(self) -> str | None:
        arg_str = f"{self.alias.data} {self.value.data}"
        return AssemblyPrinter.assembly_line(
            self.assembly_mnemonic(), arg_str, self.comment, is_indented=False
        )


@irdl_op_definition
class IllegalOp(NullaryOperation):
    """Invalid instruction that halts the sequencer with an illegal-instruction error."""

    name = "q1..illegal"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class StopIOp(IOperation):
    """Stop the sequencer using an immediate stop code."""

    name = "q1.i.stop"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        status: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(status, comment=comment)

    @property
    def status(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class StopROp(RsOperation[IntRegisterType]):
    """Stop the sequencer using a stop code read from a register."""

    name = "q1.r.stop"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        status: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(status, comment=comment)

    @property
    def status(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class StopOp(NullaryOperation):
    """Stop the sequencer with default stop code `0` (pseudo-op for `stop 0`)."""

    name = "q1..stop"

    traits = traits_def(IsTerminator())

    def __init__(self, comment: str | StringAttr | None = None):
        super().__init__(comment=comment)


@irdl_op_definition
class NopOp(NullaryOperation):
    """No-op instruction that waits one Q1 core cycle (4 ns) without changing state."""

    name = "q1..nop"

    traits = traits_def(Pure())


# endregion

# region Jump Instructions


@irdl_op_definition
class JmpIOp(IOperation):
    """Unconditional jump to an immediate address."""

    name = "q1.i.jmp"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JmpROp(RsOperation[IntRegisterType]):
    """Unconditional jump to an address stored in a register."""

    name = "q1.r.jmp"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JzIOp(IOperation):
    """Jump if `ZF == 1` to an immediate address."""

    name = "q1.i.jz"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JzROp(RsOperation[IntRegisterType]):
    """Jump if `ZF == 1` to a register address."""

    name = "q1.r.jz"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JnzIOp(IOperation):
    """Jump if `ZF == 0` to an immediate address."""

    name = "q1.i.jnz"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JnzROp(RsOperation[IntRegisterType]):
    """Jump if `ZF == 0` to a register address."""

    name = "q1.r.jnz"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JoIOp(IOperation):
    """Jump if `OF == 1` to an immediate address."""

    name = "q1.i.jo"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JoROp(RsOperation[IntRegisterType]):
    """Jump if `OF == 1` to a register address."""

    name = "q1.r.jo"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JnoIOp(IOperation):
    """Jump if `OF == 0` to an immediate address."""

    name = "q1.i.jno"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JnoROp(RsOperation[IntRegisterType]):
    """Jump if `OF == 0` to a register address."""

    name = "q1.r.jno"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JsIOp(IOperation):
    """Jump if `NF == 1` to an immediate address."""

    name = "q1.i.js"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JsROp(RsOperation[IntRegisterType]):
    """Jump if `NF == 1` to a register address."""

    name = "q1.r.js"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JnsIOp(IOperation):
    """Jump if `NF == 0` to an immediate address."""

    name = "q1.i.jns"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JnsROp(RsOperation[IntRegisterType]):
    """Jump if `NF == 0` to a register address."""

    name = "q1.r.jns"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JgIOp(IOperation):
    """Jump if signed `a > b` condition holds (`ZF == 0` and `NF == OF`)."""

    name = "q1.i.jg"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JgROp(RsOperation[IntRegisterType]):
    """Jump if signed `a > b` condition holds (`ZF == 0` and `NF == OF`)."""

    name = "q1.r.jg"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JlIOp(IOperation):
    """Jump if signed `a < b` condition holds (`NF != OF`)."""

    name = "q1.i.jl"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JlROp(RsOperation[IntRegisterType]):
    """Jump if signed `a < b` condition holds (`NF != OF`)."""

    name = "q1.r.jl"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JleIOp(IOperation):
    """Jump if signed `a <= b` condition holds (`ZF == 1` or `NF != OF`)."""

    name = "q1.i.jle"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JleROp(RsOperation[IntRegisterType]):
    """Jump if signed `a <= b` condition holds (`ZF == 1` or `NF != OF`)."""

    name = "q1.r.jle"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JaIOp(IOperation):
    """Jump if unsigned `a > b` condition holds (`ZF == 0` and `CF == 0`)."""

    name = "q1.i.ja"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JaROp(RsOperation[IntRegisterType]):
    """Jump if unsigned `a > b` condition holds (`ZF == 0` and `CF == 0`)."""

    name = "q1.r.ja"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JaeIOp(IOperation):
    """Jump if unsigned `a >= b` condition holds (`CF == 0`)."""

    name = "q1.i.jae"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JaeROp(RsOperation[IntRegisterType]):
    """Jump if unsigned `a >= b` condition holds (`CF == 0`)."""

    name = "q1.r.jae"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JbIOp(IOperation):
    """Jump if unsigned `a < b` condition holds (`CF == 1`)."""

    name = "q1.i.jb"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JbROp(RsOperation[IntRegisterType]):
    """Jump if unsigned `a < b` condition holds (`CF == 1`)."""

    name = "q1.r.jb"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JbeIOp(IOperation):
    """Jump if unsigned `a <= b` condition holds (`ZF == 1` or `CF == 1`)."""

    name = "q1.i.jbe"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JbeROp(RsOperation[IntRegisterType]):
    """Jump if unsigned `a <= b` condition holds (`ZF == 1` or `CF == 1`)."""

    name = "q1.r.jbe"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(address, comment=comment)

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JgeRIIOp(RsIIOperation[IntRegisterType]):
    """Deprecated legacy jump variant for unsigned `a >= b` with immediate address."""

    name = "q1.rii.jge"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: int | IntegerAttr[UI32],
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, address, comment=comment)

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def address(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class JgeRIROp(RsIRsOperation[IntRegisterType]):
    """Deprecated legacy jump variant for unsigned `a >= b` with register address."""

    name = "q1.rir.jge"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: int | IntegerAttr[UI32],
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, address, comment=comment)

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def address(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class JltRIIOp(RsIIOperation[IntRegisterType]):
    """Deprecated legacy jump variant for unsigned `a < b` with immediate address."""

    name = "q1.rii.jlt"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: int | IntegerAttr[UI32],
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, address, comment=comment)

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def address(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class JltRIROp(RsIRsOperation[IntRegisterType]):
    """Deprecated legacy jump variant for unsigned `a < b` with register address."""

    name = "q1.rir.jlt"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: int | IntegerAttr[UI32],
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, address, comment=comment)

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def address(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class LoopRIOp(RdIOperation[IntRegisterType]):
    """Deprecated legacy loop: decrement source and jump while the result is non-zero."""

    name = "q1.ri.loop"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        source: IntRegisterType,
        address: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(source, address, comment=comment)

    @property
    def source(self):
        """Semantic alias for the generic rd field."""

        return self.rd

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class LoopRROp(RdRsOperation[IntRegisterType]):
    """Deprecated legacy loop: decrement source and jump while the result is non-zero."""

    name = "q1.rr.loop"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        source: IntRegisterType,
        address: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(source, address, comment=comment)

    @property
    def source(self):
        """Semantic alias for the generic rd field."""

        return self.rd

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


# endregion

# region Arithmetic Instructions


@irdl_op_definition
class MoveIROp(IRdOperation[IntRegisterType]):
    """Copy an immediate source value into a destination register."""

    name = "q1.ir.move"

    traits = traits_def(Pure())

    def __init__(
        self,
        source: int | IntegerAttr[UI32],
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(source, rd, comment=comment)

    @property
    def source(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class MoveRROp(RsRdOperation[IntRegisterType]):
    """Copy a register source value into a destination register."""

    name = "q1.rr.move"

    traits = traits_def(Pure())

    def __init__(
        self,
        source: Operation | SSAValue,
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(source, rd, comment=comment)

    @property
    def source(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class NotIROp(IRdOperation[IntRegisterType]):
    """Bitwise invert an immediate source value and write the result to destination."""

    name = "q1.ir.not"

    traits = traits_def(Pure())

    def __init__(
        self,
        source: int | IntegerAttr[UI32],
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(source, rd, comment=comment)

    @property
    def source(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class NotRROp(RsRdOperation[IntRegisterType]):
    """Bitwise invert a register source value and write the result to destination."""

    name = "q1.rr.not"

    traits = traits_def(Pure())

    def __init__(
        self,
        source: Operation | SSAValue,
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(source, rd, comment=comment)

    @property
    def source(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class AddRIROp(RsIRdOperation[IntRegisterType]):
    """Add a register and immediate operand and store the result in destination."""

    name = "q1.rir.add"

    traits = traits_def(Pure())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: int | IntegerAttr[UI32],
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class AddRRROp(RsRsRdOperation[IntRegisterType]):
    """Add two register operands and store the result in destination."""

    name = "q1.rrr.add"

    traits = traits_def(Pure(), Commutative())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: Operation | SSAValue,
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class SubRIROp(RsIRdOperation[IntRegisterType]):
    """Subtract an immediate operand from a register operand into destination."""

    name = "q1.rir.sub"

    traits = traits_def(Pure())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: int | IntegerAttr[UI32],
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SubRRROp(RsRsRdOperation[IntRegisterType]):
    """Subtract a register operand from another register operand into destination."""

    name = "q1.rrr.sub"

    traits = traits_def(Pure())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: Operation | SSAValue,
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class AndRIROp(RsIRdOperation[IntRegisterType]):
    """Bitwise AND between register and immediate operands into destination."""

    name = "q1.rir.and"

    traits = traits_def(Pure())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: int | IntegerAttr[UI32],
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class AndRRROp(RsRsRdOperation[IntRegisterType]):
    """Bitwise AND between two register operands into destination."""

    name = "q1.rrr.and"

    traits = traits_def(Pure(), Commutative())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: Operation | SSAValue,
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class OrRIROp(RsIRdOperation[IntRegisterType]):
    """Bitwise OR between register and immediate operands into destination."""

    name = "q1.rir.or"

    traits = traits_def(Pure())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: int | IntegerAttr[UI32],
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class OrRRROp(RsRsRdOperation[IntRegisterType]):
    """Bitwise OR between two register operands into destination."""

    name = "q1.rrr.or"

    traits = traits_def(Pure(), Commutative())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: Operation | SSAValue,
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class XorRIROp(RsIRdOperation[IntRegisterType]):
    """Bitwise XOR between register and immediate operands into destination."""

    name = "q1.rir.xor"

    traits = traits_def(Pure())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: int | IntegerAttr[UI32],
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class XorRRROp(RsRsRdOperation[IntRegisterType]):
    """Bitwise XOR between two register operands into destination."""

    name = "q1.rrr.xor"

    traits = traits_def(Pure(), Commutative())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: Operation | SSAValue,
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class AslRIROp(RsIRdOperation[IntRegisterType]):
    """Arithmetic left shift by immediate bit-count into destination."""

    name = "q1.rir.asl"

    traits = traits_def(Pure())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: int | IntegerAttr[UI32],
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class AslRRROp(RsRsRdOperation[IntRegisterType]):
    """Arithmetic left shift by register bit-count into destination."""

    name = "q1.rrr.asl"

    traits = traits_def(Pure())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: Operation | SSAValue,
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class AsrRIROp(RsIRdOperation[IntRegisterType]):
    """Arithmetic right shift by immediate bit-count into destination."""

    name = "q1.rir.asr"

    traits = traits_def(Pure())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: int | IntegerAttr[UI32],
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class AsrRRROp(RsRsRdOperation[IntRegisterType]):
    """Arithmetic right shift by register bit-count into destination."""

    name = "q1.rrr.asr"

    traits = traits_def(Pure())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: Operation | SSAValue,
        rd: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(a, b, rd, comment=comment)

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


# endregion

# region Latched Instructions


@irdl_op_definition
class SetCondIIIIOp(IIIIOperation):
    """Configure conditional execution from trigger-network condition parameters."""

    name = "q1.iiii.set_cond"

    traits = traits_def()

    def __init__(
        self,
        cond_en: int | IntegerAttr[UI32],
        mask: int | IntegerAttr[UI32],
        op: int | IntegerAttr[UI32],
        else_cnt: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(cond_en, mask, op, else_cnt, comment=comment)

    @property
    def cond_en(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def mask(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def op(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3

    @property
    def else_cnt(self):
        """Semantic alias for the fourth generic imm field."""

        return self.imm4


@irdl_op_definition
class SetCondRRRIOp(RsRsRsIOperation[IntRegisterType]):
    """Configure conditional execution from register-based condition parameters."""

    name = "q1.rrri.set_cond"

    traits = traits_def()

    def __init__(
        self,
        cond_en: Operation | SSAValue,
        mask: Operation | SSAValue,
        op: Operation | SSAValue,
        else_cnt: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(cond_en, mask, op, else_cnt, comment=comment)

    @property
    def cond_en(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def mask(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def op(self):
        """Semantic alias for the third generic rs field."""

        return self.rs3

    @property
    def else_cnt(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetMrkIOp(IOperation):
    """Set marker output mask from an immediate source."""

    name = "q1.i.set_mrk"

    traits = traits_def()

    def __init__(
        self,
        mrk: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(mrk, comment=comment)

    @property
    def mrk(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetMrkROp(RsOperation[IntRegisterType]):
    """Set marker output mask from a register source."""

    name = "q1.r.set_mrk"

    traits = traits_def()

    def __init__(
        self,
        mrk: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(mrk, comment=comment)

    @property
    def mrk(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class SetFreqIOp(IOperation):
    """Set the latched NCO frequency from an immediate source."""

    name = "q1.i.set_freq"

    traits = traits_def()

    def __init__(
        self,
        nco_freq: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(nco_freq, comment=comment)

    @property
    def nco_freq(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetFreqROp(RsOperation[IntRegisterType]):
    """Set the latched NCO frequency from a register source."""

    name = "q1.r.set_freq"

    traits = traits_def()

    def __init__(
        self,
        nco_freq: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(nco_freq, comment=comment)

    @property
    def nco_freq(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class ResetPhOp(NullaryOperation):
    """Set the latched phase-reset flag to reset NCO phase accumulation."""

    name = "q1..reset_ph"

    traits = traits_def()


@irdl_op_definition
class SetPhIOp(IOperation):
    """Set the latched NCO phase-offset source from an immediate."""

    name = "q1.i.set_ph"

    traits = traits_def()

    def __init__(
        self,
        nco_po: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(nco_po, comment=comment)

    @property
    def nco_po(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetPhROp(RsOperation[IntRegisterType]):
    """Set the latched NCO phase-offset source from a register."""

    name = "q1.r.set_ph"

    traits = traits_def()

    def __init__(
        self,
        nco_po: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(nco_po, comment=comment)

    @property
    def nco_po(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class SetPhDeltaIOp(IOperation):
    """Set the latched instantaneous phase-kick source from an immediate."""

    name = "q1.i.set_ph_delta"

    traits = traits_def()

    def __init__(
        self,
        nco_delta_po: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(nco_delta_po, comment=comment)

    @property
    def nco_delta_po(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetPhDeltaROp(RsOperation[IntRegisterType]):
    """Set the latched instantaneous phase-kick source from a register."""

    name = "q1.r.set_ph_delta"

    traits = traits_def()

    def __init__(
        self,
        nco_delta_po: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(nco_delta_po, comment=comment)

    @property
    def nco_delta_po(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class SetAwgGainIIOp(IIOperation):
    """Set latched AWG gains for both output paths from immediate values."""

    name = "q1.ii.set_awg_gain"

    traits = traits_def()

    def __init__(
        self,
        gain0: int | IntegerAttr[UI32],
        gain1: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(gain0, gain1, comment=comment)

    @property
    def gain0(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def gain1(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class SetAwgGainRROp(RsRsOperation[IntRegisterType]):
    """Set latched AWG gains for both output paths from register values."""

    name = "q1.rr.set_awg_gain"

    traits = traits_def()

    def __init__(
        self,
        gain0: Operation | SSAValue,
        gain1: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(gain0, gain1, comment=comment)

    @property
    def gain0(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def gain1(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class SetAwgOffsIIOp(IIOperation):
    """Set latched AWG offsets for both output paths from immediate values."""

    name = "q1.ii.set_awg_offs"

    traits = traits_def()

    def __init__(
        self,
        offs0: int | IntegerAttr[UI32],
        offs1: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(offs0, offs1, comment=comment)

    @property
    def offs0(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def offs1(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class SetAwgOffsRROp(RsRsOperation[IntRegisterType]):
    """Set latched AWG offsets for both output paths from register values."""

    name = "q1.rr.set_awg_offs"

    traits = traits_def()

    def __init__(
        self,
        offs0: Operation | SSAValue,
        offs1: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(offs0, offs1, comment=comment)

    @property
    def offs0(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def offs1(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


# endregion

# region LINQ Feedback Instructions

# region Universal Receive Instructions


@irdl_op_definition
class FbPopDataIROp(IRdOperation[IntRegisterType]):
    """Pop the next entry whose id matches the immediate from the feedback queue and write
    the associated data value into the destination register."""

    name = "q1.ir.fb_pop_data"

    traits = traits_def()

    def __init__(
        self,
        id: int | IntegerAttr[UI32],
        destination: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(id, destination, comment=comment)

    @property
    def id(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def destination(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class FbPullDataRROp(RdRdOperation[IntRegisterType]):
    """Pull the first available entry from the feedback queue regardless of id, writing the
    entry's id into :attr:`destination_id` and the associated data into
    :attr:`destination`."""

    name = "q1.rr.fb_pull_data"

    traits = traits_def()

    def __init__(
        self,
        id: IntRegisterType,
        destination: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(id, destination, comment=comment)

    @property
    def destination_id(self):
        """Semantic alias for the first generic rd field."""

        return self.rd1

    @property
    def destination(self):
        """Semantic alias for the second generic rd field."""

        return self.rd2


# endregion

# region Universal Transmit Instructions


@irdl_op_definition
class FbComDataIIIOp(IIIOperation):
    """Send an immediate value over LINQ tagged with the given id and wait duration ns."""

    name = "q1.iii.fb_com_data"

    traits = traits_def()

    def __init__(
        self,
        id: int | IntegerAttr[UI32],
        value: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(id, value, duration, comment=comment)

    @property
    def id(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def value(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def duration(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3


@irdl_op_definition
class FbComDataIRIOp(IRsIOperation[IntRegisterType]):
    """Send a register value over LINQ tagged with the given id and wait duration ns."""

    name = "q1.iri.fb_com_data"

    traits = traits_def()

    def __init__(
        self,
        id: int | IntegerAttr[UI32],
        value: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(id, value, duration, comment=comment)

    @property
    def id(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def value(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class FbCmdIIIOp(IIIOperation):
    """Send an immediate command value over LINQ tagged with the given id and wait duration
    ns."""

    name = "q1.iii.fb_cmd"

    traits = traits_def()

    def __init__(
        self,
        id: int | IntegerAttr[UI32],
        value: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(id, value, duration, comment=comment)

    @property
    def id(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def value(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def duration(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3


@irdl_op_definition
class FbCmdIRIOp(IRsIOperation[IntRegisterType]):
    """Send a register command value over LINQ tagged with the given id and wait duration
    ns."""

    name = "q1.iri.fb_cmd"

    traits = traits_def()

    def __init__(
        self,
        id: int | IntegerAttr[UI32],
        value: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(id, value, duration, comment=comment)

    @property
    def id(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def value(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class FbComCfgIIIIOp(IIIIOperation):
    """Configure write-combine mode, bit position, payload length, and wait duration for
    subsequent fb_com_data transmissions."""

    name = "q1.iiii.fb_com_cfg"

    traits = traits_def()

    def __init__(
        self,
        wc: int | IntegerAttr[UI32],
        shift: int | IntegerAttr[UI32],
        length: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(wc, shift, length, duration, comment=comment)

    @property
    def wc(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def shift(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def length(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3

    @property
    def duration(self):
        """Semantic alias for the fourth generic imm field."""

        return self.imm4


@irdl_op_definition
class FbComExtraIIIOp(IIIOperation):
    """Enable or disable inclusion of extra bytes in the LINQ data payload and wait duration
    ns.

    Signature: enable: I, extra: I, duration: I
    """

    name = "q1.iii.fb_com_extra"

    traits = traits_def()

    def __init__(
        self,
        extra_vld: int | IntegerAttr[UI32],
        extra: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(extra_vld, extra, duration, comment=comment)

    @property
    def extra_vld(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def extra(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def duration(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3


# endregion

# region Readout LINQ Instructions


@irdl_op_definition
class FbAcqTbIdIIOp(IIOperation):
    """Configure the id tag attached to thresholded bits (TB) sent over LINQ (immediate
    variant) and wait duration ns.

    Setting id = 0 disables transmission.
    """

    name = "q1.ii.fb_acq_tb_id"

    traits = traits_def()

    def __init__(
        self,
        id: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(id, duration, comment=comment)

    @property
    def id(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class FbAcqTbIdRIOp(RsIOperation[IntRegisterType]):
    """Configure the id tag attached to thresholded bits (TB) sent over LINQ (register
    variant) and wait duration ns.

    Setting id = 0 disables transmission.
    """

    name = "q1.ri.fb_acq_tb_id"

    traits = traits_def()

    def __init__(
        self,
        id: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(id, duration, comment=comment)

    @property
    def id(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class FbAcqTbCfgIIIIOp(IIIIOperation):
    """Configure write-combine mode, bit position, payload length, and wait duration for
    thresholded-bit transmissions."""

    name = "q1.iiii.fb_acq_tb_cfg"

    traits = traits_def()

    def __init__(
        self,
        wc: int | IntegerAttr[UI32],
        shift: int | IntegerAttr[UI32],
        length: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(wc, shift, length, duration, comment=comment)

    @property
    def wc(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def shift(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def length(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3

    @property
    def duration(self):
        """Semantic alias for the fourth generic imm field."""

        return self.imm4


@irdl_op_definition
class FbAcqTbValidIIOp(IIOperation):
    """Configure the valid bit for thresholded bits (TB) sent over LINQ (immediate variant)
    and wait duration ns."""

    name = "q1.ii.fb_acq_tb_valid"

    traits = traits_def()

    def __init__(
        self,
        tb_valid: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(tb_valid, duration, comment=comment)

    @property
    def tb_valid(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class FbAcqTbValidRIOp(RsIOperation[IntRegisterType]):
    """Configure the valid bit for thresholded bits (TB) sent over LINQ (register variant)
    and wait duration ns."""

    name = "q1.ri.fb_acq_tb_valid"

    traits = traits_def()

    def __init__(
        self,
        tb_valid: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(tb_valid, duration, comment=comment)

    @property
    def tb_valid(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class FbAcqTbExtraIIIOp(IIIOperation):
    """Enable or disable inclusion of extra bytes in the TB payload and wait duration ns."""

    name = "q1.iii.fb_acq_tb_extra"

    traits = traits_def()

    def __init__(
        self,
        extra_vld: int | IntegerAttr[UI32],
        extra: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(extra_vld, extra, duration, comment=comment)

    @property
    def extra_vld(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def extra(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def duration(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3


@irdl_op_definition
class FbAcqTbMockIIIIOp(IIIIOperation):
    """Transmit mock thresholded bits instead of real TB data when enable = 1."""

    name = "q1.iiii.fb_acq_tb_mock"

    traits = traits_def()

    def __init__(
        self,
        mock_en: int | IntegerAttr[UI32],
        mock_vld: int | IntegerAttr[UI32],
        mock_data: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(mock_en, mock_vld, mock_data, duration, comment=comment)

    @property
    def mock_en(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def mock_vld(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def mock_data(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3

    @property
    def duration(self):
        """Semantic alias for the fourth generic imm field."""

        return self.imm4


@irdl_op_definition
class FbAcqIqIdIIOp(IIOperation):
    """Configure the id tag attached to IQ data sent over LINQ (immediate variant) and wait
    duration ns.

    Setting id = 0 disables transmission.
    """

    name = "q1.ii.fb_acq_iq_id"

    traits = traits_def()

    def __init__(
        self,
        id: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(id, duration, comment=comment)

    @property
    def id(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class FbAcqIqIdRIOp(RsIOperation[IntRegisterType]):
    """Configure the id tag attached to IQ data sent over LINQ (register variant) and wait
    duration ns.

    Setting id = 0 disables transmission.
    """

    name = "q1.ri.fb_acq_iq_id"

    traits = traits_def()

    def __init__(
        self,
        id: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(id, duration, comment=comment)

    @property
    def id(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class FbAcqIqShiftIIOp(IIOperation):
    """Right-shift IQ values by shift bits before LINQ transmission to reduce resolution,
    then wait duration ns."""

    name = "q1.ii.fb_acq_iq_shift"

    traits = traits_def()

    def __init__(
        self,
        shift: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(shift, duration, comment=comment)

    @property
    def shift(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


# endregion

# endregion

# region Q1 Real-time Instructions


@irdl_op_definition
class SetLatchEnIIOp(IIOperation):
    """Enable/Disable all trigger network address counters from an immediate value. When
    enabled counters will count all triggers on the trigger network. When disabled the
    counters hold their previous values.

    Duration specifies the amount of time spent on the instruction in ns
    """

    name = "q1.ii.set_latch_en"

    traits = traits_def()

    def __init__(
        self,
        latch_en: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(latch_en, duration, comment=comment)

    @property
    def latch_en(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class SetLatchEnRIOp(RsIOperation[IntRegisterType]):
    """Enable/Disable all trigger network address counters. When enabled counters will count
    all triggers on the trigger network. When disabled the counters hold their previous
    values.

    Duration specifies the amount of time spent on the instruction in ns
    """

    name = "q1.ri.set_latch_en"

    traits = traits_def()

    def __init__(
        self,
        latch_en: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(latch_en, duration, comment=comment)

    @property
    def latch_en(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class LatchRstIOp(IOperation):
    """Resets all trigger network address counters to 0.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.i.latch_rst"

    traits = traits_def()

    def __init__(
        self,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(duration, comment=comment)

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class LatchRstROp(RsOperation[IntRegisterType]):
    """Resets all trigger network address counters to 0.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.r.latch_rst"

    traits = traits_def()

    def __init__(
        self,
        duration: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(duration, comment=comment)

    @property
    def duration(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class WaitIOp(IOperation):
    """Waits for the specified duration in ns."""

    name = "q1.i.wait"

    traits = traits_def(Pure())

    def __init__(
        self,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(duration, comment=comment)

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class WaitROp(RsOperation[IntRegisterType]):
    """Waits for the specified duration in ns."""

    name = "q1.r.wait"

    traits = traits_def(Pure())

    def __init__(
        self,
        duration: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(duration, comment=comment)

    @property
    def duration(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class WaitTriggerIIOp(IIOperation):
    """Wait for a hardware trigger. Duration specifies the timeout in ns.

    Warning: Minimum time between wait_trigger and set_cond is 8ns.
    """

    name = "q1.ii.wait_trigger"

    traits = traits_def(Pure())

    def __init__(
        self,
        trig_addr: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(trig_addr, duration, comment=comment)

    @property
    def trig_addr(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class WaitTriggerRROp(RsRsOperation[IntRegisterType]):
    """Wait for a hardware trigger.

    Duration specifies the timeout in ns.
    """

    name = "q1.rr.wait_trigger"

    traits = traits_def(Pure())

    def __init__(
        self,
        trig_addr: Operation | SSAValue,
        duration: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(trig_addr, duration, comment=comment)

    @property
    def trig_addr(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def duration(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class WaitSyncIOp(IOperation):
    """Wait for SYNQ to complete all previous tasks of all the sequencers.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.i.wait_sync"

    traits = traits_def(Pure())

    def __init__(
        self,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(duration, comment=comment)

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class WaitSyncROp(RsOperation[IntRegisterType]):
    """Wait for SYNQ to complete all previous tasks of all the sequencers.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.r.wait_sync"

    traits = traits_def(Pure())

    def __init__(
        self,
        duration: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(duration, comment=comment)

    @property
    def duration(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class UpdParamIOp(IOperation):
    """Update the latched parameters and then wait for number of ns specified by
    duration."""

    name = "q1.i.upd_param"

    traits = traits_def()

    def __init__(
        self,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(duration, comment=comment)

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class PlayIIIOp(IIIOperation):
    """Update the latched parameters, interrupt waves being played and start playing AWG
    waveforms stored at indexes wave_0 on path 0 and wave_1 on path 1.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iii.play"

    traits = traits_def()

    def __init__(
        self,
        wave0: int | IntegerAttr[UI32],
        wave1: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(wave0, wave1, duration, comment=comment)

    @property
    def wave0(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def wave1(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def duration(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3


@irdl_op_definition
class PlayRRIOp(RsRsIOperation[IntRegisterType]):
    """Update the latched parameters, interrupt waves being played and start playing AWG
    waveforms stored at indexes wave_0 on path 0 and wave_1 on path 1.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.rri.play"

    traits = traits_def()

    def __init__(
        self,
        wave0: Operation | SSAValue,
        wave1: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(wave0, wave1, duration, comment=comment)

    @property
    def wave0(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def wave1(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def duration(self):
        """Semantic alias for the first generic imm field."""

        return self.imm


@irdl_op_definition
class AcquireIIIOp(IIIOperation):
    """Update the latched parameters, interrupt currently active acquisitions and start the
    acquisition specified and store data in index provided by bin.

    Integration is executed using a square weight with preset length from the QCoDeS
    parameter.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iii.acquire"

    traits = traits_def()

    def __init__(
        self,
        acq_idx: int | IntegerAttr[UI32],
        bin_idx: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(acq_idx, bin_idx, duration, comment=comment)

    @property
    def acq_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bin_idx(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def duration(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3


@irdl_op_definition
class AcquireIRIOp(IRsIOperation[IntRegisterType]):
    """Update the latched parameters, interrupt currently active acquisitions and start the
    acquisition specified and store data in index provided by bin.

    Integration is executed using a square weight with preset length from the QCoDeS
    parameter.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iri.acquire"

    traits = traits_def()

    def __init__(
        self,
        acq_idx: int | IntegerAttr[UI32],
        bin_idx: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(acq_idx, bin_idx, duration, comment=comment)

    @property
    def acq_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bin_idx(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class AcquireWeighedIIIIIOp(IIIIIOperation):
    """Update the latched parameters, interrupt currently active acquisitions and start the
    acquisition specified and store data in index provided by bin.

    Integration is executed using weights stored at indices weight_0 for path 0 and weight_1
    for path 1.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iiiii.acquire_weighed"

    traits = traits_def()

    def __init__(
        self,
        acq_idx: int | IntegerAttr[UI32],
        bin_idx: int | IntegerAttr[UI32],
        weight_idx0: int | IntegerAttr[UI32],
        weight_idx1: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(
            acq_idx, bin_idx, weight_idx0, weight_idx1, duration, comment=comment
        )

    @property
    def acq_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bin_idx(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def weight_idx0(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3

    @property
    def weight_idx1(self):
        """Semantic alias for the fourth generic imm field."""

        return self.imm4

    @property
    def duration(self):
        """Semantic alias for the fifth generic imm field."""

        return self.imm5


@irdl_op_definition
class AcquireWeighedIRRRIOp(IRsRsRsIOperation[IntRegisterType]):
    """Update the latched parameters, interrupt currently active acquisitions and start the
    acquisition specified and store data in index provided by bin. Integration is executed
    using weights stored at indices weight_0 for path 0 and weight_1 for path 1.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.irrri.acquire_weighed"

    traits = traits_def()

    def __init__(
        self,
        acq_idx: int | IntegerAttr[UI32],
        bin_idx: Operation | SSAValue,
        weight_idx0: Operation | SSAValue,
        weight_idx1: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(
            acq_idx, bin_idx, weight_idx0, weight_idx1, duration, comment=comment
        )

    @property
    def acq_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bin_idx(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def weight_idx0(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def weight_idx1(self):
        """Semantic alias for the third generic rs field."""

        return self.rs3

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class AcquireTtlIIIIOp(IIIIOperation):
    """Update the latched parameters, start the TTL trigger acquisition provided by the
    index in acquisition, store data in index provided by bin.

    Enable TTL trigger by writing 1 to enable, disable after by writing 0 to enable

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iiii.acquire_ttl"

    traits = traits_def()

    def __init__(
        self,
        acq_idx: int | IntegerAttr[UI32],
        bin_idx: int | IntegerAttr[UI32],
        ttl_en: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(acq_idx, bin_idx, ttl_en, duration, comment=comment)

    @property
    def acq_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bin_idx(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def ttl_en(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3

    @property
    def duration(self):
        """Semantic alias for the fourth generic imm field."""

        return self.imm4


@irdl_op_definition
class AcquireTtlIRIIOp(IRsIIOperation[IntRegisterType]):
    """Update the latched parameters, start the TTL trigger acquisition provided by the
    index in acquisition, store data in index provided by bin.

    Enable TTL trigger by writing 1 to enable, disable afterwards by writing 0 to enable

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.irii.acquire_ttl"

    traits = traits_def()

    def __init__(
        self,
        acq_idx: int | IntegerAttr[UI32],
        bin_idx: Operation | SSAValue,
        ttl_en: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(acq_idx, bin_idx, ttl_en, duration, comment=comment)

    @property
    def acq_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bin_idx(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def ttl_en(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def duration(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3


@irdl_op_definition
class AcquireTimetagsIIIIIOp(IIIIIOperation):
    """Depending on enable, open or close the time tag counting acquisition window.

    fine_delay adjusts the start of the acquisition window relative to current sequencer
    time.

    acq_idx and bin_idx are defined by a closing acquire_timetags instruction.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iiiii.acquire_timetags"

    traits = traits_def()

    def __init__(
        self,
        acq_idx: int | IntegerAttr[UI32],
        bin_idx: int | IntegerAttr[UI32],
        window_en: int | IntegerAttr[UI32],
        fine_acq_delay: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(
            acq_idx, bin_idx, window_en, fine_acq_delay, duration, comment=comment
        )

    @property
    def acq_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bin_idx(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def window_en(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3

    @property
    def fine_acq_delay(self):
        """Semantic alias for the fourth generic imm field."""

        return self.imm4

    @property
    def duration(self):
        """Semantic alias for the fifth generic imm field."""

        return self.imm5


@irdl_op_definition
class AcquireTimetagsIRIRIOp(IRsIRsIOperation[IntRegisterType]):
    """Depending on enable, open or close the time tag counting acquisition window.

    fine_delay adjusts the start of the acquisition window relative to current sequencer
    time.

    acq_idx and bin_idx are defined by a closing acquire_timetags instruction.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iriri.acquire_timetags"

    traits = traits_def()

    def __init__(
        self,
        acq_idx: int | IntegerAttr[UI32],
        bin_idx: Operation | SSAValue,
        window_en: int | IntegerAttr[UI32],
        fine_acq_delay: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(
            acq_idx, bin_idx, window_en, fine_acq_delay, duration, comment=comment
        )

    @property
    def acq_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bin_idx(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def window_en(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def fine_acq_delay(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def duration(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3


@irdl_op_definition
class AcquireDigitalIIIOp(IIIOperation):
    """Updates latched parameters, samples and records the inputs mapped to the sequencer.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iii.acquire_digital"

    traits = traits_def()

    def __init__(
        self,
        acq_idx: int | IntegerAttr[UI32],
        bin_idx: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(acq_idx, bin_idx, duration, comment=comment)

    @property
    def acq_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bin_idx(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def duration(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3


@irdl_op_definition
class AcquireDigitalIRIOp(IRsIOperation[IntRegisterType]):
    """Updates latched parameters, samples and records the inputs mapped to the sequencer.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iri.acquire_digital"

    traits = traits_def()

    def __init__(
        self,
        acq_idx: int | IntegerAttr[UI32],
        bin_idx: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(acq_idx, bin_idx, duration, comment=comment)

    @property
    def acq_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bin_idx(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class UpdThresIIIOp(IIIOperation):
    """Updates latched parameters and sets the event count threshold at the index given.
    Threshold determines how the number of detected edge events in acquire_timetags window
    maps to measurement outcome.

    Duration must be >= 4 ns.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iii.upd_thres"

    traits = traits_def()

    def __init__(
        self,
        dio_thres_idx: int | IntegerAttr[UI32],
        value: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(dio_thres_idx, value, duration, comment=comment)

    @property
    def dio_thres_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def value(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def duration(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3


@irdl_op_definition
class UpdThresIRIOp(IRsIOperation[IntRegisterType]):
    """Updates latched parameters and sets the event count threshold at the index given.
    Threshold determines how the number of detected edge events in acquire_timetags window
    maps to measurement outcome.

    Duration must be >= 4 ns.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iri.upd_thres"

    traits = traits_def()

    def __init__(
        self,
        dio_thres_idx: int | IntegerAttr[UI32],
        value: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(dio_thres_idx, value, duration, comment=comment)

    @property
    def dio_thres_idx(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def value(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


# endregion

# endregion
