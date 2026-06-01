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
    IIIIOperation,
    IIIOperation,
    IIOperation,
    IOperation,
    IRdOperation,
    IRsIOperation,
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
    RsRsOperation,
    RsRsRdOperation,
    RsRsRsIOperation,
)
from qat.experimental.dialect.q1.ir.attrs import LabelAttr
from qat.experimental.dialect.q1.ir.imm_desc import UI32
from qat.experimental.dialect.q1.ir.reg_desc import IntRegisterType

# region Q1 Core Instructions (Classical Logic & Flow)

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
        enable: int | IntegerAttr[UI32],
        mask: int | IntegerAttr[UI32],
        operator: int | IntegerAttr[UI32],
        else_duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(enable, mask, operator, else_duration, comment=comment)

    @property
    def enable(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def mask(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def operator(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3

    @property
    def else_duration(self):
        """Semantic alias for the fourth generic imm field."""

        return self.imm4


@irdl_op_definition
class SetCondRRRIOp(RsRsRsIOperation[IntRegisterType]):
    """Configure conditional execution from register-based condition parameters."""

    name = "q1.rrri.set_cond"

    traits = traits_def()

    def __init__(
        self,
        enable: Operation | SSAValue,
        mask: Operation | SSAValue,
        operator: Operation | SSAValue,
        else_duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(enable, mask, operator, else_duration, comment=comment)

    @property
    def enable(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def mask(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def operator(self):
        """Semantic alias for the third generic rs field."""

        return self.rs3

    @property
    def else_duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetMrkIOp(IOperation):
    """Set marker output mask from an immediate source."""

    name = "q1.i.set_mrk"

    traits = traits_def()

    def __init__(
        self,
        mask: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(mask, comment=comment)

    @property
    def mask(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetMrkROp(RsOperation[IntRegisterType]):
    """Set marker output mask from a register source."""

    name = "q1.r.set_mrk"

    traits = traits_def()

    def __init__(
        self,
        mask: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(mask, comment=comment)

    @property
    def mask(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class SetFreqIOp(IOperation):
    """Set the latched NCO frequency from an immediate source."""

    name = "q1.i.set_freq"

    traits = traits_def()

    def __init__(
        self,
        frequency: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(frequency, comment=comment)

    @property
    def frequency(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetFreqROp(RsOperation[IntRegisterType]):
    """Set the latched NCO frequency from a register source."""

    name = "q1.r.set_freq"

    traits = traits_def()

    def __init__(
        self,
        frequency: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(frequency, comment=comment)

    @property
    def frequency(self):
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
        phase_offset: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(phase_offset, comment=comment)

    @property
    def phase_offset(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetPhROp(RsOperation[IntRegisterType]):
    """Set the latched NCO phase-offset source from a register."""

    name = "q1.r.set_ph"

    traits = traits_def()

    def __init__(
        self,
        phase_offset: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(phase_offset, comment=comment)

    @property
    def phase_offset(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class SetPhDeltaIOp(IOperation):
    """Set the latched instantaneous phase-kick source from an immediate."""

    name = "q1.i.set_ph_delta"

    traits = traits_def()

    def __init__(
        self,
        phase_delta: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(phase_delta, comment=comment)

    @property
    def phase_delta(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetPhDeltaROp(RsOperation[IntRegisterType]):
    """Set the latched instantaneous phase-kick source from a register."""

    name = "q1.r.set_ph_delta"

    traits = traits_def()

    def __init__(
        self,
        phase_delta: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(phase_delta, comment=comment)

    @property
    def phase_delta(self):
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
        offset0: int | IntegerAttr[UI32],
        offset1: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(offset0, offset1, comment=comment)

    @property
    def offset0(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def offset1(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class SetAwgOffsRROp(RsRsOperation[IntRegisterType]):
    """Set latched AWG offsets for both output paths from register values."""

    name = "q1.rr.set_awg_offs"

    traits = traits_def()

    def __init__(
        self,
        offset0: Operation | SSAValue,
        offset1: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(offset0, offset1, comment=comment)

    @property
    def offset0(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def offset1(self):
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
    entry's id into ``destination_id`` and the associated data into `destination`."""

    name = "q1.rr.fb_pull_data"

    traits = traits_def()

    def __init__(
        self,
        destination_id: IntRegisterType,
        destination: IntRegisterType,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(destination_id, destination, comment=comment)

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
        bit_pos: int | IntegerAttr[UI32],
        length: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(wc, bit_pos, length, duration, comment=comment)

    @property
    def wc(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bit_pos(self):
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
        enable: int | IntegerAttr[UI32],
        extra: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(enable, extra, duration, comment=comment)

    @property
    def enable(self):
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
        bit_pos: int | IntegerAttr[UI32],
        length: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(wc, bit_pos, length, duration, comment=comment)

    @property
    def wc(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def bit_pos(self):
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
        valid: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(valid, duration, comment=comment)

    @property
    def valid(self):
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
        valid: Operation | SSAValue,
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(valid, duration, comment=comment)

    @property
    def valid(self):
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
        enable: int | IntegerAttr[UI32],
        extra: int | IntegerAttr[UI32],
        duration: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(enable, extra, duration, comment=comment)

    @property
    def enable(self):
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
        enable: int | IntegerAttr[UI32],
        valid: int | IntegerAttr[UI32],
        data: int | IntegerAttr[UI32],
        count: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        super().__init__(enable, valid, data, count, comment=comment)

    @property
    def enable(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def valid(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2

    @property
    def data(self):
        """Semantic alias for the third generic imm field."""

        return self.imm3

    @property
    def count(self):
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
