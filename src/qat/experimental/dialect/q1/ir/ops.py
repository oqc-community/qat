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
from xdsl.dialects.builtin import StringAttr
from xdsl.irdl import irdl_op_definition, prop_def, traits_def
from xdsl.traits import Commutative, IsTerminator, Pure

from qat.experimental.dialect.q1.ir.abstract_ops import (
    AssemblyInstructionArg,
    ImmImmImmImmImmOperation,
    ImmImmImmImmOperation,
    ImmImmImmOperation,
    ImmImmOperation,
    ImmOperation,
    ImmRdOperation,
    ImmRsImmImmOperation,
    ImmRsImmOperation,
    ImmRsOperation,
    ImmRsRdOperation,
    ImmRsRdRdOperation,
    ImmRsRsRsImmOperation,
    NullaryOperation,
    Q1AsmOperation,
    RdImmOperation,
    RdRsOperation,
    RsImmImmOperation,
    RsImmOperation,
    RsImmRdOperation,
    RsImmRdRdOperation,
    RsImmRsOperation,
    RsOperation,
    RsRdOperation,
    RsRsImmOperation,
    RsRsOperation,
    RsRsRdOperation,
    RsRsRdRdOperation,
    RsRsRsImmOperation,
)
from qat.experimental.dialect.q1.ir.attrs import LabelAttr
from qat.experimental.dialect.q1.ir.imm_desc import (
    AddressImm,
    BoolImm,
    DurationImm,
    NcoPhaseImm,
    SI16Imm,
    SI32Imm,
    SU32Imm,
    UI2Imm,
    UI3Imm,
    UI4Imm,
    UI5Imm,
    UI6Imm,
    UI7Imm,
    UI8Imm,
    UI10Imm,
    UI16Imm,
    UI24Imm,
    UI32Imm,
)
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
class StopImmOp(ImmOperation[SI32Imm]):
    """Stop the sequencer using an immediate stop code."""

    name = "q1.i.stop"

    traits = traits_def(IsTerminator())

    @property
    def status(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class StopRsOp(RsOperation[IntRegisterType]):
    """Stop the sequencer using a stop code read from a register."""

    name = "q1.r.stop"

    traits = traits_def(IsTerminator())

    @property
    def status(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class StopOp(NullaryOperation):
    """Stop the sequencer with default stop code `0` (pseudo-op for `stop 0`)."""

    name = "q1..stop"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class NopOp(NullaryOperation):
    """No-op instruction that waits one Q1 core cycle (4 ns) without changing state."""

    name = "q1..nop"

    traits = traits_def(Pure())


# endregion

# region Jump Instructions


@irdl_op_definition
class JmpImmOp(ImmOperation[AddressImm]):
    """Unconditional jump to an immediate address."""

    name = "q1.i.jmp"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JmpRsOp(RsOperation[IntRegisterType]):
    """Unconditional jump to an address stored in a register."""

    name = "q1.r.jmp"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JzImmOp(ImmOperation[AddressImm]):
    """Jump if `ZF == 1` to an immediate address."""

    name = "q1.i.jz"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JzRsOp(RsOperation[IntRegisterType]):
    """Jump if `ZF == 1` to a register address."""

    name = "q1.r.jz"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JnzImmOp(ImmOperation[AddressImm]):
    """Jump if `ZF == 0` to an immediate address."""

    name = "q1.i.jnz"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JnzRsOp(RsOperation[IntRegisterType]):
    """Jump if `ZF == 0` to a register address."""

    name = "q1.r.jnz"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JoImmOp(ImmOperation[AddressImm]):
    """Jump if `OF == 1` to an immediate address."""

    name = "q1.i.jo"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JoRsOp(RsOperation[IntRegisterType]):
    """Jump if `OF == 1` to a register address."""

    name = "q1.r.jo"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JnoImmOp(ImmOperation[AddressImm]):
    """Jump if `OF == 0` to an immediate address."""

    name = "q1.i.jno"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JnoRsOp(RsOperation[IntRegisterType]):
    """Jump if `OF == 0` to a register address."""

    name = "q1.r.jno"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JsImmOp(ImmOperation[AddressImm]):
    """Jump if `NF == 1` to an immediate address."""

    name = "q1.i.js"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JsRsOp(RsOperation[IntRegisterType]):
    """Jump if `NF == 1` to a register address."""

    name = "q1.r.js"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JnsImmOp(ImmOperation[AddressImm]):
    """Jump if `NF == 0` to an immediate address."""

    name = "q1.i.jns"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JnsRsOp(RsOperation[IntRegisterType]):
    """Jump if `NF == 0` to a register address."""

    name = "q1.r.jns"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JgImmOp(ImmOperation[AddressImm]):
    """Jump if signed `a > b` condition holds (`ZF == 0` and `NF == OF`)."""

    name = "q1.i.jg"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JgRsOp(RsOperation[IntRegisterType]):
    """Jump if signed `a > b` condition holds (`ZF == 0` and `NF == OF`)."""

    name = "q1.r.jg"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JlImmOp(ImmOperation[AddressImm]):
    """Jump if signed `a < b` condition holds (`NF != OF`)."""

    name = "q1.i.jl"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JlRsOp(RsOperation[IntRegisterType]):
    """Jump if signed `a < b` condition holds (`NF != OF`)."""

    name = "q1.r.jl"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JleImmOp(ImmOperation[AddressImm]):
    """Jump if signed `a <= b` condition holds (`ZF == 1` or `NF != OF`)."""

    name = "q1.i.jle"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JleRsOp(RsOperation[IntRegisterType]):
    """Jump if signed `a <= b` condition holds (`ZF == 1` or `NF != OF`)."""

    name = "q1.r.jle"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JaImmOp(ImmOperation[AddressImm]):
    """Jump if unsigned `a > b` condition holds (`ZF == 0` and `CF == 0`)."""

    name = "q1.i.ja"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JaRsOp(RsOperation[IntRegisterType]):
    """Jump if unsigned `a > b` condition holds (`ZF == 0` and `CF == 0`)."""

    name = "q1.r.ja"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JaeImmOp(ImmOperation[AddressImm]):
    """Jump if unsigned `a >= b` condition holds (`CF == 0`)."""

    name = "q1.i.jae"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JaeRsOp(RsOperation[IntRegisterType]):
    """Jump if unsigned `a >= b` condition holds (`CF == 0`)."""

    name = "q1.r.jae"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JbImmOp(ImmOperation[AddressImm]):
    """Jump if unsigned `a < b` condition holds (`CF == 1`)."""

    name = "q1.i.jb"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JbRsOp(RsOperation[IntRegisterType]):
    """Jump if unsigned `a < b` condition holds (`CF == 1`)."""

    name = "q1.r.jb"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JbeImmOp(ImmOperation[AddressImm]):
    """Jump if unsigned `a <= b` condition holds (`ZF == 1` or `CF == 1`)."""

    name = "q1.i.jbe"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JbeRsOp(RsOperation[IntRegisterType]):
    """Jump if unsigned `a <= b` condition holds (`ZF == 1` or `CF == 1`)."""

    name = "q1.r.jbe"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JgeImmOp(ImmOperation[AddressImm]):
    """Jump if signed `a >= b` condition holds (`NF == OF`)."""

    name = "q1.i.jge"

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class JgeRsOp(RsOperation[IntRegisterType]):
    """Jump if signed `a >= b` condition holds (`NF == OF`)."""

    name = "q1.r.jge"

    @property
    def address(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class JgeRsImmImmOp(RsImmImmOperation[IntRegisterType, UI32Imm, AddressImm]):
    """Deprecated legacy jump variant for unsigned `a >= b` with immediate address."""

    name = "q1.rii.jge"

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
class JgeRsImmRsOp(RsImmRsOperation[IntRegisterType, UI32Imm]):
    """Deprecated legacy jump variant for unsigned `a >= b` with register address."""

    name = "q1.rir.jge"

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
class JltRsImmImmOp(RsImmImmOperation[IntRegisterType, UI32Imm, AddressImm]):
    """Deprecated legacy jump variant for unsigned `a < b` with immediate address."""

    name = "q1.rii.jlt"

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
class JltRsImmRsOp(RsImmRsOperation[IntRegisterType, UI32Imm]):
    """Deprecated legacy jump variant for unsigned `a < b` with register address."""

    name = "q1.rir.jlt"

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
class LoopRdImmOp(RdImmOperation[IntRegisterType, AddressImm]):
    """Deprecated legacy loop: decrement source and jump while the result is non-zero."""

    name = "q1.ri.loop"

    @property
    def source(self):
        """Semantic alias for the generic rd field."""

        return self.rd

    @property
    def address(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class LoopRdRsOp(RdRsOperation[IntRegisterType]):
    """Deprecated legacy loop: decrement source and jump while the result is non-zero."""

    name = "q1.rr.loop"

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
class MoveImmRdOp(ImmRdOperation[IntRegisterType, SU32Imm]):
    """Copy an immediate source value into a destination register."""

    name = "q1.ir.move"

    traits = traits_def(Pure())

    @property
    def source(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class MoveRsRdOp(RsRdOperation[IntRegisterType]):
    """Copy a register source value into a destination register."""

    name = "q1.rr.move"

    traits = traits_def(Pure())

    @property
    def source(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class NotImmRdOp(ImmRdOperation[IntRegisterType, SU32Imm]):
    """Bitwise invert an immediate source value and write the result to destination."""

    name = "q1.ir.not"

    traits = traits_def(Pure())

    @property
    def source(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class NotRsRdOp(RsRdOperation[IntRegisterType]):
    """Bitwise invert a register source value and write the result to destination."""

    name = "q1.rr.not"

    traits = traits_def(Pure())

    @property
    def source(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class AddRsImmRdOp(RsImmRdOperation[IntRegisterType, SU32Imm]):
    """Add a register and immediate operand and store the result in destination."""

    name = "q1.rir.add"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class AddRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Add two register operands and store the result in destination."""

    name = "q1.rrr.add"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class SubRsImmRdOp(RsImmRdOperation[IntRegisterType, SU32Imm]):
    """Subtract an immediate operand from a register operand into destination."""

    name = "q1.rir.sub"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SubRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Subtract a register operand from another register operand into destination."""

    name = "q1.rrr.sub"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class AndRsImmRdOp(RsImmRdOperation[IntRegisterType, SU32Imm]):
    """Bitwise AND between register and immediate operands into destination."""

    name = "q1.rir.and"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class AndRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Bitwise AND between two register operands into destination."""

    name = "q1.rrr.and"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class OrRsImmRdOp(RsImmRdOperation[IntRegisterType, SU32Imm]):
    """Bitwise OR between register and immediate operands into destination."""

    name = "q1.rir.or"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class OrRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Bitwise OR between two register operands into destination."""

    name = "q1.rrr.or"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class XorRsImmRdOp(RsImmRdOperation[IntRegisterType, SU32Imm]):
    """Bitwise XOR between register and immediate operands into destination."""

    name = "q1.rir.xor"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class XorRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Bitwise XOR between two register operands into destination."""

    name = "q1.rrr.xor"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class AslRsImmRdOp(RsImmRdOperation[IntRegisterType, UI32Imm]):
    """Arithmetic left shift by immediate bit-count into destination."""

    name = "q1.rir.asl"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class AslRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Arithmetic left shift by register bit-count into destination."""

    name = "q1.rrr.asl"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class AsrRsImmRdOp(RsImmRdOperation[IntRegisterType, UI32Imm]):
    """Arithmetic right shift by immediate bit-count into destination."""

    name = "q1.rir.asr"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class AsrRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Arithmetic right shift by register bit-count into destination."""

    name = "q1.rrr.asr"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class CmpRsRsOp(RsRsOperation[IntRegisterType]):
    """Compare two register operands by computing `a - b` to update the ALU flags."""

    name = "q1.rr.cmp"

    traits = traits_def()

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class CmpRsImmOp(RsImmOperation[IntRegisterType, SU32Imm]):
    """Compare a register operand against an immediate by computing `a - b` to update the
    ALU flags."""

    name = "q1.ri.cmp"

    traits = traits_def()

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class CmpImmRsOp(ImmRsOperation[IntRegisterType, SU32Imm]):
    """Compare an immediate against a register operand by computing `b - a` to update the
    ALU flags."""

    name = "q1.ir.cmp"

    traits = traits_def()

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class TestRsRsOp(RsRsOperation[IntRegisterType]):
    """Test two register operands by computing `a & b` to update the ALU flags.

    .. note::
       Opts out of pytest collection (``__test__ = False``) because the ``Test`` prefix
       is reserved as the default pytest test-class pattern, but this is an ISA op class.
    """

    __test__ = False

    name = "q1.rr.test"

    traits = traits_def()

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class TestRsImmOp(RsImmOperation[IntRegisterType, SU32Imm]):
    """Test a register operand against an immediate by computing `a & b` to update the ALU
    flags.

    .. note::
       Opts out of pytest collection (``__test__ = False``) because the ``Test`` prefix
       is reserved as the default pytest test-class pattern, but this is an ISA op class.
    """

    __test__ = False

    name = "q1.ri.test"

    traits = traits_def()

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class TestImmRsOp(ImmRsOperation[IntRegisterType, SU32Imm]):
    """Test an immediate against a register operand by computing `b & a` to update the ALU
    flags.

    .. note::
       Opts out of pytest collection (``__test__ = False``) because the ``Test`` prefix
       is reserved as the default pytest test-class pattern, but this is an ISA op class.
    """

    __test__ = False

    name = "q1.ir.test"

    traits = traits_def()

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class LsrRsImmRdOp(RsImmRdOperation[IntRegisterType, UI32Imm]):
    """Logical right shift register operand `a` by an immediate bit-count `b` into
    destination."""

    name = "q1.rir.lsr"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class LsrRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Logical right shift register operand `a` by a register bit-count `b` into
    destination."""

    name = "q1.rrr.lsr"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class LsrImmRsRdOp(ImmRsRdOperation[IntRegisterType, UI32Imm]):
    """Logical right shift an immediate operand `b` by a register bit-count `a` into
    destination."""

    name = "q1.irr.lsr"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class LslRsImmRdOp(RsImmRdOperation[IntRegisterType, UI32Imm]):
    """Logical left shift register operand `a` by an immediate bit-count `b` into
    destination."""

    name = "q1.rir.lsl"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class LslRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Logical left shift register operand `a` by a register bit-count `b` into
    destination."""

    name = "q1.rrr.lsl"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class LslImmRsRdOp(ImmRsRdOperation[IntRegisterType, UI32Imm]):
    """Logical left shift an immediate operand `b` by a register bit-count `a` into
    destination."""

    name = "q1.irr.lsl"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Mulu16RsImmRdOp(RsImmRdOperation[IntRegisterType, UI16Imm]):
    """Unsigned 16-bit multiplication of register `a` by immediate `b` into the low 32 bits
    of destination."""

    name = "q1.rir.mulu16"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Mulu16RsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Unsigned 16-bit multiplication of two register operands into the low 32 bits of
    destination."""

    name = "q1.rrr.mulu16"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Mulu16ImmRsRdOp(ImmRsRdOperation[IntRegisterType, UI16Imm]):
    """Unsigned 16-bit multiplication of immediate `b` by register `a` into the low 32 bits
    of destination."""

    name = "q1.irr.mulu16"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Muls16RsImmRdOp(RsImmRdOperation[IntRegisterType, SI16Imm]):
    """Signed 16-bit multiplication of register `a` by immediate `b` into the low 32 bits of
    destination."""

    name = "q1.rir.muls16"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Muls16RsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Signed 16-bit multiplication of two register operands into the low 32 bits of
    destination."""

    name = "q1.rrr.muls16"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Muls16ImmRsRdOp(ImmRsRdOperation[IntRegisterType, SI16Imm]):
    """Signed 16-bit multiplication of immediate `b` by register `a` into the low 32 bits of
    destination."""

    name = "q1.irr.muls16"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Mulu32lRsImmRdOp(RsImmRdOperation[IntRegisterType, UI32Imm]):
    """Unsigned 32-bit multiplication of register `a` by immediate `b`, storing the low 32
    bits of the 64-bit result into destination."""

    name = "q1.rir.mulu32l"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Mulu32lRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Unsigned 32-bit multiplication of two register operands, storing the low 32 bits of
    the 64-bit result into destination."""

    name = "q1.rrr.mulu32l"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Mulu32lImmRsRdOp(ImmRsRdOperation[IntRegisterType, UI32Imm]):
    """Unsigned 32-bit multiplication of immediate `b` by register `a`, storing the low 32
    bits of the 64-bit result into destination."""

    name = "q1.irr.mulu32l"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Mulu32hRsImmRdOp(RsImmRdOperation[IntRegisterType, UI32Imm]):
    """Unsigned 32-bit multiplication of register `a` by immediate `b`, storing the high 32
    bits of the 64-bit result into destination."""

    name = "q1.rir.mulu32h"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_high(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Mulu32hRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Unsigned 32-bit multiplication of two register operands, storing the high 32 bits of
    the 64-bit result into destination."""

    name = "q1.rrr.mulu32h"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def dst_high(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Mulu32hImmRsRdOp(ImmRsRdOperation[IntRegisterType, UI32Imm]):
    """Unsigned 32-bit multiplication of immediate `b` by register `a`, storing the high 32
    bits of the 64-bit result into destination."""

    name = "q1.irr.mulu32h"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_high(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Muls32lRsImmRdOp(RsImmRdOperation[IntRegisterType, SI32Imm]):
    """Signed 32-bit multiplication of register `a` by immediate `b`, storing the low 32
    bits of the 64-bit result into destination."""

    name = "q1.rir.muls32l"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Muls32lRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Signed 32-bit multiplication of two register operands, storing the low 32 bits of the
    64-bit result into destination."""

    name = "q1.rrr.muls32l"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Muls32lImmRsRdOp(ImmRsRdOperation[IntRegisterType, SI32Imm]):
    """Signed 32-bit multiplication of immediate `b` by register `a`, storing the low 32
    bits of the 64-bit result into destination."""

    name = "q1.irr.muls32l"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Muls32hRsImmRdOp(RsImmRdOperation[IntRegisterType, SI32Imm]):
    """Signed 32-bit multiplication of register `a` by immediate `b`, storing the high 32
    bits of the 64-bit result into destination."""

    name = "q1.rir.muls32h"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_high(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Muls32hRsRsRdOp(RsRsRdOperation[IntRegisterType]):
    """Signed 32-bit multiplication of two register operands, storing the high 32 bits of
    the 64-bit result into destination."""

    name = "q1.rrr.muls32h"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def dst_high(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Muls32hImmRsRdOp(ImmRsRdOperation[IntRegisterType, SI32Imm]):
    """Signed 32-bit multiplication of immediate `b` by register `a`, storing the high 32
    bits of the 64-bit result into destination."""

    name = "q1.irr.muls32h"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_high(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class Mulu32RsImmRdRdOp(RsImmRdRdOperation[IntRegisterType, UI32Imm]):
    """Unsigned 32-bit multiplication of register `a` by immediate `b`, storing the low 32
    bits of the 64-bit result into `dst_low` and the high 32 bits into `dst_high`."""

    name = "q1.rirr.mulu32"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the first generic rd field."""

        return self.rd1

    @property
    def dst_high(self):
        """Semantic alias for the second generic rd field."""

        return self.rd2


@irdl_op_definition
class Mulu32RsRsRdRdOp(RsRsRdRdOperation[IntRegisterType]):
    """Unsigned 32-bit multiplication of two register operands, storing the low 32 bits of
    the 64-bit result into `dst_low` and the high 32 bits into `dst_high`."""

    name = "q1.rrrr.mulu32"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def dst_low(self):
        """Semantic alias for the first generic rd field."""

        return self.rd1

    @property
    def dst_high(self):
        """Semantic alias for the second generic rd field."""

        return self.rd2


@irdl_op_definition
class Mulu32ImmRsRdRdOp(ImmRsRdRdOperation[IntRegisterType, UI32Imm]):
    """Unsigned 32-bit multiplication of immediate `b` by register `a`, storing the low 32
    bits of the 64-bit result into `dst_low` and the high 32 bits into `dst_high`."""

    name = "q1.irrr.mulu32"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the first generic rd field."""

        return self.rd1

    @property
    def dst_high(self):
        """Semantic alias for the second generic rd field."""

        return self.rd2


@irdl_op_definition
class Muls32RsImmRdRdOp(RsImmRdRdOperation[IntRegisterType, SI32Imm]):
    """Signed 32-bit multiplication of register `a` by immediate `b`, storing the low 32
    bits of the 64-bit result into `dst_low` and the high 32 bits into `dst_high`."""

    name = "q1.rirr.muls32"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the first generic rd field."""

        return self.rd1

    @property
    def dst_high(self):
        """Semantic alias for the second generic rd field."""

        return self.rd2


@irdl_op_definition
class Muls32RsRsRdRdOp(RsRsRdRdOperation[IntRegisterType]):
    """Signed 32-bit multiplication of two register operands, storing the low 32 bits of the
    64-bit result into `dst_low` and the high 32 bits into `dst_high`."""

    name = "q1.rrrr.muls32"

    traits = traits_def(Pure(), Commutative())

    @property
    def a(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def b(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2

    @property
    def dst_low(self):
        """Semantic alias for the first generic rd field."""

        return self.rd1

    @property
    def dst_high(self):
        """Semantic alias for the second generic rd field."""

        return self.rd2


@irdl_op_definition
class Muls32ImmRsRdRdOp(ImmRsRdRdOperation[IntRegisterType, SI32Imm]):
    """Signed 32-bit multiplication of immediate `b` by register `a`, storing the low 32
    bits of the 64-bit result into `dst_low` and the high 32 bits into `dst_high`."""

    name = "q1.irrr.muls32"

    traits = traits_def(Pure())

    @property
    def a(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def b(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def dst_low(self):
        """Semantic alias for the first generic rd field."""

        return self.rd1

    @property
    def dst_high(self):
        """Semantic alias for the second generic rd field."""

        return self.rd2


# endregion

# region Latched Instructions


@irdl_op_definition
class SetCondImmImmImmImmOp(ImmImmImmImmOperation[BoolImm, UI4Imm, UI3Imm, UI16Imm]):
    """Configure conditional execution from trigger-network condition parameters."""

    name = "q1.iiii.set_cond"

    traits = traits_def()

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
class SetCondRsRsRsImmOp(RsRsRsImmOperation[IntRegisterType, UI16Imm]):
    """Configure conditional execution from register-based condition parameters."""

    name = "q1.rrri.set_cond"

    traits = traits_def()

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
class SetMrkImmOp(ImmOperation[UI4Imm]):
    """Set marker output mask from an immediate source."""

    name = "q1.i.set_mrk"

    traits = traits_def()

    @property
    def mrk(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetMrkRsOp(RsOperation[IntRegisterType]):
    """Set marker output mask from a register source."""

    name = "q1.r.set_mrk"

    traits = traits_def()

    @property
    def mrk(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class SetFreqImmOp(ImmOperation[SI32Imm]):
    """Set the latched NCO frequency from an immediate source."""

    name = "q1.i.set_freq"

    traits = traits_def()

    @property
    def nco_freq(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetFreqRsOp(RsOperation[IntRegisterType]):
    """Set the latched NCO frequency from a register source."""

    name = "q1.r.set_freq"

    traits = traits_def()

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
class SetPhImmOp(ImmOperation[NcoPhaseImm]):
    """Set the latched NCO phase-offset source from an immediate."""

    name = "q1.i.set_ph"

    traits = traits_def()

    @property
    def nco_po(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetPhRsOp(RsOperation[IntRegisterType]):
    """Set the latched NCO phase-offset source from a register."""

    name = "q1.r.set_ph"

    traits = traits_def()

    @property
    def nco_po(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class SetPhDeltaImmOp(ImmOperation[NcoPhaseImm]):
    """Set the latched instantaneous phase-kick source from an immediate."""

    name = "q1.i.set_ph_delta"

    traits = traits_def()

    @property
    def nco_delta_po(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class SetPhDeltaRsOp(RsOperation[IntRegisterType]):
    """Set the latched instantaneous phase-kick source from a register."""

    name = "q1.r.set_ph_delta"

    traits = traits_def()

    @property
    def nco_delta_po(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class SetAwgGainImmImmOp(ImmImmOperation[SI16Imm, SI16Imm]):
    """Set latched AWG gains for both output paths from immediate values."""

    name = "q1.ii.set_awg_gain"

    traits = traits_def()

    @property
    def gain0(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def gain1(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class SetAwgGainRsRsOp(RsRsOperation[IntRegisterType]):
    """Set latched AWG gains for both output paths from register values."""

    name = "q1.rr.set_awg_gain"

    traits = traits_def()

    @property
    def gain0(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def gain1(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class SetAwgOffsImmImmOp(ImmImmOperation[SI16Imm, SI16Imm]):
    """Set latched AWG offsets for both output paths from immediate values."""

    name = "q1.ii.set_awg_offs"

    traits = traits_def()

    @property
    def offs0(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def offs1(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class SetAwgOffsRsRsOp(RsRsOperation[IntRegisterType]):
    """Set latched AWG offsets for both output paths from register values."""

    name = "q1.rr.set_awg_offs"

    traits = traits_def()

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
class FbPopDataImmRdOp(ImmRdOperation[IntRegisterType, UI16Imm]):
    """Pop the next entry whose id matches the immediate from the feedback queue and write
    the associated data value into the destination register."""

    name = "q1.ir.fb_pop_data"

    traits = traits_def()

    @property
    def id(self):
        """Semantic alias for the generic imm field."""

        return self.imm

    @property
    def destination(self):
        """Semantic alias for the generic rd field."""

        return self.rd


@irdl_op_definition
class FbPullDataRsRdOp(RsRdOperation[IntRegisterType]):
    """Pull the entry whose id matches the source register from the feedback queue and write
    the associated data value into the destination register."""

    name = "q1.rr.fb_pull_data"

    traits = traits_def()

    @property
    def id(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def destination(self):
        """Semantic alias for the generic rd field."""

        return self.rd


# endregion

# region Universal Transmit Instructions


@irdl_op_definition
class FbComDataImmImmImmOp(ImmImmImmOperation[UI8Imm, UI32Imm, DurationImm]):
    """Send an immediate value over LINQ tagged with the given id and wait duration ns."""

    name = "q1.iii.fb_com_data"

    traits = traits_def()

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
class FbComDataImmRsImmOp(ImmRsImmOperation[IntRegisterType, UI8Imm, DurationImm]):
    """Send a register value over LINQ tagged with the given id and wait duration ns."""

    name = "q1.iri.fb_com_data"

    traits = traits_def()

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
class FbComCfgImmImmImmImmOp(ImmImmImmImmOperation[UI2Imm, UI10Imm, UI7Imm, DurationImm]):
    """Configure write-combine mode, bit position, payload length, and wait duration for
    subsequent fb_com_data transmissions."""

    name = "q1.iiii.fb_com_cfg"

    traits = traits_def()

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
class FbComExtraImmImmImmOp(ImmImmImmOperation[BoolImm, UI16Imm, DurationImm]):
    """Enable or disable inclusion of extra bytes in the LINQ data payload and wait duration
    ns.

    Signature: enable: I, extra: I, duration: I
    """

    name = "q1.iii.fb_com_extra"

    traits = traits_def()

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
class FbAcqTbIdImmImmOp(ImmImmOperation[UI8Imm, DurationImm]):
    """Configure the id tag attached to thresholded bits (TB) sent over LINQ (immediate
    variant) and wait duration ns.

    Setting id = 0 disables transmission.
    """

    name = "q1.ii.fb_acq_tb_id"

    traits = traits_def()

    @property
    def id(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class FbAcqTbIdRsImmOp(RsImmOperation[IntRegisterType, DurationImm]):
    """Configure the id tag attached to thresholded bits (TB) sent over LINQ (register
    variant) and wait duration ns.

    Setting id = 0 disables transmission.
    """

    name = "q1.ri.fb_acq_tb_id"

    traits = traits_def()

    @property
    def id(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class FbAcqTbCfgImmImmImmImmOp(ImmImmImmImmOperation[UI2Imm, UI10Imm, UI7Imm, DurationImm]):
    """Configure write-combine mode, bit position, payload length, and wait duration for
    thresholded-bit transmissions."""

    name = "q1.iiii.fb_acq_tb_cfg"

    traits = traits_def()

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
class FbAcqTbValidImmImmOp(ImmImmOperation[BoolImm, DurationImm]):
    """Configure the valid bit for thresholded bits (TB) sent over LINQ (immediate variant)
    and wait duration ns."""

    name = "q1.ii.fb_acq_tb_valid"

    traits = traits_def()

    @property
    def tb_valid(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class FbAcqTbValidRsImmOp(RsImmOperation[IntRegisterType, DurationImm]):
    """Configure the valid bit for thresholded bits (TB) sent over LINQ (register variant)
    and wait duration ns."""

    name = "q1.ri.fb_acq_tb_valid"

    traits = traits_def()

    @property
    def tb_valid(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class FbAcqTbExtraImmImmImmOp(ImmImmImmOperation[BoolImm, UI16Imm, DurationImm]):
    """Enable or disable inclusion of extra bytes in the TB payload and wait duration ns."""

    name = "q1.iii.fb_acq_tb_extra"

    traits = traits_def()

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
class FbAcqTbMockImmImmImmImmOp(
    ImmImmImmImmOperation[BoolImm, BoolImm, BoolImm, DurationImm]
):
    """Transmit mock thresholded bits instead of real TB data when enable = 1."""

    name = "q1.iiii.fb_acq_tb_mock"

    traits = traits_def()

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
class FbAcqIqIdImmImmOp(ImmImmOperation[UI8Imm, DurationImm]):
    """Configure the id tag attached to IQ data sent over LINQ (immediate variant) and wait
    duration ns.

    Setting id = 0 disables transmission.
    """

    name = "q1.ii.fb_acq_iq_id"

    traits = traits_def()

    @property
    def id(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class FbAcqIqIdRsImmOp(RsImmOperation[IntRegisterType, DurationImm]):
    """Configure the id tag attached to IQ data sent over LINQ (register variant) and wait
    duration ns.

    Setting id = 0 disables transmission.
    """

    name = "q1.ri.fb_acq_iq_id"

    traits = traits_def()

    @property
    def id(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class FbAcqIqShiftImmImmOp(ImmImmOperation[UI6Imm, DurationImm]):
    """Right-shift IQ values by shift bits before LINQ transmission to reduce resolution,
    then wait duration ns."""

    name = "q1.ii.fb_acq_iq_shift"

    traits = traits_def()

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
class SetLatchEnImmImmOp(ImmImmOperation[BoolImm, DurationImm]):
    """Enable/Disable all trigger network address counters from an immediate value. When
    enabled counters will count all triggers on the trigger network. When disabled the
    counters hold their previous values.

    Duration specifies the amount of time spent on the instruction in ns
    """

    name = "q1.ii.set_latch_en"

    traits = traits_def()

    @property
    def latch_en(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class SetLatchEnRsImmOp(RsImmOperation[IntRegisterType, DurationImm]):
    """Enable/Disable all trigger network address counters. When enabled counters will count
    all triggers on the trigger network. When disabled the counters hold their previous
    values.

    Duration specifies the amount of time spent on the instruction in ns
    """

    name = "q1.ri.set_latch_en"

    traits = traits_def()

    @property
    def latch_en(self):
        """Semantic alias for the generic rs field."""

        return self.rs

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class LatchRstImmOp(ImmOperation[DurationImm]):
    """Resets all trigger network address counters to 0.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.i.latch_rst"

    traits = traits_def()

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class LatchRstRsOp(RsOperation[IntRegisterType]):
    """Resets all trigger network address counters to 0.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.r.latch_rst"

    traits = traits_def()

    @property
    def duration(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class WaitImmOp(ImmOperation[DurationImm]):
    """Waits for the specified duration in ns."""

    name = "q1.i.wait"

    traits = traits_def(Pure())

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class WaitRsOp(RsOperation[IntRegisterType]):
    """Waits for the specified duration in ns."""

    name = "q1.r.wait"

    traits = traits_def(Pure())

    @property
    def duration(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class WaitTriggerImmImmOp(ImmImmOperation[UI4Imm, DurationImm]):
    """Wait for a hardware trigger. Duration specifies the timeout in ns.

    Warning: Minimum time between wait_trigger and set_cond is 8ns.
    """

    name = "q1.ii.wait_trigger"

    traits = traits_def(Pure())

    @property
    def trig_addr(self):
        """Semantic alias for the first generic imm field."""

        return self.imm1

    @property
    def duration(self):
        """Semantic alias for the second generic imm field."""

        return self.imm2


@irdl_op_definition
class WaitTriggerRsRsOp(RsRsOperation[IntRegisterType]):
    """Wait for a hardware trigger.

    Duration specifies the timeout in ns.
    """

    name = "q1.rr.wait_trigger"

    traits = traits_def(Pure())

    @property
    def trig_addr(self):
        """Semantic alias for the first generic rs field."""

        return self.rs1

    @property
    def duration(self):
        """Semantic alias for the second generic rs field."""

        return self.rs2


@irdl_op_definition
class WaitSyncImmOp(ImmOperation[DurationImm]):
    """Wait for SYNQ to complete all previous tasks of all the sequencers.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.i.wait_sync"

    traits = traits_def(Pure())

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class WaitSyncRsOp(RsOperation[IntRegisterType]):
    """Wait for SYNQ to complete all previous tasks of all the sequencers.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.r.wait_sync"

    traits = traits_def(Pure())

    @property
    def duration(self):
        """Semantic alias for the generic rs field."""

        return self.rs


@irdl_op_definition
class UpdParamImmOp(ImmOperation[DurationImm]):
    """Update the latched parameters and then wait for number of ns specified by
    duration."""

    name = "q1.i.upd_param"

    traits = traits_def()

    @property
    def duration(self):
        """Semantic alias for the generic imm field."""

        return self.imm


@irdl_op_definition
class PlayImmImmImmOp(ImmImmImmOperation[UI10Imm, UI10Imm, DurationImm]):
    """Update the latched parameters, interrupt waves being played and start playing AWG
    waveforms stored at indexes wave_0 on path 0 and wave_1 on path 1.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iii.play"

    traits = traits_def()

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
class PlayRsRsImmOp(RsRsImmOperation[IntRegisterType, DurationImm]):
    """Update the latched parameters, interrupt waves being played and start playing AWG
    waveforms stored at indexes wave_0 on path 0 and wave_1 on path 1.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.rri.play"

    traits = traits_def()

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
class AcquireImmImmImmOp(ImmImmImmOperation[UI5Imm, UI24Imm, DurationImm]):
    """Update the latched parameters, interrupt currently active acquisitions and start the
    acquisition specified and store data in index provided by bin.

    Integration is executed using a square weight with preset length from the QCoDeS
    parameter.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iii.acquire"

    traits = traits_def()

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
class AcquireImmRsImmOp(ImmRsImmOperation[IntRegisterType, UI5Imm, DurationImm]):
    """Update the latched parameters, interrupt currently active acquisitions and start the
    acquisition specified and store data in index provided by bin.

    Integration is executed using a square weight with preset length from the QCoDeS
    parameter.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iri.acquire"

    traits = traits_def()

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
class AcquireWeightedImmImmImmImmImmOp(
    ImmImmImmImmImmOperation[UI5Imm, UI24Imm, UI6Imm, UI6Imm, DurationImm]
):
    """Update the latched parameters, interrupt currently active acquisitions and start the
    acquisition specified and store data in index provided by bin.

    Integration is executed using weights stored at indices weight_0 for path 0 and weight_1
    for path 1.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iiiii.acquire_weighted"

    traits = traits_def()

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
class AcquireWeightedImmRsRsRsImmOp(
    ImmRsRsRsImmOperation[IntRegisterType, UI5Imm, DurationImm]
):
    """Update the latched parameters, interrupt currently active acquisitions and start the
    acquisition specified and store data in index provided by bin. Integration is executed
    using weights stored at indices weight_0 for path 0 and weight_1 for path 1.

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.irrri.acquire_weighted"

    traits = traits_def()

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
class AcquireTtlImmImmImmImmOp(
    ImmImmImmImmOperation[UI5Imm, UI24Imm, BoolImm, DurationImm]
):
    """Update the latched parameters, start the TTL trigger acquisition provided by the
    index in acquisition, store data in index provided by bin.

    Enable TTL trigger by writing 1 to enable, disable after by writing 0 to enable

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.iiii.acquire_ttl"

    traits = traits_def()

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
class AcquireTtlImmRsImmImmOp(
    ImmRsImmImmOperation[IntRegisterType, UI5Imm, BoolImm, DurationImm]
):
    """Update the latched parameters, start the TTL trigger acquisition provided by the
    index in acquisition, store data in index provided by bin.

    Enable TTL trigger by writing 1 to enable, disable afterwards by writing 0 to enable

    Duration specifies the amount of time spent at the beginning of the instruction in ns
    """

    name = "q1.irii.acquire_ttl"

    traits = traits_def()

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


# endregion

# endregion
