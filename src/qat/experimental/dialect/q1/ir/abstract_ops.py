# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Abstract operation formats for the QBlox Q1 ISA.

The classes in this module describe reusable operand/result formats. For example
`RsIRd`: encodes source register, immediate, destination register, in that order.
Concrete operations in the dialect inherit one of these templates and provide
the opcode-specific name and traits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from re import compile
from typing import Generic, TypeAlias

from xdsl.backend.assembly_printer import AssemblyPrinter, OneLineAssemblyPrintable
from xdsl.backend.register_allocatable import HasRegisterConstraints, RegisterConstraints
from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.ir import Operation, OpResult, SSAValue
from xdsl.irdl import IRDLOperation, operand_def, opt_prop_def, prop_def, result_def

from qat.experimental.dialect.q1.ir.imm_desc import UI32, ui32
from qat.experimental.dialect.q1.ir.reg_desc import RInvT

_Q1_OP_NAME_PATTERN = compile("^q1\\.([^.]*)\\.([^.]*)$")

AssemblyInstructionArg: TypeAlias = IntegerAttr | SSAValue | RegisterType | StringAttr | str


def _assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isinstance(arg, IntegerAttr):
        return f"{arg.value.data}"
    elif isinstance(arg, SSAValue):
        if not isinstance(t := arg.type, RegisterType):
            raise ValueError(f"Unexpected register type {t}")
        return t.register_name.data
    elif isinstance(arg, RegisterType):
        return arg.register_name.data
    elif isinstance(arg, StringAttr):
        return arg.data

    return arg


class Q1AsmOperation(IRDLOperation, OneLineAssemblyPrintable, ABC):
    """Base class for operations that can be assembly printed."""

    comment = opt_prop_def(StringAttr)

    def assembly_mnemonic(self) -> str:
        """Because operation name must match q1.<format>.<mnemonic>, the instruction name
        (mnemonic) is extracted from the operation name to be the third field."""

        m = _Q1_OP_NAME_PATTERN.match(self.name)
        if not m:
            raise ValueError(
                "Invalid operation name. Expected format: "
                f"q1.<format>.<mnemonic>, but got: {self.name}"
            )
        return m.group(2)

    @abstractmethod
    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        """Instruction arguments in the order they should be printed in the assembly."""

        raise NotImplementedError

    def assembly_line(self) -> str | None:
        """Default assembly code generator."""

        instruction_name = self.assembly_mnemonic()
        arg_str = ", ".join(
            _assembly_arg_str(arg) for arg in self.assembly_line_args() if arg is not None
        )
        return AssemblyPrinter.assembly_line(
            instruction_name, arg_str, self.comment, is_indented=False
        )


class Q1RegAllocOperation(HasRegisterConstraints, IRDLOperation, ABC):
    """Base class for operations that can take part in register allocation."""

    def get_register_constraints(self) -> RegisterConstraints:
        """Default constraints are that all operands are "in", and all results are "out"
        registers.

        If some registers are "inout" then this function must be overridden.
        """

        return RegisterConstraints(self.operands, self.results, ())


class Q1Instruction(Q1AsmOperation, Q1RegAllocOperation, ABC):
    """Base class for operations that represent an instruction in the QBlox Q1 ISA.

    These instructions have the following format:

    [label:] mnemonic argument,argument,... [comment]

    The name of the operation will be used as the QBlox Q1 assembly mnemonic. The comment
    is optional. When present, it will be printed along with the instruction.
    """


# region Nullary format


class NullaryOperation(Q1Instruction, ABC):
    """A base class for QBlox Q1 operations that have neither sources nor destinations."""

    def __init__(
        self,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(properties={"comment": comment})

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return ()


# endregion

# region Unary formats


class IOperation(Q1Instruction, ABC):
    """A base class for QBlox Q1 operations that have one immediate operand."""

    imm = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        imm: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm, int):
            imm = IntegerAttr(imm, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            properties={
                "imm": imm,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return (self.imm,)


class RsOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have one source register."""

    rs = operand_def(RInvT)

    def __init__(
        self,
        rs: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs],
            properties={"comment": comment},
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return (self.rs,)


# endregion

# region Binary formats


class IIOperation(Q1Instruction, ABC):
    """A base class for QBlox Q1 operations that have two immediate operands."""

    imm1 = prop_def(IntegerAttr[UI32])
    imm2 = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        imm1: int | IntegerAttr[UI32],
        imm2: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm1, int):
            imm1 = IntegerAttr(imm1, ui32)

        if isinstance(imm2, int):
            imm2 = IntegerAttr(imm2, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            properties={
                "imm1": imm1,
                "imm2": imm2,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.imm1, self.imm2


class IRdOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have one immediate operand followed by one
    destination register."""

    imm = prop_def(IntegerAttr[UI32])
    rd: OpResult[RInvT] = result_def(RInvT)

    def __init__(
        self,
        imm: int | IntegerAttr[UI32],
        rd: RInvT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm, int):
            imm = IntegerAttr(imm, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            properties={
                "imm": imm,
                "comment": comment,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.imm, self.rd


class RsRdOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have one source register followed by one
    destination register."""

    rs = operand_def(RInvT)
    rd: OpResult[RInvT] = result_def(RInvT)

    def __init__(
        self,
        rs: Operation | SSAValue,
        rd: RInvT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs],
            properties={"comment": comment},
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs, self.rd


class RdIOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have one destination register followed by
    one immediate operand."""

    rd: OpResult[RInvT] = result_def(RInvT)
    imm = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        rd: RInvT,
        imm: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm, int):
            imm = IntegerAttr(imm, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            properties={
                "imm": imm,
                "comment": comment,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.rd, self.imm


class RdRsOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have one destination register followed by
    one source register."""

    rd: OpResult[RInvT] = result_def(RInvT)
    rs = operand_def(RInvT)

    def __init__(
        self,
        rd: RInvT,
        rs: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs],
            properties={"comment": comment},
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.rs


class RsRsOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have two source registers."""

    rs1 = operand_def(RInvT)
    rs2 = operand_def(RInvT)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            properties={"comment": comment},
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs1, self.rs2


class RsIOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have one source register followed by one
    immediate operand."""

    rs = operand_def(RInvT)
    imm = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        rs: RInvT,
        imm: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm, int):
            imm = IntegerAttr(imm, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs],
            properties={
                "imm": imm,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs, self.imm


class RdRdOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that produce two destination registers."""

    rd1 = result_def(RInvT)
    rd2 = result_def(RInvT)

    def __init__(
        self,
        rd1: RInvT,
        rd2: RInvT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            properties={"comment": comment},
            result_types=[rd1, rd2],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd1, self.rd2


# endregion

# region Ternary formats


class RsIIOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have one source register followed by two
    immediate operands."""

    rs = operand_def(RInvT)
    imm1 = prop_def(IntegerAttr[UI32])
    imm2 = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        rs: Operation | SSAValue,
        imm1: int | IntegerAttr[UI32],
        imm2: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm1, int):
            imm1 = IntegerAttr(imm1, ui32)

        if isinstance(imm2, int):
            imm2 = IntegerAttr(imm2, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs],
            properties={
                "imm1": imm1,
                "imm2": imm2,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs, self.imm1, self.imm2


class RsIRsOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have one immediate operand surrounded by
    two source registers."""

    rs1 = operand_def(RInvT)
    imm = prop_def(IntegerAttr[UI32])
    rs2 = operand_def(RInvT)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        imm: int | IntegerAttr[UI32],
        rs2: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm, int):
            imm = IntegerAttr(imm, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            properties={
                "imm": imm,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs1, self.imm, self.rs2


class RsIRdOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have one immediate operand surrounded by a
    source register on the left and a destination register on the right."""

    rs = operand_def(RInvT)
    imm = prop_def(IntegerAttr[UI32])
    rd: OpResult[RInvT] = result_def(RInvT)

    def __init__(
        self,
        rs: Operation | SSAValue,
        imm: int | IntegerAttr[UI32],
        rd: RInvT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm, int):
            imm = IntegerAttr(imm, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs],
            properties={
                "imm": imm,
                "comment": comment,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs, self.imm, self.rd


class RsRsIOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have two source registers followed by one
    immediate operand."""

    rs1 = operand_def(RInvT)
    rs2 = operand_def(RInvT)
    imm = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        imm: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm, int):
            imm = IntegerAttr(imm, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            properties={
                "imm": imm,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs1, self.rs2, self.imm


class RsRsRdOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have two source registers followed by one
    destination register."""

    rs1 = operand_def(RInvT)
    rs2 = operand_def(RInvT)
    rd: OpResult[RInvT] = result_def(RInvT)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        rd: RInvT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            properties={"comment": comment},
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs1, self.rs2, self.rd


class IIIOperation(Q1Instruction, ABC):
    """A base class for QBlox Q1 operations that have three immediate operands."""

    imm1 = prop_def(IntegerAttr[UI32])
    imm2 = prop_def(IntegerAttr[UI32])
    imm3 = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        imm1: int | IntegerAttr[UI32],
        imm2: int | IntegerAttr[UI32],
        imm3: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm1, int):
            imm1 = IntegerAttr(imm1, ui32)

        if isinstance(imm2, int):
            imm2 = IntegerAttr(imm2, ui32)

        if isinstance(imm3, int):
            imm3 = IntegerAttr(imm3, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            properties={
                "imm1": imm1,
                "imm2": imm2,
                "imm3": imm3,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.imm1, self.imm2, self.imm3


class IRsIOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have an immediate, a source register, and
    then another immediate."""

    imm1 = prop_def(IntegerAttr[UI32])
    rs = operand_def(RInvT)
    imm2 = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        imm1: int | IntegerAttr[UI32],
        rs: RInvT,
        imm2: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm1, int):
            imm1 = IntegerAttr(imm1, ui32)

        if isinstance(imm2, int):
            imm2 = IntegerAttr(imm2, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs],
            properties={
                "imm1": imm1,
                "imm2": imm2,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.imm1, self.rs, self.imm2


# endregion

# region Quaternary formats


class IIIIOperation(Q1Instruction, ABC):
    """A base class for QBlox Q1 operations that have four immediate operands."""

    imm1 = prop_def(IntegerAttr[UI32])
    imm2 = prop_def(IntegerAttr[UI32])
    imm3 = prop_def(IntegerAttr[UI32])
    imm4 = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        imm1: int | IntegerAttr[UI32],
        imm2: int | IntegerAttr[UI32],
        imm3: int | IntegerAttr[UI32],
        imm4: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm1, int):
            imm1 = IntegerAttr(imm1, ui32)

        if isinstance(imm2, int):
            imm2 = IntegerAttr(imm2, ui32)

        if isinstance(imm3, int):
            imm3 = IntegerAttr(imm3, ui32)

        if isinstance(imm4, int):
            imm4 = IntegerAttr(imm4, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            properties={
                "imm1": imm1,
                "imm2": imm2,
                "imm3": imm3,
                "imm4": imm4,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.imm1, self.imm2, self.imm3, self.imm4


class IRsIIOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have one immediate operand followed by one
    source register followed by two immediate operands."""

    imm1 = prop_def(IntegerAttr[UI32])
    rs = operand_def(RInvT)
    imm2 = prop_def(IntegerAttr[UI32])
    imm3 = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        imm1: int | IntegerAttr[UI32],
        rs: Operation | SSAValue,
        imm2: int | IntegerAttr[UI32],
        imm3: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm1, int):
            imm1 = IntegerAttr(imm1, ui32)

        if isinstance(imm2, int):
            imm2 = IntegerAttr(imm2, ui32)

        if isinstance(imm3, int):
            imm3 = IntegerAttr(imm3, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs],
            properties={
                "comment": comment,
                "imm1": imm1,
                "imm2": imm2,
                "imm3": imm3,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.imm1, self.rs, self.imm2, self.imm3


class RsRsRsIOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have three source registers followed by one
    immediate operand."""

    rs1 = operand_def(RInvT)
    rs2 = operand_def(RInvT)
    rs3 = operand_def(RInvT)
    imm = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        rs3: Operation | SSAValue,
        imm: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm, int):
            imm = IntegerAttr(imm, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2, rs3],
            properties={
                "imm": imm,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs1, self.rs2, self.rs3, self.imm


# endregion

# region Quinary formats


class IIIIIOperation(Q1Instruction, ABC):
    """A base class for QBlox Q1 operations that have five immediate operands."""

    imm1 = prop_def(IntegerAttr[UI32])
    imm2 = prop_def(IntegerAttr[UI32])
    imm3 = prop_def(IntegerAttr[UI32])
    imm4 = prop_def(IntegerAttr[UI32])
    imm5 = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        imm1: int | IntegerAttr[UI32],
        imm2: int | IntegerAttr[UI32],
        imm3: int | IntegerAttr[UI32],
        imm4: int | IntegerAttr[UI32],
        imm5: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm1, int):
            imm1 = IntegerAttr(imm1, ui32)

        if isinstance(imm2, int):
            imm2 = IntegerAttr(imm2, ui32)

        if isinstance(imm3, int):
            imm3 = IntegerAttr(imm3, ui32)

        if isinstance(imm4, int):
            imm4 = IntegerAttr(imm4, ui32)

        if isinstance(imm5, int):
            imm5 = IntegerAttr(imm5, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            properties={
                "comment": comment,
                "imm1": imm1,
                "imm2": imm2,
                "imm3": imm3,
                "imm4": imm4,
                "imm5": imm5,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.imm1, self.imm2, self.imm3, self.imm4, self.imm5


class IRsRsRsIOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have three source registers surrounded by
    two immediate operands."""

    imm1 = prop_def(IntegerAttr[UI32])
    rs1 = operand_def(RInvT)
    rs2 = operand_def(RInvT)
    rs3 = operand_def(RInvT)
    imm2 = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        imm1: int | IntegerAttr[UI32],
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        rs3: Operation | SSAValue,
        imm2: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm1, int):
            imm1 = IntegerAttr(imm1, ui32)

        if isinstance(imm2, int):
            imm2 = IntegerAttr(imm2, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2, rs3],
            properties={
                "comment": comment,
                "imm1": imm1,
                "imm2": imm2,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.imm1, self.rs1, self.rs2, self.rs3, self.imm2


class IRsIRsIOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have an immediate operand surrounded by two
    source registers, surrounded by two immediate operands."""

    imm1 = prop_def(IntegerAttr[UI32])
    rs1 = operand_def(RInvT)
    imm2 = prop_def(IntegerAttr[UI32])
    rs2 = operand_def(RInvT)
    imm3 = prop_def(IntegerAttr[UI32])

    def __init__(
        self,
        imm1: int | IntegerAttr[UI32],
        rs1: Operation | SSAValue,
        imm2: int | IntegerAttr[UI32],
        rs2: Operation | SSAValue,
        imm3: int | IntegerAttr[UI32],
        comment: str | StringAttr | None = None,
    ):
        if isinstance(imm1, int):
            imm1 = IntegerAttr(imm1, ui32)

        if isinstance(imm2, int):
            imm2 = IntegerAttr(imm2, ui32)

        if isinstance(imm3, int):
            imm3 = IntegerAttr(imm3, ui32)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            properties={
                "comment": comment,
                "imm1": imm1,
                "imm2": imm2,
                "imm3": imm3,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.imm1, self.rs1, self.imm2, self.rs2, self.imm3


# endregion
