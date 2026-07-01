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
from collections.abc import Sequence
from re import compile
from typing import Generic, TypeAlias

from xdsl.backend.assembly_printer import AssemblyPrinter, OneLineAssemblyPrintable
from xdsl.backend.register_allocatable import HasRegisterConstraints, RegisterConstraints
from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Operation, OpResult, SSAValue
from xdsl.irdl import IRDLOperation, operand_def, opt_prop_def, prop_def, result_def
from xdsl.parser import Parser
from xdsl.printer import Printer

from qat.experimental.dialect.q1.ir.imm_desc import (
    ImmT,
    ImmT1,
    ImmT2,
    ImmT3,
    ImmT4,
    ImmT5,
    Q1Imm,
)
from qat.experimental.dialect.q1.ir.reg_desc import RInvT

_Q1_OP_NAME_PATTERN = compile("^q1\\.([^.]*)\\.([^.]*)$")

AssemblyInstructionArg: TypeAlias = Q1Imm | SSAValue | RegisterType | StringAttr | str


def _assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isinstance(arg, Q1Imm):
        return str(arg.data)
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
    """Base class for operations that can be printed and parsed.

    Printing support textual IR and assembly which are implemented by :meth:`print`
    and :meth:`assembly_line` respectively.

    Parsing currently covers parsing only textual IR which is implemented
    by :meth:`parse` and :meth:`parse_op_type`
    """

    comment = opt_prop_def(StringAttr)

    @classmethod
    def parse(cls, parser: Parser) -> Q1AsmOperation:
        """Parse a Q1 operation from IR textual form.

        Expects the MLIR generic operation format:
            op-name(operands) <{properties}> {attributes} : (input_types) -> (result_types)
        """

        operands = parser.parse_op_args_list()

        properties = parser.parse_optional_properties_dict()
        attributes = parser.parse_optional_attr_dict()

        operand_types, result_types = cls.parse_op_type(parser)
        operand_position = parser.pos

        for property_name in cls.get_irdl_definition().properties:
            if property_name in attributes:
                properties.setdefault(property_name, attributes.pop(property_name))

        resolved_operands = parser.resolve_operands(
            operands,
            operand_types,
            operand_position,
        )
        return cls.create(
            operands=resolved_operands,
            result_types=result_types,
            properties=properties,
            attributes=attributes,
        )

    @classmethod
    def parse_op_type(
        cls, parser: Parser
    ) -> tuple[Sequence[Attribute], Sequence[Attribute]]:
        """Parse the operation type (operand and result types)."""

        parser.parse_punctuation(":")
        func_type = parser.parse_function_type()
        return func_type.inputs.data, func_type.outputs.data

    def print(self, printer: Printer) -> None:
        """Print a Q1 operation in IR textual form."""

        printer.print_string(" (")
        printer.print_list(self.operands, printer.print_operand)
        printer.print_string(")")
        if self.properties:
            printer.print_string(" <")
            printer.print_attr_dict(self.properties)
            printer.print_string(">")
        if self.attributes:
            printer.print_op_attributes(self.attributes)
        self.print_op_type(printer)

    def print_op_type(self, printer: Printer) -> None:
        """Print the operation type signature."""

        printer.print_string(" : ")
        printer.print_operation_type(self)

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
        """Emits Q1 assembly instruction line corresponding to this operation."""

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


class ImmOperation(Q1Instruction, ABC, Generic[ImmT]):
    """A base class for QBlox Q1 operations that have one immediate operand."""

    imm = prop_def(ImmT)

    def __init__(
        self,
        imm: ImmT,
        comment: str | StringAttr | None = None,
    ):
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


class ImmImmOperation(Q1Instruction, ABC, Generic[ImmT1, ImmT2]):
    """A base class for QBlox Q1 operations that have two immediate operands."""

    imm1 = prop_def(ImmT1)
    imm2 = prop_def(ImmT2)

    def __init__(
        self,
        imm1: ImmT1,
        imm2: ImmT2,
        comment: str | StringAttr | None = None,
    ):
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


class ImmRdOperation(Q1Instruction, ABC, Generic[RInvT, ImmT]):
    """A base class for QBlox Q1 operations that have one immediate operand followed by one
    destination register."""

    imm = prop_def(ImmT)
    rd: OpResult[RInvT] = result_def(RInvT)

    def __init__(
        self,
        imm: ImmT,
        rd: RInvT,
        comment: str | StringAttr | None = None,
    ):
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


class RdImmOperation(Q1Instruction, ABC, Generic[RInvT, ImmT]):
    """A base class for QBlox Q1 operations that have one destination register followed by
    one immediate operand."""

    rd: OpResult[RInvT] = result_def(RInvT)
    imm = prop_def(ImmT)

    def __init__(
        self,
        rd: RInvT,
        imm: ImmT,
        comment: str | StringAttr | None = None,
    ):
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


class RsImmOperation(Q1Instruction, ABC, Generic[RInvT, ImmT]):
    """A base class for QBlox Q1 operations that have one source register followed by one
    immediate operand."""

    rs = operand_def(RInvT)
    imm = prop_def(ImmT)

    def __init__(
        self,
        rs: RInvT,
        imm: ImmT,
        comment: str | StringAttr | None = None,
    ):
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


class ImmRsOperation(Q1Instruction, ABC, Generic[RInvT, ImmT]):
    """A base class for QBlox Q1 operations that have one immediate operand followed by one
    source register."""

    imm = prop_def(ImmT)
    rs = operand_def(RInvT)

    def __init__(
        self,
        imm: ImmT,
        rs: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
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
        return self.imm, self.rs


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


class RsImmImmOperation(Q1Instruction, ABC, Generic[RInvT, ImmT1, ImmT2]):
    """A base class for QBlox Q1 operations that have one source register followed by two
    immediate operands."""

    rs = operand_def(RInvT)
    imm1 = prop_def(ImmT1)
    imm2 = prop_def(ImmT2)

    def __init__(
        self,
        rs: Operation | SSAValue,
        imm1: ImmT1,
        imm2: ImmT2,
        comment: str | StringAttr | None = None,
    ):
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


class RsImmRsOperation(Q1Instruction, ABC, Generic[RInvT, ImmT]):
    """A base class for QBlox Q1 operations that have one immediate operand surrounded by
    two source registers."""

    rs1 = operand_def(RInvT)
    imm = prop_def(ImmT)
    rs2 = operand_def(RInvT)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        imm: ImmT,
        rs2: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
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


class RsImmRdOperation(Q1Instruction, ABC, Generic[RInvT, ImmT]):
    """A base class for QBlox Q1 operations that have one immediate operand surrounded by a
    source register on the left and a destination register on the right."""

    rs = operand_def(RInvT)
    imm = prop_def(ImmT)
    rd: OpResult[RInvT] = result_def(RInvT)

    def __init__(
        self,
        rs: Operation | SSAValue,
        imm: ImmT,
        rd: RInvT,
        comment: str | StringAttr | None = None,
    ):
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


class RsRsImmOperation(Q1Instruction, ABC, Generic[RInvT, ImmT]):
    """A base class for QBlox Q1 operations that have two source registers followed by one
    immediate operand."""

    rs1 = operand_def(RInvT)
    rs2 = operand_def(RInvT)
    imm = prop_def(ImmT)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        imm: ImmT,
        comment: str | StringAttr | None = None,
    ):
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


class ImmRsRdOperation(Q1Instruction, ABC, Generic[RInvT, ImmT]):
    """A base class for QBlox Q1 operations that have one immediate operand followed by one
    source register and then one destination register."""

    imm = prop_def(ImmT)
    rs = operand_def(RInvT)
    rd: OpResult[RInvT] = result_def(RInvT)

    def __init__(
        self,
        imm: ImmT,
        rs: Operation | SSAValue,
        rd: RInvT,
        comment: str | StringAttr | None = None,
    ):
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
        return self.imm, self.rs, self.rd


class ImmImmImmOperation(Q1Instruction, ABC, Generic[ImmT1, ImmT2, ImmT3]):
    """A base class for QBlox Q1 operations that have three immediate operands."""

    imm1 = prop_def(ImmT1)
    imm2 = prop_def(ImmT2)
    imm3 = prop_def(ImmT3)

    def __init__(
        self,
        imm1: ImmT1,
        imm2: ImmT2,
        imm3: ImmT3,
        comment: str | StringAttr | None = None,
    ):
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


class ImmRsImmOperation(Q1Instruction, ABC, Generic[RInvT, ImmT1, ImmT2]):
    """A base class for QBlox Q1 operations that have an immediate, a source register, and
    then another immediate."""

    imm1 = prop_def(ImmT1)
    rs = operand_def(RInvT)
    imm2 = prop_def(ImmT2)

    def __init__(
        self,
        imm1: ImmT1,
        rs: RInvT,
        imm2: ImmT2,
        comment: str | StringAttr | None = None,
    ):
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


class ImmImmImmImmOperation(Q1Instruction, ABC, Generic[ImmT1, ImmT2, ImmT3, ImmT4]):
    """A base class for QBlox Q1 operations that have four immediate operands."""

    imm1 = prop_def(ImmT1)
    imm2 = prop_def(ImmT2)
    imm3 = prop_def(ImmT3)
    imm4 = prop_def(ImmT4)

    def __init__(
        self,
        imm1: ImmT1,
        imm2: ImmT2,
        imm3: ImmT3,
        imm4: ImmT4,
        comment: str | StringAttr | None = None,
    ):
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


class ImmRsImmImmOperation(Q1Instruction, ABC, Generic[RInvT, ImmT1, ImmT2, ImmT3]):
    """A base class for QBlox Q1 operations that have one immediate operand followed by one
    source register followed by two immediate operands."""

    imm1 = prop_def(ImmT1)
    rs = operand_def(RInvT)
    imm2 = prop_def(ImmT2)
    imm3 = prop_def(ImmT3)

    def __init__(
        self,
        imm1: ImmT1,
        rs: Operation | SSAValue,
        imm2: ImmT2,
        imm3: ImmT3,
        comment: str | StringAttr | None = None,
    ):
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


class RsRsRsImmOperation(Q1Instruction, ABC, Generic[RInvT, ImmT]):
    """A base class for QBlox Q1 operations that have three source registers followed by one
    immediate operand."""

    rs1 = operand_def(RInvT)
    rs2 = operand_def(RInvT)
    rs3 = operand_def(RInvT)
    imm = prop_def(ImmT)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        rs3: Operation | SSAValue,
        imm: ImmT,
        comment: str | StringAttr | None = None,
    ):
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


class ImmImmImmImmImmOperation(
    Q1Instruction, ABC, Generic[ImmT1, ImmT2, ImmT3, ImmT4, ImmT5]
):
    """A base class for QBlox Q1 operations that have five immediate operands."""

    imm1 = prop_def(ImmT1)
    imm2 = prop_def(ImmT2)
    imm3 = prop_def(ImmT3)
    imm4 = prop_def(ImmT4)
    imm5 = prop_def(ImmT5)

    def __init__(
        self,
        imm1: ImmT1,
        imm2: ImmT2,
        imm3: ImmT3,
        imm4: ImmT4,
        imm5: ImmT5,
        comment: str | StringAttr | None = None,
    ):
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


class ImmRsRsRsImmOperation(Q1Instruction, ABC, Generic[RInvT, ImmT1, ImmT2]):
    """A base class for QBlox Q1 operations that have three source registers surrounded by
    two immediate operands."""

    imm1 = prop_def(ImmT1)
    rs1 = operand_def(RInvT)
    rs2 = operand_def(RInvT)
    rs3 = operand_def(RInvT)
    imm2 = prop_def(ImmT2)

    def __init__(
        self,
        imm1: ImmT1,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        rs3: Operation | SSAValue,
        imm2: ImmT2,
        comment: str | StringAttr | None = None,
    ):
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


class RsRsRdRdOperation(Q1Instruction, ABC, Generic[RInvT]):
    """A base class for QBlox Q1 operations that have two source registers followed by two
    destination registers."""

    rs1 = operand_def(RInvT)
    rs2 = operand_def(RInvT)
    rd1: OpResult[RInvT] = result_def(RInvT)
    rd2: OpResult[RInvT] = result_def(RInvT)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        rd1: RInvT,
        rd2: RInvT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            properties={"comment": comment},
            result_types=[rd1, rd2],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs1, self.rs2, self.rd1, self.rd2


class RsImmRdRdOperation(Q1Instruction, ABC, Generic[RInvT, ImmT]):
    """A base class for QBlox Q1 operations that have one source register followed by one
    immediate operand and then two destination registers."""

    rs = operand_def(RInvT)
    imm = prop_def(ImmT)
    rd1: OpResult[RInvT] = result_def(RInvT)
    rd2: OpResult[RInvT] = result_def(RInvT)

    def __init__(
        self,
        rs: Operation | SSAValue,
        imm: ImmT,
        rd1: RInvT,
        rd2: RInvT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs],
            properties={
                "imm": imm,
                "comment": comment,
            },
            result_types=[rd1, rd2],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs, self.imm, self.rd1, self.rd2


class ImmRsRdRdOperation(Q1Instruction, ABC, Generic[RInvT, ImmT]):
    """A base class for QBlox Q1 operations that have one immediate operand followed by one
    source register and then two destination registers."""

    imm = prop_def(ImmT)
    rs = operand_def(RInvT)
    rd1: OpResult[RInvT] = result_def(RInvT)
    rd2: OpResult[RInvT] = result_def(RInvT)

    def __init__(
        self,
        imm: ImmT,
        rs: Operation | SSAValue,
        rd1: RInvT,
        rd2: RInvT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs],
            properties={
                "imm": imm,
                "comment": comment,
            },
            result_types=[rd1, rd2],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.imm, self.rs, self.rd1, self.rd2


# endregion
