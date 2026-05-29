# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Q1 dialect, based on the QBlox Q1 ISA
[documentation](https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html)."""

from dataclasses import dataclass
from io import StringIO
from typing import IO

from xdsl.backend.assembly_printer import AssemblyPrinter
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Dialect
from xdsl.utils.target import Target

from .ir.attrs import LabelAttr
from .ir.ops import (
    AddRIROp,
    AddRRROp,
    AndRIROp,
    AndRRROp,
    AslRIROp,
    AslRRROp,
    AsrRIROp,
    AsrRRROp,
    DefDirectiveOp,
    IllegalOp,
    JaeIOp,
    JaeROp,
    JaIOp,
    JaROp,
    JbeIOp,
    JbeROp,
    JbIOp,
    JbROp,
    JgeRIIOp,
    JgeRIROp,
    JgIOp,
    JgROp,
    JleIOp,
    JleROp,
    JlIOp,
    JlROp,
    JltRIIOp,
    JltRIROp,
    JmpIOp,
    JmpROp,
    JnoIOp,
    JnoROp,
    JnsIOp,
    JnsROp,
    JnzIOp,
    JnzROp,
    JoIOp,
    JoROp,
    JsIOp,
    JsROp,
    JzIOp,
    JzROp,
    LabelOp,
    LoopRIOp,
    LoopRROp,
    MoveIROp,
    MoveRROp,
    NopOp,
    NotIROp,
    NotRROp,
    OrRIROp,
    OrRRROp,
    ResetPhOp,
    SetAwgGainIIOp,
    SetAwgGainRROp,
    SetAwgOffsIIOp,
    SetAwgOffsRROp,
    SetCondIIIIOp,
    SetCondRRRIOp,
    SetFreqIOp,
    SetFreqROp,
    SetMrkIOp,
    SetMrkROp,
    SetPhDeltaIOp,
    SetPhDeltaROp,
    SetPhIOp,
    SetPhROp,
    StopIOp,
    StopOp,
    StopROp,
    SubRIROp,
    SubRRROp,
    XorRIROp,
    XorRRROp,
)
from .ir.reg_desc import IntRegisterType


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    """Prints a Q1 module as Q1 assembly.

    :param module: Module containing Q1 operations.
    :param output: Text stream receiving the printed assembly.
    """

    printer = AssemblyPrinter(stream=output)
    printer.print_module(module)


def q1_code(module: ModuleOp) -> str:
    """Returns the Q1 assembly text for a Q1 module.

    :param module: Module containing Q1 dialect operations.
    :returns: Rendered Q1 assembly for the module.
    """

    stream = StringIO()
    print_assembly(module, stream)
    return stream.getvalue()


@dataclass(frozen=True)
class Q1asmTarget(Target):
    name = "q1asm"

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        """Emits a Q1 module to Q1 assembly.

        :param ctx: xDSL context for the emission target.
        :param module: Module containing Q1 dialect operations.
        :param output: Text stream receiving the printed assembly.
        """

        print_assembly(module, output)


Q1 = Dialect(
    "q1",
    [
        LabelOp,
        DefDirectiveOp,
        IllegalOp,
        StopIOp,
        StopOp,
        StopROp,
        NopOp,
        JmpIOp,
        JmpROp,
        JzIOp,
        JzROp,
        JnzIOp,
        JnzROp,
        JoIOp,
        JoROp,
        JnoIOp,
        JnoROp,
        JsIOp,
        JsROp,
        JnsIOp,
        JnsROp,
        JgIOp,
        JgROp,
        JlIOp,
        JlROp,
        JleIOp,
        JleROp,
        JaIOp,
        JaROp,
        JaeIOp,
        JaeROp,
        JbIOp,
        JbROp,
        JbeIOp,
        JbeROp,
        JgeRIIOp,
        JgeRIROp,
        JltRIIOp,
        JltRIROp,
        LoopRIOp,
        LoopRROp,
        MoveIROp,
        MoveRROp,
        NotIROp,
        NotRROp,
        AddRIROp,
        AddRRROp,
        SubRIROp,
        SubRRROp,
        AndRIROp,
        AndRRROp,
        OrRIROp,
        OrRRROp,
        XorRIROp,
        XorRRROp,
        AslRIROp,
        AslRRROp,
        AsrRIROp,
        AsrRRROp,
        SetCondIIIIOp,
        SetCondRRRIOp,
        SetMrkIOp,
        SetMrkROp,
        SetFreqIOp,
        SetFreqROp,
        ResetPhOp,
        SetPhIOp,
        SetPhROp,
        SetPhDeltaIOp,
        SetPhDeltaROp,
        SetAwgGainIIOp,
        SetAwgGainRROp,
        SetAwgOffsIIOp,
        SetAwgOffsRROp,
    ],
    [
        LabelAttr,
        IntRegisterType,
    ],
)
