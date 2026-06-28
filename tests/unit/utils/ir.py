# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Test utilities for assembling xDSL IR and querying nested operations.

These helpers are shared across unit tests.
"""

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.ir import Attribute, Block, BlockArgument, Dialect, Operation, Region
from xdsl.rewriter import InsertPoint


def create_context(*dialects: Dialect) -> Context:
    """Helper function to create a context with the given dialects loaded.

    :param dialects: The dialects to load into the context.
    :return: A context with the given dialects loaded.
    """

    context = Context()
    for dialect in dialects:
        context.load_dialect(dialect)
    return context


def build_module_from_ops(ops: list[Operation]) -> ModuleOp:
    """Helper function to build a module from a list of operations.

    Intended for simple linear programs, where the operations are expected to just be a
    single block. They're added to a "main" function, which is added to a module, and the
    module is returned.
    """

    block = Block(ops)
    region = Region(block)
    func = FuncOp("main", ([], []), region)
    module = ModuleOp([func])
    return module


def build_module_with_arguments(
    *args: Attribute,
) -> tuple[ModuleOp, tuple[BlockArgument, ...], Builder]:
    """Helper function to build a module with a main function with the given arguments.

    The arguments are added to the main function, which is added to a module, and the module
    and the list of arguments are returned.

    :param args: The argument types to add to the main function.
    :return: A tuple of the module, the list of arguments, and a builder with an insertion
        point at the start of the main function's block.
    """

    func = FuncOp("main", (args, []))
    module = ModuleOp([func])
    return (
        module,
        tuple(func.regions[0].blocks[0].args),
        Builder(InsertPoint(func.regions[0].blocks[0])),
    )


def get_operations_with_type(op: Operation, op_type: type[Operation]) -> list[Operation]:
    """Helper function to walk the operation provided, and return a list of all nested
    operations that are of the given type."""
    return [nested_op for nested_op in op.walk() if isinstance(nested_op, op_type)]
