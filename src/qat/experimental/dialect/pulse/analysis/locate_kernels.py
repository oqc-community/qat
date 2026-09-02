# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Contains analysis to locate all :class:`KernelOp` operations and their call sites in a
module."""

from dataclasses import dataclass

from xdsl.ir import Operation

from qat.experimental.dialect.pulse.ir import CallKernelOp, KernelOp


@dataclass
class KernelCallSites:
    """Stores a :class:`KernelOp` operation found in a module, and each of its call sites.

    :ivar symbol_name: The name of the Kernel.
    :ivar operation: The Kernel operation.
    :ivar call_sites: A sequence of operations that invoke the Kernel.
    """

    symbol_name: str
    operation: KernelOp
    call_sites: tuple[CallKernelOp, ...]


def locate_kernels(op: Operation) -> tuple[KernelCallSites, ...]:
    """Walks ``op`` and returns a :class:`KernelCallSites` for every :class:`KernelOp`
    found, pairing each kernel with all :class:`CallKernelOp` instances that reference it.

    :param op: The root operation to walk.
    :returns: One :class:`KernelCallSites` per :class:`KernelOp`, in walk order.
    """
    kernels: dict[str, KernelOp] = {}
    call_sites: dict[str, list[CallKernelOp]] = {}

    for nested_op in op.walk():
        if isinstance(nested_op, KernelOp):
            name = nested_op.sym_name.data
            kernels[name] = nested_op
            call_sites.setdefault(name, [])
        elif isinstance(nested_op, CallKernelOp):
            name = nested_op.callee.root_reference.data
            call_sites.setdefault(name, []).append(nested_op)

    return tuple(
        KernelCallSites(
            symbol_name=name,
            operation=kernel,
            call_sites=tuple(call_sites.get(name, [])),
        )
        for name, kernel in kernels.items()
    )
