# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.ir import Block, Region

from qat.experimental.dialect.pulse.analysis.locate_kernels import locate_kernels
from qat.experimental.dialect.pulse.ir import CallKernelOp, KernelOp, ReturnOp


def _build_kernel(name: str) -> KernelOp:
    kernel_block = Block(arg_types=[i32])
    kernel_block.add_ops([ReturnOp(kernel_block.args[0])])
    return KernelOp(name, ((i32,), (i32,)), Region(kernel_block))


def _build_caller(
    name: str,
    callees: list[str],
) -> tuple[func.FuncOp, list[CallKernelOp]]:
    caller_block = Block(arg_types=[i32])
    call_ops = [CallKernelOp(callee, [caller_block.args[0]], [i32]) for callee in callees]
    caller_block.add_ops([*call_ops, func.ReturnOp(caller_block.args[0])])
    return func.FuncOp(name, ((i32,), (i32,)), Region(caller_block)), call_ops


class TestLocateKernels:
    """Tests for :func:`locate_kernels`."""

    def test_returns_empty_tuple_when_module_has_no_kernels(self):
        """Modules without kernels should produce no kernel-callsite mappings, even when
        orphan call sites are present."""

        caller, _ = _build_caller("main", ["missing_kernel"])
        module = ModuleOp([caller])

        assert locate_kernels(module) == ()

    def test_groups_call_sites_by_kernel_and_preserves_kernel_walk_order(self):
        """Located kernels should be returned in walk order, and each should carry the exact
        call sites that reference it in encounter order."""

        first_kernel = _build_kernel("first")
        second_kernel = _build_kernel("second")
        first_caller, first_calls = _build_caller("caller_one", ["second", "first"])
        second_caller, second_calls = _build_caller("caller_two", ["first"])
        module = ModuleOp([first_kernel, first_caller, second_kernel, second_caller])

        located_kernels = locate_kernels(module)

        assert [located.symbol_name for located in located_kernels] == ["first", "second"]
        assert [located.operation for located in located_kernels] == [
            first_kernel,
            second_kernel,
        ]
        assert located_kernels[0].call_sites == (first_calls[1], second_calls[0])
        assert located_kernels[1].call_sites == (first_calls[0],)

    def test_includes_kernels_without_call_sites(self):
        """A discovered kernel with no matching calls should still be returned with an empty
        call-site tuple."""

        idle_kernel = _build_kernel("idle")
        active_kernel = _build_kernel("active")
        caller, call_ops = _build_caller("main", ["active", "missing_kernel"])
        module = ModuleOp([idle_kernel, active_kernel, caller])

        located_kernels = locate_kernels(module)

        assert [located.symbol_name for located in located_kernels] == ["idle", "active"]
        assert located_kernels[0].operation is idle_kernel
        assert located_kernels[0].call_sites == ()
        assert located_kernels[1].operation is active_kernel
        assert located_kernels[1].call_sites == (call_ops[0],)
