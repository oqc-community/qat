# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Lower pulse kernels from collection-typed results to array-typed results.

This module provides the inter-dialect pass that rewrites each
:class:`~qat.experimental.dialect.pulse.ir.KernelOp` signature from
``ResultsCollectionType`` values to ``ResultsArrayType`` values, then updates all call
sites to match the expanded kernel signature.

The pass depends on the generic results conversion pass to lower the kernel body before
rewriting the kernel signature and its callers.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.ir import Attribute, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter

from qat.experimental.dialect.pulse.analysis.locate_kernels import locate_kernels
from qat.experimental.dialect.pulse.ir import CallKernelOp
from qat.experimental.dialect.results.ir import (
    CreateOp,
    ExtractOp,
    ResultsArrayType,
    ResultsCollectionType,
)
from qat.experimental.dialect.results.transforms.convert_collections_to_arrays import (
    collection_type_to_array_types,
    convert_results_collections_to_arrays,
)


@dataclass
class _KernelSignatureExpansion:
    """Pre-computed expansion plan for a kernel's collection-typed arguments and returns.

    Built once per :class:`~qat.experimental.dialect.pulse.ir.KernelOp` from its
    function type. Encapsulates the logic for expanding a single
    :class:`~qat.experimental.dialect.pulse.ir.CallKernelOp` into extract ops, a new call
    op, and reassembly ``CreateOp``s.
    """

    new_argument_types: list[Attribute]
    argument_rewrites: dict[int, tuple[tuple[str, ResultsArrayType], ...]]
    new_return_types: list[Attribute]
    return_rewrites: dict[int, tuple[tuple[str, ResultsArrayType], ...]]

    @classmethod
    def from_function_type(cls, function_type: FunctionType) -> "_KernelSignatureExpansion":
        new_argument_types: list[Attribute] = []
        argument_rewrites: dict[int, tuple[tuple[str, ResultsArrayType], ...]] = {}
        for i, arg in enumerate(function_type.inputs):
            if isinstance(arg, ResultsCollectionType):
                arrays = tuple(collection_type_to_array_types(arg).items())
                argument_rewrites[i] = arrays
                new_argument_types.extend(array_type for _, array_type in arrays)
            else:
                new_argument_types.append(arg)

        new_return_types: list[Attribute] = []
        return_rewrites: dict[int, tuple[tuple[str, ResultsArrayType], ...]] = {}
        for i, ret in enumerate(function_type.outputs):
            if isinstance(ret, ResultsCollectionType):
                arrays = tuple(collection_type_to_array_types(ret).items())
                return_rewrites[i] = arrays
                new_return_types.extend(array_type for _, array_type in arrays)
            else:
                new_return_types.append(ret)

        return cls(
            new_argument_types=new_argument_types,
            argument_rewrites=argument_rewrites,
            new_return_types=new_return_types,
            return_rewrites=return_rewrites,
        )

    @property
    def has_rewrites(self) -> bool:
        """Returns ``True`` if any argument or return requires expansion."""
        return bool(self.argument_rewrites or self.return_rewrites)


def expand_call_site(
    expansion: _KernelSignatureExpansion, call_site: CallKernelOp
) -> tuple[list[CallKernelOp | CreateOp | ExtractOp], list[SSAValue]]:
    """Builds the replacement operations and result SSA values for ``call_site``.

    Returns a tuple of ``(operations, replacement_results)`` suitable for passing
    directly to :func:`~xdsl.rewriter.Rewriter.replace_op`. ``operations`` is ordered
    as: extract ops → new call op → reassembly create ops.

    :param expansion: The precomputed kernel signature expansion.
    :param call_site: The call site to expand.
    :returns: Replacement operations and SSA values in original result order.
    """
    operations: list[CallKernelOp | CreateOp | ExtractOp] = []
    operands = []
    for index, operand in enumerate(call_site.operands):
        if (arg_arrays := expansion.argument_rewrites.get(index)) is None:
            operands.append(operand)
        else:
            for key, _ in arg_arrays:
                extract_op = ExtractOp.array_from_collection(operand, key)
                operations.append(extract_op)
                operands.append(extract_op.result)

    result_types = []
    for index, result_type in enumerate(call_site.result.types):
        if (ret_arrays := expansion.return_rewrites.get(index)) is None:
            result_types.append(result_type)
        else:
            result_types.extend(array_type for _, array_type in ret_arrays)

    new_call_op = CallKernelOp(call_site.callee, operands, result_types)
    operations.append(new_call_op)

    new_results_iter = iter(new_call_op.result)
    replacement_results: list[SSAValue] = []
    for index, _ in enumerate(call_site.result):
        if (ret_arrays := expansion.return_rewrites.get(index)) is None:
            replacement_results.append(next(new_results_iter))
        else:
            array_values = [next(new_results_iter) for _ in ret_arrays]
            create_op = CreateOp.for_collection_from_arrays(
                [key for key, _ in ret_arrays], array_values
            )
            operations.append(create_op)
            replacement_results.append(create_op.result)

    return operations, replacement_results


class LowerKernelsToResultsArrays(ModulePass):
    """Lower pulse kernel bodies, signatures, and call sites to arrays.

    The pass first rewrites collection-typed values inside each kernel body, then updates
    the kernel signature and every call site to match the lowered array-based form.

    Example:

    .. code-block:: mlir

        // Before
        module {
            pulse.kernel @accumulate() -> (!results.collection<"a": i32, "b": i64>[2]) {
                %c0 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                %acc = results.create : !results.collection<"a": i32, "b": i64>[2]
                scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %acc)
                        -> (!results.collection<"a": i32, "b": i64>[2]) {
                    %a = arith.constant 1 : i32
                    %b = arith.constant 2 : i64
                    %next = results.store %current[%i] with %a
                        : !results.collection<"a": i32, "b": i64>[2]
                    %next2 = results.store %next[%i] with %b
                        : !results.collection<"a": i32, "b": i64>[2]
                    scf.yield %next2 : !results.collection<"a": i32, "b": i64>[2]
                }
                pulse.return %acc : !results.collection<"a": i32, "b": i64>[2]
            }

            func.func @main() -> (!results.collection<"a": i32, "b": i64>[2]) {
                %result = pulse.call_kernel @accumulate() : () ->
                    (!results.collection<"a": i32, "b": i64>[2])
                func.return %result : !results.collection<"a": i32, "b": i64>[2]
            }
        }

        // After
        module {
            pulse.kernel @accumulate() -> (!results.array<i32>[2], !results.array<i64>[2]) {
                %c0 = arith.constant 0 : index
                %c2 = arith.constant 2 : index
                %c1 = arith.constant 1 : index
                %a = results.create : !results.array<i32>[2]
                %b = results.create : !results.array<i64>[2]
                %a_final, %b_final = scf.for %i = %c0 to %c2 step %c1
                        iter_args(%a_current = %a, %b_current = %b)
                        -> (!results.array<i32>[2], !results.array<i64>[2]) {
                    %a_value = arith.constant 1 : i32
                    %b_value = arith.constant 2 : i64
                    %a_next = results.store %a_current[%i] with %a_value
                        : !results.array<i32>[2]
                    %b_next = results.store %b_current[%i] with %b_value
                        : !results.array<i64>[2]
                    scf.yield %a_next, %b_next
                        : !results.array<i32>[2], !results.array<i64>[2]
                }
                pulse.return %a_final, %b_final
                    : !results.array<i32>[2], !results.array<i64>[2]
            }

            func.func @main() -> (!results.collection<"a": i32, "b": i64>[2]) {
                %a, %b = pulse.call_kernel @accumulate() : () ->
                    (!results.array<i32>[2], !results.array<i64>[2])
                %result = results.create %a, %b : !results.collection<"a": i32, "b": i64>[2]
                func.return %result : !results.collection<"a": i32, "b": i64>[2]
            }
        }
    """

    name = "lower-kernels-to-results-arrays"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Rewrite kernel signatures and all matching call sites in ``op``.

        :param ctx: The active xDSL context. It is currently unused, but required by the
            :class:`~xdsl.passes.ModulePass` interface.
        :param op: The module containing pulse kernels and their call sites.
        :returns: ``None``. The module is rewritten in place.
        """
        for kernel in locate_kernels(op):
            kernel_op = kernel.operation
            expansion = _KernelSignatureExpansion.from_function_type(
                kernel_op.function_type
            )

            convert_results_collections_to_arrays(kernel_op)
            kernel_op.function_type = FunctionType.from_lists(
                expansion.new_argument_types, expansion.new_return_types
            )

            if not kernel.call_sites or not expansion.has_rewrites:
                continue

            for call_site in kernel.call_sites:
                operations, replacement_results = expand_call_site(expansion, call_site)
                Rewriter.replace_op(call_site, operations, replacement_results)
