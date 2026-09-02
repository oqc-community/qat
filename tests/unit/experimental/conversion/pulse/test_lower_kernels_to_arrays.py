# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from xdsl.context import Context
from xdsl.dialects import func
from xdsl.dialects.builtin import IntAttr, ModuleOp, i32, i64
from xdsl.ir import Block, Region

from qat.experimental.conversion.pulse.lower_kernels_to_arrays import (
    LowerKernelsToResultsArrays,
)
from qat.experimental.dialect.pulse.ir import CallKernelOp, KernelOp, ReturnOp
from qat.experimental.dialect.results.ir import (
    CreateOp,
    ExtractOp,
    RecordFieldAttr,
    RecordSchemaAttr,
    ResultsArrayType,
    ResultsCollectionType,
)


def _build_record_schema(
    type1=i32,
    type2=i64,
    key1: str = "a",
    key2: str = "b",
) -> RecordSchemaAttr:
    return RecordSchemaAttr([RecordFieldAttr(key1, type1), RecordFieldAttr(key2, type2)])


def _apply_lower_kernels_to_results_arrays(module: ModuleOp):
    LowerKernelsToResultsArrays().apply(Context(), module)


def _assert_results_array_type(array_type, expected_element_type, expected_size: IntAttr):
    assert isinstance(array_type, ResultsArrayType)
    assert array_type.type == expected_element_type
    assert array_type.size == expected_size


def _collect_ops_of_type(block: Block, op_type: type[object]) -> list[object]:
    return [op for op in block.ops if isinstance(op, op_type)]


def _assert_extract_from_collection(
    extract_op: ExtractOp,
    expected_container,
    expected_key: str,
    expected_element_type,
    expected_size: IntAttr,
):
    assert extract_op.container is expected_container
    assert extract_op.index is None
    assert extract_op.key is not None
    assert extract_op.key.data == expected_key
    _assert_results_array_type(extract_op.result.type, expected_element_type, expected_size)


def _assert_collection_create(
    create_op: CreateOp,
    expected_keys: list[str],
    expected_values: list,
    expected_element_types: list,
    expected_size: IntAttr,
):
    assert create_op.size is None
    assert list(create_op.values) == expected_values
    assert create_op.result.type == ResultsCollectionType(
        RecordSchemaAttr(
            [
                RecordFieldAttr(key, type_)
                for key, type_ in zip(expected_keys, expected_element_types, strict=True)
            ]
        ),
        expected_size,
    )


def _build_caller_func(
    kernel_name: str,
    arg_types: tuple,
    result_types: tuple,
    name: str = "main",
) -> tuple[func.FuncOp, CallKernelOp]:
    block = Block(arg_types=list(arg_types))
    call = CallKernelOp(kernel_name, list(block.args), list(result_types))
    block.add_ops([call, func.ReturnOp(*call.results)])
    return func.FuncOp(name, (arg_types, result_types), Region(block)), call


def _collect_lowered_caller_ops(
    caller: func.FuncOp,
) -> tuple[list, list, list, list]:
    """Returns (extract_ops, call_ops, create_ops, return_ops) from the caller's block."""
    block = caller.body.block
    return (
        _collect_ops_of_type(block, ExtractOp),
        _collect_ops_of_type(block, CallKernelOp),
        _collect_ops_of_type(block, CreateOp),
        _collect_ops_of_type(block, func.ReturnOp),
    )


def _build_collection_kernel_module() -> tuple[ModuleOp, KernelOp, func.FuncOp]:
    schema = _build_record_schema()
    collection_type = ResultsCollectionType(schema, IntAttr(4))

    kernel_body = Block(arg_types=[collection_type])
    created_collection = CreateOp.for_empty_collection(schema, 4)
    kernel_body.add_ops([created_collection, ReturnOp(created_collection.result)])
    kernel = KernelOp(
        "kernel",
        ((collection_type,), (collection_type,)),
        Region(kernel_body),
    )

    caller, _ = _build_caller_func("kernel", (collection_type,), (collection_type,))

    return ModuleOp([kernel, caller]), kernel, caller


class TestLowerKernelsToResultsArrays:
    """Tests for :class:`LowerKernelsToResultsArrays`."""

    def test_collection_kernel_arguments_and_returns_are_lowered(self):
        """Collection-typed kernel signatures should expand to arrays, and the caller should
        be rewritten to extract and reassemble those arrays."""

        module, kernel, caller = _build_collection_kernel_module()

        _apply_lower_kernels_to_results_arrays(module)

        expected_input_types = [
            ResultsArrayType(i32, IntAttr(4)),
            ResultsArrayType(i64, IntAttr(4)),
        ]
        expected_output_types = [
            ResultsArrayType(i32, IntAttr(4)),
            ResultsArrayType(i64, IntAttr(4)),
        ]

        assert list(kernel.function_type.inputs) == expected_input_types
        assert list(kernel.function_type.outputs) == expected_output_types

        kernel_block = kernel.body.block
        assert len(kernel_block.args) == 2
        _assert_results_array_type(kernel_block.args[0].type, i32, IntAttr(4))
        _assert_results_array_type(kernel_block.args[1].type, i64, IntAttr(4))

        kernel_create_ops = _collect_ops_of_type(kernel_block, CreateOp)
        kernel_return_ops = _collect_ops_of_type(kernel_block, ReturnOp)
        assert len(kernel_create_ops) == 2
        assert len(kernel_return_ops) == 1
        assert kernel_create_ops[0].size is None
        assert list(kernel_create_ops[0].values) == []
        assert kernel_create_ops[1].size is None
        assert list(kernel_create_ops[1].values) == []
        _assert_results_array_type(kernel_create_ops[0].result.type, i32, IntAttr(4))
        _assert_results_array_type(kernel_create_ops[1].result.type, i64, IntAttr(4))
        assert list(kernel_return_ops[0].arguments) == [
            kernel_create_ops[0].result,
            kernel_create_ops[1].result,
        ]

        caller_block = caller.body.block
        caller_extract_ops, caller_call_ops, caller_create_ops, caller_return_ops = (
            _collect_lowered_caller_ops(caller)
        )

        assert len(caller_extract_ops) == 2
        _assert_extract_from_collection(
            caller_extract_ops[0], caller_block.args[0], "a", i32, IntAttr(4)
        )
        _assert_extract_from_collection(
            caller_extract_ops[1], caller_block.args[0], "b", i64, IntAttr(4)
        )
        assert len(caller_call_ops) == 1
        assert len(caller_create_ops) == 1
        assert len(caller_return_ops) == 1

        lowered_call = caller_call_ops[0]
        lowered_arguments = list(lowered_call.arguments)
        lowered_results = list(lowered_call.result)
        assert len(lowered_arguments) == 2
        _assert_results_array_type(lowered_arguments[0].type, i32, IntAttr(4))
        _assert_results_array_type(lowered_arguments[1].type, i64, IntAttr(4))
        assert len(lowered_results) == 2
        _assert_results_array_type(lowered_results[0].type, i32, IntAttr(4))
        _assert_results_array_type(lowered_results[1].type, i64, IntAttr(4))
        _assert_collection_create(
            caller_create_ops[0],
            ["a", "b"],
            lowered_results,
            [i32, i64],
            IntAttr(4),
        )
        assert caller_return_ops[0].arguments[0] is caller_create_ops[0].result

    def test_plain_kernel_is_unchanged(self):
        """A kernel without collection-typed values should remain structurally unchanged."""

        kernel_body = Block(arg_types=[i32])
        kernel_body.add_ops([ReturnOp(kernel_body.args[0])])
        kernel = KernelOp("plain", ((i32,), (i32,)), Region(kernel_body))

        caller_block = Block(arg_types=[i32])
        call = CallKernelOp("plain", [caller_block.args[0]], [i32])
        caller_block.add_ops([call, func.ReturnOp(call.result[0])])
        caller = func.FuncOp("main", ((i32,), (i32,)), Region(caller_block))

        module = ModuleOp([kernel, caller])
        before = module.clone()

        _apply_lower_kernels_to_results_arrays(module)

        assert before.is_structurally_equivalent(module)

    def test_kernel_with_single_collection_operand_and_result(self):
        """Tests the lowering when just a single collection is fed as an operand to a
        kernel, and when the kernel returns a single collection."""

        schema = _build_record_schema()
        collection_type = ResultsCollectionType(schema, IntAttr(3))

        kernel_body = Block(arg_types=[collection_type])
        kernel_body.add_ops([ReturnOp(kernel_body.args[0])])
        kernel = KernelOp(
            "kernel", ((collection_type,), (collection_type,)), Region(kernel_body)
        )

        caller, _ = _build_caller_func("kernel", (collection_type,), (collection_type,))

        module = ModuleOp([kernel, caller])

        _apply_lower_kernels_to_results_arrays(module)

        expected_arrays = [
            ResultsArrayType(i32, IntAttr(3)),
            ResultsArrayType(i64, IntAttr(3)),
        ]
        assert list(kernel.function_type.inputs) == expected_arrays
        assert list(kernel.function_type.outputs) == expected_arrays

        lowered_kernel_block = kernel.body.block
        assert len(lowered_kernel_block.args) == 2
        _assert_results_array_type(lowered_kernel_block.args[0].type, i32, IntAttr(3))
        _assert_results_array_type(lowered_kernel_block.args[1].type, i64, IntAttr(3))

        kernel_create_ops = _collect_ops_of_type(lowered_kernel_block, CreateOp)
        kernel_return_ops = _collect_ops_of_type(lowered_kernel_block, ReturnOp)
        assert kernel_create_ops == []
        assert len(kernel_return_ops) == 1
        assert list(kernel_return_ops[0].arguments) == list(lowered_kernel_block.args)

        lowered_caller_block = caller.body.block
        caller_extract_ops, caller_call_ops, caller_create_ops, caller_return_ops = (
            _collect_lowered_caller_ops(caller)
        )

        assert len(caller_extract_ops) == 2
        _assert_extract_from_collection(
            caller_extract_ops[0], lowered_caller_block.args[0], "a", i32, IntAttr(3)
        )
        _assert_extract_from_collection(
            caller_extract_ops[1], lowered_caller_block.args[0], "b", i64, IntAttr(3)
        )
        assert len(caller_call_ops) == 1
        assert len(caller_create_ops) == 1
        assert len(caller_return_ops) == 1

        lowered_call = caller_call_ops[0]
        assert list(lowered_call.arguments) == [
            caller_extract_ops[0].result,
            caller_extract_ops[1].result,
        ]
        assert list(lowered_call.result.types) == expected_arrays
        _assert_collection_create(
            caller_create_ops[0],
            ["a", "b"],
            list(lowered_call.result),
            [i32, i64],
            IntAttr(3),
        )
        assert list(caller_return_ops[0].arguments) == [caller_create_ops[0].result]

    def test_kernel_with_single_collection_operand_and_result_within_other_args(self):
        """Tests the lowering when just a single collection is fed as an operand to a
        kernel, and when the kernel returns a single collection, but the operand and result
        is one of many operand and results.

        That is, there are many operand and results, but only one of each are collections.
        """

        schema = _build_record_schema()
        collection_type = ResultsCollectionType(schema, IntAttr(5))

        kernel_body = Block(arg_types=[i32, collection_type, i64])
        kernel_body.add_ops(
            [ReturnOp(kernel_body.args[2], kernel_body.args[1], kernel_body.args[0])]
        )
        kernel = KernelOp(
            "kernel",
            ((i32, collection_type, i64), (i64, collection_type, i32)),
            Region(kernel_body),
        )

        caller, _ = _build_caller_func(
            "kernel",
            (i32, collection_type, i64),
            (i64, collection_type, i32),
        )

        module = ModuleOp([kernel, caller])

        _apply_lower_kernels_to_results_arrays(module)

        expected_input_types = [
            i32,
            ResultsArrayType(i32, IntAttr(5)),
            ResultsArrayType(i64, IntAttr(5)),
            i64,
        ]
        expected_output_types = [
            i64,
            ResultsArrayType(i32, IntAttr(5)),
            ResultsArrayType(i64, IntAttr(5)),
            i32,
        ]
        assert list(kernel.function_type.inputs) == expected_input_types
        assert list(kernel.function_type.outputs) == expected_output_types

        lowered_kernel_block = kernel.body.block
        assert len(lowered_kernel_block.args) == 4
        assert lowered_kernel_block.args[0].type == i32
        _assert_results_array_type(lowered_kernel_block.args[1].type, i32, IntAttr(5))
        _assert_results_array_type(lowered_kernel_block.args[2].type, i64, IntAttr(5))
        assert lowered_kernel_block.args[3].type == i64

        kernel_create_ops = _collect_ops_of_type(lowered_kernel_block, CreateOp)
        kernel_return_ops = _collect_ops_of_type(lowered_kernel_block, ReturnOp)
        assert kernel_create_ops == []
        assert len(kernel_return_ops) == 1
        assert list(kernel_return_ops[0].arguments) == [
            lowered_kernel_block.args[3],
            lowered_kernel_block.args[1],
            lowered_kernel_block.args[2],
            lowered_kernel_block.args[0],
        ]

        lowered_caller_block = caller.body.block
        caller_extract_ops, caller_call_ops, caller_create_ops, caller_return_ops = (
            _collect_lowered_caller_ops(caller)
        )

        assert len(caller_extract_ops) == 2
        _assert_extract_from_collection(
            caller_extract_ops[0], lowered_caller_block.args[1], "a", i32, IntAttr(5)
        )
        _assert_extract_from_collection(
            caller_extract_ops[1], lowered_caller_block.args[1], "b", i64, IntAttr(5)
        )
        assert len(caller_call_ops) == 1
        assert len(caller_create_ops) == 1
        assert len(caller_return_ops) == 1

        lowered_call = caller_call_ops[0]
        assert list(lowered_call.arguments) == [
            lowered_caller_block.args[0],
            caller_extract_ops[0].result,
            caller_extract_ops[1].result,
            lowered_caller_block.args[2],
        ]
        assert list(lowered_call.result.types) == expected_output_types
        _assert_collection_create(
            caller_create_ops[0],
            ["a", "b"],
            [lowered_call.result[1], lowered_call.result[2]],
            [i32, i64],
            IntAttr(5),
        )
        assert list(caller_return_ops[0].arguments) == [
            lowered_call.result[0],
            caller_create_ops[0].result,
            lowered_call.result[3],
        ]

    def test_kernel_with_multiple_collection_operands_and_results(self):
        """Tests the lowering when just multiple collections are fed as operands to a
        kernel, and when the kernel returns many collections."""

        lhs_schema = _build_record_schema()
        rhs_schema = _build_record_schema(type1=i64, type2=i32, key1="x", key2="y")
        lhs_collection = ResultsCollectionType(lhs_schema, IntAttr(2))
        rhs_collection = ResultsCollectionType(rhs_schema, IntAttr(2))

        kernel_body = Block(arg_types=[lhs_collection, i32, rhs_collection])
        kernel_body.add_ops([ReturnOp(kernel_body.args[2], kernel_body.args[0])])
        kernel = KernelOp(
            "kernel",
            ((lhs_collection, i32, rhs_collection), (rhs_collection, lhs_collection)),
            Region(kernel_body),
        )

        caller, _ = _build_caller_func(
            "kernel",
            (lhs_collection, i32, rhs_collection),
            (rhs_collection, lhs_collection),
        )

        module = ModuleOp([kernel, caller])

        _apply_lower_kernels_to_results_arrays(module)

        expected_input_types = [
            ResultsArrayType(i32, IntAttr(2)),
            ResultsArrayType(i64, IntAttr(2)),
            i32,
            ResultsArrayType(i64, IntAttr(2)),
            ResultsArrayType(i32, IntAttr(2)),
        ]
        expected_output_types = [
            ResultsArrayType(i64, IntAttr(2)),
            ResultsArrayType(i32, IntAttr(2)),
            ResultsArrayType(i32, IntAttr(2)),
            ResultsArrayType(i64, IntAttr(2)),
        ]
        assert list(kernel.function_type.inputs) == expected_input_types
        assert list(kernel.function_type.outputs) == expected_output_types

        lowered_kernel_block = kernel.body.block
        assert len(lowered_kernel_block.args) == 5
        _assert_results_array_type(lowered_kernel_block.args[0].type, i32, IntAttr(2))
        _assert_results_array_type(lowered_kernel_block.args[1].type, i64, IntAttr(2))
        assert lowered_kernel_block.args[2].type == i32
        _assert_results_array_type(lowered_kernel_block.args[3].type, i64, IntAttr(2))
        _assert_results_array_type(lowered_kernel_block.args[4].type, i32, IntAttr(2))

        kernel_create_ops = _collect_ops_of_type(lowered_kernel_block, CreateOp)
        kernel_return_ops = _collect_ops_of_type(lowered_kernel_block, ReturnOp)
        assert kernel_create_ops == []
        assert len(kernel_return_ops) == 1
        assert list(kernel_return_ops[0].arguments) == [
            lowered_kernel_block.args[3],
            lowered_kernel_block.args[4],
            lowered_kernel_block.args[0],
            lowered_kernel_block.args[1],
        ]

        lowered_caller_block = caller.body.block
        caller_extract_ops, caller_call_ops, caller_create_ops, caller_return_ops = (
            _collect_lowered_caller_ops(caller)
        )

        assert len(caller_extract_ops) == 4
        _assert_extract_from_collection(
            caller_extract_ops[0], lowered_caller_block.args[0], "a", i32, IntAttr(2)
        )
        _assert_extract_from_collection(
            caller_extract_ops[1], lowered_caller_block.args[0], "b", i64, IntAttr(2)
        )
        _assert_extract_from_collection(
            caller_extract_ops[2], lowered_caller_block.args[2], "x", i64, IntAttr(2)
        )
        _assert_extract_from_collection(
            caller_extract_ops[3], lowered_caller_block.args[2], "y", i32, IntAttr(2)
        )
        assert len(caller_call_ops) == 1
        assert len(caller_create_ops) == 2
        assert len(caller_return_ops) == 1

        lowered_call = caller_call_ops[0]
        assert list(lowered_call.arguments) == [
            caller_extract_ops[0].result,
            caller_extract_ops[1].result,
            lowered_caller_block.args[1],
            caller_extract_ops[2].result,
            caller_extract_ops[3].result,
        ]
        assert list(lowered_call.result.types) == expected_output_types
        _assert_collection_create(
            caller_create_ops[0],
            ["x", "y"],
            [lowered_call.result[0], lowered_call.result[1]],
            [i64, i32],
            IntAttr(2),
        )
        _assert_collection_create(
            caller_create_ops[1],
            ["a", "b"],
            [lowered_call.result[2], lowered_call.result[3]],
            [i32, i64],
            IntAttr(2),
        )
        assert list(caller_return_ops[0].arguments) == [
            caller_create_ops[0].result,
            caller_create_ops[1].result,
        ]

    def test_collection_kernel_with_no_call_sites_is_still_lowered(self):
        """A collection-typed kernel that has no call sites should still have its body and
        signature rewritten."""

        schema = _build_record_schema()
        collection_type = ResultsCollectionType(schema, IntAttr(2))

        kernel_body = Block(arg_types=[collection_type])
        created = CreateOp.for_empty_collection(schema, 2)
        kernel_body.add_ops([created, ReturnOp(created.result)])
        kernel = KernelOp(
            "orphan",
            ((collection_type,), (collection_type,)),
            Region(kernel_body),
        )

        module = ModuleOp([kernel])

        _apply_lower_kernels_to_results_arrays(module)

        expected_arrays = [
            ResultsArrayType(i32, IntAttr(2)),
            ResultsArrayType(i64, IntAttr(2)),
        ]
        assert list(kernel.function_type.inputs) == expected_arrays
        assert list(kernel.function_type.outputs) == expected_arrays

        lowered_block = kernel.body.block
        assert len(lowered_block.args) == 2
        _assert_results_array_type(lowered_block.args[0].type, i32, IntAttr(2))
        _assert_results_array_type(lowered_block.args[1].type, i64, IntAttr(2))

        create_ops = _collect_ops_of_type(lowered_block, CreateOp)
        return_ops = _collect_ops_of_type(lowered_block, ReturnOp)
        assert len(create_ops) == 2
        assert len(return_ops) == 1
        _assert_results_array_type(create_ops[0].result.type, i32, IntAttr(2))
        _assert_results_array_type(create_ops[1].result.type, i64, IntAttr(2))

    def test_multiple_call_sites_for_same_kernel_are_all_rewritten(self):
        """Each call site referencing the same collection kernel should be independently
        expanded into extract ops, a new call op, and reassembly create ops."""

        schema = _build_record_schema()
        collection_type = ResultsCollectionType(schema, IntAttr(3))

        kernel_body = Block(arg_types=[collection_type])
        kernel_body.add_ops([ReturnOp(kernel_body.args[0])])
        kernel = KernelOp(
            "kernel",
            ((collection_type,), (collection_type,)),
            Region(kernel_body),
        )

        first_caller, _ = _build_caller_func(
            "kernel", (collection_type,), (collection_type,), name="caller_a"
        )
        second_caller, _ = _build_caller_func(
            "kernel", (collection_type,), (collection_type,), name="caller_b"
        )

        module = ModuleOp([kernel, first_caller, second_caller])

        _apply_lower_kernels_to_results_arrays(module)

        expected_arrays = [
            ResultsArrayType(i32, IntAttr(3)),
            ResultsArrayType(i64, IntAttr(3)),
        ]

        for caller_func in (first_caller, second_caller):
            lowered_block = caller_func.body.block
            extract_ops = _collect_ops_of_type(lowered_block, ExtractOp)
            call_ops = _collect_ops_of_type(lowered_block, CallKernelOp)
            create_ops = _collect_ops_of_type(lowered_block, CreateOp)

            assert len(extract_ops) == 2
            assert len(call_ops) == 1
            assert len(create_ops) == 1
            _assert_extract_from_collection(
                extract_ops[0], lowered_block.args[0], "a", i32, IntAttr(3)
            )
            _assert_extract_from_collection(
                extract_ops[1], lowered_block.args[0], "b", i64, IntAttr(3)
            )
            assert list(call_ops[0].result.types) == expected_arrays
            _assert_collection_create(
                create_ops[0], ["a", "b"], list(call_ops[0].result), [i32, i64], IntAttr(3)
            )

    def test_collection_input_only_kernel_expands_arguments_but_not_returns(self):
        """A kernel whose inputs include a collection but whose return is a plain type
        should have its arguments expanded and no reassembly create ops at call sites."""

        schema = _build_record_schema()
        collection_type = ResultsCollectionType(schema, IntAttr(6))

        kernel_body = Block(arg_types=[collection_type, i32])
        kernel_body.add_ops([ReturnOp(kernel_body.args[1])])
        kernel = KernelOp(
            "kernel",
            ((collection_type, i32), (i32,)),
            Region(kernel_body),
        )

        caller, _ = _build_caller_func("kernel", (collection_type, i32), (i32,))

        module = ModuleOp([kernel, caller])

        _apply_lower_kernels_to_results_arrays(module)

        expected_input_types = [
            ResultsArrayType(i32, IntAttr(6)),
            ResultsArrayType(i64, IntAttr(6)),
            i32,
        ]
        assert list(kernel.function_type.inputs) == expected_input_types
        assert list(kernel.function_type.outputs) == [i32]

        lowered_caller_block = caller.body.block
        extract_ops, call_ops, create_ops, _ = _collect_lowered_caller_ops(caller)

        assert len(extract_ops) == 2
        assert len(call_ops) == 1
        assert len(create_ops) == 0
        _assert_extract_from_collection(
            extract_ops[0], lowered_caller_block.args[0], "a", i32, IntAttr(6)
        )
        _assert_extract_from_collection(
            extract_ops[1], lowered_caller_block.args[0], "b", i64, IntAttr(6)
        )
        lowered_call = call_ops[0]
        assert list(lowered_call.arguments) == [
            extract_ops[0].result,
            extract_ops[1].result,
            lowered_caller_block.args[1],
        ]
        assert list(lowered_call.result.types) == [i32]

    def test_collection_return_only_kernel_expands_returns_but_not_arguments(self):
        """A kernel whose return includes a collection but whose inputs are plain types
        should have its returns expanded and no extract ops at call sites."""

        schema = _build_record_schema()
        collection_type = ResultsCollectionType(schema, IntAttr(7))

        kernel_body = Block(arg_types=[i32])
        kernel_body.add_ops([ReturnOp(kernel_body.args[0])])
        kernel = KernelOp(
            "kernel",
            ((i32,), (i32, collection_type)),
            Region(kernel_body),
        )

        caller, _ = _build_caller_func("kernel", (i32,), (i32, collection_type))

        module = ModuleOp([kernel, caller])

        _apply_lower_kernels_to_results_arrays(module)

        expected_output_types = [
            i32,
            ResultsArrayType(i32, IntAttr(7)),
            ResultsArrayType(i64, IntAttr(7)),
        ]
        assert list(kernel.function_type.inputs) == [i32]
        assert list(kernel.function_type.outputs) == expected_output_types

        lowered_caller_block = caller.body.block
        extract_ops, call_ops, create_ops, return_ops = _collect_lowered_caller_ops(caller)

        assert len(extract_ops) == 0
        assert len(call_ops) == 1
        assert len(create_ops) == 1
        assert len(return_ops) == 1

        lowered_call = call_ops[0]
        assert list(lowered_call.arguments) == [lowered_caller_block.args[0]]
        assert list(lowered_call.result.types) == expected_output_types
        _assert_collection_create(
            create_ops[0],
            ["a", "b"],
            [lowered_call.result[1], lowered_call.result[2]],
            [i32, i64],
            IntAttr(7),
        )
        assert list(return_ops[0].arguments) == [
            lowered_call.result[0],
            create_ops[0].result,
        ]
