# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects import cf, func, scf
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import IndexType, IntAttr, i32, i64
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.results.ir import (
    CreateOp,
    ExtractOp,
    RecordFieldAttr,
    RecordSchemaAttr,
    RecordType,
    ResultsArrayType,
    ResultsCollectionType,
    StoreOp,
)
from qat.experimental.dialect.results.transforms.convert_collections_to_arrays import (
    collection_type_to_array_types,
    convert_results_collections_to_arrays,
)


@irdl_op_definition
class _MockTypedSourceOp(IRDLOperation):
    name = "results.test_typed_source"
    result = result_def()

    def __init__(self, result_type):
        super().__init__(result_types=[result_type])


@irdl_op_definition
class _MockNonVariadicConsumerOp(IRDLOperation):
    name = "results.test_non_variadic_consumer"
    value = operand_def()

    def __init__(self, value: Operation | SSAValue):
        super().__init__(operands=[value])


@irdl_op_definition
class _MockVariadicConsumerOp(IRDLOperation):
    name = "results.test_variadic_consumer"
    values = var_operand_def()

    def __init__(self, *values: Operation | SSAValue):
        super().__init__(operands=[list(values)])


@irdl_op_definition
class _MockVariadicProducerOp(IRDLOperation):
    name = "results.test_variadic_producer"
    values = var_result_def()

    def __init__(self, result_types):
        super().__init__(result_types=[list(result_types)])


@irdl_op_definition
class _MockMixedProducerOp(IRDLOperation):
    """An op with both a non-variadic result and a variadic result, used to exercise the
    non-variadic branch of :class:`_OpSpec.from_op`."""

    name = "results.test_mixed_producer"
    fixed = result_def()
    values = var_result_def()

    def __init__(self, fixed_type, variadic_types):
        super().__init__(result_types=[fixed_type, list(variadic_types)])


@irdl_op_definition
class _SingleRegionWrapperOp(IRDLOperation):
    name = "results.test_single_region_wrapper"
    body = region_def("single_block")

    def __init__(self, block: Block):
        super().__init__(regions=[Region(block)])


def _schema(
    type1=i32,
    type2=i64,
    key1: str = "a",
    key2: str = "b",
) -> RecordSchemaAttr:
    return RecordSchemaAttr([RecordFieldAttr(key1, type1), RecordFieldAttr(key2, type2)])


def _const_i32(value: int = 1) -> ArithConstantOp:
    return ArithConstantOp.from_int_and_width(value, i32)


def _const_i64(value: int = 2) -> ArithConstantOp:
    return ArithConstantOp.from_int_and_width(value, i64)


def _func_with_ops(*ops: Operation):
    return func.FuncOp("main", ((), ()), Region(Block([*ops, func.ReturnOp()])))


def _apply_in_function(*ops: Operation) -> Block:
    fn = _func_with_ops(*ops)
    convert_results_collections_to_arrays(fn)
    return fn.body.block


def _ops_of_type(block: Block, op_type: type[Operation]) -> list[Operation]:
    return [op for op in block.ops if isinstance(op, op_type)]


def _assert_array_type(array_type, expected_element_type, expected_size: IntAttr):
    assert isinstance(array_type, ResultsArrayType)
    assert array_type.type == expected_element_type
    assert array_type.size == expected_size


class TestCollectionTypeToArrayTypes:
    """Tests for :func:`collection_type_to_array_types`."""

    def test_returns_one_entry_per_schema_field(self):
        """Each schema field produces exactly one array type in the result."""
        schema = _schema(type1=i32, type2=i64)
        collection = ResultsCollectionType(schema, IntAttr(4))
        result = collection_type_to_array_types(collection)
        assert list(result.keys()) == ["a", "b"]

    def test_array_types_carry_correct_element_types(self):
        """Each array type's element type matches the corresponding schema field type."""
        schema = _schema(type1=i32, type2=i64)
        collection = ResultsCollectionType(schema, IntAttr(4))
        result = collection_type_to_array_types(collection)
        assert result["a"] == ResultsArrayType(i32, IntAttr(4))
        assert result["b"] == ResultsArrayType(i64, IntAttr(4))

    def test_array_types_inherit_static_collection_size(self):
        """A static collection size is propagated to every array type."""
        schema = _schema(type1=i32, type2=i64)
        collection = ResultsCollectionType(schema, IntAttr(7))
        result = collection_type_to_array_types(collection)
        assert all(v.size == IntAttr(7) for v in result.values())

    def test_single_field_schema(self):
        """A schema with one field produces exactly one array type."""
        schema = RecordSchemaAttr([RecordFieldAttr("x", i32)])
        collection = ResultsCollectionType(schema, IntAttr(1))
        result = collection_type_to_array_types(collection)
        assert list(result.keys()) == ["x"]
        assert result["x"] == ResultsArrayType(i32, IntAttr(1))

    def test_field_order_is_preserved(self):
        """The returned dict preserves schema field insertion order."""
        schema = RecordSchemaAttr(
            [
                RecordFieldAttr("z", i64),
                RecordFieldAttr("a", i32),
                RecordFieldAttr("m", i64),
            ]
        )
        collection = ResultsCollectionType(schema, IntAttr(2))
        result = collection_type_to_array_types(collection)
        assert list(result.keys()) == ["z", "a", "m"]


class TestConvertCreateOps:
    """Tests that CreateOps are dealt with in the expected ways."""

    def test_create_collection_op_decomposes_into_create_array_ops(self):
        """Tests that a CreateOp with a collection type is decomposed into multiple
        CreateOps with array types."""

        type1 = i32
        type2 = i64
        size = 3

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        block = _apply_in_function(collection)

        create_ops = _ops_of_type(block, CreateOp)
        collection_creates = [
            op for op in create_ops if isinstance(op.result.type, ResultsCollectionType)
        ]
        assert len(collection_creates) == 0
        assert len(create_ops) == 2
        _assert_array_type(create_ops[0].result.type, type1, IntAttr(size))
        _assert_array_type(create_ops[1].result.type, type2, IntAttr(size))

    def test_create_array_op_is_unchanged(self):
        """Tests that a CreateOp with an array type is unchanged by the conversion."""

        array_create = CreateOp.for_array(i32, 3)
        before = array_create.clone()
        block = _apply_in_function(array_create)

        assert array_create in list(block.ops)
        assert before.is_structurally_equivalent(array_create)

    def test_create_record_op_is_unchanged(self):
        """Tests that a CreateOp with a record type is unchanged by the conversion."""

        c1 = _const_i32(1)
        c2 = _const_i64(2)
        record_create = CreateOp.for_record(["a", "b"], [c1.result, c2.result])
        before = record_create.clone()
        block = _apply_in_function(c1, c2, record_create)

        assert record_create in list(block.ops)
        assert before.is_structurally_equivalent(record_create)

    def test_collection_from_arrays_op_is_removed(self):
        """If we're creating a collection from the arrays directly, then the arrays already
        exist and we just need to replace them."""

        a1 = CreateOp.for_array(i32, 3)
        a2 = CreateOp.for_array(i64, 3)
        collection = CreateOp.for_collection_from_arrays(["a", "b"], [a1.result, a2.result])
        extract = ExtractOp.array_from_collection(collection.result, "a")
        consumer = _MockVariadicConsumerOp(extract.result)
        block = _apply_in_function(a1, a2, collection, extract, consumer)
        [new_consumer] = _ops_of_type(block, _MockVariadicConsumerOp)
        consumer_values = list(new_consumer.values)
        collection_creates = [
            op
            for op in _ops_of_type(block, CreateOp)
            if isinstance(op.result.type, ResultsCollectionType)
        ]

        assert len(collection_creates) == 0
        assert a1 in list(block.ops)
        assert a2 in list(block.ops)
        assert len(_ops_of_type(block, ExtractOp)) == 0
        assert len(consumer_values) == 1
        assert consumer_values[0] is a1.result

    def test_collection_with_no_values_is_removed(self):
        """If we're creating a collection with no values, then we can just remove the
        operation."""

        type1 = i32
        type2 = i64

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), 5)
        block = _apply_in_function(collection)
        collection_creates = [
            op
            for op in _ops_of_type(block, CreateOp)
            if isinstance(op.result.type, ResultsCollectionType)
        ]

        assert len(collection_creates) == 0


class TestConvertStoreOps:
    """Tests that StoreOps are dealt with in the expected ways."""

    def test_value_into_collection_decomposes_into_single_store_op(self):
        """Tests that a StoreOp with a collection result / container type which stores a
        value at an index is decomposed into a store op for that value in the correct
        array."""

        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        index = _const_i32(0)
        value = _const_i32(7)
        store = StoreOp.value_in_collection(
            collection.result, index.result, "a", value.result
        )

        block = _apply_in_function(index, value, collection, store)
        store_ops = _ops_of_type(block, StoreOp)
        [lowered_store] = store_ops

        collection_container_stores = [
            op for op in store_ops if isinstance(op.container.type, ResultsCollectionType)
        ]
        assert len(collection_container_stores) == 0
        assert len(store_ops) == 1
        _assert_array_type(lowered_store.container.type, type1, IntAttr(size))
        assert lowered_store.index is index.result
        assert lowered_store.value is value.result

    def test_value_into_array_is_unchanged(self):
        """Tests that a StoreOp with an array result / container type which stores a value
        at an index is unchanged by the conversion."""

        type1 = i32
        size = 4

        array = CreateOp.for_array(type1, size)
        index = _const_i32(0)
        value = _const_i32(7)
        store = StoreOp.value_in_array(array.result, index.result, value.result)
        before = store.clone()

        block = _apply_in_function(index, value, array, store)

        assert store in list(block.ops)
        assert before.is_structurally_equivalent(store)

    def test_record_op_into_collection_decomposes_into_many_store_ops(self):
        """Tests that a StoreOp with a record type into a collection result / container type
        which stores a value at an index is decomposed into many store ops for each field in
        the record."""

        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        index = _const_i32(0)
        v1 = _const_i32(1)
        v2 = _const_i64(2)
        record = CreateOp.for_record(["a", "b"], [v1.result, v2.result])
        store = StoreOp.record_in_collection(collection.result, index.result, record.result)

        block = _apply_in_function(index, v1, v2, collection, record, store)
        store_ops = _ops_of_type(block, StoreOp)

        collection_container_stores = [
            op for op in store_ops if isinstance(op.container.type, ResultsCollectionType)
        ]
        assert len(collection_container_stores) == 0
        assert len(store_ops) == 2
        i32_stores = [op for op in store_ops if op.value.type == type1]
        i64_stores = [op for op in store_ops if op.value.type == type2]

        assert len(i32_stores) == 1
        assert len(i64_stores) == 1
        _assert_array_type(i32_stores[0].container.type, type1, IntAttr(size))
        _assert_array_type(i64_stores[0].container.type, type2, IntAttr(size))
        assert i32_stores[0].value is v1.result
        assert i64_stores[0].value is v2.result
        assert i32_stores[0].index is index.result
        assert i64_stores[0].index is index.result

    def test_record_op_into_collection_with_unknown_source_creates_extract_ops(self):
        """Tests that a StoreOp with a record type into a collection result / container type
        with a record that is not from a CreateOp is accessed through ExtractOps for each
        field in the record and then stored into the correct arrays."""

        type1 = i32
        type2 = i64
        size = 4

        schema = _schema(type1=type1, type2=type2)
        collection = CreateOp.for_empty_collection(schema, size)
        index = _const_i32(0)
        record_src = _MockTypedSourceOp(RecordType(schema))
        store = StoreOp.record_in_collection(
            collection.result, index.result, record_src.result
        )

        block = _apply_in_function(index, collection, record_src, store)
        extract_ops = _ops_of_type(block, ExtractOp)
        store_ops = _ops_of_type(block, StoreOp)
        record_sources = _ops_of_type(block, _MockTypedSourceOp)

        collection_container_stores = [
            op for op in store_ops if isinstance(op.container.type, ResultsCollectionType)
        ]
        extracts_by_key = {op.properties["key"].data: op for op in extract_ops}
        i32_stores = [op for op in store_ops if op.value.type == type1]
        i64_stores = [op for op in store_ops if op.value.type == type2]

        assert len(collection_container_stores) == 0
        assert len(record_sources) == 1
        assert isinstance(record_sources[0].result.type, RecordType)
        assert all(extract.container is record_sources[0].result for extract in extract_ops)
        assert set(extracts_by_key) == {"a", "b"}
        assert extracts_by_key["a"].result.type == type1
        assert extracts_by_key["b"].result.type == type2
        assert len(extract_ops) == 2
        assert len(store_ops) == 2
        assert len(i32_stores) == 1
        assert len(i64_stores) == 1
        _assert_array_type(i32_stores[0].container.type, type1, IntAttr(size))
        _assert_array_type(i64_stores[0].container.type, type2, IntAttr(size))
        assert i32_stores[0].value is extracts_by_key["a"].result
        assert i64_stores[0].value is extracts_by_key["b"].result
        assert i32_stores[0].index is index.result
        assert i64_stores[0].index is index.result


class TestConvertExtractOps:
    """Tests that ExtractOps are dealt with in the expected ways."""

    def test_from_array_is_unchanged(self):
        """Tests that an ExtractOp with an array result / container type which extracts a
        value at an index is unchanged by the conversion."""

        type1 = i32
        size = 4

        array = CreateOp.for_array(type1, size)
        index = _const_i32(0)
        extract = ExtractOp.value_from_array(array.result, index.result)
        before = extract.clone()

        block = _apply_in_function(index, array, extract)

        assert extract in list(block.ops)
        assert before.is_structurally_equivalent(extract)

    def test_value_from_collection_decomposes_into_single_extract_op(self):
        """Tests that an ExtractOp with a collection result / container type which extracts
        a value at an index is decomposed into an extract op for that value in the correct
        array."""

        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        index = _const_i32(0)
        extract = ExtractOp.value_from_collection(collection.result, "a", index.result)

        block = _apply_in_function(index, collection, extract)
        create_ops = _ops_of_type(block, CreateOp)
        extract_ops = _ops_of_type(block, ExtractOp)
        [lowered_extract] = extract_ops
        array_creates = [
            op for op in create_ops if isinstance(op.result.type, ResultsArrayType)
        ]
        i32_arrays = [op for op in array_creates if op.result.type.type == type1]
        i64_arrays = [op for op in array_creates if op.result.type.type == type2]

        collection_container_extracts = [
            op for op in extract_ops if isinstance(op.container.type, ResultsCollectionType)
        ]
        collection_creates = [
            op for op in create_ops if isinstance(op.result.type, ResultsCollectionType)
        ]
        assert len(collection_creates) == 0
        assert len(array_creates) == 2
        assert len(i32_arrays) == 1
        assert len(i64_arrays) == 1
        _assert_array_type(i32_arrays[0].result.type, type1, IntAttr(size))
        _assert_array_type(i64_arrays[0].result.type, type2, IntAttr(size))
        assert len(collection_container_extracts) == 0
        assert len(extract_ops) == 1
        _assert_array_type(lowered_extract.container.type, type1, IntAttr(size))
        assert lowered_extract.container is i32_arrays[0].result
        assert lowered_extract.index is index.result
        assert lowered_extract.result.type == type1

    def test_array_from_collection_erases_operation(self):
        """Tests that an ExtractOp with a collection result / container type which extracts
        an array at an index is replaced with the original array, and the ExtractOp is
        removed as part of the intended transformation."""

        type1 = i32
        type2 = i64
        size = 3

        a1 = CreateOp.for_array(type1, size)
        a2 = CreateOp.for_array(type2, size)
        collection = CreateOp.for_collection_from_arrays(["a", "b"], [a1.result, a2.result])
        extract = ExtractOp.array_from_collection(collection.result, "a")
        consumer = _MockVariadicConsumerOp(extract.result)

        block = _apply_in_function(a1, a2, collection, extract, consumer)
        create_ops = _ops_of_type(block, CreateOp)
        extract_ops = _ops_of_type(block, ExtractOp)
        [new_consumer] = _ops_of_type(block, _MockVariadicConsumerOp)
        consumer_values = list(new_consumer.values)
        array_creates = [
            op for op in create_ops if isinstance(op.result.type, ResultsArrayType)
        ]
        i32_arrays = [op for op in array_creates if op.result.type.type == type1]
        i64_arrays = [op for op in array_creates if op.result.type.type == type2]

        assert len(extract_ops) == 0
        assert len(array_creates) == 2
        assert len(i32_arrays) == 1
        assert len(i64_arrays) == 1
        _assert_array_type(i32_arrays[0].result.type, type1, IntAttr(size))
        _assert_array_type(i64_arrays[0].result.type, type2, IntAttr(size))
        assert len(consumer_values) == 1
        assert consumer_values[0] is i32_arrays[0].result

    def test_record_from_collection_decomposes_into_extracts_and_create(self):
        """Transforming an extraction of a record from a collection should decompose into
        many ExtractOps for each field in the record from the appropriate arrays, then a
        CreateOp to create the record from the extracted fields."""

        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        index = _const_i32(0)
        extract = ExtractOp.record_from_collection(collection.result, index.result)

        block = _apply_in_function(index, collection, extract)
        create_ops = _ops_of_type(block, CreateOp)
        extract_ops = _ops_of_type(block, ExtractOp)
        record_creates = [op for op in create_ops if isinstance(op.result.type, RecordType)]
        array_creates = [
            op for op in create_ops if isinstance(op.result.type, ResultsArrayType)
        ]
        i32_arrays = [op for op in array_creates if op.result.type.type == type1]
        i64_arrays = [op for op in array_creates if op.result.type.type == type2]
        extracts_by_type = {op.result.type: op for op in extract_ops}

        collection_container_extracts = [
            op for op in extract_ops if isinstance(op.container.type, ResultsCollectionType)
        ]
        collection_creates = [
            op for op in create_ops if isinstance(op.result.type, ResultsCollectionType)
        ]
        assert len(collection_creates) == 0
        assert len(array_creates) == 2
        assert len(i32_arrays) == 1
        assert len(i64_arrays) == 1
        _assert_array_type(i32_arrays[0].result.type, type1, IntAttr(size))
        _assert_array_type(i64_arrays[0].result.type, type2, IntAttr(size))
        assert len(collection_container_extracts) == 0
        assert set(extracts_by_type) == {type1, type2}
        assert extracts_by_type[type1].container is i32_arrays[0].result
        assert extracts_by_type[type2].container is i64_arrays[0].result
        assert extracts_by_type[type1].index is index.result
        assert extracts_by_type[type2].index is index.result
        assert len(extract_ops) == 2
        assert len(record_creates) == 1
        assert list(record_creates[0].values) == [
            extracts_by_type[type1].result,
            extracts_by_type[type2].result,
        ]


class TestConvertCollectionOperandsAndResults:
    """Tests the general type conversion for generic operations with operands and results
    that are the collection type."""

    def test_operation_with_no_collection_operands_or_results_is_unchanged(self):
        """Only results collection types should be changed."""

        op = _MockVariadicProducerOp([i32, i64])
        before = op.clone()
        block = _apply_in_function(op)

        assert op in list(block.ops)
        assert before.is_structurally_equivalent(op)

    def test_arbitrary_collection_consumer_with_none_variadic_argument_raises(self):
        """A consumer of a collection that doesn't have a variadic operand cannot be reduced
        to many operands and the pass always raises."""

        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        consumer = _MockNonVariadicConsumerOp(collection.result.owner)

        with pytest.raises(PassFailedException, match="failed to convert all"):
            _apply_in_function(collection, consumer)

    def test_arbitrary_collection_producer_with_non_variadic_argument_raises(self):
        """An unknown producer of a collection that doesn't have a variadic result cannot be
        reduced to many results and the pass always raises."""

        type1 = i32
        type2 = i64
        size = 4

        producer = _MockTypedSourceOp(
            ResultsCollectionType(_schema(type1=type1, type2=type2), IntAttr(size))
        )
        with pytest.raises(PassFailedException, match="failed to convert all"):
            _apply_in_function(producer)

    def test_variadic_results_with_single_collection_is_changed(self):
        """Tests that an operation with a variadic results which contains a collection type
        is correctly decomposed."""

        type1 = i32
        type2 = i64
        size = 4

        result_types = [
            type1,
            ResultsCollectionType(_schema(type1=type1, type2=type2), IntAttr(size)),
            i64,
        ]
        producer = _MockVariadicProducerOp(result_types)

        block = _apply_in_function(producer)
        [new_producer] = _ops_of_type(block, _MockVariadicProducerOp)

        result_types = [res.type for res in new_producer.values]
        assert len(result_types) == 4
        _assert_array_type(result_types[1], type1, IntAttr(size))
        _assert_array_type(result_types[2], type2, IntAttr(size))

    def test_variadic_operands_with_single_collection_is_changed(self):
        """Tests that an operation that has variadic operands which contains a collection
        type is correctly decomposed."""

        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        c1 = _const_i32(1)
        c2 = _const_i64(2)
        consumer = _MockVariadicConsumerOp(c1, collection.result, c2)

        block = _apply_in_function(c1, c2, collection, consumer)
        [new_consumer] = _ops_of_type(block, _MockVariadicConsumerOp)
        new_values = list(new_consumer.values)

        assert len(new_values) == 4
        _assert_array_type(new_values[1].type, type1, IntAttr(size))
        _assert_array_type(new_values[2].type, type2, IntAttr(size))

    def test_variadic_results_with_multiple_collections_are_changed(self):
        """Tests that an operation with a variadic result that contains multiple collection
        types is correctly decomposed."""

        type1 = i32
        type2 = i64
        size = 4

        result_types = [
            ResultsCollectionType(_schema(type1=type1, type2=type2), IntAttr(size)),
            type1,
            ResultsCollectionType(_schema(type1=type1, type2=type2), IntAttr(size)),
        ]
        producer = _MockVariadicProducerOp(result_types)
        block = _apply_in_function(producer)
        [new_producer] = _ops_of_type(block, _MockVariadicProducerOp)

        assert len(list(new_producer.values)) == 5

    def test_variadic_operands_with_multiple_collections_are_changed(self):
        """Tests that an operation with a variadic operands that contains multiple
        collection types is correctly decomposed."""

        type1 = i32
        type2 = i64
        size = 4

        c = _const_i32(1)
        col1 = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        col2 = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        consumer = _MockVariadicConsumerOp(c, col1.result, col2.result)

        block = _apply_in_function(c, col1, col2, consumer)
        [new_consumer] = _ops_of_type(block, _MockVariadicConsumerOp)

        assert len(list(new_consumer.values)) == 5

    def test_variadic_results_preserve_relative_result_order(self):
        """Expanded array results should preserve ordering relative to non-collection
        results around them."""

        type1 = i32
        type2 = i64
        size = 4

        result_types = [
            type1,
            ResultsCollectionType(_schema(type1=type1, type2=type2), IntAttr(size)),
            i64,
        ]
        producer = _MockVariadicProducerOp(result_types)
        block = _apply_in_function(producer)
        [new_producer] = _ops_of_type(block, _MockVariadicProducerOp)

        new_types = [res.type for res in new_producer.values]
        assert new_types[0] == type1
        _assert_array_type(new_types[1], type1, IntAttr(size))
        _assert_array_type(new_types[2], type2, IntAttr(size))
        assert new_types[3] == i64

    def test_variadic_operands_preserve_relative_operand_order(self):
        """Expanded array operands should preserve ordering relative to non-collection
        operands around them."""

        type1 = i32
        type2 = i64
        size = 4

        c1 = _const_i32(11)
        c2 = _const_i64(22)
        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        consumer = _MockVariadicConsumerOp(c1, collection.result, c2)

        block = _apply_in_function(c1, c2, collection, consumer)
        [new_consumer] = _ops_of_type(block, _MockVariadicConsumerOp)
        operands = list(new_consumer.values)

        assert operands[0] is c1.result
        _assert_array_type(operands[1].type, type1, IntAttr(size))
        _assert_array_type(operands[2].type, type2, IntAttr(size))
        assert operands[3] is c2.result

    def test_op_with_non_variadic_and_variadic_results_lowers_collection(self):
        """An op with both a non-variadic result and a variadic result containing a
        collection is correctly decomposed; the non-variadic result is preserved unchanged
        and the collection in the variadic slot is expanded to arrays."""

        type1 = i32
        type2 = i64
        size = 4

        producer = _MockMixedProducerOp(
            i32,
            [ResultsCollectionType(_schema(type1=type1, type2=type2), IntAttr(size))],
        )
        block = _apply_in_function(producer)
        [new_producer] = _ops_of_type(block, _MockMixedProducerOp)

        assert new_producer.fixed.type == i32
        variadic_types = [res.type for res in new_producer.values]
        assert len(variadic_types) == 2
        _assert_array_type(variadic_types[0], type1, IntAttr(size))
        _assert_array_type(variadic_types[1], type2, IntAttr(size))


class TestConvertCollectionBlockArguments:
    """Tests the general type conversion for generic operations with block arguments that
    are the collection type."""

    def test_operation_with_no_collection_block_args_is_unchanged(self):
        """Only results collection types should be changed."""

        type1 = i32
        type2 = i64

        block = Block(arg_types=[type1, type2])
        consumer = _MockVariadicConsumerOp()
        block.add_op(consumer)
        wrapper = _SingleRegionWrapperOp(block)
        before = wrapper.clone()

        convert_results_collections_to_arrays(wrapper)

        assert before.is_structurally_equivalent(wrapper)

    def test_single_collection_block_arg_with_single_user_is_converted(self):
        """Creates a collection block argument that has two fields in it with different
        types and a static index, and then uses that block argument within the block.

        This test checks that the block argument is replaced by two block arguments, which
        are used by the consumer.
        """

        type1 = i32
        type2 = i64

        collection_type = ResultsCollectionType(
            _schema(type1=type1, type2=type2), IntAttr(3)
        )
        block = Block(arg_types=[collection_type])
        block.add_op(_MockVariadicConsumerOp(block.args[0]))
        wrapper = _SingleRegionWrapperOp(block)

        convert_results_collections_to_arrays(wrapper)

        assert len(block.args) == 2
        _assert_array_type(block.args[0].type, type1, IntAttr(3))
        _assert_array_type(block.args[1].type, type2, IntAttr(3))

    def test_multiple_collection_block_args_with_single_user_is_converted(self):
        """When multiple collections are used, they should be individually converted."""

        type1 = i32
        type2 = i64

        ct = ResultsCollectionType(_schema(type1=type1, type2=type2), IntAttr(3))
        block = Block(arg_types=[ct, type1, ct])
        block.add_op(_MockVariadicConsumerOp(block.args[0], block.args[2]))
        wrapper = _SingleRegionWrapperOp(block)

        convert_results_collections_to_arrays(wrapper)

        [consumer] = _ops_of_type(block, _MockVariadicConsumerOp)
        consumer_values = list(consumer.values)

        assert len(block.args) == 5
        _assert_array_type(block.args[0].type, type1, IntAttr(3))
        _assert_array_type(block.args[1].type, type2, IntAttr(3))
        _assert_array_type(block.args[3].type, type1, IntAttr(3))
        _assert_array_type(block.args[4].type, type2, IntAttr(3))
        assert consumer_values == [
            block.args[0],
            block.args[1],
            block.args[3],
            block.args[4],
        ]

    def test_block_argument_not_double_converted_on_revisit(self):
        """A block argument already expanded in a previous traversal should not be expanded
        again in later fixed-point iterations."""

        type1 = i32
        type2 = i64

        collection_type = ResultsCollectionType(
            _schema(type1=type1, type2=type2), IntAttr(3)
        )
        block = Block(arg_types=[collection_type])
        block.add_op(_MockVariadicConsumerOp(block.args[0]))
        wrapper = _SingleRegionWrapperOp(block)

        convert_results_collections_to_arrays(wrapper)
        first_pass_args = len(block.args)
        convert_results_collections_to_arrays(wrapper)

        assert first_pass_args == 2
        assert len(block.args) == 2


class TestCompositeEdgeCases:
    """Tests composite use of CreateOps, StoreOps and ExtractOps in special edge cases to
    check we get predictable behaviour."""

    def test_create_from_arrays_with_extract_array_erases_create_and_extract(self):
        """If we have a CreateOp that creates a collection from arrays, then look to extract
        the array from the collection, we should be able to replace the extract with the
        original array, and the create op is removed as part of the intended
        transformation."""

        a1 = CreateOp.for_array(i32, 3)
        a2 = CreateOp.for_array(i64, 3)
        collection = CreateOp.for_collection_from_arrays(["a", "b"], [a1.result, a2.result])
        extract = ExtractOp.array_from_collection(collection.result, "a")
        user = _MockVariadicConsumerOp(extract.result)

        block = _apply_in_function(a1, a2, collection, extract, user)

        collection_creates = [
            op
            for op in _ops_of_type(block, CreateOp)
            if isinstance(op.result.type, ResultsCollectionType)
        ]
        assert len(collection_creates) == 0
        assert len(_ops_of_type(block, ExtractOp)) == 0

    def test_pass_is_idempotent_on_already_lowered_ir(self):
        """Running the pass twice should leave already-lowered IR unchanged on the second
        run."""

        type1 = i32
        type2 = i64

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), 3)
        fn = _func_with_ops(collection)
        convert_results_collections_to_arrays(fn)
        before_second_pass = fn.clone()
        convert_results_collections_to_arrays(fn)

        assert before_second_pass.is_structurally_equivalent(fn)

    def test_dynamic_size_collection_creates_dynamic_size_arrays(self):
        """When lowering an empty collection with dynamic size, every created array should
        use the same dynamic size SSA."""

        type1 = i32
        type2 = i64

        dyn = _const_i32(9)
        collection = CreateOp.for_empty_collection(
            _schema(type1=type1, type2=type2), dyn.result
        )
        block = _apply_in_function(dyn, collection)
        create_ops = _ops_of_type(block, CreateOp)

        assert len(create_ops) == 2
        assert all(op.size is dyn.result for op in create_ops)

    def test_static_size_collection_creates_static_size_arrays(self):
        """When lowering an empty collection with static size, every created array should
        preserve that static size."""

        type1 = i32
        type2 = i64

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), 6)
        block = _apply_in_function(collection)
        create_ops = _ops_of_type(block, CreateOp)

        assert all(op.result.type.size == IntAttr(6) for op in create_ops)

    def test_record_extract_reconstruction_preserves_schema_field_order(self):
        """Record extraction from a collection should reconstruct the record with values in
        schema field order."""

        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        index = _const_i32(0)
        extract = ExtractOp.record_from_collection(collection.result, index.result)
        block = _apply_in_function(index, collection, extract)
        record_creates = [
            op
            for op in _ops_of_type(block, CreateOp)
            if isinstance(op.result.type, RecordType)
        ]

        assert len(record_creates) == 1
        assert [
            field.key.data for field in record_creates[0].result.type.schema.fields
        ] == [
            "a",
            "b",
        ]

    def test_create_from_arrays_schema_mismatch_fails_loudly(self):
        """Malformed collection-from-arrays fixtures with mismatched schema/value arity
        should fail loudly rather than partially rewriting."""

        type1 = i32
        type2 = i64

        schema = _schema(type1=type1, type2=type2)
        arr = CreateOp.for_array(type1, 3)
        bad = CreateOp(ResultsCollectionType(schema, IntAttr(3)), values=[arr.result])

        with pytest.raises(ValueError, match="zip"):
            _apply_in_function(arr, bad)

    def test_non_variadic_collection_use_raises_with_error_context(self):
        """Unsupported non-variadic collection usage always raises, and the error message
        identifies the violating operation."""

        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        user = _MockNonVariadicConsumerOp(collection.result.owner)

        with pytest.raises(PassFailedException, match="Cannot convert"):
            _apply_in_function(collection, user)

    def test_record_producer_detached_when_only_used_by_rewritten_store(self):
        """A record-producing CreateOp should be detached when its only users are rewritten
        away by store lowering."""

        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        index = _const_i32(0)
        v1 = _const_i32(1)
        v2 = _const_i64(2)
        record = CreateOp.for_record(["a", "b"], [v1.result, v2.result])
        store = StoreOp.record_in_collection(collection.result, index.result, record.result)

        block = _apply_in_function(index, v1, v2, collection, record, store)
        record_creates = [
            op
            for op in _ops_of_type(block, CreateOp)
            if isinstance(op.result.type, RecordType)
        ]

        assert len(record_creates) == 0

    def test_record_producer_not_detached_when_other_uses_remain(self):
        """A record-producing CreateOp should remain when at least one non-rewritten user
        still references it."""

        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        index = _const_i32(0)
        v1 = _const_i32(1)
        v2 = _const_i64(2)
        record = CreateOp.for_record(["a", "b"], [v1.result, v2.result])
        store = StoreOp.record_in_collection(collection.result, index.result, record.result)
        user = _MockVariadicConsumerOp(record.result.owner)

        block = _apply_in_function(index, v1, v2, collection, record, store, user)

        assert record in list(block.ops)


class TestStructuredControlFlowSequences:
    """Runs tests for structured control-flow use cases with real scf operations."""

    def test_collection_with_records_added_from_for_loop(self):
        """An empty collection is created, followed by an scf.for body that adds records.

        Should convert to entirely arrays.
        """

        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        lower = ArithConstantOp.from_int_and_width(0, IndexType())
        upper = ArithConstantOp.from_int_and_width(size, IndexType())
        step = ArithConstantOp.from_int_and_width(1, IndexType())
        v1 = _const_i32(1)
        v2 = _const_i64(2)
        loop_body = Block(arg_types=[IndexType(), collection.result.type])
        iv = loop_body.args[0]
        loop_collection = loop_body.args[1]
        record = CreateOp.for_record(["a", "b"], [v1.result, v2.result])
        store = StoreOp.record_in_collection(loop_collection, iv, record.result)
        loop_body.add_ops([record, store, scf.YieldOp(store.result)])
        loop = scf.ForOp(lower, upper, step, [collection], loop_body)

        block = _apply_in_function(lower, upper, step, v1, v2, collection, loop)
        ops = list(block.ops)
        create_ops = _ops_of_type(block, CreateOp)
        [for_op] = _ops_of_type(block, scf.ForOp)
        create_0_index = ops.index(create_ops[0])
        create_1_index = ops.index(create_ops[1])

        assert len(create_ops) == 2
        assert create_0_index < create_1_index
        assert isinstance(create_ops[0].result.type, ResultsArrayType)
        assert isinstance(create_ops[1].result.type, ResultsArrayType)
        assert ops[create_1_index + 1] is for_op
        assert list(for_op.iter_args) == [create_ops[0].result, create_ops[1].result]

        loop_args = list(for_op.body.block.args)
        assert len(loop_args) == 3
        assert loop_args[0].type == IndexType()
        assert loop_args[1].type == create_ops[0].result.type
        assert loop_args[2].type == create_ops[1].result.type

        [store_a, store_b, yield_op] = list(for_op.body.block.ops)
        assert isinstance(store_a, StoreOp)
        assert isinstance(store_b, StoreOp)
        assert isinstance(yield_op, scf.YieldOp)
        assert store_a.container is loop_args[1]
        assert store_b.container is loop_args[2]
        assert list(yield_op.arguments) == [store_a.result, store_b.result]

        assert len(list(for_op.results)) == 2
        assert for_op.results[0].type == create_ops[0].result.type
        assert for_op.results[1].type == create_ops[1].result.type

        all_ops = [nested for op in block.ops for nested in op.walk()]
        assert all(
            not isinstance(res.type, ResultsCollectionType)
            for op in all_ops
            for res in op.results
        )

    def test_collection_made_from_arrays_which_are_added_from_within_for_loop(self):
        """Creates empty arrays and writes to them in an scf.for region, then creates a
        collection from those arrays.

        Should convert to returning the arrays.
        """

        type1 = i32
        type2 = i64
        size = 4

        lower = ArithConstantOp.from_int_and_width(0, IndexType())
        upper = ArithConstantOp.from_int_and_width(size, IndexType())
        step = ArithConstantOp.from_int_and_width(1, IndexType())
        a1 = CreateOp.for_array(type1, size)
        a2 = CreateOp.for_array(type2, size)
        v1 = _const_i32(5)
        v2 = _const_i64(6)
        loop_body = Block(arg_types=[IndexType(), a1.result.type, a2.result.type])
        iv = loop_body.args[0]
        loop_a1 = loop_body.args[1]
        loop_a2 = loop_body.args[2]
        s1 = StoreOp.value_in_array(loop_a1, iv, v1.result)
        s2 = StoreOp.value_in_array(loop_a2, iv, v2.result)
        loop_body.add_ops([s1, s2, scf.YieldOp(s1.result, s2.result)])
        loop = scf.ForOp(lower, upper, step, [a1, a2], loop_body)
        collection = CreateOp.for_collection_from_arrays(
            ["a", "b"], [loop.results[0], loop.results[1]]
        )

        block = _apply_in_function(lower, upper, step, v1, v2, a1, a2, loop, collection)
        ops = list(block.ops)
        create_ops = _ops_of_type(block, CreateOp)
        [for_op] = _ops_of_type(block, scf.ForOp)
        create_0_index = ops.index(create_ops[0])
        create_1_index = ops.index(create_ops[1])
        collection_creates = [
            op for op in create_ops if isinstance(op.result.type, ResultsCollectionType)
        ]

        assert len(create_ops) == 2
        assert create_0_index < create_1_index
        assert isinstance(create_ops[0].result.type, ResultsArrayType)
        assert isinstance(create_ops[1].result.type, ResultsArrayType)
        assert ops[create_1_index + 1] is for_op
        assert list(for_op.iter_args) == [create_ops[0].result, create_ops[1].result]

        loop_args = list(for_op.body.block.args)
        assert len(loop_args) == 3
        assert loop_args[0].type == IndexType()
        assert loop_args[1].type == create_ops[0].result.type
        assert loop_args[2].type == create_ops[1].result.type

        [store_a, store_b, yield_op] = list(for_op.body.block.ops)
        assert isinstance(store_a, StoreOp)
        assert isinstance(store_b, StoreOp)
        assert isinstance(yield_op, scf.YieldOp)
        assert store_a.container is loop_args[1]
        assert store_b.container is loop_args[2]
        assert list(yield_op.arguments) == [store_a.result, store_b.result]

        assert len(list(for_op.results)) == 2
        assert for_op.results[0].type == create_ops[0].result.type
        assert for_op.results[1].type == create_ops[1].result.type
        assert len(collection_creates) == 0

    def test_collection_made_within_if_else_is_converted(self):
        """Branches create collections from arrays and yield through a real scf.if.

        Should just return arrays.
        """

        type1 = i32
        type2 = i64
        size = 4
        cond = ArithConstantOp.from_int_and_width(1, 1)
        schema = _schema(type1=type1, type2=type2)
        collection_type = ResultsCollectionType(schema, IntAttr(size))

        a1 = CreateOp.for_array(type1, size)
        a2 = CreateOp.for_array(type2, size)
        a3 = CreateOp.for_array(type1, size)
        c_then = CreateOp.for_collection_from_arrays(["a", "b"], [a1.result, a2.result])
        c_else = CreateOp.for_collection_from_arrays(["a", "b"], [a3.result, a2.result])

        then_block = Block([c_then, scf.YieldOp(c_then)])
        else_block = Block([c_else, scf.YieldOp(c_else)])
        if_op = scf.IfOp(cond, [collection_type], [then_block], [else_block])
        use = _MockVariadicConsumerOp(if_op)

        block = _apply_in_function(cond, a1, a2, a3, if_op, use)
        ops = list(block.ops)
        create_ops = _ops_of_type(block, CreateOp)
        [lowered_if] = _ops_of_type(block, scf.IfOp)
        [consumer] = _ops_of_type(block, _MockVariadicConsumerOp)
        create_0_index = ops.index(create_ops[0])
        create_1_index = ops.index(create_ops[1])
        create_2_index = ops.index(create_ops[2])
        all_ops = [nested for op in block.ops for nested in op.walk()]
        collection_creates = [
            op
            for op in all_ops
            if isinstance(op, CreateOp)
            if isinstance(op.result.type, ResultsCollectionType)
        ]

        assert len(create_ops) == 3
        assert create_0_index < create_1_index < create_2_index
        assert all(isinstance(op.result.type, ResultsArrayType) for op in create_ops)
        assert ops[create_2_index + 1] is lowered_if

        then_ops = list(lowered_if.true_region.block.ops)
        else_ops = list(lowered_if.false_region.block.ops)
        assert len(then_ops) == 1
        assert len(else_ops) == 1
        assert isinstance(then_ops[0], scf.YieldOp)
        assert isinstance(else_ops[0], scf.YieldOp)

        then_args = list(then_ops[0].arguments)
        else_args = list(else_ops[0].arguments)
        assert then_args == [create_ops[0].result, create_ops[1].result]
        assert else_args == [create_ops[2].result, create_ops[1].result]

        assert len(list(lowered_if.results)) == 2
        assert lowered_if.results[0].type == create_ops[0].result.type
        assert lowered_if.results[1].type == create_ops[1].result.type
        assert list(consumer.values) == [lowered_if.results[0], lowered_if.results[1]]
        assert len(collection_creates) == 0
        assert len(_ops_of_type(block, _MockVariadicConsumerOp)) == 1


class TestControlFlowSequences:
    """Tests operation orders that require fixed-point rewrite revisits."""

    def test_store_a_record_from_a_create_op(self):
        """Multi-block: collection flows as entry-block arg, stored via a record-CreateOp
        in a producer block and consumed in a successor block.
        """
        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        idx = _const_i32(0)
        v1 = _const_i32(1)
        v2 = _const_i64(2)

        consumer = Block(arg_types=[collection.result.type])
        consumer.add_ops(
            [
                _MockVariadicConsumerOp(consumer.args[0]),
                func.ReturnOp(),
            ]
        )

        producer = Block(
            arg_types=[
                collection.result.type,
                idx.result.type,
                v1.result.type,
                v2.result.type,
            ]
        )
        record = CreateOp.for_record(["a", "b"], [producer.args[2], producer.args[3]])
        store = StoreOp.record_in_collection(
            producer.args[0], producer.args[1], record.result
        )
        producer.add_ops([record, store, cf.BranchOp(consumer, store.result)])

        entry = Block(
            [
                collection,
                idx,
                v1,
                v2,
                cf.BranchOp(producer, collection.result, idx.result, v1.result, v2.result),
            ]
        )
        fn = func.FuncOp("main", ((), ()), Region([entry, consumer, producer]))
        convert_results_collections_to_arrays(fn)

        blocks = list(fn.body.blocks)
        entry_block, consumer_block, producer_block = blocks
        entry_create_ops = _ops_of_type(entry_block, CreateOp)
        producer_store_ops = _ops_of_type(producer_block, StoreOp)

        assert blocks == [entry, consumer, producer]
        assert len(entry_create_ops) == 2
        assert all(isinstance(op.result.type, ResultsArrayType) for op in entry_create_ops)
        assert len(producer_store_ops) == 2
        assert all(
            isinstance(op.container.type, ResultsArrayType) for op in producer_store_ops
        )
        assert len(_ops_of_type(consumer_block, _MockVariadicConsumerOp)) == 1
        assert len(consumer_block.args) == 2
        assert consumer_block.args[0].type == entry_create_ops[0].result.type
        assert consumer_block.args[1].type == entry_create_ops[1].result.type
        [consumer_op] = _ops_of_type(consumer_block, _MockVariadicConsumerOp)
        assert list(consumer_op.values) == [consumer_block.args[0], consumer_block.args[1]]
        assert len(_ops_of_type(entry_block, StoreOp)) == 0

    def test_store_arbitrary_record(self):
        """Multi-block: collection flows as entry-block arg, stored via an arbitrary record
        source across blocks.
        """
        type1 = i32
        type2 = i64
        size = 4

        schema = _schema(type1=type1, type2=type2)
        collection = CreateOp.for_empty_collection(schema, size)
        idx = _const_i32(0)
        source = _MockTypedSourceOp(RecordType(schema))

        consumer = Block(arg_types=[collection.result.type])
        consumer.add_ops(
            [
                _MockVariadicConsumerOp(consumer.args[0]),
                func.ReturnOp(),
            ]
        )

        producer = Block(
            arg_types=[collection.result.type, idx.result.type, source.result.type]
        )
        store = StoreOp.record_in_collection(
            producer.args[0], producer.args[1], producer.args[2]
        )
        producer.add_ops([store, cf.BranchOp(consumer, store.result)])

        entry = Block(
            [
                collection,
                idx,
                source,
                cf.BranchOp(producer, collection.result, idx.result, source.result),
            ]
        )
        fn = func.FuncOp("main", ((), ()), Region([entry, consumer, producer]))
        convert_results_collections_to_arrays(fn)

        blocks = list(fn.body.blocks)
        entry_block, consumer_block, producer_block = blocks
        producer_extract_ops = _ops_of_type(producer_block, ExtractOp)
        entry_create_ops = _ops_of_type(entry_block, CreateOp)

        assert blocks == [entry, consumer, producer]
        assert len(entry_create_ops) == 2
        assert len(producer_extract_ops) == 2
        assert len(_ops_of_type(consumer_block, _MockVariadicConsumerOp)) == 1
        assert len(consumer_block.args) == 2
        assert all(
            arg.type == create.result.type
            for arg, create in zip(consumer_block.args, entry_create_ops, strict=False)
        )

    def test_store_value_op(self):
        """Multi-block: collection flows as entry-block arg, a keyed scalar is stored in a
        producer block and the result passes to an already-lowered consumer block.
        """
        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        idx = _const_i32(0)
        value = _const_i32(99)

        consumer = Block(
            arg_types=[
                ResultsArrayType(type1, IntAttr(size)),
                ResultsArrayType(type2, IntAttr(size)),
            ]
        )
        consumer.add_ops(
            [
                _MockVariadicConsumerOp(consumer.args[0], consumer.args[1]),
                func.ReturnOp(),
            ]
        )

        producer = Block(
            arg_types=[collection.result.type, idx.result.type, value.result.type]
        )
        store = StoreOp.value_in_collection(
            producer.args[0], producer.args[1], "a", producer.args[2]
        )
        producer.add_ops([store, cf.BranchOp(consumer, store.result)])

        entry = Block(
            [
                collection,
                idx,
                value,
                cf.BranchOp(producer, collection.result, idx.result, value.result),
            ]
        )
        fn = func.FuncOp("main", ((), ()), Region([entry, consumer, producer]))
        convert_results_collections_to_arrays(fn)

        blocks = list(fn.body.blocks)
        entry_block, consumer_block, producer_block = blocks

        assert blocks == [entry, consumer, producer]
        assert len(_ops_of_type(entry_block, CreateOp)) == 2
        assert len(_ops_of_type(producer_block, StoreOp)) == 1
        assert len(_ops_of_type(consumer_block, _MockVariadicConsumerOp)) == 1
        assert len(consumer_block.args) == 2
        assert isinstance(consumer_block.args[0].type, ResultsArrayType)
        assert isinstance(consumer_block.args[1].type, ResultsArrayType)

    def test_extract_array_op(self):
        """Multi-block: collection flows as entry-block arg, a full array is extracted and
        passed to a consumer; the extract is eliminated by direct SSA rewiring.
        """
        a1 = CreateOp.for_array(i32, 3)
        a2 = CreateOp.for_array(i64, 3)
        collection = CreateOp.for_collection_from_arrays(["a", "b"], [a1.result, a2.result])

        consumer = Block(arg_types=[ResultsArrayType(i32, IntAttr(3))])
        consumer.add_ops(
            [
                _MockVariadicConsumerOp(consumer.args[0]),
                func.ReturnOp(),
            ]
        )

        producer = Block(arg_types=[collection.result.type])
        extract = ExtractOp.array_from_collection(producer.args[0], "a")
        producer.add_ops([extract, cf.BranchOp(consumer, extract.result)])

        entry = Block([a1, a2, collection, cf.BranchOp(producer, collection.result)])
        fn = func.FuncOp("main", ((), ()), Region([entry, consumer, producer]))
        convert_results_collections_to_arrays(fn)

        blocks = list(fn.body.blocks)
        entry_block, consumer_block, producer_block = blocks

        assert blocks == [entry, consumer, producer]
        assert len(_ops_of_type(entry_block, CreateOp)) == 2
        assert len(_ops_of_type(producer_block, ExtractOp)) == 0
        assert len(consumer_block.args) == 1
        assert isinstance(consumer_block.args[0].type, ResultsArrayType)
        assert len(_ops_of_type(consumer_block, _MockVariadicConsumerOp)) == 1

    def test_extract_value_op(self):
        """Multi-block: collection flows as entry-block arg, scalar value extracted in a
        producer block and passed through to a single-arg consumer.
        """
        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        idx = _const_i32(0)

        consumer = Block(arg_types=[type1])
        consumer.add_ops([_MockNonVariadicConsumerOp(consumer.args[0]), func.ReturnOp()])

        producer = Block(arg_types=[collection.result.type, idx.result.type])
        extract = ExtractOp.value_from_collection(producer.args[0], "a", producer.args[1])
        producer.add_ops([extract, cf.BranchOp(consumer, extract.result)])

        entry = Block(
            [collection, idx, cf.BranchOp(producer, collection.result, idx.result)]
        )
        fn = func.FuncOp("main", ((), ()), Region([entry, consumer, producer]))
        convert_results_collections_to_arrays(fn)

        blocks = list(fn.body.blocks)
        entry_block, consumer_block, producer_block = blocks

        assert blocks == [entry, consumer, producer]
        assert len(_ops_of_type(entry_block, CreateOp)) == 2
        assert len(_ops_of_type(producer_block, ExtractOp)) == 1
        assert len(consumer_block.args) == 1
        assert consumer_block.args[0].type == type1

    def test_extract_record_op(self):
        """Multi-block: collection flows as entry-block arg, a record is extracted in a
        producer block and passed through to a single-arg consumer.
        """
        type1 = i32
        type2 = i64
        size = 4

        collection = CreateOp.for_empty_collection(_schema(type1=type1, type2=type2), size)
        idx = _const_i32(0)

        consumer = Block(arg_types=[RecordType(_schema(type1=type1, type2=type2))])
        consumer.add_ops([_MockNonVariadicConsumerOp(consumer.args[0]), func.ReturnOp()])

        producer = Block(arg_types=[collection.result.type, idx.result.type])
        extract = ExtractOp.record_from_collection(producer.args[0], producer.args[1])
        producer.add_ops([extract, cf.BranchOp(consumer, extract.result)])

        entry = Block(
            [collection, idx, cf.BranchOp(producer, collection.result, idx.result)]
        )
        fn = func.FuncOp("main", ((), ()), Region([entry, consumer, producer]))
        convert_results_collections_to_arrays(fn)

        blocks = list(fn.body.blocks)
        entry_block, consumer_block, producer_block = blocks

        assert blocks == [entry, consumer, producer]
        assert len(_ops_of_type(entry_block, CreateOp)) == 2
        assert len(_ops_of_type(producer_block, ExtractOp)) == 2
        assert len(consumer_block.args) == 1
        assert consumer_block.args[0].type == RecordType(_schema(type1=type1, type2=type2))
