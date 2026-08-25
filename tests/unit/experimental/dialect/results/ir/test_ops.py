# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the operations in the results dialect."""

import pytest
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import DYNAMIC_INDEX, IntAttr, StringAttr, TupleType, i32, i64
from xdsl.ir import Block, Region, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    result_def,
    traits_def,
)
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.results.ir import (
    CreateOp,
    ExtractOp,
    GroupEntriesOp,
    IntegerStatePredicateAttr,
    MapOp,
    PostSelectOp,
    PostSelectPredicateAttr,
    RecordFieldAttr,
    RecordSchemaAttr,
    RecordType,
    ReduceOp,
    ResultsArrayType,
    ResultsCollectionType,
    StoreOp,
    YieldOp,
)

_DEFAULT_SCHEMA = RecordSchemaAttr([RecordFieldAttr("a", i32)])


@irdl_op_definition
class _MockRecordOp(IRDLOperation):
    """A minimal mock op that produces a RecordType SSA result for testing."""

    name = "results.test_mock_record"
    res = result_def(RecordType)

    def __init__(self, schema: RecordSchemaAttr = _DEFAULT_SCHEMA):
        super().__init__(result_types=[RecordType(schema)])


@irdl_op_definition
class _MockCollectionOp(IRDLOperation):
    """A minimal mock op that produces a ResultsCollectionType SSA result for testing."""

    name = "results.test_mock_collection"
    res = result_def(ResultsCollectionType)

    def __init__(self, schema: RecordSchemaAttr | None = None, size: IntAttr | None = None):
        schema = schema if schema is not None else _DEFAULT_SCHEMA
        size = size if size is not None else IntAttr(DYNAMIC_INDEX)
        super().__init__(result_types=[ResultsCollectionType(schema, size)])


@irdl_op_definition
class _MockArrayOp(IRDLOperation):
    """A minimal mock op that produces a ResultsArrayType SSA result for testing."""

    name = "results.test_mock_array"
    res = result_def(ResultsArrayType)

    def __init__(
        self,
        type_: TypeAttribute,
        size: IntAttr,
    ):
        super().__init__(result_types=[ResultsArrayType(type_, size)])


@irdl_op_definition
class _MockTerminatorOp(IRDLOperation):
    """A minimal mock op that acts as a terminator for testing."""

    name = "results.test_mock_terminator"
    traits = traits_def(IsTerminator())


class TestCreateOp:
    """Tests the merged CreateOp semantics."""

    def test_for_record_initialization_and_properties(self):
        value1 = ArithConstantOp.from_int_and_width(1, i32)
        value2 = ArithConstantOp.from_int_and_width(2, i64)

        op = CreateOp.for_record(["a", "b"], [value1.result, value2.result])
        op.verify()

        expected_schema = RecordSchemaAttr(
            [
                RecordFieldAttr("a", i32),
                RecordFieldAttr("b", i64),
            ]
        )
        assert op.result.type == RecordType(expected_schema)
        assert tuple(op.values) == (value1.result, value2.result)

    def test_for_record_fails_when_keys_and_values_length_mismatch(self):
        value = ArithConstantOp.from_int_and_width(1, i32)

        with pytest.raises(ValueError, match="does not match number of values"):
            CreateOp.for_record(["a", "b"], [value.result])

    def test_for_array_static_size_initialization_and_properties(self):
        op = CreateOp.for_array(i32, 2)
        op.verify()

        assert op.result.type == ResultsArrayType(i32, IntAttr(2))

    def test_for_array_requires_size_argument(self):
        with pytest.raises(TypeError):
            CreateOp.for_array(i32)

    def test_for_array_accepts_int_attr_size(self):
        op = CreateOp.for_array(i32, IntAttr(3))
        op.verify()

        assert op.result.type == ResultsArrayType(i32, IntAttr(3))

    def test_for_array_rejects_invalid_size_type(self):
        with pytest.raises(
            TypeError,
            match="Size must be an int, IntAttr, Operation, or SSAValue\\[IntegerType\\]",
        ):
            CreateOp.for_array(i32, "invalid")

    def test_for_array_dynamic_size_initialization_and_properties(self):
        dyn_size = ArithConstantOp.from_int_and_width(5, i32)
        op = CreateOp.for_array(i32, dyn_size.result)
        op.verify()

        assert op.size is dyn_size.result
        assert op.result.type == ResultsArrayType(i32, IntAttr(DYNAMIC_INDEX))

    def test_for_array_accepts_operation_size_operand(self):
        dyn_size = ArithConstantOp.from_int_and_width(5, i32)
        op = CreateOp(ResultsArrayType(i32, IntAttr(DYNAMIC_INDEX)), size=dyn_size)
        op.verify()

        assert op.size is dyn_size.result
        assert op.result.type == ResultsArrayType(i32, IntAttr(DYNAMIC_INDEX))

    def test_for_collection_from_arrays_initialization_and_properties(self):
        arr1 = _MockArrayOp(i32, IntAttr(1))
        arr2 = _MockArrayOp(i32, IntAttr(1))

        op = CreateOp.for_collection_from_arrays(["a", "b"], [arr1.res, arr2.res])
        op.verify()

        expected_schema = RecordSchemaAttr(
            [
                RecordFieldAttr("a", i32),
                RecordFieldAttr("b", i32),
            ]
        )
        assert op.result.type == ResultsCollectionType(expected_schema, IntAttr(1))

    def test_for_collection_from_arrays_fails_when_arrays_are_empty(self):
        with pytest.raises(ValueError, match="requires at least one array"):
            CreateOp.for_collection_from_arrays([], [])

    def test_for_collection_from_arrays_fails_when_keys_and_arrays_length_mismatch(self):
        arr = _MockArrayOp(i32, IntAttr(1))

        with pytest.raises(ValueError, match="does not match number of arrays"):
            CreateOp.for_collection_from_arrays(["a", "b"], [arr.res])

    def test_for_collection_from_arrays_fails_when_sizes_mismatch(self):
        arr1 = _MockArrayOp(i32, IntAttr(1))
        arr2 = _MockArrayOp(i32, IntAttr(2))

        with pytest.raises(ValueError, match="same size"):
            CreateOp.for_collection_from_arrays(["a", "b"], [arr1.res, arr2.res])

    def test_for_empty_collection_static_size_initialization_and_properties(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])

        op = CreateOp.for_empty_collection(schema, 3)
        op.verify()

        assert op.result.type == ResultsCollectionType(schema, IntAttr(3))

    def test_for_empty_collection_dynamic_size_initialization_and_properties(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        dyn_size = ArithConstantOp.from_int_and_width(5, i32)

        op = CreateOp.for_empty_collection(schema, dyn_size.result)
        op.verify()

        assert op.size is dyn_size.result
        assert op.result.type == ResultsCollectionType(schema, IntAttr(DYNAMIC_INDEX))

    def test_for_empty_collection_rejects_invalid_size_type(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])

        with pytest.raises(
            TypeError,
            match="Size must be an int, IntAttr, Operation, or SSAValue\\[IntegerType\\]",
        ):
            CreateOp.for_empty_collection(schema, "invalid")

    def test_verify_array_fails_when_values_are_provided(self):
        value = ArithConstantOp.from_int_and_width(1, i32)
        op = CreateOp(ResultsArrayType(i32, IntAttr(1)), values=[value.result])

        with pytest.raises(VerifyException, match="only supports empty arrays"):
            op.verify()

    def test_verify_dynamic_array_requires_size_operand(self):
        op = CreateOp(ResultsArrayType(i32, IntAttr(DYNAMIC_INDEX)))

        with pytest.raises(VerifyException, match="requires a size operand"):
            op.verify()

    def test_verify_static_array_rejects_size_operand(self):
        dyn_size = ArithConstantOp.from_int_and_width(2, i32)
        op = CreateOp(ResultsArrayType(i32, IntAttr(2)), size=dyn_size.result)

        with pytest.raises(VerifyException, match="does not use a size operand"):
            op.verify()

    def test_verify_collection_from_arrays_requires_array_operands(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        value = ArithConstantOp.from_int_and_width(1, i32)
        op = CreateOp(ResultsCollectionType(schema, IntAttr(1)), values=[value.result])

        with pytest.raises(VerifyException, match="must be of type ResultsArrayType"):
            op.verify()

    def test_verify_collection_from_arrays_rejects_mismatched_field_type(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        array = _MockArrayOp(i64, IntAttr(1))
        op = CreateOp(ResultsCollectionType(schema, IntAttr(1)), values=[array.res])

        with pytest.raises(VerifyException, match="does not match the expected type"):
            op.verify()

    def test_verify_dynamic_empty_collection_requires_size_operand(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        op = CreateOp(ResultsCollectionType(schema, IntAttr(DYNAMIC_INDEX)))

        with pytest.raises(VerifyException, match="requires a size operand"):
            op.verify()

    def test_verify_record_fails_when_size_operand_is_present(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        value = ArithConstantOp.from_int_and_width(1, i32)
        dyn_size = ArithConstantOp.from_int_and_width(2, i32)
        op = CreateOp(RecordType(schema), values=[value.result], size=dyn_size.result)

        with pytest.raises(VerifyException, match="does not use a size operand"):
            op.verify()

    def test_verify_record_fails_when_values_length_mismatches_schema(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32), RecordFieldAttr("b", i32)])
        value = ArithConstantOp.from_int_and_width(1, i32)
        op = CreateOp(RecordType(schema), values=[value.result])

        with pytest.raises(VerifyException, match="does not match number of values"):
            op.verify()

    def test_verify_record_fails_when_value_type_mismatches_schema(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i64)])
        value = ArithConstantOp.from_int_and_width(1, i32)
        op = CreateOp(RecordType(schema), values=[value.result])

        with pytest.raises(VerifyException, match="does not match the expected type"):
            op.verify()

    def test_verify_collection_from_arrays_fails_when_size_operand_present(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        array = _MockArrayOp(i32, IntAttr(1))
        dyn_size = ArithConstantOp.from_int_and_width(1, i32)
        op = CreateOp(
            ResultsCollectionType(schema, IntAttr(1)),
            values=[array.res],
            size=dyn_size.result,
        )

        with pytest.raises(VerifyException, match="does not use a size operand"):
            op.verify()

    def test_verify_collection_from_arrays_fails_when_schema_fields_length_mismatches(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32), RecordFieldAttr("b", i32)])
        array = _MockArrayOp(i32, IntAttr(1))
        op = CreateOp(ResultsCollectionType(schema, IntAttr(1)), values=[array.res])

        with pytest.raises(VerifyException, match="does not match number of arrays"):
            op.verify()

    def test_verify_collection_from_arrays_fails_when_array_size_mismatches(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        array = _MockArrayOp(i32, IntAttr(2))
        op = CreateOp(ResultsCollectionType(schema, IntAttr(1)), values=[array.res])

        with pytest.raises(VerifyException, match="same size"):
            op.verify()

    def test_verify_empty_collection_rejects_size_operand_for_static_size(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        dyn_size = ArithConstantOp.from_int_and_width(1, i32)
        op = CreateOp(ResultsCollectionType(schema, IntAttr(1)), size=dyn_size.result)

        with pytest.raises(VerifyException, match="does not use a size operand"):
            op.verify()

    def test_verify_tuple_fails_when_size_operand_present(self):
        value = ArithConstantOp.from_int_and_width(1, i32)
        dyn_size = ArithConstantOp.from_int_and_width(2, i32)
        op = CreateOp(TupleType((i32,)), values=[value.result], size=dyn_size.result)

        with pytest.raises(VerifyException, match="does not use a size operand"):
            op.verify()

    def test_verify_tuple_fails_when_values_length_mismatches(self):
        op = CreateOp(TupleType((i32, i64)), values=[])

        with pytest.raises(VerifyException, match="does not match number of values"):
            op.verify()

    def test_verify_tuple_fails_when_value_type_mismatches(self):
        value = ArithConstantOp.from_int_and_width(1, i32)
        op = CreateOp(TupleType((i64,)), values=[value.result])

        with pytest.raises(VerifyException, match="does not match the expected tuple"):
            op.verify()

    def test_for_tuple_initialization_and_properties(self):
        value1 = ArithConstantOp.from_int_and_width(1, i32)
        value2 = ArithConstantOp.from_int_and_width(2, i64)

        op = CreateOp.for_tuple([value1.result, value2.result])
        op.verify()

        assert tuple(op.values) == (value1.result, value2.result)
        assert op.result.type == TupleType((i32, i64))


class TestStoreOp:
    """Tests the merged StoreOp semantics."""

    def test_for_collection_record_factory_does_not_set_key(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema)
        record = _MockRecordOp(schema=schema)
        index = ArithConstantOp.from_int_and_width(0, i32)

        op = StoreOp.record_in_collection(collection.res, index.result, record.res)
        op.verify()

        assert op.key is None

    def test_for_collection_value_factory_sets_key(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema)
        index = ArithConstantOp.from_int_and_width(0, i32)
        value = ArithConstantOp.from_int_and_width(1, i32)

        op = StoreOp.value_in_collection(
            collection.res,
            index.result,
            key="a",
            value=value.result,
        )
        op.verify()

        assert op.key == StringAttr("a")

    def test_for_array_factory_does_not_set_key(self):
        array = _MockArrayOp(i32, IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        value = ArithConstantOp.from_int_and_width(1, i32)

        op = StoreOp.value_in_array(array.res, index.result, value.result)
        op.verify()

        assert op.key is None

    def test_store_record_in_collection_passes_for_matching_schema(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema)
        record = _MockRecordOp(schema=schema)
        index = ArithConstantOp.from_int_and_width(0, i32)

        op = StoreOp(collection.res, index.result, record.res)
        op.verify()

        assert op.result.type == collection.res.type

    def test_store_record_in_collection_fails_for_schema_mismatch(self):
        collection_schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        record_schema = RecordSchemaAttr([RecordFieldAttr("b", i32)])
        collection = _MockCollectionOp(schema=collection_schema)
        record = _MockRecordOp(schema=record_schema)
        index = ArithConstantOp.from_int_and_width(0, i32)

        op = StoreOp(collection.res, index.result, record.res)

        with pytest.raises(VerifyException, match="does not match schema"):
            op.verify()

    def test_store_value_in_collection_fails_when_key_missing(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema)
        index = ArithConstantOp.from_int_and_width(0, i32)
        value = ArithConstantOp.from_int_and_width(1, i32)

        op = StoreOp(collection.res, index.result, value.result)

        with pytest.raises(VerifyException, match="requires a key"):
            op.verify()

    def test_store_value_in_collection_fails_when_key_not_in_schema(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema)
        index = ArithConstantOp.from_int_and_width(0, i32)
        value = ArithConstantOp.from_int_and_width(1, i32)

        op = StoreOp(collection.res, index.result, value.result, key="missing")

        with pytest.raises(VerifyException, match="does not exist in the schema"):
            op.verify()

    def test_store_value_in_collection_fails_when_value_type_mismatch(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema)
        index = ArithConstantOp.from_int_and_width(0, i32)
        value = ArithConstantOp.from_int_and_width(1, i64)

        op = StoreOp(collection.res, index.result, value.result, key="a")

        with pytest.raises(VerifyException, match="does not match the expected type"):
            op.verify()

    def test_store_value_in_collection_passes_when_type_matches(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema)
        index = ArithConstantOp.from_int_and_width(0, i32)
        value = ArithConstantOp.from_int_and_width(1, i32)

        op = StoreOp(collection.res, index.result, value.result, key="a")
        op.verify()

    def test_store_value_in_array_fails_when_value_type_mismatch(self):
        array = _MockArrayOp(i32, IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        value = ArithConstantOp.from_int_and_width(1, i64)

        op = StoreOp(array.res, index.result, value.result)

        with pytest.raises(VerifyException, match="does not match the expected type"):
            op.verify()

    def test_store_value_in_array_passes_when_value_type_matches(self):
        array = _MockArrayOp(i32, IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        value = ArithConstantOp.from_int_and_width(1, i32)

        op = StoreOp(array.res, index.result, value.result)
        op.verify()

    def test_store_record_in_collection_fails_when_key_is_set(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema)
        record = _MockRecordOp(schema=schema)
        index = ArithConstantOp.from_int_and_width(0, i32)

        op = StoreOp(collection.res, index.result, record.res, key="a")

        with pytest.raises(VerifyException, match="does not require a key"):
            op.verify()

    def test_store_value_in_array_fails_when_key_is_set(self):
        array = _MockArrayOp(i32, IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        value = ArithConstantOp.from_int_and_width(1, i32)

        op = StoreOp(array.res, index.result, value.result, key="a")

        with pytest.raises(VerifyException, match="does not require a key"):
            op.verify()

    def test_store_result_type_must_match_container_type(self):
        array = _MockArrayOp(i32, IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        value = ArithConstantOp.from_int_and_width(1, i32)
        op = StoreOp.create(
            operands=[array.res, index.result, value.result],
            result_types=[i32],
        )

        with pytest.raises(VerifyException, match="Result type"):
            op.verify()


class TestExtractOp:
    """Tests the merged ExtractOp semantics."""

    def test_value_from_record_factory_initialization(self):
        value = ArithConstantOp.from_int_and_width(1, i32)
        record = CreateOp.for_record(["a"], [value.result])

        op = ExtractOp.value_from_record(record.result, "a")
        op.verify()

        assert op.result.type == i32
        assert op.key == StringAttr("a")

    def test_value_from_record_factory_fails_when_key_missing(self):
        value = ArithConstantOp.from_int_and_width(1, i32)
        record = CreateOp.for_record(["a"], [value.result])

        with pytest.raises(ValueError, match="does not exist in the schema"):
            ExtractOp.value_from_record(record.result, "missing")

    def test_verification_fails_when_record_result_type_mismatches_schema(self):
        value = ArithConstantOp.from_int_and_width(1, i32)
        record = CreateOp.for_record(["a"], [value.result])

        op = ExtractOp(record.result, i64, key="a")

        with pytest.raises(VerifyException, match="does not match the expected type"):
            op.verify()

    def test_verification_fails_when_key_property_mismatches_schema(self):
        value = ArithConstantOp.from_int_and_width(1, i32)
        record = CreateOp.for_record(["a"], [value.result])

        op = ExtractOp(record.result, i32, key="a")
        op.properties["key"] = StringAttr("missing")

        with pytest.raises(VerifyException, match="does not exist in the schema"):
            op.verify()

    def test_value_from_array_factory_initialization(self):
        array = _MockArrayOp(i32, IntAttr(2))
        index = ArithConstantOp.from_int_and_width(0, i32)

        op = ExtractOp.value_from_array(array.res, index.result)
        op.verify()

        assert op.result.type == i32

    def test_extract_from_array_without_index_fails(self):
        array = _MockArrayOp(i32, IntAttr(1))
        op = ExtractOp(array.res, i32)

        with pytest.raises(VerifyException, match="requires an index"):
            op.verify()

    def test_record_from_collection_factory_initialization(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)

        op = ExtractOp.record_from_collection(collection.res, index.result)
        op.verify()

        assert op.result.type == RecordType(schema)

    def test_array_from_collection_factory_initialization(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))

        op = ExtractOp.array_from_collection(collection.res, key="a")
        op.verify()

        assert op.result.type == ResultsArrayType(i32, IntAttr(1))

    def test_value_from_collection_factory_initialization(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)

        op = ExtractOp.value_from_collection(collection.res, key="a", index=index.result)
        op.verify()

        assert op.result.type == i32

    def test_extract_record_from_collection_rejects_key(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        op = ExtractOp(collection.res, RecordType(schema), key="a", index=index.result)

        with pytest.raises(VerifyException, match="does not use a key"):
            op.verify()

    def test_extract_array_from_collection_rejects_index(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        op = ExtractOp(
            collection.res,
            ResultsArrayType(i32, IntAttr(1)),
            key="a",
            index=index.result,
        )

        with pytest.raises(VerifyException, match="does not use an index"):
            op.verify()

    def test_extract_value_from_collection_requires_both_selectors(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        op = ExtractOp(collection.res, i32, key="a")

        with pytest.raises(VerifyException, match="requires both key and index"):
            op.verify()

    def test_extract_value_from_collection_result_type_mismatch_fails(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        op = ExtractOp(collection.res, i64, key="a", index=index.result)

        with pytest.raises(VerifyException, match="does not match the expected type"):
            op.verify()

    def test_array_from_collection_factory_fails_when_key_missing(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))

        with pytest.raises(ValueError, match="does not exist in the schema"):
            ExtractOp.array_from_collection(collection.res, key="missing")

    def test_value_from_collection_factory_fails_when_key_missing(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)

        with pytest.raises(ValueError, match="does not exist in the schema"):
            ExtractOp.value_from_collection(
                collection.res, key="missing", index=index.result
            )

    def test_extract_from_record_requires_key(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        record = _MockRecordOp(schema=schema)
        op = ExtractOp(record.res, i32)

        with pytest.raises(VerifyException, match="requires a key"):
            op.verify()

    def test_extract_from_record_rejects_index(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        record = _MockRecordOp(schema=schema)
        index = ArithConstantOp.from_int_and_width(0, i32)
        op = ExtractOp(record.res, i32, key="a", index=index.result)

        with pytest.raises(VerifyException, match="does not use an index"):
            op.verify()

    def test_extract_from_array_rejects_key(self):
        array = _MockArrayOp(i32, IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        op = ExtractOp(array.res, i32, key="a", index=index.result)

        with pytest.raises(VerifyException, match="does not use a key"):
            op.verify()

    def test_extract_from_array_result_type_mismatch_fails(self):
        array = _MockArrayOp(i32, IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        op = ExtractOp(array.res, i64, index=index.result)

        with pytest.raises(VerifyException, match="does not match the expected type"):
            op.verify()

    def test_extract_record_from_collection_requires_index(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        op = ExtractOp(collection.res, RecordType(schema))

        with pytest.raises(VerifyException, match="requires an index"):
            op.verify()

    def test_extract_record_from_collection_result_type_mismatch_fails(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        wrong_schema = RecordSchemaAttr([RecordFieldAttr("b", i32)])
        op = ExtractOp(collection.res, RecordType(wrong_schema), index=index.result)

        with pytest.raises(VerifyException, match="does not match the expected type"):
            op.verify()

    def test_extract_array_from_collection_requires_key(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        op = ExtractOp(collection.res, ResultsArrayType(i32, IntAttr(1)))

        with pytest.raises(VerifyException, match="requires a key"):
            op.verify()

    def test_extract_array_from_collection_fails_when_key_not_in_schema(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        op = ExtractOp(collection.res, ResultsArrayType(i32, IntAttr(1)), key="missing")

        with pytest.raises(VerifyException, match="does not exist in the schema"):
            op.verify()

    def test_extract_array_from_collection_result_type_mismatch_fails(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        op = ExtractOp(collection.res, ResultsArrayType(i64, IntAttr(1)), key="a")

        with pytest.raises(VerifyException, match="does not match the expected type"):
            op.verify()

    def test_extract_value_from_collection_fails_when_key_not_in_schema(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        index = ArithConstantOp.from_int_and_width(0, i32)
        op = ExtractOp(collection.res, i32, key="missing", index=index.result)

        with pytest.raises(VerifyException, match="does not exist in the schema"):
            op.verify()


class TestPostSelectOp:
    """Tests the PostSelectOp."""

    @irdl_attr_definition
    class _DummyPredicateAttr(PostSelectPredicateAttr):
        """A dummy predicate attribute for testing purposes."""

        name = "results.test_dummy_predicate"

    def test_initialization_and_properties(self):
        collection = _MockCollectionOp()
        predicate = IntegerStatePredicateAttr("state", [0])

        op = PostSelectOp(collection.res, predicate)
        op.verify()

        assert op.collection is collection.res
        assert tuple(op.predicates) == (predicate,)
        assert op.result.type == ResultsCollectionType(
            collection.res.type.schema,
            IntAttr(DYNAMIC_INDEX),
        )
        assert op.result.type.size == IntAttr(DYNAMIC_INDEX)

    def test_initialization_accepts_predicate_subclass(self):
        collection = _MockCollectionOp()
        predicate = self._DummyPredicateAttr()

        op = PostSelectOp(collection.res, predicate)
        op.verify()

        assert tuple(op.predicates) == (predicate,)
        assert op.result.type.size == IntAttr(DYNAMIC_INDEX)


class TestGroupEntriesOp:
    """Tests the GroupEntriesOp."""

    def test_initialization_builds_grouped_schema(self):
        value1 = ArithConstantOp.from_int_and_width(1, i32)
        value2 = ArithConstantOp.from_int_and_width(2, i64)
        record = CreateOp.for_record(["a", "b"], [value1.result, value2.result])

        op = GroupEntriesOp(record.result, ["a", "b"], "grouped")
        op.verify()

        expected_schema = RecordSchemaAttr(
            [RecordFieldAttr("grouped", TupleType((i32, i64)))]
        )
        assert op.result.type == RecordType(expected_schema)

    def test_initialization_fails_when_group_keys_are_missing(self):
        value = ArithConstantOp.from_int_and_width(1, i32)
        record = CreateOp.for_record(["a"], [value.result])

        with pytest.raises(VerifyException, match="All keys to group must exist"):
            GroupEntriesOp(record.result, ["missing"], "grouped")

    def test_verification_fails_with_empty_keys(self):
        record = _MockRecordOp()
        op = GroupEntriesOp(record.res, [], "grouped")

        with pytest.raises(VerifyException, match="requires at least one key to group"):
            op.verify()


class TestReduceOp:
    """Tests the ReduceOp."""

    def test_initialization_builds_reduced_schema(self):
        value1 = ArithConstantOp.from_int_and_width(1, i32)
        value2 = ArithConstantOp.from_int_and_width(2, i64)
        record = CreateOp.for_record(["a", "b"], [value1.result, value2.result])

        op = ReduceOp(record.result, ["b"])
        op.verify()

        expected_schema = RecordSchemaAttr([RecordFieldAttr("b", i64)])
        assert op.result.type == RecordType(expected_schema)

    def test_initialization_fails_when_reduce_keys_are_missing(self):
        value = ArithConstantOp.from_int_and_width(1, i32)
        record = CreateOp.for_record(["a"], [value.result])

        with pytest.raises(VerifyException, match="All keys to retain must exist"):
            ReduceOp(record.result, ["missing"])

    def test_verification_fails_with_empty_keys(self):
        record = _MockRecordOp()
        op = ReduceOp(record.res, [])

        with pytest.raises(VerifyException, match="requires at least one key to retain"):
            op.verify()


class TestYieldOp:
    """Tests the YieldOp."""

    def test_initialization_and_properties(self):
        record = _MockRecordOp()

        op = YieldOp(record.res)
        op.verify()

        assert op.record is record.res


class TestMapOp:
    """Tests the MapOp initialization and verification branches."""

    def _make_valid_block(self, schema: RecordSchemaAttr) -> Block:
        record_type = RecordType(schema)
        block = Block(arg_types=(record_type,))
        block.add_ops([YieldOp(block.args[0])])
        return block

    def test_initialization_with_region_passes_verification(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection_type = ResultsCollectionType.dynamic_size(schema)
        block = self._make_valid_block(schema=schema)
        region = Region(blocks=[block])
        collection_op = _MockCollectionOp(schema=schema)

        map_op = MapOp(collection_op.res, region, collection_type)
        map_op.verify()

        assert map_op.value is collection_op.res
        assert map_op.body is region
        assert map_op.result.type == collection_type

    def test_initialization_with_block_passes_verification(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection_type = ResultsCollectionType.dynamic_size(schema)
        block = self._make_valid_block(schema=schema)
        collection_op = _MockCollectionOp(schema=schema)

        map_op = MapOp(collection_op.res, block, collection_type)
        map_op.verify()

        assert map_op.value is collection_op.res
        assert map_op.body.blocks[0] is block

    def test_verification_with_two_block_arguments_fails(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection_type = ResultsCollectionType.dynamic_size(schema)
        block = Block(arg_types=(RecordType(schema), RecordType(schema)))
        block.add_ops([YieldOp(block.args[0])])
        collection_op = _MockCollectionOp(schema=schema)
        map_op = MapOp(collection_op.res, block, collection_type)

        with pytest.raises(
            VerifyException, match="must have a single argument of type RecordType"
        ):
            map_op.verify()

    def test_verification_with_non_yield_last_op_fails(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection_type = ResultsCollectionType.dynamic_size(schema)
        block = Block(arg_types=(RecordType(schema),))
        block.add_ops([_MockTerminatorOp()])
        collection_op = _MockCollectionOp(schema=schema)
        map_op = MapOp(collection_op.res, block, collection_type)

        with pytest.raises(VerifyException, match="must be a YieldOp"):
            map_op.verify()

    def test_verification_when_yielded_record_type_mismatches_result_schema(self):
        input_schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        output_schema = RecordSchemaAttr([RecordFieldAttr("b", i64)])

        collection_type = ResultsCollectionType.dynamic_size(output_schema)
        block = self._make_valid_block(schema=input_schema)
        collection_op = _MockCollectionOp(schema=input_schema)
        map_op = MapOp(collection_op.res, block, collection_type)

        with pytest.raises(VerifyException, match="must match the schema"):
            map_op.verify()

    def test_verification_when_input_schema_mismatches_collection_schema(self):
        input_schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])
        collection_schema = RecordSchemaAttr([RecordFieldAttr("b", i32)])

        collection_type = ResultsCollectionType.dynamic_size(collection_schema)
        block = self._make_valid_block(schema=input_schema)
        collection_op = _MockCollectionOp(schema=collection_schema)
        map_op = MapOp(collection_op.res, block, collection_type)

        with pytest.raises(
            VerifyException, match="must match the schema of the input collection"
        ):
            map_op.verify()

    def test_verification_when_result_size_mismatches_input_size(self):
        schema = RecordSchemaAttr([RecordFieldAttr("a", i32)])

        input_collection = _MockCollectionOp(schema=schema, size=IntAttr(1))
        output_collection_type = ResultsCollectionType(schema, IntAttr(2))
        block = self._make_valid_block(schema=schema)
        map_op = MapOp(input_collection.res, block, output_collection_type)

        with pytest.raises(
            VerifyException, match="must match the size of the input collection"
        ):
            map_op.verify()
