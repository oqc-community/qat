# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the operations in the results dialect."""

import pytest
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr, i32
from xdsl.ir import Block, Region
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
    AddRecordOp,
    CreateRecordOp,
    CreateResultsArrayOp,
    CreateResultsCollectionOp,
    ExtractOp,
    GroupEntriesOp,
    IntegerStatePredicateAttr,
    MapOp,
    PostSelectOp,
    ReduceOp,
    YieldOp,
)
from qat.experimental.dialect.results.ir.attributes import PostSelectPredicateAttr
from qat.experimental.dialect.results.ir.types import RecordType, ResultsCollectionType


@irdl_op_definition
class _MockRecordOp(IRDLOperation):
    """A minimal mock op that produces a RecordType SSA result for testing."""

    name = "results.test_mock_record"
    res = result_def(RecordType)

    def __init__(self):
        super().__init__(result_types=[RecordType()])


@irdl_op_definition
class _MockCollectionOp(IRDLOperation):
    """A minimal mock op that produces a ResultsCollectionType SSA result for testing."""

    name = "results.test_mock_collection"
    res = result_def(ResultsCollectionType)

    def __init__(self):
        super().__init__(result_types=[ResultsCollectionType()])


@irdl_op_definition
class _MockTerminatorOp(IRDLOperation):
    """A minimal mock op that acts as a terminator for testing."""

    name = "results.test_mock_terminator"
    traits = traits_def(IsTerminator())


class TestCreateRecordOp:
    """Tests the CreateRecordOp, which creates a results record."""

    def test_initialization_and_properties(self):
        """Tests that the CreateRecordOp can be initialized with a list of entries and that
        its properties return the expected values."""
        value1 = ArithConstantOp.from_int_and_width(1, i32)
        value2 = ArithConstantOp.from_int_and_width(2, i32)

        op = CreateRecordOp(["a", "b"], [value1.result, value2.result])
        op.verify()

        assert op.keys == ArrayAttr([StringAttr("a"), StringAttr("b")])
        assert tuple(op.values) == (value1.result, value2.result)
        assert op.result.type == RecordType()

    def test_verification_fails_when_keys_and_values_length_mismatch(self):
        """Tests that verification fails when the number of keys and values do not match,
        raising a VerifyException with the expected error message."""
        value = ArithConstantOp.from_int_and_width(1, i32)
        op = CreateRecordOp(["a", "b"], [value.result])

        with pytest.raises(VerifyException, match="does not match number of values"):
            op.verify()

    def test_verification_fails_with_duplicate_keys(self):
        """Tests that verification fails when there are duplicate keys, raising a
        VerifyException with the expected error message."""
        value1 = ArithConstantOp.from_int_and_width(1, i32)
        value2 = ArithConstantOp.from_int_and_width(2, i32)
        op = CreateRecordOp(["dup", "dup"], [value1.result, value2.result])

        with pytest.raises(VerifyException, match="Keys must be unique"):
            op.verify()


class TestCreateResultsArrayOp:
    """Tests the CreateResultsArrayOp, which creates a results array."""

    def test_initialization_and_properties(self):
        """Tests that the CreateResultsArrayOp can be initialized with SSA values and that
        its properties return the expected values."""
        value1 = ArithConstantOp.from_int_and_width(1, i32)
        value2 = ArithConstantOp.from_int_and_width(2, i32)

        op = CreateResultsArrayOp([value1.result, value2.result])
        op.verify()
        assert tuple(op.values) == (value1.result, value2.result)


class TestCreateResultsCollectionOp:
    """Tests the CreateResultsCollectionOp, which creates a results collection."""

    def test_initialization_and_properties(self):
        """Tests that the CreateResultsCollectionOp can be initialized and that its
        properties return the expected values."""
        op = CreateResultsCollectionOp()
        op.verify()
        assert op.result.type == ResultsCollectionType()


class TestAddRecordOp:
    """Tests the AddRecordOp, which adds a record to a results collection."""

    def test_initialization_and_properties(self):
        """Tests that the AddRecordOp can be initialized with a collection and a record and
        that its properties return the expected values."""
        collection = _MockCollectionOp()
        record = _MockRecordOp()

        op = AddRecordOp(collection.res, record.res)
        op.verify()

        assert op.collection is collection.res
        assert op.record is record.res
        assert op.result.type == ResultsCollectionType()


class TestPostSelectOp:
    """Tests the PostSelectOp, which filters a results collection based on a predicate."""

    @irdl_attr_definition
    class _DummyPredicateAttr(PostSelectPredicateAttr):
        """A dummy predicate attribute for testing purposes."""

        name = "results.test_dummy_predicate"

    def test_initialization_and_properties(self):
        """Tests that the PostSelectOp can be initialized with a collection and a predicate
        and that its properties return the expected values."""
        collection = _MockCollectionOp()
        predicate = IntegerStatePredicateAttr("state", [0])

        op = PostSelectOp(collection.res, predicate)
        op.verify()

        assert op.collection is collection.res
        assert tuple(op.predicates) == (predicate,)
        assert op.result.type == ResultsCollectionType()

    def test_initialization_accepts_predicate_subclass(self):
        """Tests that the PostSelectOp accepts arbitrary PostSelectPredicateAttr
        subclasses."""
        collection = _MockCollectionOp()
        predicate = self._DummyPredicateAttr()

        op = PostSelectOp(collection.res, predicate)
        op.verify()

        assert tuple(op.predicates) == (predicate,)

    def test_verification_fails_with_non_predicate_attribute(self):
        """Tests that verification fails when a non-predicate attribute is provided, raising
        a VerifyException with the expected error message."""
        collection = _MockCollectionOp()
        non_predicate_attr = IntAttr(42)

        op = PostSelectOp(collection.res, non_predicate_attr)

        with pytest.raises(
            VerifyException,
            match="should be of base attribute results.post_select_predicate",
        ):
            op.verify()


class TestGroupEntriesOp:
    """Tests the GroupEntriesOp, which groups entries in a results record."""

    def test_initialization_and_properties_with_record(self):
        """Tests that the GroupEntriesOp can be initialized with a collection and a key and
        that its properties return the expected values."""
        record = _MockRecordOp()

        op = GroupEntriesOp(record.res, ["a", "b"], "grouped")
        op.verify()

        assert op.collection is record.res
        assert op.keys == ArrayAttr([StringAttr("a"), StringAttr("b")])
        assert op.group_key == StringAttr("grouped")
        assert op.result.type == RecordType()

    def test_verification_fails_with_empty_keys(self):
        """Tests that verification fails when no keys are provided, raising a
        VerifyException with the expected error message."""
        record = _MockRecordOp()
        op = GroupEntriesOp(record.res, [], "grouped")

        with pytest.raises(VerifyException, match="requires at least one key to group"):
            op.verify()


class TestReduceOp:
    """Tests the ReduceOp, which reduces a results record to selected keys."""

    def test_initialization_and_properties_with_record(self):
        """Tests that the ReduceOp can be initialized with a record and that its properties
        return the expected values."""
        record = _MockRecordOp()

        op = ReduceOp(record.res, ["a", "b"])
        op.verify()

        assert op.collection is record.res
        assert op.keys == ArrayAttr([StringAttr("a"), StringAttr("b")])
        assert op.result.type == RecordType()

    def test_verification_fails_with_empty_keys(self):
        """Tests that verification fails when no keys are provided, raising a
        VerifyException with the expected error message."""
        record = _MockRecordOp()
        op = ReduceOp(record.res, [])

        with pytest.raises(VerifyException, match="requires at least one key to retain"):
            op.verify()


class TestExtractOp:
    """Tests the ExtractOp, in particular its ability to be parameterised with different
    results types."""

    def test_initialization_from_result(self):
        """Tests that the ExtractOp can be initialized with a record and a result type and
        that its properties return the expected values."""
        record_op = _MockRecordOp()
        result_type = i32
        op = ExtractOp(record_op.res, "key", i32)
        op.verify()

        assert op.result.type == result_type
        assert isinstance(op.key, StringAttr)
        assert op.key.data == "key"
        assert op.record is record_op.res

    def test_initialization_from_operation(self):
        """Tests that the ExtractOp can be initialized with an operation producing a record
        and a result type and that its properties return the expected values."""
        record_op = _MockRecordOp()
        result_type = i32
        op = ExtractOp(record_op, "key", i32)
        op.verify()

        assert op.result.type == result_type
        assert isinstance(op.key, StringAttr)
        assert op.key.data == "key"
        assert op.record is record_op.res

    def test_verification_fails_with_non_type_result_attribute(self):
        """Tests that verification fails when the result is not a type attribute."""
        record_op = _MockRecordOp()
        op = ExtractOp(record_op.res, "key", IntAttr(1))

        with pytest.raises(
            VerifyException, match="ExtractOp result must be a type attribute"
        ):
            op.verify()


class TestYieldOp:
    """Tests the YieldOp, which yields a results record from a region."""

    def test_initialization_and_properties(self):
        """Tests that the YieldOp can be initialized with a record and that its properties
        return the expected values."""
        record = _MockRecordOp()

        op = YieldOp(record.res)
        op.verify()

        assert op.record is record.res


class TestMapOp:
    """Tests the many ways to initialize the MapOp, and its verification."""

    def _make_block(self) -> Block:
        """Creates a block with extract, record creation, and yield operations."""

        block = Block(arg_types=(RecordType(),))
        block_arg = block.args[0]
        res = ExtractOp(block_arg, "key", i32)
        new_record = CreateRecordOp(["key"], [res.result])
        yield_op = YieldOp(new_record.result)
        block.add_ops([res, new_record, yield_op])
        return block

    def test_initialization_with_region_passes_verification(self):
        """Tests providing a region to the MapOp passes verification."""

        block = self._make_block()
        region = Region(blocks=[block])
        collection_op = _MockCollectionOp()
        map_op = MapOp(collection_op.res, region)
        map_op.verify()
        assert map_op.value is collection_op.res
        assert map_op.body is region
        assert map_op.result.type == ResultsCollectionType()

    def test_initialization_with_block_passes_verification(self):
        """Tests providing a block to the MapOp passes verification."""
        block = self._make_block()
        collection_op = _MockCollectionOp()
        map_op = MapOp(collection_op.res, block)
        map_op.verify()
        assert map_op.value is collection_op.res
        assert map_op.body.blocks[0] is block
        assert map_op.result.type == ResultsCollectionType()

    def test_initialization_with_sequence_of_blocks_passes_verification(self):
        """Tests providing a sequence of blocks to the MapOp passes verification."""
        block = self._make_block()
        collection_op = _MockCollectionOp()
        map_op = MapOp(collection_op.res, (block,))
        map_op.verify()
        assert map_op.value is collection_op.res
        assert map_op.body.blocks[0] is block
        assert map_op.result.type == ResultsCollectionType()

    def test_initialization_with_collection_op_passes_verification(self):
        """Tests providing a collection op to the MapOp passes verification."""
        block = self._make_block()
        collection_op = _MockCollectionOp()
        map_op = MapOp(collection_op, block)
        map_op.verify()
        assert map_op.value is collection_op.res
        assert map_op.body.blocks[0] is block
        assert map_op.result.type == ResultsCollectionType()

    def test_verification_with_no_block_arguments_fails_verification(self):
        """Tests that verification fails when the MapOp's block has no arguments, raising a
        VerifyException with the expected error message."""
        block = Block()
        create_record_op = _MockRecordOp()
        yield_op = YieldOp(create_record_op.res)
        block.add_ops([create_record_op, yield_op])
        collection_op = _MockCollectionOp()
        map_op = MapOp(collection_op.res, block)
        with pytest.raises(
            VerifyException, match="must have a single argument of type RecordType"
        ):
            map_op.verify()

    def test_verification_with_two_block_arguments_fails_verification(self):
        """Tests that verification fails when the MapOp's block has two arguments, raising a
        VerifyException with the expected error message."""
        block = Block(arg_types=(RecordType(), RecordType()))
        create_record_op = _MockRecordOp()
        yield_op = YieldOp(create_record_op.res)
        block.add_ops([create_record_op, yield_op])
        collection_op = _MockCollectionOp()
        map_op = MapOp(collection_op.res, block)
        with pytest.raises(
            VerifyException, match="must have a single argument of type RecordType"
        ):
            map_op.verify()

    def test_verification_with_non_record_argument_fails_verification(self):
        """Tests that verification fails when the MapOp's block has a non-RecordType
        argument, raising a VerifyException with the expected error message."""
        block = Block(arg_types=(i32,))
        create_record_op = _MockRecordOp()
        yield_op = YieldOp(create_record_op.res)
        block.add_ops([create_record_op, yield_op])
        collection_op = _MockCollectionOp()
        map_op = MapOp(collection_op.res, block)
        with pytest.raises(
            VerifyException, match="must have a single argument of type RecordType"
        ):
            map_op.verify()

    def test_verification_with_non_yield_last_op_fails_verification(self):
        """Tests that verification fails when the last operation in the MapOp's block is not
        a YieldOp, raising a VerifyException with the expected error message."""
        block = Block(arg_types=(RecordType(),))
        block.add_ops([_MockTerminatorOp()])
        collection_op = _MockCollectionOp()
        map_op = MapOp(collection_op.res, block)
        with pytest.raises(VerifyException, match="must be a YieldOp"):
            map_op.verify()

    def test_verification_with_empty_region_fails_verification(self):
        """Tests that verification fails when the MapOp body region is empty."""
        collection_op = _MockCollectionOp()
        map_op = MapOp(collection_op.res, Region())
        with pytest.raises(VerifyException, match="expected a single block"):
            map_op.verify()
