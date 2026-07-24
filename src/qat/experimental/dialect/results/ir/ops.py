# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Models the operations in the results dialect, which are used to store and manipulate
collections of results."""

from collections.abc import Sequence

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import Attribute, Block, Operation, Region, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    SSAValue,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import IsolatedFromAbove, IsTerminator, Pure
from xdsl.utils.exceptions import VerifyException

from .attributes import PostSelectPredicateAttr
from .types import RecordType, ResultsArrayType, ResultsCollectionType


@irdl_op_definition
class CreateRecordOp(IRDLOperation):
    """Creates a results record, which can be added to a results collection.

    The operation contains a sequence of keys and values, which are used to construct a
    results record. The keys must be unique, and each key must have a corresponding value.

    Since a record takes dictionary semantics, the keys and values are variable-length.

    :ivar keys: An array of string attributes representing the keys for the record.
    :ivar values: A sequence of SSA values representing the values for the record.
    :ivar result: The resulting record type, which is a RecordType.
    """

    name = "results.create_record"

    keys: ArrayAttr[StringAttr] = prop_def(ArrayAttr[StringAttr])
    values = var_operand_def()
    result = result_def(RecordType)

    def __init__(self, keys: Sequence[str], values: Sequence[SSAValue]):
        """Initializes the CreateRecordOp with the given keys and values.

        :param keys: A sequence of strings representing the keys for the record.
        :param values: A sequence of SSA values representing the values for the record.
        """
        keys_attr = ArrayAttr([StringAttr(key) for key in keys])
        values = list(values)
        return super().__init__(
            properties={"keys": keys_attr},
            operands=[values],
            result_types=[RecordType()],
        )

    def verify_(self) -> None:
        """Verifies that the number of keys and values match."""
        if len(self.keys.data) != len(self.values):
            raise VerifyException(
                f"Number of keys ({len(self.keys.data)}) does not match number of values "
                f"({len(self.values)})."
            )

        key_names = [key.data for key in self.keys.data]
        seen_keys: set[str] = set()
        duplicates: list[str] = []
        for key in key_names:
            if key in seen_keys and key not in duplicates:
                duplicates.append(key)
            seen_keys.add(key)

        if duplicates:
            raise VerifyException(
                f"Keys must be unique. Duplicates: {', '.join(duplicates)}"
            )


@irdl_op_definition
class CreateResultsArrayOp(IRDLOperation):
    """Creates a results array, which can be added to a results collection.

    The operation contains a sequence of values, which are used to construct a results
    array. The values can be of any type.

    :ivar values: A sequence of SSA values representing the values for the array.
    :ivar result: The resulting array type, which is a ResultsArrayType.
    """

    name = "results.create_array"

    values = var_operand_def()
    result = result_def(ResultsArrayType)

    def __init__(self, values: Sequence[SSAValue]):
        """Initializes the CreateResultsArrayOp with the given values.

        :param values: A sequence of SSA values representing the values for the array.
        """
        values = list(values)
        return super().__init__(operands=[values], result_types=[ResultsArrayType()])


@irdl_op_definition
class CreateResultsCollectionOp(IRDLOperation):
    """Creates a results collection, which can be added to and filtered with given
    operations.

    The operation creates an empty results collection that can be populated by later
    operations such as :class:`AddRecordOp`.

    :ivar result: The resulting empty collection, which is a ResultsCollectionType.
    """

    name = "results.create_collection"

    result = result_def(ResultsCollectionType)

    def __init__(self):
        """Initializes the CreateResultsCollectionOp with no values, as the collection is
        empty upon creation."""
        return super().__init__(result_types=[ResultsCollectionType()])


@irdl_op_definition
class AddRecordOp(IRDLOperation):
    """Adds a results record to a results collection.

    The operation takes a results collection and a results record as operands, and produces
    a new results collection that includes the added record.

    :ivar collection: The SSA value representing the existing results collection.
    :ivar record: The SSA value representing the results record to be added.
    :ivar result: The resulting collection type, which is a ResultsCollectionType.
    """

    name = "results.add_record"

    collection = operand_def(ResultsCollectionType)
    record = operand_def(RecordType)
    result = result_def(ResultsCollectionType)

    def __init__(
        self,
        collection: SSAValue[ResultsCollectionType] | Operation,
        record: SSAValue[RecordType] | Operation,
    ):
        """Initializes the AddRecordOp with the given collection and record.

        :param collection: The SSA value representing the existing results collection.
        :param record: The SSA value representing the results record to be added.
        """
        collection = SSAValue.get(collection, type=ResultsCollectionType)
        record = SSAValue.get(record, type=RecordType)
        return super().__init__(
            operands=[collection, record],
            result_types=[ResultsCollectionType()],
        )


@irdl_op_definition
class PostSelectOp(IRDLOperation):
    """Filters a results collection based on a given predicate, producing a new collection
    that only includes records satisfying the predicate.

    The operation takes a results collection operand and a predicates property, and produces
    a new results collection that includes only the records that satisfy all configured
    predicates.

    This is used to filter a results collection to records that satisfy the predicate. This
    is modelled around legacy runtime implementations of post-selection, which post-selects
    on an entire collection of records, and filters them down. Lowering paths could be
    implemented to allow for an on-the-fly implementation of post-selection making use of
    classical control flow, given hardware compatibility.

    :ivar collection: The operand representing the existing results collection.
    :ivar predicates: The predicate attributes used to filter records in the collection.
    :ivar result: The resulting collection type, which is a ResultsCollectionType.
    """

    # TODO: Add a canonicalization hook that merges consecutive PostSelectOps into a single
    # operation with each of the predicates. COMPILER-1375

    name = "results.post_select"

    collection = operand_def(ResultsCollectionType)
    predicates: ArrayAttr[PostSelectPredicateAttr] = prop_def(
        ArrayAttr[PostSelectPredicateAttr]
    )
    result = result_def(ResultsCollectionType)

    def __init__(
        self,
        collection: SSAValue[ResultsCollectionType] | Operation,
        *predicates: PostSelectPredicateAttr,
    ):
        """Initializes the PostSelectOp with the given collection and predicates.

        :param collection: The SSA value representing the existing results collection.
        :param predicates: The array of predicate attributes to filter records.
        """
        collection = SSAValue.get(collection, type=ResultsCollectionType)
        return super().__init__(
            operands=[collection],
            result_types=[ResultsCollectionType()],
            properties={"predicates": ArrayAttr(list(predicates))},
        )


@irdl_op_definition
class GroupEntriesOp(IRDLOperation):
    """Groups entries in a record into a single entry, producing an array of those entries,
    ordered by the given keys, with the array stored with a provided key in the record.

    This operation is roughly equivalent to the Assign instruction in legacy IR. This
    operation has a lowering path which could make use of :class:`ExtractOp` to extract the
    values from the record, and then a make a new record with the grouped entries. But
    currently, this operation is more useful to the current runtime.

    .. note::

        This operation exists to support legacy runtime implementations of post-processing.
        Going forward, it is highly encouraged to assemble records in the structure that is
        desired to promote proper dataflow semantics.

    .. warning::

        This operation is likely to be flagged for deprecation in the future.
    """

    name = "results.group_entries"

    collection = operand_def(RecordType)
    keys: ArrayAttr[StringAttr] = prop_def(ArrayAttr[StringAttr])
    group_key: StringAttr = prop_def(StringAttr)
    result = result_def(RecordType)

    def __init__(
        self,
        collection: SSAValue[RecordType] | Operation,
        keys: Sequence[str],
        group_key: str,
    ):
        """Initializes the GroupEntriesOp with the given collection, keys, and group key.

        :param collection: The SSA value representing the existing results record.
        :param keys: A list of strings representing the keys to group in the new entry.
        :param group_key: A string representing the key for the new grouped entry.
        """
        collection = SSAValue.get(collection, type=RecordType)
        keys_attr = ArrayAttr([StringAttr(key) for key in keys])
        return super().__init__(
            operands=[collection],
            result_types=[RecordType()],
            properties={"keys": keys_attr, "group_key": StringAttr(group_key)},
        )

    def verify_(self) -> None:
        if len(self.keys.data) == 0:
            raise VerifyException("GroupEntriesOp requires at least one key to group.")


@irdl_op_definition
class ReduceOp(IRDLOperation):
    """Reduces a record, down to a subset of the entries in the record, producing a new
    record or collection of records.

    We often want to gather a number of measurements to use in post-processing for use cases
    such as post-selection, and more general error mitigation methods. But not each of these
    measurements are the measurements that are requested in the original circuit. After the
    post-processing has completed and we have no need for these measurements, we can reduce
    records down to only the entries we would like to return.

    This is a high-level operation that is roughly equivalent to the Return instruction in
    legacy IR. This has a lowering path to extract the entries from the record, and then
    create a new record with only the entries that are requested. But currently, this
    operation is more useful to the current runtime.


    .. note::

        This operation exists to support legacy runtime implementations of post-processing.
        Going forward, it is highly encouraged to assemble records in the structure that is
        desired to promote proper dataflow semantics.

    .. warning::

        This operation is likely to be flagged for deprecation in the future.
    """

    name = "results.reduce"

    collection = operand_def(RecordType)
    keys: ArrayAttr[StringAttr] = prop_def(ArrayAttr[StringAttr])
    result = result_def(RecordType)

    def __init__(
        self,
        collection: SSAValue[RecordType] | Operation,
        keys: Sequence[str],
    ):
        """Initializes the ReduceOp with the given collection and keys.

        :param collection: The SSA value representing the existing results record.
        :param keys: A list of strings representing the keys to retain in the reduced
            records.
        """
        collection = SSAValue.get(collection, type=RecordType)
        keys_attr = ArrayAttr([StringAttr(key) for key in keys])
        return super().__init__(
            operands=[collection],
            result_types=[RecordType()],
            properties={"keys": keys_attr},
        )

    def verify_(self) -> None:
        if len(self.keys.data) == 0:
            raise VerifyException("ReduceOp requires at least one key to retain.")


@irdl_op_definition
class ExtractOp(IRDLOperation):
    """Extracts an entry from a record from a string key.

    As part of building the operation, the type of the extracted object must be specified.
    """

    name = "results.extract"

    record = operand_def(RecordType)
    key = prop_def(StringAttr)
    result = result_def()

    def __init__(
        self, record: SSAValue[RecordType] | Operation, key: str, result_type: Attribute
    ):
        """Initializes the ExtractOp with the given record, key, and result type.

        :param record: The SSA value representing the existing results record.
        :param key: A string representing the key to extract from the record.
        :param result_type: The type of the extracted value, for example a built-in scalar
            type, RecordType, or ResultsArrayType.
        """
        record = SSAValue.get(record, type=RecordType)
        return super().__init__(
            operands=[record],
            result_types=[result_type],
            properties={"key": StringAttr(key)},
        )

    def verify_(self) -> None:
        if not isinstance(self.result.type, TypeAttribute):
            raise VerifyException("ExtractOp result must be a type attribute.")


@irdl_op_definition
class YieldOp(IRDLOperation):
    """Yields a record from a region, which can be used to produce a new collection of
    records.

    This operation is used to yield a record from a region, which can be used to produce a
    new collection of records. The yielded record must be of type RecordType.

    :ivar record: The SSA value representing the record to be yielded.
    """

    name = "results.yield"
    traits = traits_def(IsTerminator())

    record = operand_def(RecordType)

    def __init__(self, record: SSAValue[RecordType] | Operation):
        """Initializes the YieldOp with the given record.

        :param record: The SSA value representing the record to be yielded.
        """
        record = SSAValue.get(record, type=RecordType)
        return super().__init__(operands=[record])


@irdl_op_definition
class MapOp(IRDLOperation):
    """Maps a transformation over a record to a collection of records, producing a new
    collection of records.

    The operation contains a single region, which contains the operations that transform
    a record into a new record. The region has a block argument of type :class:`RecordType`,
    which represents the input record, and must yield a value of type :class:`RecordType`.
    The operation takes a :class:`ResultsCollectionType` operand, which represents the
    collection of records to be transformed, and produces a new
    :class:`ResultsCollectionType` result. The implication is that the transformation is
    applied to each record in the collection.

    The operation is modelled as pure to not allow for any side effects to be introduced
    within the region, and is enforced to be isolated from above to ensure that the region
    does not have access to any values outside of the region, which could introduce side
    effects.

    This is intended to allow for a granular post-processing chain to be implemented that
    acts locally to a record. It intentionally does not specify any details of how this is
    implemented, e.g., in parallel or sequentially, apply every operation to each record
    before moving onto the next or going operation-by-operation (on every record). Those
    details are left to the runtime implementation, or lowering if relevant.

    Within this block, you might expect to see operations such as :class:`ExtractOp` to
    extract values from the record, and then post-processing chains (such as those defined
    in the pulse dialect) to transform the values, and then a :class:`CreateRecordOp` to
    create a new record. You might also expect to see operations such as
    :class:`GroupEntriesOp` to group entries in the record, or a :class:`ReduceOp` to filter
    out entries.

    :ivar value: The SSA value representing the existing results collection.
    :ivar body: The region containing the operations that transform a record into a new
        record.
    :ivar result: The resulting collection type, which is a ResultsCollectionType.
    """

    name = "results.map"
    traits = traits_def(Pure(), IsolatedFromAbove())

    value = operand_def(ResultsCollectionType)
    result = result_def(ResultsCollectionType)
    body = region_def("single_block")

    def __init__(
        self,
        value: SSAValue[ResultsCollectionType] | Operation,
        body: Block | Region | Sequence[Block],
    ):
        """Initializes the MapOp with the given collection and body.

        :param value: The SSA value representing the existing results collection.
        :param body: The region or block(s) containing the operations that transform a
            record into a new record.
        """
        body = [body] if isinstance(body, Block) else body
        value = SSAValue.get(value, type=ResultsCollectionType)
        return super().__init__(
            operands=[value],
            result_types=[ResultsCollectionType()],
            regions=[body],
        )

    def verify_(self):
        """Verifies that the region begins with a block that has a single argument of type
        RecordType, and that the region yields a value of type RecordType."""

        # The region def enforces a single block
        first_block = self.body.blocks[0]
        if len(first_block.args) != 1 or not isinstance(
            first_block.args[0].type, RecordType
        ):
            raise VerifyException(
                "The block of the MapOp body must have a single argument of type "
                "RecordType."
            )

        last_op = first_block.ops.last if first_block.ops else None
        if not isinstance(last_op, YieldOp):
            raise VerifyException(
                "The last operation in the block of the MapOp body must be a YieldOp."
            )
