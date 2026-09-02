# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Models the operations in the results dialect, which are used to store and manipulate
collections of results."""

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    ArrayAttr,
    IndexType,
    IntAttr,
    IntegerType,
    StringAttr,
    TupleType,
)
from xdsl.ir import Block, Operation, Region, TypeAttribute
from xdsl.irdl import (
    AnyOf,
    AttrSizedOperandSegments,
    IRDLOperation,
    SSAValue,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import IsolatedFromAbove, IsTerminator, Pure
from xdsl.utils.exceptions import VerifyException

from .attributes import PostSelectPredicateAttr, RecordFieldAttr, RecordSchemaAttr
from .types import RecordType, ResultsArrayType, ResultsCollectionType


def _normalise_size_operand(
    size: Operation | SSAValue[IntegerType] | IntAttr | int,
) -> tuple[SSAValue[IntegerType] | None, IntAttr]:
    """Normalise static/dynamic size inputs into operand and attribute form."""

    if isinstance(size, int):
        return None, IntAttr(size)
    if isinstance(size, Operation | SSAValue):
        return SSAValue.get(size, type=IntegerType), IntAttr(DYNAMIC_INDEX)
    if isinstance(size, IntAttr):
        return None, size
    raise TypeError("Size must be an int, IntAttr, Operation, or SSAValue[IntegerType].")


@irdl_op_definition
class CreateOp(IRDLOperation):
    """Creates a value of a results type from provided values and an optional size.

    The semantics are determined by the result type:

    * :class:`RecordType`: Creates a record from ``values``. The values must match the
      field types encoded by the record schema in the result type. No ``size`` operand.
    * :class:`ResultsArrayType`:
        * Only empty arrays are supported. ``values`` must be empty.
        * If the result size is ``DYNAMIC_INDEX``, a ``size`` operand must be provided.
        * If the result size is static or unspecified, a ``size`` operand must not be
          provided.
    * :class:`ResultsCollectionType`:
        * With ``values`` (all of type :class:`ResultsArrayType`): creates a collection from
          arrays. Arrays must match the result schema field types and all have the same size
          as the collection result type. No ``size`` operand.
        * Without ``values``: creates an empty collection. If the result size is
          ``DYNAMIC_INDEX``, a ``size`` operand must be provided.
    * :class:`~xdsl.dialects.builtin.TupleType`: Creates a tuple from ``values``. No
      ``size`` operand.

    Use the factory class methods for ergonomic construction.

    :ivar size: Optional dynamic size operand used for dynamic-size empty arrays and
            collections.
    :ivar values: Variadic operands used to populate records, collections, and tuples.
    :ivar result: The created value. Must be one of RecordType, ResultsArrayType,
            ResultsCollectionType, or TupleType.
    """

    name = "results.create"
    irdl_options = (AttrSizedOperandSegments(),)

    size = opt_operand_def(IntegerType)
    values = var_operand_def()
    result = result_def(
        AnyOf((RecordType, ResultsArrayType, ResultsCollectionType, TupleType))
    )

    def __init__(
        self,
        result_type: TypeAttribute,
        values: Sequence[SSAValue | Operation] = (),
        size: Operation | SSAValue[IntegerType] | None = None,
    ):
        """Initialises CreateOp with a pre-built result type and operands.

        :param result_type: The result type to create. Use the factory class methods to
            build this automatically from the input values.
        :param values: Values to create from (fields for records, elements for arrays and
            tuples, or arrays for collections).
        :param size: Optional dynamic size operand for empty arrays and collections.
        """
        values_ssa = [SSAValue.get(v) for v in values]
        size_ssa = SSAValue.get(size, type=IntegerType) if size is not None else None
        return super().__init__(
            operands=[size_ssa, values_ssa],
            result_types=[result_type],
        )

    @classmethod
    def for_record(
        cls, keys: Sequence[str], values: Sequence[SSAValue | Operation]
    ) -> "CreateOp":
        """Create a :class:`RecordType` from keys and values.

        The result schema is derived from the keys and value types.

        :param keys: Keys for the record fields.
        :param values: Values for the record fields.
        """
        values_ssa = [SSAValue.get(v) for v in values]
        if len(keys) != len(values_ssa):
            raise ValueError(
                f"Number of keys ({len(keys)}) does not match number of values "
                f"({len(values_ssa)})."
            )
        fields = [
            RecordFieldAttr(key=k, type_=v.type)
            for k, v in zip(keys, values_ssa, strict=False)
        ]
        result_type = RecordType(RecordSchemaAttr(fields))
        return cls(result_type, values=values_ssa)

    @classmethod
    def for_array(
        cls,
        type_: TypeAttribute,
        size: Operation | SSAValue[IntegerType] | IntAttr | int,
    ) -> "CreateOp":
        """Create an empty :class:`ResultsArrayType`.

        :param type_: The element type of the array.
        :param size: The size of the array. Can be a static integer, an ``IntAttr``, or a
            dynamic SSA value.
        """
        size_op, size_attr = _normalise_size_operand(size)
        result_type = ResultsArrayType(type_, size_attr)
        return cls(result_type, size=size_op)

    @classmethod
    def for_collection_from_arrays(
        cls,
        keys: Sequence[str | StringAttr],
        arrays: Sequence[SSAValue[ResultsArrayType] | Operation],
    ) -> "CreateOp":
        """Create a :class:`ResultsCollectionType` from keyed arrays.

        :param keys: Keys for each array.
        :param arrays: The arrays to create the collection from. Must be non-empty, of
            equal size, and all of type :class:`ResultsArrayType`.
        """
        arrays_ssa = [SSAValue.get(a, type=ResultsArrayType) for a in arrays]
        if not arrays_ssa:
            raise ValueError(
                "CreateOp.for_collection_from_arrays requires at least one array."
            )
        if len(keys) != len(arrays_ssa):
            raise ValueError(
                f"Number of keys ({len(keys)}) does not match number of arrays "
                f"({len(arrays_ssa)})."
            )
        size = arrays_ssa[0].type.size
        for array in arrays_ssa[1:]:
            if array.type.size != size:
                raise ValueError(
                    "All arrays must have the same size to create a collection."
                )
        fields = [
            RecordFieldAttr(key=k, type_=a.type.type)
            for k, a in zip(keys, arrays_ssa, strict=False)
        ]
        result_type = ResultsCollectionType(RecordSchemaAttr(fields), size)
        return cls(result_type, values=arrays_ssa)

    @classmethod
    def for_empty_collection(
        cls,
        schema: RecordSchemaAttr,
        size: Operation | SSAValue[IntegerType] | IntAttr | int,
    ) -> "CreateOp":
        """Create an empty :class:`ResultsCollectionType`.

        :param schema: The schema of the collection.
        :param size: The size of the collection. Can be a static integer, an ``IntAttr``,
            or a dynamic SSA value.
        """
        size_op, size_attr = _normalise_size_operand(size)
        result_type = ResultsCollectionType(schema, size_attr)
        return cls(result_type, size=size_op)

    @classmethod
    def for_tuple(cls, values: Sequence[SSAValue | Operation]) -> "CreateOp":
        """Create a :class:`~xdsl.dialects.builtin.TupleType` from values.

        :param values: Values to populate the tuple with.
        """
        values_ssa = [SSAValue.get(v) for v in values]
        result_type = TupleType(tuple(v.type for v in values_ssa))
        return cls(result_type, values=values_ssa)

    def verify_(self):
        """Verifies the :class:`CreateOp` by dispatching the verification method based on
        the result of the operation."""

        result_type = self.result.type
        values = list(self.values)

        if isinstance(result_type, RecordType):
            self._verify_record(result_type, values)
        elif isinstance(result_type, ResultsArrayType):
            self._verify_array(result_type, values)
        elif isinstance(result_type, ResultsCollectionType) and values:
            self._verify_collection_from_arrays(result_type, values)
        elif isinstance(result_type, ResultsCollectionType):
            self._verify_empty_collection(result_type)
        elif isinstance(result_type, TupleType):
            self._verify_tuple(result_type, values)

    def _verify_record(self, result_type: RecordType, values: list):
        """Verify record creation values/types and disallow size operands."""

        if self.size is not None:
            raise VerifyException("Record creation does not use a size operand.")

        record_attributes = result_type.schema.fields.data
        if len(record_attributes) != len(values):
            raise VerifyException(
                f"Number of schema fields ({len(record_attributes)}) does not match "
                f"number of values ({len(values)})."
            )
        for i, (field, value) in enumerate(zip(record_attributes, values, strict=False)):
            if value.type != field.type:
                raise VerifyException(
                    f"Type of value at index {i} ({value.type}) does not match the "
                    f"expected type ({field.type}) for key '{field.key.data}' in the "
                    f"record schema."
                )

    def _verify_array(self, result_type: ResultsArrayType, values: list):
        """Verify that only empty-array creation is used and size semantics are valid."""

        if values:
            raise VerifyException("Array creation only supports empty arrays.")

        size_attr = result_type.size
        if isinstance(size_attr, IntAttr) and size_attr.data == DYNAMIC_INDEX:
            if self.size is None:
                raise VerifyException(
                    "Dynamic size array creation requires a size operand."
                )
        elif self.size is not None:
            raise VerifyException("Static size array creation does not use a size operand.")

    def _verify_collection_from_arrays(
        self, result_type: ResultsCollectionType, values: list
    ):
        """Verify collection creation from keyed result arrays."""

        if self.size is not None:
            raise VerifyException(
                "Collection creation from arrays does not use a size operand."
            )

        schema_fields = result_type.schema.fields.data
        if len(schema_fields) != len(values):
            raise VerifyException(
                f"Number of schema fields ({len(schema_fields)}) does not "
                f"match number of arrays ({len(values)})."
            )

        expected_size = result_type.size
        for i, (field, value) in enumerate(zip(schema_fields, values, strict=False)):
            if not isinstance(value.type, ResultsArrayType):
                raise VerifyException(
                    f"Value at index {i} must be of type ResultsArrayType, got "
                    f"{value.type}."
                )
            if value.type.size != expected_size:
                raise VerifyException(
                    "All arrays must have the same size to create a collection."
                )
            if value.type.type != field.type:
                raise VerifyException(
                    f"Type of array at index {i} ({value.type.type}) does not "
                    f"match the expected type ({field.type}) for key "
                    f"'{field.key.data}' in the collection schema."
                )

    def _verify_empty_collection(self, result_type: ResultsCollectionType):
        """Verify empty-collection creation size semantics."""

        size_attr = result_type.size
        if isinstance(size_attr, IntAttr) and size_attr.data == DYNAMIC_INDEX:
            if self.size is None:
                raise VerifyException(
                    "Dynamic size collection creation requires a size operand."
                )
        elif self.size is not None:
            raise VerifyException(
                "Static size collection creation does not use a size operand."
            )

    def _verify_tuple(self, result_type: TupleType, values: list) -> None:
        if self.size is not None:
            raise VerifyException("Tuple creation does not use a size operand.")
        expected_types = result_type.types.data
        if len(expected_types) != len(values):
            raise VerifyException(
                f"Number of expected tuple types ({len(expected_types)}) does not "
                f"match number of values ({len(values)})."
            )
        for i, (expected, value) in enumerate(zip(expected_types, values, strict=False)):
            if value.type != expected:
                raise VerifyException(
                    f"Type of value at index {i} ({value.type}) does not match the "
                    f"expected tuple element type ({expected})."
                )


@irdl_op_definition
class StoreOp(IRDLOperation):
    """Stores a value into a results container at a given index.

    The semantics are determined by the container and value types:

    * ``container`` is :class:`ResultsCollectionType`, ``value`` is :class:`RecordType`:
      stores the record at ``index``. The record schema must match the collection schema.
      No ``key`` property.
    * ``container`` is :class:`ResultsCollectionType`, ``value`` is any other type: stores
      the value at ``key`` and ``index``. A ``key`` property must be provided and must
      exist in the collection schema with the correct type.
    * ``container`` is :class:`ResultsArrayType`: stores the value at ``index``. The value
      type must match the array element type. No ``key`` property.

    :ivar container: The container operand to write into.
    :ivar index: The integer-or-index operand identifying the element/shot to overwrite.
    :ivar value: The value operand to write.
    :ivar key: Optional key selector used only for keyed collection value stores.
    :ivar result: The updated container value. Must match ``container`` type.
    """

    name = "results.store"

    container = operand_def(AnyOf((ResultsCollectionType, ResultsArrayType)))
    index = operand_def(AnyOf((IntegerType, IndexType)))
    value = operand_def()
    key = opt_prop_def(StringAttr)
    result = result_def()

    def __init__(
        self,
        container: SSAValue | Operation,
        index: SSAValue | Operation,
        value: SSAValue | Operation,
        key: str | StringAttr | None = None,
    ):
        """Initialises the StoreOp.

        :param container: The container to store into (collection or array).
        :param index: The index at which to store the value.
        :param value: The value to store.
        :param key: Optional key for storing a value by key in a collection.
        """
        container_ssa = SSAValue.get(container)
        index_ssa = SSAValue.get(index)
        if not isinstance(index_ssa.type, IntegerType | IndexType):
            raise TypeError("Index must be of type IntegerType or IndexType for StoreOp.")
        value_ssa = SSAValue.get(value)
        key_attr = StringAttr(key) if isinstance(key, str) else key
        return super().__init__(
            operands=[container_ssa, index_ssa, value_ssa],
            properties=({"key": key_attr} if key_attr is not None else {}),
            result_types=[container_ssa.type],
        )

    @classmethod
    def value_in_array(
        cls,
        array: SSAValue[ResultsArrayType] | Operation,
        index: SSAValue | Operation,
        value: SSAValue | Operation,
    ) -> "StoreOp":
        """Store a value into an array at ``index``.

        :param array: The results array to store into.
        :param index: The array index to write.
        :param value: The value to store.
        """
        return cls(array, index, value)

    @classmethod
    def record_in_collection(
        cls,
        collection: SSAValue[ResultsCollectionType] | Operation,
        index: SSAValue | Operation,
        record: SSAValue[RecordType] | Operation,
    ) -> "StoreOp":
        """Store a full record into a collection at ``index``.

        :param collection: The collection to store into.
        :param index: The shot index to write.
        :param record: The record to store.
        """
        return cls(collection, index, record)

    @classmethod
    def value_in_collection(
        cls,
        collection: SSAValue[ResultsCollectionType] | Operation,
        index: SSAValue | Operation,
        key: str | StringAttr,
        value: SSAValue | Operation,
    ) -> "StoreOp":
        """Store a field value into a collection at ``key`` and ``index``.

        :param collection: The collection to store into.
        :param index: The shot index to write.
        :param key: The field key in the collection schema.
        :param value: The value to store for ``key``.
        """
        return cls(collection, index, value, key=key)

    def verify_(self):
        """Verifies the StoreOp based on the container type."""
        container_type = self.container.type

        if isinstance(container_type, ResultsCollectionType):
            self._verify_collection_store(container_type)
        elif isinstance(container_type, ResultsArrayType):
            self._verify_array_store(container_type)

        if self.result.type != container_type:
            raise VerifyException(
                f"Result type ({self.result.type}) must match the container type "
                f"({container_type})."
            )

    def _verify_collection_store(self, container_type: ResultsCollectionType) -> None:
        if isinstance(self.value.type, RecordType):
            self._verify_collection_record_store(container_type)
            return
        self._verify_collection_value_store(container_type)

    def _verify_collection_record_store(
        self, container_type: ResultsCollectionType
    ) -> None:
        if self.key is not None:
            raise VerifyException(
                "Storing a record in a collection does not require a key."
            )
        collection_schema = container_type.schema
        record_schema = self.value.type.schema
        if collection_schema != record_schema:
            raise VerifyException(
                f"Schema of the record ({record_schema}) does not match schema of "
                f"the collection ({collection_schema})."
            )

    def _verify_collection_value_store(self, container_type: ResultsCollectionType) -> None:
        if self.key is None:
            raise VerifyException("Storing a value in a collection requires a key.")
        schema_dict = container_type.schema.as_dict()
        key = self.key.data
        if key not in schema_dict:
            raise VerifyException(
                f"Key '{key}' does not exist in the schema of the collection."
            )
        expected_type = schema_dict[key]
        if self.value.type != expected_type:
            raise VerifyException(
                f"Type of the value ({self.value.type}) does not match the expected "
                f"type ({expected_type}) for key '{key}' in the collection schema."
            )

    def _verify_array_store(self, container_type: ResultsArrayType) -> None:
        if self.key is not None:
            raise VerifyException("Storing a value in an array does not require a key.")
        expected_type = container_type.type
        if self.value.type != expected_type:
            raise VerifyException(
                f"Type of the value ({self.value.type}) does not match the expected "
                f"type ({expected_type}) for the results array."
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
        result_type = ResultsCollectionType(collection.type.schema, IntAttr(DYNAMIC_INDEX))
        return super().__init__(
            operands=[collection],
            result_types=[result_type],
            properties={"predicates": ArrayAttr(list(predicates))},
        )


@irdl_op_definition
class GroupEntriesOp(IRDLOperation):
    """Groups entries in a record into a single entry, producing a tuple of those entries,
    ordered by the given keys, with the tuple stored with a provided key in the record.

    The operation creates a record type that replaces the provided keys with a single key,
    and its corresponding type is a tuple of the grouped field types.

    This operation is roughly equivalent to the Assign instruction in legacy IR. This
    operation has a lowering path which could make use of :class:`ExtractOp`
    to extract the values from the record, and then make a new record with the
    grouped entries. But currently, this operation is more useful to the current
    runtime.

    .. note::

        This operation exists to support legacy runtime implementations of post-processing.
        Going forward, it is highly encouraged to assemble records in the structure that is
        desired to promote proper dataflow semantics.

    .. warning::

        This operation is likely to be flagged for deprecation in the future.

    :ivar record: The input record to regroup.
    :ivar keys: The set of keys to group into a tuple-valued entry.
    :ivar group_key: The key name assigned to the new grouped tuple entry.
    :ivar result: The transformed record type after grouping.
    """

    name = "results.group_entries"

    record = operand_def(RecordType)
    keys: ArrayAttr[StringAttr] = prop_def(ArrayAttr[StringAttr])
    group_key: StringAttr = prop_def(StringAttr)
    result = result_def(RecordType)

    def __init__(
        self,
        record: SSAValue[RecordType] | Operation,
        keys: Sequence[str],
        group_key: str,
    ):
        """Initializes the GroupEntriesOp with the given record, keys, and group key.

        :param record: The SSA value representing the existing results record.
        :param keys: A list of strings representing the keys to group in the new entry.
        :param group_key: A string representing the key for the new grouped entry.
        """
        record = SSAValue.get(record, type=RecordType)
        keys_attr = ArrayAttr([StringAttr(key) for key in keys])

        record_type = record.type
        schema = record_type.schema
        record_keys = [field.key.data for field in schema.fields.data]
        if not all(key in record_keys for key in keys):
            raise VerifyException(
                f"All keys to group must exist in the record schema. "
                f"Record schema keys: {record_keys}, keys to group: {keys}"
            )

        # Split into non-grouped keys & values and grouped keys & values
        schema_dict = schema.as_dict()
        non_grouped_keys = [key for key in record_keys if key not in keys]
        non_grouped_types = [schema_dict[key] for key in non_grouped_keys]
        grouped_types = tuple(schema_dict[key] for key in keys)
        grouped_tuple_type = TupleType(grouped_types)

        new_schema = RecordSchemaAttr(
            [
                *[
                    RecordFieldAttr(key=key, type_=typ)
                    for key, typ in zip(non_grouped_keys, non_grouped_types, strict=False)
                ],
                RecordFieldAttr(key=group_key, type_=grouped_tuple_type),
            ]
        )
        new_record_type = RecordType(new_schema)

        return super().__init__(
            operands=[record],
            result_types=[new_record_type],
            properties={"keys": keys_attr, "group_key": StringAttr(group_key)},
        )

    def verify_(self) -> None:
        if len(self.keys.data) == 0:
            raise VerifyException("GroupEntriesOp requires at least one key to group.")


@irdl_op_definition
class ReduceOp(IRDLOperation):
    """Reduces a record, down to a subset of the entries in the record, producing a new
    record.

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

    :ivar record: The input record to reduce.
    :ivar keys: The keys to retain in the reduced record.
    :ivar result: The reduced record type containing only retained keys.
    """

    name = "results.reduce"

    record = operand_def(RecordType)
    keys: ArrayAttr[StringAttr] = prop_def(ArrayAttr[StringAttr])
    result = result_def(RecordType)

    def __init__(
        self,
        record: SSAValue[RecordType] | Operation,
        keys: Sequence[str],
    ):
        """Initializes the ReduceOp with the given record and keys.

        :param record: The SSA value representing the existing results record.
        :param keys: A list of strings representing the keys to retain in the reduced
            record.
        """
        record = SSAValue.get(record, type=RecordType)
        keys_attr = ArrayAttr([StringAttr(key) for key in keys])

        record_type = record.type
        schema = record_type.schema
        record_keys = [field.key.data for field in schema.fields.data]
        if not all(key in record_keys for key in keys):
            raise VerifyException(
                f"All keys to retain must exist in the record schema. "
                f"Record schema keys: {record_keys}, keys to retain: {keys}"
            )

        # Create a new schema with only the retained keys and their types
        schema_dict = schema.as_dict()
        retained_types = [schema_dict[key] for key in keys]
        new_schema = RecordSchemaAttr(
            [
                RecordFieldAttr(key=key, type_=typ)
                for key, typ in zip(keys, retained_types, strict=False)
            ]
        )
        new_record_type = RecordType(new_schema)

        return super().__init__(
            operands=[record],
            result_types=[new_record_type],
            properties={"keys": keys_attr},
        )

    def verify_(self) -> None:
        if len(self.keys.data) == 0:
            raise VerifyException("ReduceOp requires at least one key to retain.")


@irdl_op_definition
class ExtractOp(IRDLOperation):
    """Extracts a value from a results container.

    The extraction semantics are determined by the container type and selectors:

    * A ``RecordType`` can be extracted from a ``ResultsCollectionType`` at a given
      ``index``.
    * A ``ResultsArrayType`` can be extracted from a ``ResultsCollectionType`` at a given
      ``key``.
    * A field value can be extracted from a ``ResultsCollectionType`` at a given ``key``
      and ``index``.
    * A field value can be extracted from a ``RecordType`` at a given ``key``.
    * An element can be extracted from a ``ResultsArrayType`` at a given ``index``.

    Prefer the factory methods for common extraction shapes.

    :ivar container: The container operand to extract from.
    :ivar index: Optional index selector used for array access and collection shot access.
    :ivar key: Optional key selector used for record/collection field selection.
    :ivar result: The extracted value whose type must match the selected extraction mode.
    """

    name = "results.extract"

    container = operand_def(AnyOf((RecordType, ResultsArrayType, ResultsCollectionType)))
    index = opt_operand_def(AnyOf((IntegerType, IndexType)))
    key = opt_prop_def(StringAttr)
    result = result_def()

    def __init__(
        self,
        container: SSAValue | Operation,
        result_type: TypeAttribute,
        key: str | StringAttr | None = None,
        index: SSAValue | Operation | None = None,
    ):
        """Initializes the ExtractOp with explicit selectors and result type.

        :param container: The container to extract from.
        :param result_type: Explicit result type for the extraction.
        :param key: Optional key selector.
        :param index: Optional index selector.
        """

        container_ssa = SSAValue.get(container)
        key_attr = StringAttr(key) if isinstance(key, str) else key
        index_ssa = SSAValue.get(index) if index is not None else None
        if index_ssa is not None and not isinstance(
            index_ssa.type, IntegerType | IndexType
        ):
            raise TypeError("Index must be of type IntegerType or IndexType for ExtractOp.")

        return super().__init__(
            operands=[container_ssa, index_ssa],
            result_types=[result_type],
            properties=({"key": key_attr} if key_attr is not None else {}),
        )

    @classmethod
    def value_from_record(
        cls,
        record: SSAValue[RecordType] | Operation,
        key: str | StringAttr,
    ) -> "ExtractOp":
        """Extract a field value from a record by key."""
        record_ssa = SSAValue.get(record, type=RecordType)
        key_attr = StringAttr(key) if isinstance(key, str) else key
        schema_dict = record_ssa.type.schema.as_dict()
        if key_attr.data not in schema_dict:
            raise ValueError(
                f"Key '{key_attr.data}' does not exist in the schema of the record."
            )
        return cls(record_ssa, schema_dict[key_attr.data], key=key_attr)

    @classmethod
    def value_from_array(
        cls,
        array: SSAValue[ResultsArrayType] | Operation,
        index: SSAValue | Operation,
    ) -> "ExtractOp":
        """Extract an element value from an array by index."""
        array_ssa = SSAValue.get(array, type=ResultsArrayType)
        return cls(array_ssa, array_ssa.type.type, index=index)

    @classmethod
    def record_from_collection(
        cls,
        collection: SSAValue[ResultsCollectionType] | Operation,
        index: SSAValue | Operation,
    ) -> "ExtractOp":
        """Extract a full record from a collection by index."""
        collection_ssa = SSAValue.get(collection, type=ResultsCollectionType)
        return cls(collection_ssa, RecordType(collection_ssa.type.schema), index=index)

    @classmethod
    def array_from_collection(
        cls,
        collection: SSAValue[ResultsCollectionType] | Operation,
        key: str | StringAttr,
    ) -> "ExtractOp":
        """Extract a full field array from a collection by key."""
        collection_ssa = SSAValue.get(collection, type=ResultsCollectionType)
        key_attr = StringAttr(key) if isinstance(key, str) else key
        schema_dict = collection_ssa.type.schema.as_dict()
        if key_attr.data not in schema_dict:
            raise ValueError(
                f"Key '{key_attr.data}' does not exist in the schema of the collection."
            )
        result_type = ResultsArrayType(schema_dict[key_attr.data], collection_ssa.type.size)
        return cls(collection_ssa, result_type, key=key_attr)

    @classmethod
    def value_from_collection(
        cls,
        collection: SSAValue[ResultsCollectionType] | Operation,
        key: str | StringAttr,
        index: SSAValue | Operation,
    ) -> "ExtractOp":
        """Extract a field value from a collection by key and index."""
        collection_ssa = SSAValue.get(collection, type=ResultsCollectionType)
        key_attr = StringAttr(key) if isinstance(key, str) else key
        schema_dict = collection_ssa.type.schema.as_dict()
        if key_attr.data not in schema_dict:
            raise ValueError(
                f"Key '{key_attr.data}' does not exist in the schema of the collection."
            )
        return cls(collection_ssa, schema_dict[key_attr.data], key=key_attr, index=index)

    def verify_(self):
        """Verifies selectors and result type against the container semantics."""

        container_type = self.container.type
        result_type = self.result.type
        if isinstance(container_type, RecordType):
            self._verify_extract_from_record(container_type)
        elif isinstance(container_type, ResultsArrayType):
            self._verify_extract_from_array(container_type)
        elif isinstance(container_type, ResultsCollectionType):
            if isinstance(result_type, RecordType):
                self._verify_record_from_collection(container_type)
            elif isinstance(result_type, ResultsArrayType):
                self._verify_array_from_collection(container_type)
            else:
                self._verify_value_from_collection(container_type)

    def _verify_extract_from_record(self, container_type: RecordType):
        """Verifies that the key is provided and exists in the record schema, and that no
        index is used."""

        key = self.key
        if key is None:
            raise VerifyException("Extracting from a record requires a key.")
        if self.index is not None:
            raise VerifyException("Extracting from a record does not use an index.")

        schema_dict = container_type.schema.as_dict()
        if key.data not in schema_dict:
            raise VerifyException(
                f"Key '{key.data}' does not exist in the schema of the record."
            )
        expected_type = schema_dict[key.data]
        if self.result.type != expected_type:
            raise VerifyException(
                f"Type of the extracted value ({self.result.type}) does not match the "
                f"expected type ({expected_type}) for this extract operation."
            )

    def _verify_extract_from_array(self, container_type: ResultsArrayType):
        """Verifies that no key is used, an index is provided, and the result type matches
        the array element type."""

        if self.key is not None:
            raise VerifyException("Extracting from an array does not use a key.")
        if self.index is None:
            raise VerifyException("Extracting from an array requires an index.")

        expected_type = container_type.type
        if self.result.type != expected_type:
            raise VerifyException(
                f"Type of the extracted value ({self.result.type}) does not match the "
                f"expected type ({expected_type}) for this extract operation."
            )

    def _verify_record_from_collection(self, container_type: ResultsCollectionType):
        """Verifies that no key is used, an index is provided, and the result type matches
        the record type of the collection."""

        if self.key is not None:
            raise VerifyException(
                "Extracting a record from a collection does not use a key."
            )
        if self.index is None:
            raise VerifyException(
                "Extracting a record from a collection requires an index."
            )

        expected_type = RecordType(container_type.schema)
        if self.result.type != expected_type:
            raise VerifyException(
                f"Type of the extracted value ({self.result.type}) does not match the "
                f"expected type ({expected_type}) for this extract operation."
            )

    def _verify_array_from_collection(self, container_type: ResultsCollectionType):
        """Verifies that a key is provided, no index is used, and the result type matches
        the array type of the collection for the given key."""

        key = self.key
        if key is None:
            raise VerifyException("Extracting an array from a collection requires a key.")
        if self.index is not None:
            raise VerifyException(
                "Extracting an array from a collection does not use an index."
            )

        schema_dict = container_type.schema.as_dict()
        if key.data not in schema_dict:
            raise VerifyException(
                f"Key '{key.data}' does not exist in the schema of the collection."
            )
        expected_type = ResultsArrayType(schema_dict[key.data], container_type.size)
        if self.result.type != expected_type:
            raise VerifyException(
                f"Type of the extracted value ({self.result.type}) does not match the "
                f"expected type ({expected_type}) for this extract operation."
            )

    def _verify_value_from_collection(self, container_type: ResultsCollectionType):
        """Verifies that both a key and an index are provided, and that the result type
        matches the type of the collection for the given key."""
        key = self.key
        index = self.index
        if key is None or index is None:
            raise VerifyException(
                "Extracting a value from a collection requires both key and index."
            )

        schema_dict = container_type.schema.as_dict()
        if key.data not in schema_dict:
            raise VerifyException(
                f"Key '{key.data}' does not exist in the schema of the collection."
            )
        expected_type = schema_dict[key.data]
        if self.result.type != expected_type:
            raise VerifyException(
                f"Type of the extracted value ({self.result.type}) does not match the "
                f"expected type ({expected_type}) for this extract operation."
            )


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
    in the pulse dialect) to transform the values, and then a :class:`CreateOp`
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
        results_collection_type: ResultsCollectionType,
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
            result_types=[results_collection_type],
            regions=[body],
        )

    def verify_(self):
        """Verifies that the region begins with a block that has a single argument of type
        RecordType, and that the region yields a value of type RecordType.

        It then validates that the yielded results type matches the expected results type of
        the MapOp.
        """

        # The region def enforces a single block
        first_block = self.body.blocks[0]
        if len(first_block.args) != 1 or not isinstance(
            first_block.args[0].type, RecordType
        ):
            raise VerifyException(
                "The block of the MapOp body must have a single argument of type "
                "RecordType."
            )

        input_record_type = first_block.args[0].type
        if input_record_type.schema != self.value.type.schema:
            raise VerifyException(
                "The schema of the record argument in the MapOp body must match the "
                "schema of the input collection."
            )

        last_op = first_block.ops.last if first_block.ops else None
        if not isinstance(last_op, YieldOp):
            raise VerifyException(
                "The last operation in the block of the MapOp body must be a YieldOp."
            )

        expected_record_type = RecordType(self.result.type.schema)
        if last_op.record.type != expected_record_type:
            raise VerifyException(
                "The type of the record yielded by the YieldOp must match the schema "
                "of the MapOp result type."
            )

        if self.result.type.size != self.value.type.size:
            raise VerifyException(
                "The size of the MapOp result type must match the size of the input "
                "collection."
            )
