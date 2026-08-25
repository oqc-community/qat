# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Models the types in the results dialect, which are used for dataflow of results to
collect and manipulate."""

from xdsl.dialects.builtin import DYNAMIC_INDEX, IntAttr
from xdsl.ir import ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition, param_def

from .attributes import RecordSchemaAttr, _TypeConstraint


@irdl_attr_definition
class RecordType(ParametrizedAttribute, TypeAttribute):
    """A type that carries a results record that can be added to a results collection.

    This is used to represent a single record of results, e.g., from a single shot. It has
    the semantics of a dictionary, where the entries are accessed by string keys, and it can
    store arbitrary data types.

    :class:`RecordType` is modelled to be immutable, and must be complete when constructed.

    :ivar schema: The schema of the record, which defines the keys and types of the entries
        in the record.
    """

    name = "results.record"
    schema: RecordSchemaAttr

    def __init__(self, schema: RecordSchemaAttr):
        """Initializes a record type with a schema."""
        super().__init__(schema)


@irdl_attr_definition
class ResultsArrayType(ParametrizedAttribute, TypeAttribute):
    """A type that represents an array of results, which can be added to and filtered with
    given operations.

    This is used to represent a collection of results that can be indexed into, e.g., a
    register of classical bits measured from a quantum circuit. It takes standard array
    semantics, holding an ordered list of results.

    :class:`ResultsArrayType` is modelled to be immutable, and must be complete when
    constructed.

    :ivar type: The type of the entries in the array.
    :ivar size: The size of the array. The special value ``DYNAMIC_INDEX`` indicates
        runtime-dynamic size.
    """

    name = "results.array"
    type: TypeAttribute = param_def(_TypeConstraint())
    size: IntAttr

    def __init__(
        self,
        type_: TypeAttribute,
        size: IntAttr,
    ):
        """Initializes an array type with element type and size."""
        super().__init__(type_, size)

    @classmethod
    def dynamic_size(cls, type_: TypeAttribute) -> "ResultsArrayType":
        """Construct a :class:`ResultsArrayType` with a dynamic size."""
        return cls(type_, IntAttr(DYNAMIC_INDEX))


@irdl_attr_definition
class ResultsCollectionType(ParametrizedAttribute, TypeAttribute):
    """A type that represents a collection of results, which can be added to and filtered
    with given operations.

    The collection is modelled as a collection of data which are referenced by two
    identifiers:

    * An integer index, which for example, might refer to the shot number.
    * A string key, which for example, might refer to the name of the result.

    The actual data type is not specified to be index-major or key-major. This allows us to
    support flexible results acquisition. For example, you could treat this as a dictionary
    of arrays, and append a result to each array for each shot. Or you could treat this as
    a list of records, and append a new record for each shot.

    :ivar schema: The schema of the records in the collection, which defines the keys and
        types.
    :ivar size: The size of the collection. The special value ``DYNAMIC_INDEX`` indicates
        runtime-dynamic size.
    """

    name = "results.collection"
    schema: RecordSchemaAttr
    size: IntAttr

    def __init__(
        self,
        schema: RecordSchemaAttr,
        size: IntAttr,
    ):
        """Initializes a collection type with schema and size."""
        super().__init__(schema, size)

    @classmethod
    def dynamic_size(cls, schema: RecordSchemaAttr) -> "ResultsCollectionType":
        """Construct a :class:`ResultsCollectionType` with a dynamic size."""
        return cls(schema, IntAttr(DYNAMIC_INDEX))
