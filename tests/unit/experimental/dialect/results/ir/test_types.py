# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the types that belong to the results dialect."""

import pytest
from xdsl.dialects.builtin import DYNAMIC_INDEX, IntAttr, IntegerType
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.results.ir import (
    RecordFieldAttr,
    RecordType,
    ResultsArrayType,
    ResultsCollectionType,
)
from qat.experimental.dialect.results.ir.attributes import RecordSchemaAttr


class TestRecordType:
    """Tests the RecordType, which models a record in the results dialect."""

    def test_initialization_and_properties_with_schema(self):
        """Tests that the RecordType can be initialized with a schema and that its
        properties return the expected values."""
        schema = RecordSchemaAttr(
            [
                RecordFieldAttr("field1", IntegerType(1)),
                RecordFieldAttr("field2", IntegerType(8)),
            ]
        )
        record_type = RecordType(schema)

        assert isinstance(record_type.schema, RecordSchemaAttr)
        assert record_type.schema == schema


class TestResultsArrayType:
    """Tests the ResultsArrayType, which models an array of results in the results
    dialect."""

    def test_initialization_and_properties_with_type_and_size(self):
        """Tests that the ResultsArrayType can be initialized with a type and size and that
        its properties return the expected values."""
        array_type = ResultsArrayType(IntegerType(1), IntAttr(10))

        assert isinstance(array_type.type, IntegerType)
        assert array_type.type == IntegerType(1)
        assert isinstance(array_type.size, IntAttr)
        assert array_type.size == IntAttr(10)

    def test_initialization_with_none_size_raises_type_error(self):
        """Tests that the ResultsArrayType rejects None size."""
        with pytest.raises(VerifyException, match="base attribute builtin.int"):
            ResultsArrayType(IntegerType(1), None)

    def test_dynamic_size_gives_correct_attribute(self):
        """Tests that the dynamic_size class method returns a ResultsArrayType with the
        correct type and size attributes."""
        array_type = ResultsArrayType.dynamic_size(IntegerType(1))
        assert isinstance(array_type.type, IntegerType)
        assert isinstance(array_type.size, IntAttr)
        assert array_type.size.data == DYNAMIC_INDEX

    def test_dynamic_size_gives_dynamic_index(self):
        """Tests that dynamic_size sets DYNAMIC_INDEX explicitly."""
        array_type = ResultsArrayType.dynamic_size(IntegerType(1))
        assert isinstance(array_type.type, IntegerType)
        assert array_type.size == IntAttr(DYNAMIC_INDEX)


class TestResultsCollectionType:
    """Tests the ResultsCollectionType, which models a collection of results in the results
    dialect."""

    def test_initialization_and_properties_with_schema_and_size(self):
        """Tests that the ResultsCollectionType can be initialized with a schema and size
        and that its properties return the expected values."""
        schema = RecordSchemaAttr(
            [
                RecordFieldAttr("field1", IntegerType(1)),
                RecordFieldAttr("field2", IntegerType(8)),
            ]
        )
        collection_type = ResultsCollectionType(schema, IntAttr(10))

        assert isinstance(collection_type.schema, RecordSchemaAttr)
        assert collection_type.schema == schema
        assert isinstance(collection_type.size, IntAttr)
        assert collection_type.size == IntAttr(10)

    def test_initialization_with_none_size_raises_type_error(self):
        """Tests that the ResultsCollectionType rejects None size."""
        schema = RecordSchemaAttr([RecordFieldAttr("field1", IntegerType(1))])
        with pytest.raises(VerifyException, match="base attribute builtin.int"):
            ResultsCollectionType(schema, None)

    def test_dynamic_size_gives_correct_attribute(self):
        """Tests that the dynamic_size class method returns a ResultsCollectionType with the
        correct schema and size attributes."""
        schema = RecordSchemaAttr([RecordFieldAttr("field1", IntegerType(1))])
        collection_type = ResultsCollectionType.dynamic_size(schema)
        assert isinstance(collection_type.schema, RecordSchemaAttr)
        assert isinstance(collection_type.size, IntAttr)
        assert collection_type.size.data == DYNAMIC_INDEX

    def test_dynamic_size_gives_dynamic_index(self):
        """Tests that dynamic_size sets DYNAMIC_INDEX explicitly."""
        schema = RecordSchemaAttr([RecordFieldAttr("field1", IntegerType(1))])
        collection_type = ResultsCollectionType.dynamic_size(schema)
        assert isinstance(collection_type.schema, RecordSchemaAttr)
        assert collection_type.size == IntAttr(DYNAMIC_INDEX)
