# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the attributes in the results dialect."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, IntegerType, StringAttr
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.results.ir import (
    IntegerStatePredicateAttr,
    RecordFieldAttr,
    RecordSchemaAttr,
)


class TestRecordFieldAttr:
    """Tests the RecordFieldAttr, which models a field in a record in the results
    dialect."""

    def test_initialization_from_string(self):
        """Tests that the RecordFieldAttr can be initialized with a string key and a
        type."""
        attr = RecordFieldAttr("field1", IntegerType(1))

        assert attr.key == StringAttr("field1")
        assert attr.type == IntegerType(1)

    def test_initialization_from_string_attr(self):
        """Tests that the RecordFieldAttr can be initialized with a StringAttr key."""
        attr = RecordFieldAttr(StringAttr("field2"), IntegerType(8))

        assert attr.key == StringAttr("field2")
        assert attr.type == IntegerType(8)

    def test_initialization_with_non_type_attribute_raises(self):
        """Tests that initializing the RecordFieldAttr with a non-TypeAttribute raises a
        VerifyException."""
        with pytest.raises(VerifyException, match="should be of base attribute"):
            RecordFieldAttr("field3", IntAttr(32))


class TestRecordSchemaAttr:
    """Tests the RecordSchemaAttr, which models the schema of a record in the results
    dialect."""

    def test_initialization_and_properties(self):
        """Tests that the RecordSchemaAttr can be initialized with a list of RecordFieldAttr
        and that its properties return the expected values."""
        field1 = RecordFieldAttr("field1", IntegerType(1))
        field2 = RecordFieldAttr("field2", IntegerType(8))
        schema = RecordSchemaAttr([field1, field2])

        assert isinstance(schema.fields, ArrayAttr)
        assert schema.fields == ArrayAttr([field1, field2])

    def test_verify_fails_when_duplicate_keys_exist(self):
        """Tests that verification fails when there are duplicate keys in the schema."""
        field1 = RecordFieldAttr("field1", IntegerType(1))
        field2 = RecordFieldAttr("field1", IntegerType(8))  # Duplicate key

        with pytest.raises(
            VerifyException,
            match=r"Duplicate key\(s\) found in record schema: field1",
        ):
            RecordSchemaAttr([field1, field2])

    def test_verify_duplicate_key_error_is_sorted(self):
        """Tests duplicate-key errors are deterministic and sorted by key."""
        fields = [
            RecordFieldAttr("b", IntegerType(1)),
            RecordFieldAttr("a", IntegerType(1)),
            RecordFieldAttr("b", IntegerType(8)),
            RecordFieldAttr("a", IntegerType(8)),
        ]

        with pytest.raises(
            VerifyException,
            match=r"Duplicate key\(s\) found in record schema: a, b",
        ):
            RecordSchemaAttr(fields)

    def test_as_dict_returns_expected_mapping(self):
        """Tests that the as_dict method returns a dictionary mapping keys to types."""
        field1 = RecordFieldAttr("field1", IntegerType(1))
        field2 = RecordFieldAttr("field2", IntegerType(8))
        schema = RecordSchemaAttr([field1, field2])

        expected_dict = {
            "field1": IntegerType(1),
            "field2": IntegerType(8),
        }
        assert schema.as_dict() == expected_dict

    def test_equality_is_true_for_identical_schemas(self):
        """Tests that two RecordSchemaAttr instances with identical fields are considered
        equal."""
        field1 = RecordFieldAttr("field1", IntegerType(1))
        field2 = RecordFieldAttr("field2", IntegerType(8))
        field3 = RecordFieldAttr("field1", IntegerType(1))
        field4 = RecordFieldAttr("field2", IntegerType(8))
        schema1 = RecordSchemaAttr([field1, field2])
        schema2 = RecordSchemaAttr([field3, field4])

        assert schema1 == schema2

    def test_equality_is_true_for_different_order_of_fields(self):
        """Tests that two RecordSchemaAttr instances with the same fields in different
        orders are considered equal."""
        field1 = RecordFieldAttr("field1", IntegerType(1))
        field2 = RecordFieldAttr("field2", IntegerType(8))
        schema1 = RecordSchemaAttr([field1, field2])
        schema2 = RecordSchemaAttr([field2, field1])

        assert schema1 == schema2

    def test_equality_is_false_for_different_schemas(self):
        """Tests that two RecordSchemaAttr instances with different fields are not
        considered equal."""
        field1 = RecordFieldAttr("field1", IntegerType(1))
        field2 = RecordFieldAttr("field2", IntegerType(8))
        field3 = RecordFieldAttr("field3", IntegerType(2))
        schema1 = RecordSchemaAttr([field1, field2])
        schema2 = RecordSchemaAttr([field1, field3])

        assert schema1 != schema2

    def test_equality_is_false_for_non_schema_objects(self):
        """Tests that a RecordSchemaAttr instance is not considered equal to a non-schema
        object."""
        field1 = RecordFieldAttr("field1", IntegerType(1))
        field2 = RecordFieldAttr("field2", IntegerType(8))
        schema = RecordSchemaAttr([field1, field2])

        assert schema != "not-a-schema"


class TestIntegerStatePredicateAttr:
    """Tests the IntegerStatePredicateAttr, which models a predicate for post-selecting
    results based on an integer state."""

    def test_initialization_and_properties(self):
        """Tests that the IntegerStatePredicateAttr can be initialized with a key and a list
        of disallowed values, and that its properties return the expected values."""
        attr = IntegerStatePredicateAttr("state", [0, 1, 2])

        assert attr.key == StringAttr("state")
        assert isinstance(attr.disallowed_values, ArrayAttr)
        assert attr.disallowed_values == ArrayAttr([IntAttr(0), IntAttr(1), IntAttr(2)])
        attr.verify()

    def test_verify_fails_when_disallowed_values_are_not_all_int_attrs(self):
        """Tests that verification fails when the disallowed values are not all IntAttr."""
        with pytest.raises(
            VerifyException,
            match="should be of base attribute",
        ):
            IntegerStatePredicateAttr(
                "state",
                ArrayAttr([IntAttr(0), StringAttr("not-an-int")]),
            )
