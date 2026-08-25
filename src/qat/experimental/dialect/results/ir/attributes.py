# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Models the attributes in the results dialect.

This includes attributes that model post-selection predicates.
"""

from abc import ABC
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.ir import Attribute, TypeAttribute
from xdsl.irdl import (
    AttrConstraint,
    ConstraintContext,
    ParametrizedAttribute,
    irdl_attr_definition,
    param_def,
)
from xdsl.utils.exceptions import VerifyException


@dataclass(frozen=True)
class _TypeConstraint(AttrConstraint):
    """Constraint that accepts any TypeAttribute.

    This is required because ``TypeAttribute`` is abstract and has no ``.name`` attribute,
    which prevents using ``BaseAttr(TypeAttribute)`` directly. This acts as a workaround.
    """

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, TypeAttribute):
            raise VerifyException(f"{attr} should be of base attribute TypeAttribute")

    def mapping_type_vars(self, type_var_mapping: object) -> "_TypeConstraint":
        return self


@irdl_attr_definition
class RecordFieldAttr(ParametrizedAttribute):
    """Models a field in a record in the results dialect, to be used by records and
    collections of records.

    Contains a key and a type.

    :ivar key: The key of the entry in the record.
    :ivar type: The type of the entry in the record.
    """

    name = "results.record_field"
    key: StringAttr
    type: Attribute = param_def(_TypeConstraint())

    def __init__(self, key: str | StringAttr, type_: TypeAttribute):
        """Initializes the RecordFieldAttr with the given key and type.

        :param key: The key of the entry in the record.
        :param type_: The type of the entry in the record.
        """
        key_attr = StringAttr(key) if isinstance(key, str) else key
        return super().__init__(key_attr, type_)


@irdl_attr_definition
class RecordSchemaAttr(ParametrizedAttribute):
    """Models the schema of a record in the results dialect, to be used by records and
    collections of records.

    :ivar fields: An array of record field attributes, which individually define the key and
        type of each field in the record.
    """

    name = "results.record_schema"
    fields: ArrayAttr[RecordFieldAttr]

    def __init__(self, fields: ArrayAttr[RecordFieldAttr] | Sequence[RecordFieldAttr]):
        """Initializes the RecordSchemaAttr with the given fields.

        :param fields: An array of record field attributes, which individually define the
            key and type of each field in the record.
        """
        fields_attr = fields if isinstance(fields, ArrayAttr) else ArrayAttr(fields)
        return super().__init__(fields_attr)

    def verify(self):
        """Verify that there are no duplicate keys in the schema."""

        keys = [f.key.data for f in self.fields.data]
        if len(keys) != len(set(keys)):
            duplicate_keys = sorted({k for k in keys if keys.count(k) > 1})
            raise VerifyException(
                f"Duplicate key(s) found in record schema: {', '.join(duplicate_keys)}"
            )

    def as_dict(self) -> dict[str, TypeAttribute]:
        """Return the schema as a dictionary mapping keys to types."""
        return {f.key.data: f.type for f in self.fields.data}

    def __eq__(self, other: object) -> bool:
        """Check equality with another RecordSchemaAttr without requiring the order to be
        the same."""
        if not isinstance(other, RecordSchemaAttr):
            return False
        return self.as_dict() == other.as_dict()


class PostSelectPredicateAttr(ParametrizedAttribute, ABC):
    """Models a predicate for post selecting results.

    Eventually, this class could be extended to provide a lowering hook.
    """

    name = "results.post_select_predicate"


@irdl_attr_definition
class IntegerStatePredicateAttr(PostSelectPredicateAttr):
    """Models a predicate for post selecting results based on an integer state.

    This attribute is used to filter results based on a specific integer state value. It is
    described by a key and a list of disallowed integer values. The key refers to the entry
    in a record.

    :ivar key: The key of the entry in the record that post-selection is performed on.
    :ivar disallowed_values: The list of values that are disallowed and will result in the
        record being post-selected out of the results collection.
    """

    name = "results.integer_state_predicate"

    key: StringAttr
    disallowed_values: ArrayAttr[IntAttr]

    def __init__(
        self,
        key: str | StringAttr,
        disallowed_values: Iterable[int | IntAttr] | ArrayAttr[IntAttr],
    ):
        """Initializes the IntegerStatePredicateAttr with the given key and disallowed
        values.

        :param key: The key of the entry in the record that post-selection is performed on.
        :param disallowed_values: The list of values that are disallowed and will result in
            the record being post-selected out of the results collection.
        """
        key_attr = StringAttr(key) if isinstance(key, str) else key

        if isinstance(disallowed_values, ArrayAttr):
            disallowed_values_attr = disallowed_values
        else:
            disallowed_values_attr = ArrayAttr(
                [
                    IntAttr(value) if isinstance(value, int) else value
                    for value in disallowed_values
                ]
            )

        return super().__init__(key_attr, disallowed_values_attr)
