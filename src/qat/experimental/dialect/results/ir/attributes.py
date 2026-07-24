# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Models the attributes in the results dialect.

This includes attributes that model post-selection predicates.
"""

from abc import ABC
from collections.abc import Iterable

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.irdl import ParametrizedAttribute, irdl_attr_definition


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
