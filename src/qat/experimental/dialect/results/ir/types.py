# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Models the types in the results dialect, which are used for dataflow of results to
collect and manipulate."""

from xdsl.ir import ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition


@irdl_attr_definition
class RecordType(ParametrizedAttribute, TypeAttribute):
    """A type that carries a results record that can be added to a results collection.

    This is used to represent a single record of results, e.g., from a single shot. It has
    the semantics of a dictionary, where the entries are accessed by string keys, and it can
    store arbitrary data types.
    """

    name = "results.record"


@irdl_attr_definition
class ResultsArrayType(ParametrizedAttribute, TypeAttribute):
    """A type that represents an array of results, which can be added to and filtered with
    given operations.

    This is used to represent a collection of results that can be indexed into, e.g., a
    register of classical bits measured from a quantum circuit. It takes standard array
    semantics, holding an ordered list of results.

    :class:`ResultsArrayType` is modelled to be immutable, and must be complete when
    constructed.
    """

    name = "results.array"


@irdl_attr_definition
class ResultsCollectionType(ParametrizedAttribute, TypeAttribute):
    """A type that represents a collection of results, which can be added to and filtered
    with given operations.

    Each entry is expected to hold the results of a single circuit execution, such as a
    single shot. Results can dynamically be added to the collection, and filtered away with
    given operations.
    """

    name = "results.collection"
