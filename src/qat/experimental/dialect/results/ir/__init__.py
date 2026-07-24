# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""The Results dialect is used to model the collection and manipulation of results from
quantum programs.

It provides types and attributes to represent result records, arrays, and collections. The
dialect also defines operations for creating and transforming those structures, including
post-selection and record reshaping operations needed by existing lowering and runtime
paths.

This dialect is written heavily to be compatible with the legacy QAT runtime, but is
expected to rapidly evolve to support more general and flexible results collection and
manipulation in the future.


The dialect is centered around collections of results:

* :class:`ResultsArrayType` allows us to represent a list-like collection of results,
  grouping them together in a single structure.
* :class:`RecordType` represents the entire results of a single shot with dictionary-like
  semantics, allowing each piece of data to be stored by a string key.
* :class:`ResultsCollectionType` allows us to represent a collection of results, enabling
  dynamic addition of records, such as adding results from a single shot after each shot
  completes.

The types are not specific to any particular result data type, and allow us to store
arbitrary types of data (e.g. signals, bits, integers, IQ values). Along with operations to
assemble these data structures, the dialect also provides operations to manipulate records.
For example, an operation to group entries in a record into one single entry, with the
original entries placed within a :class:`ResultsArrayType`, and an operation to reduce
records to selected data.

The operations to create data structures and add them to the collections allows us to
store results from each shot as we go. This suits the representation of programs at a high
level, such as the gate-level or pulse-level. However, this is not always the way results
are represented closer to the hardware. For example, a particular control system might not
support pulse post-processing operations, and might just return raw IQ values. Then we would
need to retroactively apply the post-processing chain once the whole ensemble of results are
returned. To support this, the dialect also provides operations to apply a post-processing
chain to an entire collection of results. The :class:`MapOp` allows us to specify a
chain of operations that act on a :class:`RecordType`, and produce a new
:class:`RecordType`. The :class:`MapOp` then maps this chain of operations to every record
in a :class:`ResultsCollectionType`. This allows us to transform from a representation that
applies post-processing to results as they're collected, to a representation that applies
them retroactively to the entire collection of results.

Additionally, the dialect provides a post-selection operation which can be parameterised
with predicates to filter out results from a collection that do not satisfy the given
predicates.

**Mapping to post-processing in the runtime**

The runtime operates on legacy instructions. The instructions within this dialect allow us
to map those post-processing instructions:

* The :class:`GroupEntriesOp` maps to an Assign.
* The :class:`ReduceOp` maps to a Return.
"""

# TODO: Reassess __init__ with COMPILER-1350

from xdsl.ir import Dialect

from .attributes import IntegerStatePredicateAttr, PostSelectPredicateAttr
from .ops import (
    AddRecordOp,
    CreateRecordOp,
    CreateResultsArrayOp,
    CreateResultsCollectionOp,
    ExtractOp,
    GroupEntriesOp,
    MapOp,
    PostSelectOp,
    ReduceOp,
    YieldOp,
)
from .types import RecordType, ResultsArrayType, ResultsCollectionType

_ops = [
    AddRecordOp,
    CreateRecordOp,
    CreateResultsArrayOp,
    CreateResultsCollectionOp,
    ExtractOp,
    GroupEntriesOp,
    MapOp,
    PostSelectOp,
    ReduceOp,
    YieldOp,
]

_types = [RecordType, ResultsArrayType, ResultsCollectionType]

_attributes = [IntegerStatePredicateAttr]

Results = Dialect("results", _ops, _types + _attributes)

_all_classes = _ops + _types + _attributes

__all__ = ["PostSelectPredicateAttr", "Results"] + [cls_.__name__ for cls_ in _all_classes]
