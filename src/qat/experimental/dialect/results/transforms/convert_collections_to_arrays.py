# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements the ``convert_results_collections_to_arrays`` pass, which lowers
:class:`~qat.experimental.dialect.results.ir.ResultsCollectionType` SSA values into
individual :class:`~qat.experimental.dialect.results.ir.ResultsArrayType` SSA values
throughout an operation tree.

The pass rewrites :class:`~qat.experimental.dialect.results.ir.CreateOp`,
:class:`~qat.experimental.dialect.results.ir.StoreOp`, and
:class:`~qat.experimental.dialect.results.ir.ExtractOp` that operate on collections,
and applies a generic decomposition for any other operation that carries collection-typed
variadic operands, results, or block arguments.

Entry point: :func:`convert_results_collections_to_arrays`.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from xdsl.builder import InsertPoint
from xdsl.ir import Attribute, Block, BlockArgument, Operation, Region, TypeAttribute
from xdsl.irdl import IRDLOperation, SSAValue
from xdsl.irdl.operations import VarOperandDef, VarResultDef
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.results.ir import (
    CreateOp,
    ExtractOp,
    RecordSchemaAttr,
    RecordType,
    ResultsArrayType,
    ResultsCollectionType,
    StoreOp,
)

_OpT = TypeVar("_OpT", bound=Operation)


class _ResultsCollectionToArrayMap:
    """Stores a mapping of results collection SSA values to their corresponding results
    array SSA values which they are replaced with.

    A results collection in this context is treated as a dictionary of results arrays, with
    keys to identify the results array (which is the field name).

    When a new results collection SSA is created, it takes on the values of the results
    arrays from any possible previous results collection SSA values that it was created
    from, but allows for particular results arrays to be replaced with new results arrays.

    Since this is modelling collection SSA values, the mappings of collection to arrays SSAs
    are, by definition, immutable.
    """

    def __init__(self):
        self._map: dict[
            SSAValue[ResultsCollectionType], dict[str, SSAValue[ResultsArrayType]]
        ] = {}

    def add(
        self,
        collection: SSAValue[ResultsCollectionType],
        arrays: dict[str, SSAValue[ResultsArrayType]],
        previous_collection: SSAValue[ResultsCollectionType] | None = None,
    ):
        """Adds a mapping of a results collection SSA value to its corresponding results
        array SSA values.

        A dictionary of key to results array SSA values can be provided, in addition to the
        previous results collection SSA value whose arrays are used as the base. If both are
        provided, the previous collection's results arrays are used as a base and the
        provided arrays replace any entries with the same key.
        """

        if previous_collection is not None:
            # Shallow copy to create a new mapping, but use the same objects
            # Orchestrator guarantees that we've already seen this
            base = copy(self._map[previous_collection])
        else:
            base = {}
        base.update(arrays)
        self._map[collection] = base

    def get(
        self, collection: SSAValue[ResultsCollectionType]
    ) -> dict[str, SSAValue[ResultsArrayType]] | None:
        """Returns the mapping of results array SSA values for a given results collection
        SSA value, or ``None`` if the collection has not yet been registered."""
        return self._map.get(collection, None)


@dataclass
class _RewriteCandidate(ABC, Generic[_OpT]):
    """Analysis result for an operation that carries a :class:`ResultsCollectionType` as an
    operand, result, or block argument.

    Subclasses implement :meth:`build` to analyse a concrete operation and :meth:`rewrite`
    to transform it once all of its collection-typed dependencies have already been
    lowered.

    Fields: ``operation`` is the op under analysis; ``operands``, ``results``, and
    ``block_args`` collect the collection-typed SSA values; ``rewritable`` is ``False``
    when the op has non-variadic collection-typed operands or results that the pass
    cannot decompose.
    """

    operation: _OpT
    operands: tuple[SSAValue[ResultsCollectionType]]
    results: tuple[SSAValue[ResultsCollectionType]]
    block_args: tuple[BlockArgument[ResultsCollectionType]]
    rewritable: bool

    @classmethod
    @abstractmethod
    def build(cls, operation: _OpT) -> "_RewriteCandidate":
        """Analyses ``operation`` and returns a populated candidate instance."""
        ...

    @abstractmethod
    def rewrite(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrites the operation, replacing collection-typed values with array values.

        Expects ``mapping`` to already contain entries for all collection-typed operands.
        """
        ...

    def can_rewrite(self, mapping: _ResultsCollectionToArrayMap) -> bool:
        """Returns ``True`` if all collection-typed operands have already been lowered."""

        return all(mapping.get(operand) is not None for operand in self.operands)


@dataclass
class _CreateOpRewrite(_RewriteCandidate[CreateOp]):
    """Rewrite candidate for :class:`~qat.experimental.dialect.results.ir.CreateOp`
    operations that produce a :class:`ResultsCollectionType` result.

    An empty-collection ``CreateOp`` is replaced by one ``CreateOp`` per schema field,
    each producing a :class:`ResultsArrayType`. A collection-from-arrays ``CreateOp`` is
    dissolved by mapping the collection result directly to its constituent array operands;
    no new operations are emitted.
    """

    @classmethod
    def build(cls, operation: CreateOp) -> "_CreateOpRewrite":
        """Performs the analysis and creates the result from the operation."""

        return cls(
            operation=operation,
            operands=(),
            results=(operation.result,),
            block_args=(),
            rewritable=True,
        )

    def rewrite(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrites the operation using the results from dependent operations."""

        if len(self.operation.values) == 0:
            self._rewrite_create_empty_collection(mapping)
        else:
            self._rewrite_create_collection_from_arrays(mapping)

    def _rewrite_create_empty_collection(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrite an empty-collection ``CreateOp`` into per-field array creates.

        Inserts one :class:`CreateOp` for each schema field, records the
        collection-to-arrays mapping, then detaches the original operation.
        """
        op = self.operation
        schema: RecordSchemaAttr = op.result.type.schema
        size = op.size or op.result.type.size  # Dynamic SSA or static value

        key_to_array_map = {}
        for schema_field in schema.fields:
            new_op = CreateOp.for_array(schema_field.type, size)
            key_to_array_map[schema_field.key.data] = new_op.result
            Rewriter.insert_op(new_op, InsertPoint.before(op))
        mapping.add(op.result, key_to_array_map)
        op.detach()

    def _rewrite_create_collection_from_arrays(self, mapping: _ResultsCollectionToArrayMap):
        """Dissolve a collection-from-arrays ``CreateOp`` by mapping its result directly to
        its existing array operands; no new operations are emitted."""
        op = self.operation
        schema: RecordSchemaAttr = op.result.type.schema
        key_to_array_map = {
            field.key.data: value
            for field, value in zip(schema.fields, op.values, strict=True)
        }
        mapping.add(op.result, key_to_array_map)
        op.detach()


@dataclass
class _StoreOpRewrite(_RewriteCandidate[StoreOp]):
    """Rewrite candidate for :class:`~qat.experimental.dialect.results.ir.StoreOp`
    operations whose container is a :class:`ResultsCollectionType`.

    Three sub-cases are handled:

    * Storing a record produced by a :class:`CreateOp` — field values are stored directly
      into the corresponding per-field arrays, and the ``CreateOp`` is detached if it
      becomes dead.
    * Storing a record produced by an arbitrary operation — per-field
      :class:`ExtractOp` / :class:`StoreOp` pairs are emitted to scatter the record into
      the individual arrays.
    * Storing a single keyed scalar value — a single array :class:`StoreOp` is emitted
      for the appropriate field array.
    """

    @classmethod
    def build(cls, operation: StoreOp) -> "_StoreOpRewrite":
        """Performs the analysis and creates the result from the operation."""
        operands = (
            (operation.container,)
            if isinstance(operation.container.type, ResultsCollectionType)
            else ()
        )
        results = (
            (operation.result,)
            if isinstance(operation.result.type, ResultsCollectionType)
            else ()
        )

        return cls(
            operation=operation,
            operands=operands,
            results=results,
            block_args=(),
            rewritable=True,
        )

    def rewrite(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrites the operation using the results from dependent operations."""
        op = self.operation

        if isinstance(op.value.type, RecordType) and isinstance(op.value.owner, CreateOp):
            self._rewrite_store_create_record_op(mapping)
        elif isinstance(op.value.type, RecordType):
            self._rewrite_store_arbitrary_record_op(mapping)
        else:
            self._rewrite_store_value_op(mapping)

    def _rewrite_store_create_record_op(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrite record store when the record comes from :class:`CreateOp`."""

        op = self.operation
        producer: CreateOp = op.value.owner
        collection = op.container
        arrays = mapping.get(collection)
        schema = op.container.type.schema

        key_to_array_map = {}
        for schema_field, value in zip(schema.fields, producer.values, strict=True):
            store_op = StoreOp.value_in_array(
                arrays[schema_field.key.data], op.index, value
            )
            key_to_array_map[schema_field.key.data] = store_op.result
            Rewriter.insert_op(store_op, InsertPoint.before(op))

        mapping.add(op.result, key_to_array_map, previous_collection=collection)
        op.detach()

        # Detach the record producer if its result is only used by detached ops.
        if all(user.operation.parent_block() is None for user in producer.result.uses):
            producer.detach()

    def _rewrite_store_arbitrary_record_op(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrite record store when the record is produced by a non-Create operation."""

        op = self.operation
        collection = op.container
        arrays = mapping.get(collection)
        index = op.index
        schema = collection.type.schema
        record = op.value

        key_to_array_map = {}
        for schema_field in schema.fields:
            extract_op = ExtractOp.value_from_record(record, schema_field.key)
            store_op = StoreOp.value_in_array(
                arrays[schema_field.key.data], index, extract_op.result
            )
            Rewriter.insert_op(extract_op, InsertPoint.before(op))
            Rewriter.insert_op(store_op, InsertPoint.before(op))
            key_to_array_map[schema_field.key.data] = store_op.result

        mapping.add(op.result, key_to_array_map, previous_collection=collection)
        op.detach()

    def _rewrite_store_value_op(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrite keyed scalar stores into array stores."""

        op = self.operation
        collection = op.container
        arrays = mapping.get(collection)
        key = op.key.data
        index = op.index

        new_op = StoreOp.value_in_array(arrays[key], index, op.value)
        mapping.add(op.result, {key: new_op.result}, previous_collection=collection)
        Rewriter.insert_op(new_op, InsertPoint.before(op))
        op.detach()


@dataclass
class _ExtractOpRewrite(_RewriteCandidate[ExtractOp]):
    """Rewrite candidate for :class:`~qat.experimental.dialect.results.ir.ExtractOp`
    operations whose container is a :class:`ResultsCollectionType`.

    Three sub-cases are handled:

    * Extracting a full :class:`ResultsArrayType` — the result is replaced directly with
      the mapped array SSA value; no new operations are emitted.
    * Extracting a scalar field value — replaced with an array :class:`ExtractOp` on the
      appropriate field array.
    * Extracting a :class:`RecordType` — replaced with per-field array extracts followed
      by a :class:`CreateOp` that assembles the record.
    """

    @classmethod
    def build(cls, operation: ExtractOp) -> "_ExtractOpRewrite":
        """Performs the analysis and creates the result from the operation."""
        operands = (
            (operation.container,)
            if isinstance(operation.container.type, ResultsCollectionType)
            else ()
        )
        results = (
            (operation.result,)
            if isinstance(operation.result.type, ResultsCollectionType)
            else ()
        )

        return cls(
            operation=operation,
            operands=operands,
            results=results,
            block_args=(),
            rewritable=True,
        )

    def rewrite(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrites the operation using the results from dependent operations."""
        op = self.operation

        if isinstance(op.result.type, ResultsArrayType):
            self._rewrite_extract_array_op(mapping)
        elif isinstance(op.result.type, RecordType):
            self._rewrite_extract_record_op(mapping)
        else:
            self._rewrite_extract_value_op(mapping)

    def _rewrite_extract_array_op(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrite extraction of a full results array from a collection."""

        op = self.operation
        collection = op.container
        arrays = mapping.get(collection)
        key = op.key.data
        Rewriter.replace_op(op, [], [arrays[key]])

    def _rewrite_extract_value_op(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrite extraction of a scalar field value from a collection."""

        op = self.operation
        collection = op.container
        arrays = mapping.get(collection)
        key = op.key.data
        index = op.index

        extract_op = ExtractOp.value_from_array(arrays[key], index)
        Rewriter.replace_op(op, extract_op)

    def _rewrite_extract_record_op(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrite extraction of a record from a collection via per-field extracts."""

        op = self.operation
        collection = op.container
        arrays = mapping.get(collection)
        index = op.index
        schema: RecordSchemaAttr = op.result.type.schema
        keys = [field.key.data for field in schema.fields]

        ops = []
        for key in keys:
            extract_op = ExtractOp.value_from_array(arrays[key], index)
            ops.append(extract_op)

        create_op = CreateOp.for_record(keys, ops)
        ops.append(create_op)
        Rewriter.replace_op(op, ops, [create_op.result])


@dataclass
class _OpSpec:
    """Mutable snapshot of an :class:`~xdsl.irdl.IRDLOperation`'s builder arguments.

    Captures all information required to call ``type.build(...)`` and reconstructs the
    operation with modified operands or result types. Used by
    :class:`_CollectionTypeRewrite` to expand collection-typed variadic positions into
    individual array positions before rebuilding the operation.

    ``operands`` and ``result_types`` are parallel to ``operand_names`` and
    ``result_names`` respectively; variadic positions are stored as sequences. ``regions``
    are detached from the original operation when :meth:`build` is called, so a spec
    instance must only be built once.
    """

    type: type[IRDLOperation]
    operand_names: list[str]
    operands: list[SSAValue | Sequence[SSAValue] | None]
    result_names: list[str]
    result_types: list[TypeAttribute | Sequence[TypeAttribute] | None]
    properties: dict[str, Attribute]
    successors: list[Block]
    regions: list[Region]
    _built: bool = field(default=False, init=False, repr=False, compare=False)

    @classmethod
    def from_op(cls, op: IRDLOperation) -> "_OpSpec":
        """Constructs an :class:`_OpSpec` by extracting builder arguments from ``op``."""

        irdl_def = type(op).get_irdl_definition()

        operands_sequence: list[SSAValue | Sequence[SSAValue]] = []
        operand_names: list[str] = []
        for name, operand_def in irdl_def.operands:
            operand_value = getattr(op, name)
            operand_value = (
                (operand_value,)
                if isinstance(operand_def, VarOperandDef)
                and not isinstance(operand_value, Sequence)
                else operand_value
            )
            operands_sequence.append(operand_value)
            operand_names.append(name)

        result_names: list[str] = []
        results_types_sequence: list[TypeAttribute | Sequence[TypeAttribute]] = []
        for name, result_def in irdl_def.results:
            result_value = getattr(op, name)
            result_names.append(name)
            if isinstance(result_def, VarResultDef):
                result_value = (
                    result_value if isinstance(result_value, Sequence) else (result_value,)
                )
                result_types = tuple(result.type for result in result_value)
                results_types_sequence.append(result_types)
            else:
                results_types_sequence.append(result_value.type)

        return cls(
            type=type(op),
            operand_names=operand_names,
            operands=operands_sequence,
            result_names=result_names,
            result_types=results_types_sequence,
            properties=op.properties,
            successors=op.successors,
            regions=op.regions,
        )

    def build(self) -> IRDLOperation:
        """Builds and returns a new operation from the current spec.

        The spec's regions are detached from any existing parent before being passed to the
        builder, so this method must only be called once per spec instance.
        """
        if self._built:
            raise AssertionError("_OpSpec.build() must only be called once per instance.")
        self._built = True

        for region in self.regions:
            region.parent_op().detach_region(region)

        return self.type.build(
            operands=self.operands,
            result_types=self.result_types,
            properties=self.properties,
            successors=self.successors,
            regions=self.regions,
        )


@dataclass
class _ResultsArraySpec:
    """Specifies where in an operation results arrays are located that correspond to a
    particular results collection."""

    field: str
    index: int
    number_of_arrays: int
    keys: list[str]

    def get_values(self, op: Operation) -> list[SSAValue[ResultsArrayType]]:
        """Returns the slice of array-typed SSA values from ``op``'s variadic result field
        at the position recorded by this spec."""

        var_results: Sequence[SSAValue[ResultsArrayType]] = getattr(op, self.field)
        return list(var_results[self.index : self.index + self.number_of_arrays])


def _has_non_variadic_collection_type(operation: IRDLOperation) -> bool:
    """Returns ``True`` if any non-variadic operand or result of ``operation`` has a
    :class:`ResultsCollectionType`; indicates the operation cannot be lowered by
    :class:`_CollectionTypeRewrite`."""
    irdl_def = type(operation).get_irdl_definition()
    for name, operand_def in irdl_def.operands:
        if isinstance(operand_def, VarOperandDef):
            continue
        value = getattr(operation, name)
        if value is not None and isinstance(value.type, ResultsCollectionType):
            return True
    for name, result_def in irdl_def.results:
        if isinstance(result_def, VarResultDef):
            continue
        value = getattr(operation, name)
        if value is not None and isinstance(value.type, ResultsCollectionType):
            return True
    return False


def collection_type_to_array_types(
    collection: ResultsCollectionType,
) -> dict[str, ResultsArrayType]:
    """Returns an ordered mapping of field keys to :class:`ResultsArrayType` for each field
    in ``collection``.

    The keys are ordered by schema field declaration order, which is the canonical ordering
    used throughout this pass when expanding a collection into individual arrays. Callers
    that depend on positional correspondence between fields and array values must preserve
    this order.

    Each array type inherits the collection's size.

    :param collection: The collection type to decompose.
    :returns: An ordered dictionary mapping each field key to its corresponding
        :class:`ResultsArrayType`, in schema field declaration order.
    """
    return {
        schema_field.key.data: ResultsArrayType(
            type_=schema_field.type, size=collection.size
        )
        for schema_field in collection.schema.fields
    }


@dataclass
class _CollectionTypeRewrite(_RewriteCandidate[IRDLOperation]):
    """Rewrite candidate for arbitrary :class:`~xdsl.irdl.IRDLOperation` operations that
    carry :class:`ResultsCollectionType` values only in variadic operand or result
    positions, or as block arguments within their regions.

    This is the fallback handler used for operations that are not one of the three
    dialect-specific ops (:class:`CreateOp`, :class:`StoreOp`, :class:`ExtractOp`). An
    operation is considered rewritable only when every non-variadic operand and result is
    free of :class:`ResultsCollectionType`; variadic positions are expanded in place.

    Block arguments of type :class:`ResultsCollectionType` in the operation's regions are
    replaced with N individual array-typed arguments (one per schema field) before the
    operand/result decomposition is applied.
    """

    @classmethod
    def build(cls, operation: IRDLOperation) -> "_CollectionTypeRewrite":
        """Performs the analysis and creates the result from the operation."""
        operands = tuple(
            operand
            for operand in operation.operands
            if isinstance(operand.type, ResultsCollectionType)
        )
        results = tuple(
            result
            for result in operation.results
            if isinstance(result.type, ResultsCollectionType)
        )
        block_args = tuple(
            arg
            for region in operation.regions
            for block in region.blocks
            for arg in block.args
            if isinstance(arg.type, ResultsCollectionType)
        )

        rewritable = not _has_non_variadic_collection_type(operation)

        return cls(
            operation=operation,
            operands=operands,
            results=results,
            block_args=block_args,
            rewritable=rewritable,
        )

    def rewrite(self, mapping: _ResultsCollectionToArrayMap):
        """Rewrites the operation using the results from dependent operations."""
        op = self.operation

        if op.regions:
            self._replace_results_collection_block_arguments(op, mapping)

        if not (
            any(isinstance(operand.type, ResultsCollectionType) for operand in op.operands)
            or any(isinstance(result.type, ResultsCollectionType) for result in op.results)
        ):
            # Only block args required expansion; operands/results are already lowered.
            return

        self._decompose_results_collection_operands_and_results(op, mapping)

    def _replace_results_collection_block_arguments(
        self,
        op: IRDLOperation,
        mapping: _ResultsCollectionToArrayMap,
    ):
        """Replaces collection-typed block arguments with array-typed arguments."""

        for region in op.regions:
            for block in region.blocks:
                block_args = [
                    (index, arg)
                    for index, arg in enumerate(block.args)
                    if isinstance(arg.type, ResultsCollectionType)
                ]
                if not block_args:
                    continue

                # Iterate backwards so insertion does not disturb pending indices.
                for index, arg in reversed(block_args):
                    key_to_array_map = {}
                    for i, (key, type_) in enumerate(
                        collection_type_to_array_types(arg.type).items()
                    ):
                        block_arg = block.insert_arg(type_, index + i)
                        key_to_array_map[key] = block_arg
                    mapping.add(arg, key_to_array_map)

    def _decompose_results_collection_operands_and_results(
        self,
        op: IRDLOperation,
        mapping: _ResultsCollectionToArrayMap,
    ):
        """Decomposes collection-typed variadic operands and results into arrays."""

        op_spec = _OpSpec.from_op(op)
        for i, operand in enumerate(op_spec.operands):
            if not isinstance(operand, Sequence):
                continue

            variadic_operands = []
            for sub_operand in operand:
                if not isinstance(sub_operand.type, ResultsCollectionType):
                    variadic_operands.append(sub_operand)
                    continue

                arrays = mapping.get(sub_operand)
                variadic_operands.extend(arrays.values())

            op_spec.operands[i] = variadic_operands

        collection_results_specs: dict[
            SSAValue[ResultsCollectionType], _ResultsArraySpec
        ] = {}
        for i, result in enumerate(op_spec.result_types):
            if not isinstance(result, Sequence):
                continue

            result_name = op_spec.result_names[i]
            original_result_values: Sequence[SSAValue] = getattr(op, result_name)

            variadic_results = []
            for sub_result, original_ssa in zip(
                result, original_result_values, strict=True
            ):
                if not isinstance(sub_result, ResultsCollectionType):
                    variadic_results.append(sub_result)
                    continue

                array_types = collection_type_to_array_types(sub_result)
                expanded_index = len(variadic_results)
                variadic_results.extend(array_types.values())
                collection_results_specs[original_ssa] = _ResultsArraySpec(
                    field=result_name,
                    index=expanded_index,
                    number_of_arrays=len(array_types),
                    keys=list(array_types.keys()),
                )

            op_spec.result_types[i] = variadic_results

        new_op = op_spec.build()
        Rewriter.insert_op(new_op, InsertPoint.before(op))
        op.detach()

        for collection_result, array_spec in collection_results_specs.items():
            arrays = array_spec.get_values(new_op)
            mapping.add(collection_result, dict(zip(array_spec.keys, arrays, strict=True)))


# Ops absent from this mapping fall through to _CollectionTypeRewrite, which handles
# collection-typed values only in variadic positions. New dialect ops with non-variadic
# collection-typed operands or results must be registered here with a dedicated handler.
_CANDIDATES: dict[type[Operation], type[_RewriteCandidate]] = {
    CreateOp: _CreateOpRewrite,
    StoreOp: _StoreOpRewrite,
    ExtractOp: _ExtractOpRewrite,
}


def _has_results_collection_value(op: Operation) -> bool:
    """Returns ``True`` if ``op`` has any :class:`ResultsCollectionType` operand, result, or
    block argument across all regions."""
    has_col_operands = any(
        isinstance(operand.type, ResultsCollectionType) for operand in op.operands
    )
    has_col_results = any(
        isinstance(result.type, ResultsCollectionType) for result in op.results
    )
    has_col_block_args = any(
        isinstance(arg.type, ResultsCollectionType)
        for region in op.regions
        for block in region.blocks
        for arg in block.args
    )
    return has_col_operands or has_col_results or has_col_block_args


def _rewrite_analysis(op: Operation) -> list[_RewriteCandidate]:
    """Walks ``op`` and builds a rewrite candidate for every nested operation that carries a
    :class:`ResultsCollectionType` value.

    Dialect-specific operations (:class:`CreateOp`, :class:`StoreOp`,
    :class:`ExtractOp`) are assigned their dedicated candidate types; all other
    qualifying operations use :class:`_CollectionTypeRewrite`. Candidates are returned
    in IR walk order.
    """
    candidates = []
    for nested_op in op.walk():
        if _has_results_collection_value(nested_op):
            candidate = _CANDIDATES.get(type(nested_op), _CollectionTypeRewrite).build(
                nested_op
            )
            candidates.append(candidate)
    return candidates


def _throw_if_not_rewritable(candidates: list[_RewriteCandidate]):
    """Raises :class:`~xdsl.utils.exceptions.PassFailedException` if any candidate has
    ``rewritable=False``."""
    non_rewritable_ops = [candidate for candidate in candidates if not candidate.rewritable]
    if non_rewritable_ops:
        raise PassFailedException(
            f"The pass convert_results_collections_to_arrays failed to convert all "
            f"results collection types to results array types. Cannot convert the "
            f"following operations: {', '.join(str(c.operation) for c in non_rewritable_ops)}"
        )


def _remove_old_collection_block_arguments(
    candidates: list[_RewriteCandidate], mapping: _ResultsCollectionToArrayMap
):
    """Removes collection-typed block arguments that have been replaced by array arguments.

    This is run as a cleanup pass after all operations have been rewritten, at which point
    all uses of the old collection block arguments should be dead.

    Reuses the block_args gathered during analysis rather than re-walking the IR.
    """
    blocks_to_args: dict[Block, list[BlockArgument]] = {}
    for candidate in candidates:
        for arg in candidate.block_args:
            if mapping.get(arg) is not None:
                blocks_to_args.setdefault(arg.block, []).append(arg)

    for block, args in blocks_to_args.items():
        # Sort descending so erasure starts from the highest index, avoiding index shifts.
        for arg in sorted(args, key=lambda a: a.index, reverse=True):
            # By this point all live users should have been rewritten
            block.erase_arg(arg, safe_erase=False)


def _rewrite_candidates(
    candidates: list[_RewriteCandidate], mapping: _ResultsCollectionToArrayMap
):
    """Rewrites candidates in dependency order using a fixed-point loop.

    On each iteration, candidates whose collection-typed operands have not yet been lowered
    are deferred to the next round. Raises if an iteration makes no progress, which would
    indicate a cycle or a missing analysis entry.
    """
    while candidates:
        non_rewrite_candidates = []
        for candidate in candidates:
            if not candidate.can_rewrite(mapping):
                non_rewrite_candidates.append(candidate)
                continue
            candidate.rewrite(mapping)

        if len(candidates) == len(non_rewrite_candidates):
            # Guard rail; practically the analysis should reveal if we can't rewrite all
            # operations, but we have this here to prevent infinite loops
            raise PassFailedException(
                "Pass convert_results_collections_to_arrays failed as not all operations "
                "with collection types could be rewritten under a fixed point iteration."
            )

        candidates = non_rewrite_candidates


def convert_results_collections_to_arrays(op: Operation):
    """Lowers all :class:`ResultsCollectionType` SSA values within ``op`` to individual
    :class:`ResultsArrayType` SSA values.

    The pass walks ``op``, identifies every nested operation that carries a
    :class:`ResultsCollectionType` operand, result, or block argument, and rewrites each
    in dependency order so that no collection-typed values remain in the IR on completion.

    Operations are rewritten in a fixed-point loop: candidates whose collection-typed
    operands have not yet been lowered are deferred to the next iteration. The loop
    terminates when all candidates have been rewritten, or raises if no progress is made.

    :param op: The root operation whose nested IR is transformed in place.
    :raises PassFailedException: If any operation cannot be lowered (e.g. it has a
        non-variadic collection-typed operand or result), or if the fixed-point iteration
        stalls without making progress.
    """

    all_candidates = _rewrite_analysis(op)
    _throw_if_not_rewritable(all_candidates)
    mapping = _ResultsCollectionToArrayMap()
    _rewrite_candidates(all_candidates, mapping)
    _remove_old_collection_block_arguments(all_candidates, mapping)
