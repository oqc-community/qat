# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Extract runtime-ready post-processing metadata from experimental results IR.

This module analyses results-dialect operations (for example ``results.map`` and
``results.post_select``) and produces a compact representation used by runtime execution.
The extracted model includes per-measurement acquire metadata, record-level post-selection
rules, record assignments, and the set of returned measurement aliases.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import singledispatchmethod

from compiler_config.config import InlineResultsProcessing
from xdsl.ir import Operation, SSAValue, TypeAttribute
from xdsl.utils.exceptions import PassFailedException

from qat.executables import AcquireData
from qat.experimental.dialect.pulse.ir import (
    AcquisitionType,
    DiscriminateOp,
    EqualiseOp,
    IQResultType,
    MaximumLikelihoodPolicyAttr,
    RealThresholdPolicyAttr,
)
from qat.experimental.dialect.results.ir import (
    CreateRecordOp,
    CreateResultsArrayOp,
    ExtractOp,
    GroupEntriesOp,
    IntegerStatePredicateAttr,
    MapOp,
    PostSelectOp,
    ReduceOp,
    YieldOp,
)
from qat.ir.instructions import Assign
from qat.ir.measure import (
    AcquireMode,
    Discriminate,
    Equalise,
    GranularPostProcessInstruction,
    PostSelect,
)
from qat.model.post_processing import MaxLikelihoodMethod, MLDiscriminateParams


@dataclass
class _ResultsProcessingOperations:
    """Holds the operations that represents results processing operations used by the
    :func:`extract_post_processing_instructions` function to extract the post-processing
    instructions from a module.

    :ivar map_ops: Holds the :class:`MapOp` operations that are used to represent the
        processing chains for individual measurements, and operations that happen
        collectively on the record.
    :ivar post_select_ops: Holds the :class:`PostSelectOp` operations that are used to
        filter away any records from a collection of records that do not meet the
        predicates.
    """

    map_ops: list[MapOp] = field(default_factory=list)
    post_select_ops: list[PostSelectOp] = field(default_factory=list)

    @classmethod
    def from_operation(cls, operation: Operation) -> "_ResultsProcessingOperations":
        """Constructs the results processing operations from an operation that contains a
        region.

        :param operation: The operation to extract the results processing operations from.
        :returns: The results processing operations extracted from the operation.
        """
        map_ops = []
        post_select_ops = []

        for op in operation.walk():
            if isinstance(op, MapOp):
                map_ops.append(op)
            elif isinstance(op, PostSelectOp):
                post_select_ops.append(op)

        return cls(map_ops=map_ops, post_select_ops=post_select_ops)


@dataclass
class PostProcessingAnalysis:
    """Container for post-processing metadata extracted from results IR.

    The fields in this model are runtime-oriented:

    * ``acquire_data`` captures per-measurement acquire mode, shape, and granular
      post-processing chain.
    * ``post_selects`` captures record-level filtering predicates.
    * ``assigns`` captures record alias materialisation (including grouped and array-style
      aliases).
    * ``returns`` captures the final set of record aliases to emit.

    .. note::

        ``post_selects`` are intentionally kept separate from
        ``acquire_data[alias].post_processing``. Even when a predicate references a
        single alias, post-selection is applied to the full record, not to an
        individual measurement processing chain.

    :ivar acquire_data: Runtime acquire metadata by measurement alias.
    :ivar post_selects: Record-level post-selection rules.
    :ivar assigns: Runtime ``Assign`` instructions derived from record construction
        and grouping operations.
    :ivar returns: Final set of measurement aliases to include in output records.
    """

    acquire_data: dict[str, AcquireData] = field(default_factory=dict)
    post_selects: list[PostSelect] = field(default_factory=list)
    assigns: list[Assign] = field(default_factory=list)
    returns: set[str] = field(default_factory=set)


class _MeasurementAliasTracker:
    """Tracks measurement aliases to SSA values, processing any results operations and
    keeping track of per-measurement processing chains, assigns and returns.

    This is intended to be used to walk a :class:`MapOp` and produce the operations that
    can be used at runtime. So if a chain of operations occurs that is unsupported by the
    runtime, it will raise an exception.

    The runtime implements a single-chain pipeline for each measurement; applying the
    post-processing mutates that measurement, so we can not yet have branching pipelines for
    a single measurement.
    """

    def __init__(self):
        self._alias_acquire_modes: dict[str, AcquireMode] = {}
        self._key_to_value: dict[str, SSAValue] = {}
        self._value_to_key: dict[SSAValue, str | None] = {}
        self._post_processing_steps: defaultdict[
            str, list[GranularPostProcessInstruction]
        ] = defaultdict(list)
        self._assigns: list[Assign] = []
        self._returns: set[str] = set()

    @singledispatchmethod
    def process_operation(self, op: Operation):
        """Processes an operation and updates the internal state of the tracker.

        :param op: The operation to process.
        """
        raise PassFailedException(
            f"Unsupported operation {op} in results processing chain."
        )

    def finalise(self, shape: tuple[int, ...]) -> PostProcessingAnalysis:
        """Returns the collected post-processing instructions, assigns and returns.

        :returns: The post-processing analysis containing acquire data, assigns and returns.
        """

        acquire_data = {}
        for alias, acquire_mode in self._alias_acquire_modes.items():
            # The physical channel is used for association of readout -> qubit which is
            # used with error mitigation which we don't need to support at this point. But
            # also, this doesn't work in a world with multiplexing. We need to have better
            # IR and runtime implementations for error mitigation. Anyway, this makes it
            # safe to set the physical channel to an empty string for now.

            # The inline results processing also needs rethinking in SIR world, so let's
            # just set it to RAW for now.

            acquire_data[alias] = AcquireData(
                mode=acquire_mode,
                shape=shape,
                post_processing=self._post_processing_steps[alias],
                results_processing=InlineResultsProcessing.Raw,
                physical_channel="",
            )

        return PostProcessingAnalysis(
            acquire_data=acquire_data,
            assigns=self._assigns,
            returns=self._returns,
        )

    def _get_measurement_alias(self, value: SSAValue) -> str | None:
        """Gets the measurement alias for an SSA value, if it exists.

        :param value: The SSA value to get the measurement alias for.
        :returns: The measurement alias for the SSA value, or ``None`` if it does not exist.
        """
        return self._value_to_key.get(value, None)

    def _track_measurement_alias(
        self, value: SSAValue, measurement_alias: str | None = None
    ):
        """Tracks a measurement alias to an SSA value.

        If no measurement alias is provided, the SSA value is tracked without a name, and is
        updated later upon usage.
        """
        if measurement_alias is not None and measurement_alias in self._key_to_value:
            old_value = self._key_to_value[measurement_alias]
            del self._value_to_key[old_value]
        self._value_to_key[value] = measurement_alias
        if measurement_alias is not None:
            self._key_to_value[measurement_alias] = value

    def _check_result_only_has_single_use(self, value: SSAValue):
        """Checks that the SSA value has only a single use, and raises an exception if it
        has multiple uses.

        This is used to ensure that the runtime can support the processing chain for a
        measurement, as it can only support a single chain of operations for each
        measurement.
        """
        if value.uses.get_length() > 1:
            raise PassFailedException(
                f"Results from operation {value.owner} has multiple uses, which is not "
                f"supported."
            )

    def _determine_acquire_mode(self, type_: TypeAttribute) -> AcquireMode:
        """Determines the acquire mode the runtime uses to understand how the acquisition
        has been processed on the hardware."""

        match type_:
            case IQResultType():
                return AcquireMode.INTEGRATOR
            case AcquisitionType():
                return AcquireMode.RAW
            case _:
                raise PassFailedException(
                    f"Unsupported SSA type {type_} for acquire operation in results "
                    f"processing chain."
                )

    @process_operation.register
    def _process_extract(self, op: ExtractOp):
        """Process the :class:`ExtractOp` operation, translating it to the runtime
        representation and updating the internal state of the tracker.

        Extracting it tracks the measurement alias to the SSA value for subsequent
        operations.
        """
        measurement_alias = op.key.data

        if measurement_alias in self._key_to_value:
            raise PassFailedException(
                f"Multiple ExtractOps found for the same measurement alias "
                f"{measurement_alias}, which "
                f"is not supported."
            )
        self._alias_acquire_modes[measurement_alias] = self._determine_acquire_mode(
            op.result.type
        )
        self._check_result_only_has_single_use(op.result)
        self._track_measurement_alias(op.result, measurement_alias)

    @process_operation.register
    def _process_discriminate(self, op: DiscriminateOp):
        """Process the :class:`DiscriminateOp` operation, translating it to the runtime
        representation and updating the internal state of the tracker."""

        self._check_result_only_has_single_use(op.result)
        # result cannot be None due to properties of map op and the supported operations
        measurement_alias: str = self._get_measurement_alias(op.value)
        self._track_measurement_alias(op.result, measurement_alias)

        match op.policy:
            case RealThresholdPolicyAttr(threshold=threshold):
                discriminate_instr = Discriminate(
                    output_variable=measurement_alias,
                    threshold=threshold.data,
                    method=None,
                )
            case MaximumLikelihoodPolicyAttr(
                state_centers=state_centers,
                noise_estimate=noise_estimate,
                p_min=p_min,
            ):
                states = {
                    i: MLDiscriminateParams(location=location.data)
                    for i, location in enumerate(state_centers)
                }

                discriminate_instr = Discriminate(
                    output_variable=measurement_alias,
                    threshold=None,
                    method=MaxLikelihoodMethod(
                        noise_est=noise_estimate.data,
                        states=states,
                        p_min=p_min.data,
                    ),
                )
            case _:
                raise PassFailedException(
                    f"Unsupported policy {op.policy} for DiscriminateOp in results "
                    f"processing chain."
                )
        self._post_processing_steps[measurement_alias].append(discriminate_instr)

    @process_operation.register
    def _process_equalise(self, op: EqualiseOp):
        """Process the :class:`EqualiseOp` operation, translating it to the runtime
        respresentation and updating the internal state of the tracker."""

        self._check_result_only_has_single_use(op.result)
        # result cannot be None due to properties of map op and the supported operations
        measurement_alias: str = self._get_measurement_alias(op.value)
        self._track_measurement_alias(op.result, measurement_alias)

        equalise_instr = Equalise(
            output_variable=measurement_alias,
            transform=op.affine_transform.linear_matrix,
            offset=op.affine_transform.translation_vector,
        )
        self._post_processing_steps[measurement_alias].append(equalise_instr)

    @process_operation.register
    def _process_create_results_array(self, op: CreateResultsArrayOp):
        """Process the :class:`CreateResultsArrayOp` operation, translating it to the
        runtime representation and updating the internal state of the tracker.

        Doing so creates an array with an unknown alias; we can look ahead to see if this
        operation is used within a :class:`CreateRecordOp` to determine the alias, or assign
        it a temporary alias if it's used in another :class:`CreateResultsArrayOp`.

        The result of this operation is to make an assign instruction.
        """

        # TODO: any renaming of measurement alias must make use of an assign.

        self._check_result_only_has_single_use(op.result)
        uses = list(op.result.uses)

        alias = None
        if len(uses) == 1:
            use = uses[0]
            if isinstance(use.operation, CreateRecordOp):
                alias = use.operation.keys.data[use.index].data
        alias = alias or f"_temp_{hash(op)}"

        # Get the alias for each SSA value
        aliases = [self._get_measurement_alias(value) for value in op.values]
        if any(a is None for a in aliases):
            raise PassFailedException(
                "CreateResultsArrayOp contains values that are not associated with "
                "measurement aliases."
            )

        self._track_measurement_alias(op.result, alias)
        self._assigns.append(Assign(name=alias, value=aliases))

    @process_operation.register
    def _process_create_record(self, op: CreateRecordOp):
        """Process the :class:`CreateRecordOp` operation, translating it to the runtime
        representation and updating the internal state of the tracker.

        If the record created is used by the :class:`YieldOp` operation, then this is the
        dictionary that is returned by the results processing. Otherwise, it's used by the
        operations that act on a record. Regardless, we use this to determine which
        measurement aliases are returned, which might be subsequently reduced later.
        """

        self._check_result_only_has_single_use(op.result)
        if self._returns:
            raise PassFailedException(
                "Multiple CreateRecordOps found in the results processing chain, which is "
                "not supported."
            )

        self._returns.update([key.data for key in op.keys])

    @process_operation.register
    def _process_group_entries(self, op: GroupEntriesOp):
        """Process the :class:`GroupEntriesOp` operation, translating it to the runtime
        representation and updating the internal state of the tracker.

        The group entries is a high-level operation that will act as an assign, moving the
        measurement aliases into a new alias that is a list of the grouped aliases. We need
        to adjust the returns to remove the aliases within the group, and add the new alias.
        """

        self._check_result_only_has_single_use(op.result)

        group_keys = [key.data for key in op.keys]
        assign_key = op.group_key.data

        # check the keys are in the returns
        if set(group_keys).difference(self._returns):
            raise PassFailedException(
                f"GroupEntriesOp found in the results processing chain with keys "
                f"{group_keys} that are not in the returns {self._returns}, which is not "
                f"supported."
            )

        self._returns.difference_update(group_keys)
        self._returns.add(assign_key)

        self._assigns.append(Assign(name=assign_key, value=group_keys))

    @process_operation.register
    def _process_reduce(self, op: ReduceOp):
        """Process the :class:`ReduceOp` operation, translating it to the runtime
        representation and updating the internal state of the tracker.

        The reduce operation is a high-level operation that will filter measurements in a
        record down to the keys that are given in the operation. This updates the returns,
        and
        """

        self._check_result_only_has_single_use(op.result)

        reduce_keys = [key.data for key in op.keys]
        if not set(reduce_keys).issubset(self._returns):
            raise PassFailedException(
                f"ReduceOp found in the results processing chain with keys "
                f"{reduce_keys} that are not in the returns {self._returns}, which is not "
                f"supported."
            )

        self._returns = set(reduce_keys)

    @process_operation.register
    def _process_yield(self, _op: YieldOp):
        """Process the :class:`YieldOp` operation, translating it to the runtime
        representation and updating the internal state of the tracker.

        The yield operation does nothing in terms of the runtime representation.
        """
        pass


def extract_post_processing_instructions(
    top_level: Operation, acquire_shape: int | tuple[int, ...]
) -> PostProcessingAnalysis:
    """Extract post-processing metadata from a top-level IR operation.

    The traversal supports either a module-like container or a function-like container as
    long as it contains at most one ``results.map`` operation and any number of
    ``results.post_select`` operations. This is designed for the current "legacy"
    implementation of runtime, which uses post-processing instructions in a prescribed way
    that is not very extensible. This analysis will ensure those restrictions are met, and
    will raise an exception if they are not.

    This analysis pass will not have a place in the future with a more sophisticated
    runtime, and will be replaced with something more flexible and extensible.

    The returned :class:`PostProcessingAnalysis` is ready to be embedded in an
    executable for runtime consumption.

    :param top_level: The operation to extract the post-processing instructions from.
    :param acquire_shape: Acquisition shape to attach to each extracted
        :class:`~qat.executables.AcquireData`. An ``int`` is normalised to a single-element
        tuple.
    :returns: The post-processing instructions extracted from the operation.
    :raises PassFailedException: If unsupported operations, predicates, or incompatible
        result-flow patterns are encountered.
    """

    acquire_shape = (acquire_shape,) if isinstance(acquire_shape, int) else acquire_shape
    results_processing_ops = _ResultsProcessingOperations.from_operation(top_level)

    if len(results_processing_ops.map_ops) > 1:
        raise PassFailedException(
            "Multiple MapOps found in the top-level operation. Only one is allowed."
        )

    tracker = _MeasurementAliasTracker()
    if len(results_processing_ops.map_ops) == 1:
        for op in results_processing_ops.map_ops[0].body.ops:
            tracker.process_operation(op)
    analysis = tracker.finalise(shape=acquire_shape)

    # Deal with the post-select instructions
    post_selects = []
    for post_select_op in results_processing_ops.post_select_ops:
        predicates = post_select_op.predicates
        for predicate in predicates:
            if not isinstance(predicate, IntegerStatePredicateAttr):
                raise PassFailedException(
                    f"Unsupported predicate {predicate} in PostSelectOp. Only "
                    f"IntegerStatePredicateAttr is supported."
                )

            post_selects.append(
                PostSelect(
                    output_variable=predicate.key.data,
                    additional_disallowed=[val.data for val in predicate.disallowed_values],
                )
            )

    analysis.post_selects = post_selects
    return analysis
