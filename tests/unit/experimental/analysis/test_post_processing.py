# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the analysis to extract post-processing utilities from the IR to be applied at
runtime."""

import pytest
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.analysis.post_processing import extract_post_processing_instructions
from qat.experimental.dialect.pulse.ir import (
    AcquisitionType,
    DiscriminateOp,
    DiscriminatorPolicyAttr,
    EqualiseAttr,
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
from qat.experimental.dialect.results.ir.attributes import PostSelectPredicateAttr
from qat.experimental.dialect.results.ir.types import RecordType, ResultsCollectionType
from qat.ir.instructions import Assign
from qat.ir.measure import AcquireMode, Discriminate, Equalise, PostSelect
from qat.model.post_processing import MaxLikelihoodMethod, MLDiscriminateParams


@irdl_op_definition
class _MockCollectionOp(IRDLOperation):
    """Produces a ResultsCollectionType SSA value for tests."""

    name = "test.post_processing_mock_collection"
    res = result_def(ResultsCollectionType)

    def __init__(self) -> None:
        super().__init__(result_types=[ResultsCollectionType()])


@irdl_op_definition
class _MockTransformOp(IRDLOperation):
    """Unsupported IQ-to-IQ transform op for testing unregistered operation handling."""

    name = "test.post_processing_mock_transform"
    value = operand_def(IQResultType)
    res = result_def(IQResultType)

    def __init__(self, value: SSAValue) -> None:
        super().__init__(
            operands=[SSAValue.get(value, type=IQResultType)],
            result_types=[IQResultType()],
        )


@irdl_attr_definition
class _MockDiscriminatorPolicyAttr(DiscriminatorPolicyAttr):
    """Mock discriminator policy for testing unsupported policy handling."""

    name = "test.post_processing_mock_discriminator_policy"

    @property
    def state_range(self) -> tuple[int, int]:
        return (0, 1)


@irdl_attr_definition
class _MockPredicateAttr(PostSelectPredicateAttr):
    """Mock post-select predicate for testing unsupported predicate handling."""

    name = "test.post_processing_mock_predicate"


def _build_module(*fn_ops: Operation) -> ModuleOp:
    """Wraps operations inside a 'main' FuncOp inside a ModuleOp."""
    return ModuleOp([func.FuncOp("main", ((), ()), Region(Block(list(fn_ops))))])


def _make_two_acquire_fn() -> tuple[func.FuncOp, EqualiseAttr, EqualiseAttr, float, float]:
    """Builds a FuncOp containing a two-acquire MapOp pipeline with equalise, discriminate,
    and a post-select on the first acquire.

    Returns the FuncOp along with the EqualiseAttrs and discriminate thresholds used, so the
    caller can construct matching expected values.
    """
    eq_attr1 = EqualiseAttr(1 + 0j, 0 + 0j, 0 + 0j)
    eq_attr2 = EqualiseAttr(1 + 0j, 0 + 0.5j, 1 + 1j)
    threshold1 = 0.5
    threshold2 = -0.3

    body = Block(arg_types=[RecordType()])
    record_arg = body.args[0]
    ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
    ext2 = ExtractOp(record_arg, "acquire2", IQResultType())
    eq1 = EqualiseOp(ext1.result, eq_attr1)
    eq2 = EqualiseOp(ext2.result, eq_attr2)
    disc1 = DiscriminateOp(eq1.result, RealThresholdPolicyAttr(threshold1))
    disc2 = DiscriminateOp(eq2.result, RealThresholdPolicyAttr(threshold2))
    rec = CreateRecordOp(["acquire1", "acquire2"], [disc1.result, disc2.result])
    body.add_ops([ext1, ext2, eq1, eq2, disc1, disc2, rec, YieldOp(rec.result)])

    collection = _MockCollectionOp()
    map_op = MapOp(collection.res, body)
    post_select = PostSelectOp(map_op.result, IntegerStatePredicateAttr("acquire1", [1]))
    fn = func.FuncOp(
        "main",
        ((), ()),
        Region(Block([collection, map_op, post_select, func.ReturnOp()])),
    )
    return fn, eq_attr1, eq_attr2, threshold1, threshold2


def _assert_two_acquire_result(result, eq_attr1, eq_attr2, threshold1, threshold2):
    """Asserts the expected shape of a two-acquire pipeline analysis result."""
    assert set(result.acquire_data.keys()) == {"acquire1", "acquire2"}

    acquire1 = result.acquire_data["acquire1"]
    assert acquire1.mode == AcquireMode.INTEGRATOR
    assert acquire1.shape == (1000,)
    assert acquire1.post_processing == [
        Equalise(
            output_variable="acquire1",
            transform=eq_attr1.linear_matrix,
            offset=eq_attr1.translation_vector,
        ),
        Discriminate(output_variable="acquire1", threshold=threshold1),
    ]

    acquire2 = result.acquire_data["acquire2"]
    assert acquire2.mode == AcquireMode.INTEGRATOR
    assert acquire2.shape == (1000,)
    assert acquire2.post_processing == [
        Equalise(
            output_variable="acquire2",
            transform=eq_attr2.linear_matrix,
            offset=eq_attr2.translation_vector,
        ),
        Discriminate(output_variable="acquire2", threshold=threshold2),
    ]

    assert result.returns == {"acquire1", "acquire2"}
    assert result.assigns == []
    assert result.post_selects == [
        PostSelect(output_variable="acquire1", additional_disallowed={1})
    ]


class TestExtractPostProcessingOnIR:
    """Tests this works when the operation fed into the analysis is either a module or
    function."""

    def test_extract_post_processing_on_module_gives_expected_results(self):
        """Tests that the analysis correctly extracts post-processing utilities from a
        module operation."""

        fn, eq_attr1, eq_attr2, threshold1, threshold2 = _make_two_acquire_fn()
        module = ModuleOp([fn])
        result = extract_post_processing_instructions(module, (1000,))
        _assert_two_acquire_result(result, eq_attr1, eq_attr2, threshold1, threshold2)

    def test_extract_post_processing_on_function_gives_expected_results(self):
        """Tests that the analysis correctly extracts post-processing utilities from a
        function operation."""

        fn, eq_attr1, eq_attr2, threshold1, threshold2 = _make_two_acquire_fn()
        result = extract_post_processing_instructions(fn, (1000,))
        _assert_two_acquire_result(result, eq_attr1, eq_attr2, threshold1, threshold2)


class TestExtractPostProcessingErrorHandling:
    """Tests that the analysis correctly raises errors when the IR is not as expected."""

    def test_unsupported_operation_in_map_op_raises(self):
        """Tests that the analysis raises a PassFailedException when the MapOp contains an
        unsupported operation."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        mock_op = _MockTransformOp(ext1.result)
        rec = CreateRecordOp(["acquire1"], [mock_op.res])
        body.add_ops([ext1, mock_op, rec, YieldOp(rec.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        with pytest.raises(PassFailedException, match="Unsupported operation"):
            extract_post_processing_instructions(module, (1000,))

    def test_result_with_multiple_uses_in_map_op_raises(self):
        """Tests that the analysis raises a PassFailedException when the MapOp contains a
        result with multiple uses."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        eq_attr = EqualiseAttr(1 + 0j, 0 + 0j, 0 + 0j)
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        eq1 = EqualiseOp(ext1.result, eq_attr)
        eq2 = EqualiseOp(ext1.result, eq_attr)  # second use of ext1.result
        rec = CreateRecordOp(["acquire1"], [eq1.result])
        body.add_ops([ext1, eq1, eq2, rec, YieldOp(rec.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        with pytest.raises(PassFailedException, match="multiple uses"):
            extract_post_processing_instructions(module, (1000,))

    def test_unsupported_type_from_extract_raises(self):
        """Tests that the analysis raises a PassFailedException when the MapOp contains an
        extract operation that returns an unsupported type."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", i32)
        rec = CreateRecordOp(["acquire1"], [ext1.result])
        body.add_ops([ext1, rec, YieldOp(rec.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        with pytest.raises(PassFailedException, match="Unsupported SSA type"):
            extract_post_processing_instructions(module, (1000,))

    def test_multiple_extracts_with_same_alias_raises(self):
        """Tests that the analysis raises a PassFailedException when the MapOp contains
        multiple extract operations with the same alias."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        ext2 = ExtractOp(record_arg, "acquire1", IQResultType())  # duplicate alias
        rec = CreateRecordOp(["acquire1", "acquire1b"], [ext1.result, ext2.result])
        body.add_ops([ext1, ext2, rec, YieldOp(rec.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        with pytest.raises(PassFailedException, match="Multiple ExtractOps"):
            extract_post_processing_instructions(module, (1000,))

    def test_unsupported_discrimination_policy_raises(self):
        """Tests that the analysis raises a PassFailedException when the MapOp contains a
        discrimination operation with an unsupported policy."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        disc_op = DiscriminateOp(ext1.result, _MockDiscriminatorPolicyAttr())
        rec = CreateRecordOp(["acquire1"], [disc_op.result])
        body.add_ops([ext1, disc_op, rec, YieldOp(rec.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        with pytest.raises(PassFailedException, match="Unsupported policy"):
            extract_post_processing_instructions(module, (1000,))

    def test_multiple_create_record_ops_in_map_op_raises(self):
        """Tests that the analysis raises a PassFailedException when the MapOp contains
        multiple create record operations."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        ext2 = ExtractOp(record_arg, "acquire2", IQResultType())
        rec1 = CreateRecordOp(["acquire1"], [ext1.result])
        rec2 = CreateRecordOp(["acquire2"], [ext2.result])
        body.add_ops([ext1, ext2, rec1, rec2, YieldOp(rec2.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        with pytest.raises(PassFailedException, match="Multiple CreateRecordOps"):
            extract_post_processing_instructions(module, (1000,))

    def test_group_entries_with_non_existent_aliases_raises(self):
        """Tests that the analysis raises a PassFailedException when the MapOp contains a
        group entries operation with non-existent aliases."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        rec = CreateRecordOp(["acquire1"], [ext1.result])
        group_op = GroupEntriesOp(rec.result, ["nonexistent1", "nonexistent2"], "grouped")
        body.add_ops([ext1, rec, group_op, YieldOp(group_op.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        with pytest.raises(PassFailedException, match="not in the returns"):
            extract_post_processing_instructions(module, (1000,))

    def test_reduce_entries_with_non_existent_aliases_raises(self):
        """Tests that the analysis raises a PassFailedException when the MapOp contains a
        reduce entries operation with non-existent aliases."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        rec = CreateRecordOp(["acquire1"], [ext1.result])
        reduce_op = ReduceOp(rec.result, ["nonexistent"])
        body.add_ops([ext1, rec, reduce_op, YieldOp(reduce_op.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        with pytest.raises(PassFailedException, match="not in the returns"):
            extract_post_processing_instructions(module, (1000,))

    def test_multiple_map_ops_in_module_raises(self):
        """Tests that the analysis raises a PassFailedException when the module contains
        multiple MapOps."""
        body1 = Block(arg_types=[RecordType()])
        ext1 = ExtractOp(body1.args[0], "a", IQResultType())
        rec1 = CreateRecordOp(["a"], [ext1.result])
        body1.add_ops([ext1, rec1, YieldOp(rec1.result)])

        body2 = Block(arg_types=[RecordType()])
        ext2 = ExtractOp(body2.args[0], "b", IQResultType())
        rec2 = CreateRecordOp(["b"], [ext2.result])
        body2.add_ops([ext2, rec2, YieldOp(rec2.result)])

        collection = _MockCollectionOp()
        map_op1 = MapOp(collection.res, body1)
        map_op2 = MapOp(map_op1.result, body2)
        module = _build_module(collection, map_op1, map_op2, func.ReturnOp())

        with pytest.raises(PassFailedException, match="Multiple MapOps"):
            extract_post_processing_instructions(module, (1000,))

    def test_non_integer_state_predicate_raises(self):
        """Tests that the analysis raises a PassFailedException when the post-select
        operation contains a non-integer state predicate."""
        collection = _MockCollectionOp()
        post_select = PostSelectOp(collection.res, _MockPredicateAttr())
        module = _build_module(collection, post_select, func.ReturnOp())

        with pytest.raises(PassFailedException, match="Unsupported predicate"):
            extract_post_processing_instructions(module, (1000,))


class TestExtractPostProcessingSpecialCases:
    """Tests that the analysis correctly handles special cases."""

    def test_with_real_threshold_policy(self):
        """Tests that the analysis correctly handles a real threshold policy in a
        discrimination operation."""
        eq_attr = EqualiseAttr(1 + 0j, 0 + 0j, 0 + 0j)
        threshold = 0.5

        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        eq1 = EqualiseOp(ext1.result, eq_attr)
        disc1 = DiscriminateOp(eq1.result, RealThresholdPolicyAttr(threshold))
        rec = CreateRecordOp(["acquire1"], [disc1.result])
        body.add_ops([ext1, eq1, disc1, rec, YieldOp(rec.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        result = extract_post_processing_instructions(module, (1000,))

        assert result.acquire_data["acquire1"].post_processing == [
            Equalise(
                output_variable="acquire1",
                transform=eq_attr.linear_matrix,
                offset=eq_attr.translation_vector,
            ),
            Discriminate(output_variable="acquire1", threshold=threshold),
        ]

    def test_with_maximum_likelihood_policy(self):
        """Tests that the analysis correctly handles a maximum likelihood policy in a
        discrimination operation."""
        eq_attr = EqualiseAttr(1 + 0j, 0 + 0j, 0 + 0j)
        state_centers = [0 + 0j, 1 + 0j]
        noise_est = 0.1
        p_min = 0.0
        ml_policy = MaximumLikelihoodPolicyAttr(
            state_centers=state_centers,
            noise_estimate=noise_est,
            p_min=p_min,
        )

        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        eq1 = EqualiseOp(ext1.result, eq_attr)
        disc1 = DiscriminateOp(eq1.result, ml_policy)
        rec = CreateRecordOp(["acquire1"], [disc1.result])
        body.add_ops([ext1, eq1, disc1, rec, YieldOp(rec.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        result = extract_post_processing_instructions(module, (1000,))

        expected_method = MaxLikelihoodMethod(
            states={
                i: MLDiscriminateParams(location=c) for i, c in enumerate(state_centers)
            },
            noise_est=noise_est,
            p_min=p_min,
        )
        assert result.acquire_data["acquire1"].post_processing == [
            Equalise(
                output_variable="acquire1",
                transform=eq_attr.linear_matrix,
                offset=eq_attr.translation_vector,
            ),
            Discriminate(output_variable="acquire1", method=expected_method),
        ]

    def test_create_results_array_creates_assign(self):
        """Tests that the analysis correctly handles a create results array operation."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        ext2 = ExtractOp(record_arg, "acquire2", IQResultType())
        arr_op = CreateResultsArrayOp([ext1.result, ext2.result])
        rec = CreateRecordOp(["grouped"], [arr_op.result])
        body.add_ops([ext1, ext2, arr_op, rec, YieldOp(rec.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        result = extract_post_processing_instructions(module, (1000,))

        assert result.assigns == [Assign(name="grouped", value=["acquire1", "acquire2"])]
        assert result.returns == {"grouped"}

    def test_create_nested_results_arrays_creates_multiple_assigns(self):
        """Tests that the analysis correctly handles a create nested results array
        operation."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext_a = ExtractOp(record_arg, "a", IQResultType())
        ext_b = ExtractOp(record_arg, "b", IQResultType())
        ext_c = ExtractOp(record_arg, "c", IQResultType())
        ext_d = ExtractOp(record_arg, "d", IQResultType())
        arr1 = CreateResultsArrayOp([ext_a.result, ext_b.result])
        arr2 = CreateResultsArrayOp([ext_c.result, ext_d.result])
        arr3 = CreateResultsArrayOp([arr1.result, arr2.result])
        rec = CreateRecordOp(["nested"], [arr3.result])
        body.add_ops(
            [ext_a, ext_b, ext_c, ext_d, arr1, arr2, arr3, rec, YieldOp(rec.result)]
        )

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        result = extract_post_processing_instructions(module, (1000,))

        assert len(result.assigns) == 3
        outer_assign = next(a for a in result.assigns if a.name == "nested")
        inner_assigns = [a for a in result.assigns if a.name != "nested"]
        assert sorted([sorted(a.value) for a in inner_assigns]) == [["a", "b"], ["c", "d"]]
        assert set(outer_assign.value) == {a.name for a in inner_assigns}

    def test_group_entries_creates_assign(self):
        """Tests that the analysis correctly handles a group entries operation."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        ext2 = ExtractOp(record_arg, "acquire2", IQResultType())
        rec = CreateRecordOp(["acquire1", "acquire2"], [ext1.result, ext2.result])
        group_op = GroupEntriesOp(rec.result, ["acquire1", "acquire2"], "grouped")
        body.add_ops([ext1, ext2, rec, group_op, YieldOp(group_op.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        result = extract_post_processing_instructions(module, (1000,))

        assert result.assigns == [Assign(name="grouped", value=["acquire1", "acquire2"])]
        assert result.returns == {"grouped"}

    def test_reduce_entries_limits_return_variables(self):
        """Tests that the analysis correctly handles a reduce entries operation."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        ext2 = ExtractOp(record_arg, "acquire2", IQResultType())
        rec = CreateRecordOp(["acquire1", "acquire2"], [ext1.result, ext2.result])
        reduce_op = ReduceOp(rec.result, ["acquire1"])
        body.add_ops([ext1, ext2, rec, reduce_op, YieldOp(reduce_op.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        result = extract_post_processing_instructions(module, (1000,))

        assert result.returns == {"acquire1"}

    def test_create_record_with_not_all_entries_gives_reduced_returns(self):
        """Tests that the analysis correctly handles a create record operation with not all
        entries."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        ext2 = ExtractOp(record_arg, "acquire2", IQResultType())
        rec = CreateRecordOp(["acquire1"], [ext1.result])
        body.add_ops([ext1, ext2, rec, YieldOp(rec.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        result = extract_post_processing_instructions(module, (1000,))

        assert result.returns == {"acquire1"}

    def test_post_selects_with_multiple_integer_predicates_gives_multiple_post_selects(
        self,
    ):
        """Tests that the analysis correctly handles a post-select operation with multiple
        integer predicates."""
        collection = _MockCollectionOp()
        post_select = PostSelectOp(
            collection.res,
            IntegerStatePredicateAttr("acquire1", [1]),
            IntegerStatePredicateAttr("acquire2", [0, 1]),
        )
        module = _build_module(collection, post_select, func.ReturnOp())

        result = extract_post_processing_instructions(module, (1000,))

        assert len(result.post_selects) == 2
        assert result.post_selects[0] == PostSelect(
            output_variable="acquire1", additional_disallowed={1}
        )
        assert result.post_selects[1] == PostSelect(
            output_variable="acquire2", additional_disallowed={0, 1}
        )

    def test_multiple_post_selects_gives_multiple_post_selects(self):
        """Tests that the analysis correctly handles multiple post-select operations."""
        collection = _MockCollectionOp()
        ps_op1 = PostSelectOp(collection.res, IntegerStatePredicateAttr("acquire1", [1]))
        ps_op2 = PostSelectOp(ps_op1.result, IntegerStatePredicateAttr("acquire2", [0]))
        module = _build_module(collection, ps_op1, ps_op2, func.ReturnOp())

        result = extract_post_processing_instructions(module, (1000,))

        assert len(result.post_selects) == 2
        assert result.post_selects[0] == PostSelect(
            output_variable="acquire1", additional_disallowed={1}
        )
        assert result.post_selects[1] == PostSelect(
            output_variable="acquire2", additional_disallowed={0}
        )

    def test_extract_with_acquisition_type_gives_raw_acquire_mode(self):
        """Tests that the analysis correctly handles an extract operation with an
        acquisition type."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", AcquisitionType())
        rec = CreateRecordOp(["acquire1"], [ext1.result])
        body.add_ops([ext1, rec, YieldOp(rec.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        result = extract_post_processing_instructions(module, (1000,))

        assert result.acquire_data["acquire1"].mode == AcquireMode.RAW

    def test_create_results_array_with_no_use_gives_assign_with_random_name(self):
        """Tests that the analysis correctly handles a create results array operation with
        no uses, giving an assign with a random name."""
        body = Block(arg_types=[RecordType()])
        record_arg = body.args[0]
        ext1 = ExtractOp(record_arg, "acquire1", IQResultType())
        ext2 = ExtractOp(record_arg, "acquire2", IQResultType())
        arr_op = CreateResultsArrayOp([ext1.result, ext2.result])
        rec = CreateRecordOp([], [])
        body.add_ops([ext1, ext2, arr_op, rec, YieldOp(rec.result)])

        collection = _MockCollectionOp()
        map_op = MapOp(collection.res, body)
        module = _build_module(collection, map_op, func.ReturnOp())

        result = extract_post_processing_instructions(module, (1000,))

        assert len(result.assigns) == 1
        assign_name = result.assigns[0].name
        assert assign_name.startswith("_temp_")
