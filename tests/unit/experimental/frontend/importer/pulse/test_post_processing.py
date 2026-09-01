# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import warnings

import pytest

from qat.experimental.dialect.results.ir import PostSelectOp, RecordSchemaAttr
from qat.experimental.dialect.results.ir.ops import CreateOp
from qat.experimental.frontend.importer.pulse.post_processing import (
    PostSelectionBuilder,
    _build_post_select_op,
)
from qat.experimental.system_data.pulse.post_processing import PostProcessing


def _make_post_processing(
    channel_id: str = "ch0",
    disallowed_states: set[int] = frozenset({-1}),
) -> PostProcessing:
    return PostProcessing(
        channel_to_disallowed_states=(
            {channel_id: set(disallowed_states)} if disallowed_states else {}
        ),
        known_channel_ids=frozenset({channel_id}),
    )


def _make_collection_ssa_value():
    return CreateOp.for_empty_collection(RecordSchemaAttr(()), 0).result


class TestBuildPostSelectOp:
    def test_returns_post_select_op_when_disallowed_states_present(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()
        result = _build_post_select_op(collection, pp, {"meas0": "ch0"})
        assert isinstance(result, PostSelectOp)

    def test_returns_collection_unchanged_when_no_disallowed_states(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset())
        collection = _make_collection_ssa_value()
        result = _build_post_select_op(collection, pp, {"meas0": "ch0"})
        assert result is collection

    def test_returns_collection_unchanged_when_no_acquires(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()
        result = _build_post_select_op(collection, pp, {})
        assert result is collection

    def test_returns_collection_unchanged_when_channel_not_in_post_processing(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()

        with pytest.warns(UserWarning, match="Unmatched channels"):
            result = _build_post_select_op(collection, pp, {"meas0": "ch99"})
        assert result is collection

    def test_post_select_op_predicates_contain_acquire_key(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()
        op = _build_post_select_op(collection, pp, {"meas0": "ch0"})
        assert isinstance(op, PostSelectOp)
        predicate_keys = [p.key.data for p in op.predicates.data]
        assert "meas0" in predicate_keys

    def test_post_select_op_predicate_disallowed_states_match(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1, -2}))
        collection = _make_collection_ssa_value()
        op = _build_post_select_op(collection, pp, {"meas0": "ch0"})
        assert isinstance(op, PostSelectOp)
        predicate = next(p for p in op.predicates.data if p.key.data == "meas0")
        disallowed = {s.data for s in predicate.disallowed_values.data}
        assert disallowed == {-1, -2}

    def test_two_acquires_produce_two_predicates(self):
        pp = PostProcessing(
            channel_to_disallowed_states={
                "ch0": {-1},
                "ch1": {-1},
            },
            known_channel_ids=frozenset({"ch0", "ch1"}),
        )
        collection = _make_collection_ssa_value()
        op = _build_post_select_op(collection, pp, {"meas0": "ch0", "meas1": "ch1"})
        assert isinstance(op, PostSelectOp)
        assert len(op.predicates.data) == 2


class TestPostSelectionBuilder:
    def test_disabled_returns_collection_unchanged(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()
        builder = PostSelectionBuilder(pp, enabled=False)
        result = builder.apply(collection, {"meas0": "ch0"})
        assert result is collection

    def test_enabled_no_disallowed_states_returns_collection_unchanged(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset())
        collection = _make_collection_ssa_value()
        builder = PostSelectionBuilder(pp, enabled=True)
        result = builder.apply(collection, {"meas0": "ch0"})
        assert result is collection

    def test_enabled_with_disallowed_states_returns_post_select_op(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()
        builder = PostSelectionBuilder(pp, enabled=True)
        result = builder.apply(collection, {"meas0": "ch0"})
        assert isinstance(result, PostSelectOp)

    def test_enabled_true_by_default(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        builder = PostSelectionBuilder(pp)
        collection = _make_collection_ssa_value()
        result = builder.apply(collection, {"meas0": "ch0"})
        assert isinstance(result, PostSelectOp)

    def test_disabled_skips_all_predicate_checks(self):
        pp = _make_post_processing(
            channel_id="ch0", disallowed_states=frozenset({-1, -2, -3})
        )
        collection = _make_collection_ssa_value()
        builder = PostSelectionBuilder(pp, enabled=False)
        result = builder.apply(collection, {"meas0": "ch0"})
        assert result is collection
        assert not isinstance(result, PostSelectOp)

    def test_channel_id_mismatch_emits_user_warning(self):
        """When acquires are present but no channel IDs match the post-processing data, a
        UserWarning is raised to surface the likely ID mismatch."""
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()
        builder = PostSelectionBuilder(pp, enabled=True)
        with pytest.warns(UserWarning, match="Unmatched channels"):
            result = builder.apply(collection, {"meas0": "ch_unknown"})
        assert result is collection

    def test_no_warning_when_label_to_channel_is_empty(self):
        """No warning is emitted when there are simply no acquires to match."""
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()
        builder = PostSelectionBuilder(pp, enabled=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = builder.apply(collection, {})
        assert not any("Unmatched channels" in str(w.message) for w in caught)
        assert result is collection
