# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import warnings

import pytest

from qat.experimental.dialect.pulse.ir import MaximumLikelihoodPolicyAttr
from qat.experimental.dialect.results.ir import PostSelectOp, RecordSchemaAttr
from qat.experimental.dialect.results.ir.ops import CreateOp
from qat.experimental.frontend.importer.pulse.post_processing import (
    PostProcessingFactory,
    _build_post_select_op,
)
from qat.experimental.system_data.pulse.post_processing import (
    DiscriminateData,
    PostProcessingView,
)


def _make_discriminate_data(
    noise_est: float = 1.0,
    p_min: float = 0.0,
    centroids: tuple[complex, ...] = (),
) -> DiscriminateData:
    return DiscriminateData(noise_est=noise_est, p_min=p_min, state_centroids=centroids)


def _make_post_processing(
    channel_id: str = "ch0",
    disallowed_states: set[int] = frozenset({-1}),
    discriminate_data: DiscriminateData | None = None,
) -> PostProcessingView:
    return PostProcessingView(
        channel_to_disallowed_states=(
            {channel_id: set(disallowed_states)} if disallowed_states else {}
        ),
        known_channel_ids=frozenset({channel_id}),
        channel_to_discriminate_data=(
            {channel_id: discriminate_data} if discriminate_data is not None else {}
        ),
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
        pp = PostProcessingView(
            channel_to_disallowed_states={
                "ch0": {-1},
                "ch1": {-1},
            },
            known_channel_ids=frozenset({"ch0", "ch1"}),
            channel_to_discriminate_data={},
        )
        collection = _make_collection_ssa_value()
        op = _build_post_select_op(collection, pp, {"meas0": "ch0", "meas1": "ch1"})
        assert isinstance(op, PostSelectOp)
        assert len(op.predicates.data) == 2


class TestPostProcessingFactoryPostSelect:
    def test_disabled_returns_collection_unchanged(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()
        factory = PostProcessingFactory(pp, post_selection_enabled=False)
        result = factory.post_select(collection, {"meas0": "ch0"})
        assert result is collection

    def test_enabled_no_disallowed_states_returns_collection_unchanged(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset())
        collection = _make_collection_ssa_value()
        factory = PostProcessingFactory(pp, post_selection_enabled=True)
        result = factory.post_select(collection, {"meas0": "ch0"})
        assert result is collection

    def test_enabled_with_disallowed_states_returns_post_select_op(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()
        factory = PostProcessingFactory(pp, post_selection_enabled=True)
        result = factory.post_select(collection, {"meas0": "ch0"})
        assert isinstance(result, PostSelectOp)

    def test_enabled_true_by_default(self):
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        factory = PostProcessingFactory(pp)
        collection = _make_collection_ssa_value()
        result = factory.post_select(collection, {"meas0": "ch0"})
        assert isinstance(result, PostSelectOp)

    def test_disabled_skips_all_predicate_checks(self):
        pp = _make_post_processing(
            channel_id="ch0", disallowed_states=frozenset({-1, -2, -3})
        )
        collection = _make_collection_ssa_value()
        factory = PostProcessingFactory(pp, post_selection_enabled=False)
        result = factory.post_select(collection, {"meas0": "ch0"})
        assert result is collection
        assert not isinstance(result, PostSelectOp)

    def test_channel_id_mismatch_emits_user_warning(self):
        """When acquires are present but no channel IDs match the post-processing data, a
        UserWarning is raised to surface the likely ID mismatch."""
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()
        factory = PostProcessingFactory(pp, post_selection_enabled=True)
        with pytest.warns(UserWarning, match="Unmatched channels"):
            result = factory.post_select(collection, {"meas0": "ch_unknown"})
        assert result is collection

    def test_no_warning_when_label_to_channel_is_empty(self):
        """No warning is emitted when there are simply no acquires to match."""
        pp = _make_post_processing(channel_id="ch0", disallowed_states=frozenset({-1}))
        collection = _make_collection_ssa_value()
        factory = PostProcessingFactory(pp, post_selection_enabled=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = factory.post_select(collection, {})
        assert not any("Unmatched channels" in str(w.message) for w in caught)
        assert result is collection


class TestPostProcessingFactoryPolicyFor:
    """Tests resolving a channel's calibration into a discrimination policy."""

    def test_returns_none_for_none_channel(self):
        pp = _make_post_processing(
            discriminate_data=_make_discriminate_data(1.0, 0.0, (1 + 0j, -1 + 0j))
        )
        assert PostProcessingFactory(pp).policy_for(None) is None

    def test_returns_none_for_uncalibrated_channel(self):
        pp = _make_post_processing(
            channel_id="ch0",
            discriminate_data=_make_discriminate_data(1.0, 0.0, (1 + 0j, -1 + 0j)),
        )
        assert PostProcessingFactory(pp).policy_for("ch_unknown") is None

    def test_returns_none_when_no_discriminate_data(self):
        pp = _make_post_processing(channel_id="ch0")
        assert PostProcessingFactory(pp).policy_for("ch0") is None

    def test_returns_none_when_states_are_empty(self):
        pp = _make_post_processing(
            channel_id="ch0", discriminate_data=_make_discriminate_data(1.0, 0.0, ())
        )
        assert PostProcessingFactory(pp).policy_for("ch0") is None

    def test_returns_maximum_likelihood_policy(self):
        pp = _make_post_processing(
            channel_id="ch0",
            discriminate_data=_make_discriminate_data(0.5, 0.2, (1 + 0j, -1 + 0j)),
        )
        policy = PostProcessingFactory(pp).policy_for("ch0")
        assert isinstance(policy, MaximumLikelihoodPolicyAttr)

    def test_state_centers_preserve_calibration_order(self):
        pp = _make_post_processing(
            channel_id="ch0",
            discriminate_data=_make_discriminate_data(1.0, 0.0, (1 + 0j, -1 + 0j, 0 + 1j)),
        )
        policy = PostProcessingFactory(pp).policy_for("ch0")
        assert [c.data for c in policy.state_centers] == [1 + 0j, -1 + 0j, 0 + 1j]

    def test_disallowed_state_centroids_are_included(self):
        """Centroids for disallowed states must still compete during discrimination.

        Discrimination assigns an IQ value to its nearest centroid; dropping the disallowed
        centroids would silently absorb those shots into the allowed states and leave post-
        selection with nothing to filter.
        """
        pp = PostProcessingView(
            channel_to_disallowed_states={"ch0": {-2}},
            known_channel_ids=frozenset({"ch0"}),
            channel_to_discriminate_data={
                "ch0": _make_discriminate_data(1.0, 0.0, (1 + 0j, -1 + 0j, 0 + 1j))
            },
        )
        policy = PostProcessingFactory(pp).policy_for("ch0")
        assert len(policy.state_centers) == 3
        assert 0 + 1j in [c.data for c in policy.state_centers]

    def test_noise_estimate_and_p_min_are_carried_through(self):
        pp = _make_post_processing(
            channel_id="ch0",
            discriminate_data=_make_discriminate_data(0.5, 0.2, (1 + 0j, -1 + 0j)),
        )
        policy = PostProcessingFactory(pp).policy_for("ch0")
        assert policy.noise_estimate.data == pytest.approx(0.5)
        assert policy.p_min.data == pytest.approx(0.2)

    def test_state_range_spans_unmapped_to_last_state(self):
        pp = _make_post_processing(
            channel_id="ch0",
            discriminate_data=_make_discriminate_data(1.0, 0.0, (1 + 0j, -1 + 0j)),
        )
        policy = PostProcessingFactory(pp).policy_for("ch0")
        assert policy.state_range == (-1, 1)

    def test_policy_is_independent_of_post_selection_being_disabled(self):
        """Discrimination is not optional; disabling post-selection must not remove it."""
        pp = _make_post_processing(
            channel_id="ch0",
            discriminate_data=_make_discriminate_data(1.0, 0.0, (1 + 0j, -1 + 0j)),
        )
        factory = PostProcessingFactory(pp, post_selection_enabled=False)
        assert isinstance(factory.policy_for("ch0"), MaximumLikelihoodPolicyAttr)
