# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import warnings

from xdsl.ir import SSAValue

from qat.experimental.dialect.pulse.ir import (
    DiscriminatorPolicyAttr,
    MaximumLikelihoodPolicyAttr,
)
from qat.experimental.dialect.results.ir import PostSelectOp, ResultsCollectionType
from qat.experimental.dialect.results.ir.attributes import IntegerStatePredicateAttr
from qat.experimental.system_data.pulse.post_processing import PostProcessingView


class PostProcessingFactory:
    """Creates the results post-processing IR described by a derived
    :class:`PostProcessingView`.

    * **Discrimination**, applied per acquisition inside the results map, turning an IQ
      value into an integer state label by picking that channel's nearest calibrated
      centroid.
    * **Post-selection**, applied once to the whole discriminated collection, discarding
      shots whose emitted state labels are disallowed for their channel.

    :param post_processing: Derived post-processing view.
    :param post_selection_enabled: Whether post-selection is enabled. When ``False``,
        :meth:`post_select` returns the passed collection unchanged.
    """

    def __init__(
        self,
        post_processing: PostProcessingView,
        post_selection_enabled: bool = True,
    ):
        """Initialise the post-processing factory.

        :param post_processing: Derived post-processing view.
        :param post_selection_enabled: Whether post-selection is enabled. Defaults to
            ``True``.
        """
        self._post_processing = post_processing
        self._post_selection_enabled = post_selection_enabled

    def policy_for(self, channel_id: str | None) -> DiscriminatorPolicyAttr | None:
        """Resolve the discrimination policy calibrated for a logical channel.

        :param channel_id: The logical channel identifier (``mode.channel_id``) the
            acquisition was made on, or ``None`` if it could not be resolved.
        :returns: A :class:`MaximumLikelihoodPolicyAttr` built from the channel's
            calibration, or ``None`` when the channel has no max-likelihood calibration.
        """
        if channel_id is None:
            return None

        entry = self._post_processing.channel_to_discriminate_data.get(channel_id)
        if entry is None:
            return None

        if not entry.state_centroids:
            return None

        return MaximumLikelihoodPolicyAttr(
            list(entry.state_centroids),
            noise_estimate=entry.noise_est,
            p_min=entry.p_min,
        )

    def post_select(
        self,
        collection: SSAValue[ResultsCollectionType],
        label_to_channel: dict[str, str],
    ) -> SSAValue[ResultsCollectionType] | PostSelectOp:
        """Apply post-selection to a discriminated results collection.

        :param collection: The discriminated :class:`ResultsCollectionType` SSA value to
            filter, typically ``map_op.result``.
        :param label_to_channel: Mapping from acquire output-variable name to the logical
            channel identifier (``mode.channel_id``) for that acquisition. Built by the
            importer from the purr IR before the kernel is emitted.
        :returns: The unmodified ``collection`` if post-selection is disabled or there are
            no disallowed states. Otherwise a :class:`PostSelectOp` wrapping the filtered
            collection.
        """
        if not self._post_selection_enabled:
            return collection
        return _build_post_select_op(collection, self._post_processing, label_to_channel)


def _build_post_select_op(
    collection: SSAValue[ResultsCollectionType],
    post_processing: PostProcessingView,
    label_to_channel: dict[str, str],
) -> SSAValue[ResultsCollectionType] | PostSelectOp:
    """Build a :class:`PostSelectOp` wrapping a discriminated results collection.

    Resolves disallowed states for each acquire label via its logical channel identifier.

    :param collection: The discriminated results collection SSA value to filter.
    :param post_processing: Derived post-processing view keyed on logical channel id.
    :param label_to_channel: Mapping from acquire output-variable name to logical channel
        identifier.
    :returns: A :class:`PostSelectOp` if any acquire key has disallowed states, otherwise
        the original ``collection`` SSA value unchanged.
    """
    predicates: list[IntegerStatePredicateAttr] = []
    for acquire_key, channel_id in label_to_channel.items():
        disallowed = post_processing.disallowed_states_for_channel(channel_id)
        if disallowed:
            predicates.append(IntegerStatePredicateAttr(acquire_key, sorted(disallowed)))

    channel_ids = list(label_to_channel.values())
    if channel_ids:
        unmatched = post_processing.unmatched_channels(channel_ids)
        if unmatched:
            warnings.warn(
                "Post-selection is enabled but the following acquire channel IDs "
                "were not found in the post-processing system data. "
                f"Unmatched channels: {unmatched}. "
                f"Known channels: "
                f"{sorted(post_processing.known_channel_ids)}.",
                UserWarning,
                stacklevel=3,
            )

    if not predicates:
        return collection

    return PostSelectOp(collection, *predicates)
