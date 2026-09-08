# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from dataclasses import dataclass

from qat.experimental.system_data.canonical.schema import (
    CanonicalSystemData,
    MaxLikelihoodMethodData,
)

UNMAPPED_STATE_LABEL = -1
"""The state label a maximum-likelihood discriminator emits when the winning normalised
likelihood falls below ``p_min``, i.e. when no centroid can be assigned confidently."""


@dataclass(frozen=True, slots=True)
class DiscriminateData:
    """The max-likelihood payload a channel is calibrated with.

    :ivar noise_est: Global Gaussian noise variance used in likelihood evaluation.
    :ivar p_min: Minimum normalised likelihood required for acceptance. When non-zero, a
        shot whose winning likelihood falls below it is emitted as
        :data:`UNMAPPED_STATE_LABEL`.
    :ivar state_centroids: The calibrated IQ-plane centroids, in calibration order. A
        discriminator labels a shot by the **position** of the centroid it is nearest.
    """

    noise_est: float
    p_min: float
    state_centroids: tuple[complex, ...]


def _discriminate_data(method: MaxLikelihoodMethodData) -> DiscriminateData:
    """Extract the discriminate payload for a calibrated max-likelihood method.

    Every calibrated centroid is kept, in calibration order. Discrimination assigns an
    IQ value to whichever centroid it is nearest.

    :param method: The channel's calibrated max-likelihood method.
    :returns: The channel's :class:`DiscriminateData`.
    """
    return DiscriminateData(
        noise_est=method.noise_est,
        p_min=method.p_min,
        state_centroids=tuple(params.location for _, params in method.states),
    )


def _disallowed_labels(method: MaxLikelihoodMethodData) -> set[int]:
    """Map a calibrated method onto the state labels post-selection must discard.

    A discriminator built from this method labels its centroids **positionally**: the
    ``i``-th centroid in calibration order is emitted as label ``i``.

    When ``p_min`` is non-zero the discriminator can additionally decline to classify a
    shot, emitting :data:`UNMAPPED_STATE_LABEL`. Those shots are ambiguous by
    construction and are always discarded.

    :param method: The channel's calibrated max-likelihood method.
    :returns: The set of emitted state labels that should be post-selected out.
    """
    labels = {index for index, (key, _) in enumerate(method.states) if key < 0}
    if method.p_min > 0.0:
        labels.add(UNMAPPED_STATE_LABEL)
    return labels


@dataclass(frozen=True, slots=True, kw_only=True)
class PostProcessingView:
    """Post-processing configuration derived from a :class:`CanonicalSystemData`.

    A discriminator only assigns an IQ value to its nearest calibrated centroid,
    labelling centroids by their position in calibration order; it carries no notion
    of which outcomes are wanted.
    The calibrated integer keys, which encode exactly that, are resolved here into the
    emitted labels post-selection must discard.

    :ivar channel_to_disallowed_states: Mapping from logical channel identifier
        (``mode.channel_id`` in the canonical schema) to the set of **emitted state
        labels** that should be post-processed out. These are positions in the
        ``channel_to_discriminate_data`` centroid ordering, not the calibrated keys they
        were derived from, plus :data:`UNMAPPED_STATE_LABEL` when ``p_min`` is non-zero.
        Channels with nothing to discard are omitted.
    :ivar channel_to_discriminate_data: Mapping from logical channel identifier
        (``mode.channel_id`` in the canonical schema) to the single
        :class:`DiscriminateData` calibrated for that channel. Present for every channel
        carrying a max-likelihood calibration, independently of whether that channel has
        anything to post-select out.
    :ivar known_channel_ids: All channel IDs present in the canonical system data.
    """

    channel_to_disallowed_states: dict[str, set[int]]
    known_channel_ids: frozenset[str]
    channel_to_discriminate_data: dict[str, DiscriminateData]

    @classmethod
    def derive(cls, parent: CanonicalSystemData) -> "PostProcessingView":
        """Builds the post-processing configuration from a canonical hardware model.

        Walks all qubits and their modes, keyed directly by ``mode.channel_id``. Several
        modes may share a channel, and a channel resolves to exactly one max-likelihood
        method: modes sharing a channel must carry structurally identical calibration,
        otherwise there is no basis for choosing between them and this raises. Disallowed
        labels are aligned with the centroid ordering they index into.

        Channels with no max-likelihood calibration are absent from both mappings, as
        there is nothing to discriminate against. Every channel that *does* carry a
        calibration gets a discriminate payload, whether or not it has anything to
        post-select out; only :attr:`channel_to_disallowed_states` additionally omits
        channels whose disallowed-label set is empty.

        :param parent: The canonical hardware model to derive post-processing from.
        :returns: The derived post-processing configuration for the system.
        :raises ValueError: If modes sharing a channel carry different max-likelihood
            methods.
        """
        channel_to_method: dict[str, MaxLikelihoodMethodData] = {}
        for qubit in parent.qubits:
            for mode in qubit.modes:
                method = mode.post_process_method
                if not isinstance(method, MaxLikelihoodMethodData):
                    continue
                existing = channel_to_method.setdefault(mode.channel_id, method)
                if existing != method:
                    raise ValueError(
                        f"Modes on channel '{mode.channel_id}' carry different "
                        "max-likelihood methods."
                    )

        return cls(
            channel_to_disallowed_states={
                channel_id: labels
                for channel_id, method in channel_to_method.items()
                if (labels := _disallowed_labels(method))
            },
            channel_to_discriminate_data={
                channel_id: _discriminate_data(method)
                for channel_id, method in channel_to_method.items()
            },
            known_channel_ids=frozenset(ch.id for ch in parent.channels),
        )

    def disallowed_states_for_channel(self, channel_id: str) -> set[int]:
        """Returns the disallowed emitted state labels for a given logical channel.

        :param channel_id: The logical channel identifier to look up.
        :returns: The set of disallowed state labels, or an empty set if the channel has
            nothing to post-select out.
        """
        return self.channel_to_disallowed_states.get(channel_id, set())

    def unmatched_channels(self, channel_ids: list[str]) -> list[str]:
        """Return the channel IDs that are not known to the system data at all.

        A channel is considered *unmatched* only when it is entirely absent from the
        canonical system data (i.e. not in :attr:`known_channel_ids`).

        :param channel_ids: The logical channel identifiers to check.
        :returns: A list of channel IDs from *channel_ids* that are absent from
            :attr:`known_channel_ids`. Empty if all are present.
        """
        return [ch for ch in channel_ids if ch not in self.known_channel_ids]
