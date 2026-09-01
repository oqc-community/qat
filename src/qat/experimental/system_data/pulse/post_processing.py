# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from collections import defaultdict
from dataclasses import dataclass

from qat.experimental.system_data.canonical.schema import (
    CanonicalSystemData,
    MaxLikelihoodMethodData,
    ModeData,
)


def _disallowed_states_for_mode(mode: ModeData) -> set[int]:
    """Extracts the disallowed integer state labels for a mode.

    Only :class:`MaxLikelihoodMethodData` carries explicit integer state labels.
    Negative keys in ``states`` are the disallowed/background states per the schema
    convention. :class:`LinearMapToRealMethodData` produces a thresholded real value
    rather than a discrete state key, so it contributes no disallowed states.

    :param mode: The mode to extract disallowed states from.
    :returns: The set of integer state labels that should be post-selected out.
    """
    match mode.post_process_method:
        case MaxLikelihoodMethodData(states=states):
            return {key for key, _ in states if key < 0}
        case _:
            return set()


@dataclass(frozen=True, slots=True, kw_only=True)
class PostProcessing:
    """Post-selection configuration derived from a :class:`CanonicalSystemData`.

    Provides a mapping from logical channel identifier to the set of integer
    state labels that should be discarded during post-selection. Only channels whose mode
    has at least one disallowed state are included.

    :ivar channel_to_disallowed_states: Mapping from logical channel identifier
        (``mode.channel_id`` in the canonical schema) to the set of integer state
        labels that should be post-selected out.
    :ivar known_channel_ids: All channel IDs present in the canonical system data.
    """

    channel_to_disallowed_states: dict[str, set[int]]
    known_channel_ids: frozenset[str]

    @classmethod
    def derive(cls, parent: CanonicalSystemData) -> "PostProcessing":
        """Builds the post-selection configuration from a canonical hardware model.

        Walks all qubits and their modes and collects disallowed integer state labels
        keyed directly by ``mode.channel_id``. Channels with no disallowed states are
        omitted.

        :param parent: The canonical hardware model to derive post-selection from.
        :returns: The derived post-selection configuration for the system.
        """
        channel_to_disallowed_states: dict[str, set[int]] = defaultdict(set)
        for qubit in parent.qubits:
            for mode in qubit.modes:
                if disallowed := _disallowed_states_for_mode(mode):
                    channel_to_disallowed_states[mode.channel_id].update(disallowed)
        return cls(
            channel_to_disallowed_states=dict(channel_to_disallowed_states),
            known_channel_ids=frozenset(ch.id for ch in parent.channels),
        )

    def disallowed_states_for_channel(self, channel_id: str) -> set[int]:
        """Returns the disallowed integer state labels for a given logical channel.

        :param channel_id: The logical channel identifier to look up.
        :returns: The set of disallowed integer state labels, or an empty set if the channel
            has no disallowed states.
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
