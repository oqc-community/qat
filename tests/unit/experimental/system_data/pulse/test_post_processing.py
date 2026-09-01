# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.canonical.schema import (
    CanonicalSystemData,
    ChannelData,
    LinearMapToRealMethodData,
    MaxLikelihoodDiscriminateParams,
    MaxLikelihoodMethodData,
    ModeData,
    QubitData,
)
from qat.experimental.system_data.pulse.post_processing import PostProcessing


def _make_system_data(
    channel_id: str = "ch0",
    port_id: str = "port0",
    mode_id: str = "q0/acquire",
    post_process_method=None,
) -> CanonicalSystemData:
    mode = ModeData(
        id=mode_id, channel_id=channel_id, post_process_method=post_process_method
    )
    qubit = QubitData(id="q0", index=0, modes=(mode,))
    channel = ChannelData(id=channel_id, port_id=port_id, frequency=8_800_000_000)
    return CanonicalSystemData(channels=(channel,), qubits=(qubit,))


def _max_likelihood(states: dict[int, complex]) -> MaxLikelihoodMethodData:
    return MaxLikelihoodMethodData(
        states=tuple(
            (k, MaxLikelihoodDiscriminateParams(location=v)) for k, v in states.items()
        )
    )


class TestPostProcessingDerive:
    def test_empty_system_returns_empty_mapping(self):
        pp = PostProcessing.derive(CanonicalSystemData())
        assert pp.channel_to_disallowed_states == {}

    def test_mode_with_no_post_process_method_excluded(self):
        pp = PostProcessing.derive(_make_system_data(post_process_method=None))
        assert pp.channel_to_disallowed_states == {}

    def test_mode_with_linear_map_method_excluded(self):
        pp = PostProcessing.derive(
            _make_system_data(post_process_method=LinearMapToRealMethodData())
        )
        assert pp.channel_to_disallowed_states == {}

    def test_mode_with_max_likelihood_no_negative_keys_excluded(self):
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j})
        pp = PostProcessing.derive(_make_system_data(post_process_method=method))
        assert pp.channel_to_disallowed_states == {}

    def test_mode_with_single_negative_key(self):
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j, -1: 0 + 0j})
        pp = PostProcessing.derive(
            _make_system_data(channel_id="ch0", post_process_method=method)
        )
        assert pp.channel_to_disallowed_states == {"ch0": {-1}}

    def test_mode_with_multiple_negative_keys(self):
        method = _max_likelihood({0: 1 + 0j, -1: 0 + 0j, -2: 0.5 + 0j})
        pp = PostProcessing.derive(
            _make_system_data(channel_id="ch0", post_process_method=method)
        )
        assert pp.channel_to_disallowed_states == {"ch0": {-1, -2}}

    def test_multiple_qubits_only_channels_with_disallowed_states_included(self):
        method = _max_likelihood({0: 1 + 0j, -1: 0 + 0j})
        mode0 = ModeData(id="q0/acquire", channel_id="ch0", post_process_method=method)
        mode1 = ModeData(id="q1/acquire", channel_id="ch1")
        qubits = (
            QubitData(id="q0", index=0, modes=(mode0,)),
            QubitData(id="q1", index=1, modes=(mode1,)),
        )
        channels = (
            ChannelData(id="ch0", port_id="port0", frequency=8_800_000_000),
            ChannelData(id="ch1", port_id="port1", frequency=8_900_000_000),
        )
        pp = PostProcessing.derive(CanonicalSystemData(channels=channels, qubits=qubits))
        assert pp.channel_to_disallowed_states == {"ch0": {-1}}

    def test_multiple_qubits_both_channels_with_disallowed_states(self):
        method = _max_likelihood({0: 1 + 0j, -1: 0 + 0j})
        mode0 = ModeData(id="q0/acquire", channel_id="ch0", post_process_method=method)
        mode1 = ModeData(id="q1/acquire", channel_id="ch1", post_process_method=method)
        qubits = (
            QubitData(id="q0", index=0, modes=(mode0,)),
            QubitData(id="q1", index=1, modes=(mode1,)),
        )
        channels = (
            ChannelData(id="ch0", port_id="port0", frequency=8_800_000_000),
            ChannelData(id="ch1", port_id="port1", frequency=8_900_000_000),
        )
        pp = PostProcessing.derive(CanonicalSystemData(channels=channels, qubits=qubits))
        assert pp.channel_to_disallowed_states == {
            "ch0": {-1},
            "ch1": {-1},
        }


class TestPostProcessingLookup:
    @pytest.fixture
    def post_processing(self) -> PostProcessing:
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j, -1: 0 + 0j})
        return PostProcessing.derive(
            _make_system_data(channel_id="ch0", post_process_method=method)
        )

    def test_known_channel_returns_disallowed_states(self, post_processing):
        assert post_processing.disallowed_states_for_channel("ch0") == {-1}

    def test_unknown_channel_returns_empty_set(self, post_processing):
        assert post_processing.disallowed_states_for_channel("unknown") == set()
