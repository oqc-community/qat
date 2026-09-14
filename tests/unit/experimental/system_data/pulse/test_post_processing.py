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
from qat.experimental.system_data.pulse.post_processing import (
    UNMAPPED_STATE_LABEL,
    DiscriminateData,
    PostProcessingView,
)


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


def _max_likelihood(
    states: dict[int, complex], p_min: float = 0.0
) -> MaxLikelihoodMethodData:
    return MaxLikelihoodMethodData(
        p_min=p_min,
        states=tuple(
            (k, MaxLikelihoodDiscriminateParams(location=v)) for k, v in states.items()
        ),
    )


class TestPostProcessingDerive:
    def test_empty_system_returns_empty_mapping(self):
        pp = PostProcessingView.derive(CanonicalSystemData())
        assert pp.channel_to_disallowed_states == {}
        assert pp.channel_to_discriminate_data == {}

    def test_mode_with_no_post_process_method_excluded(self):
        pp = PostProcessingView.derive(_make_system_data(post_process_method=None))
        assert pp.channel_to_disallowed_states == {}
        assert pp.channel_to_discriminate_data == {}

    def test_mode_with_linear_map_method_excluded(self):
        pp = PostProcessingView.derive(
            _make_system_data(post_process_method=LinearMapToRealMethodData())
        )
        assert pp.channel_to_disallowed_states == {}
        assert pp.channel_to_discriminate_data == {}

    def test_mode_with_max_likelihood_no_negative_keys_excluded(self):
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j})
        pp = PostProcessingView.derive(_make_system_data(post_process_method=method))
        assert pp.channel_to_disallowed_states == {}
        assert pp.channel_to_discriminate_data == {
            "ch0": DiscriminateData(
                noise_est=1.0, p_min=0.0, state_centroids=(1 + 0j, -1 + 0j)
            )
        }

    def test_mode_with_single_negative_key(self):
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j, -1: 0 + 0j})
        pp = PostProcessingView.derive(
            _make_system_data(channel_id="ch0", post_process_method=method)
        )
        assert pp.channel_to_disallowed_states == {"ch0": {2}}
        assert pp.channel_to_discriminate_data == {
            "ch0": DiscriminateData(
                noise_est=1.0, p_min=0.0, state_centroids=(1 + 0j, -1 + 0j, 0 + 0j)
            )
        }

    def test_mode_with_multiple_negative_keys(self):
        method = _max_likelihood({0: 1 + 0j, -1: 0 + 0j, -2: 0.5 + 0j})
        pp = PostProcessingView.derive(
            _make_system_data(channel_id="ch0", post_process_method=method)
        )
        assert pp.channel_to_disallowed_states == {"ch0": {1, 2}}

    @pytest.mark.parametrize(
        "states, expected",
        [
            ({-1: 0 + 0j, 0: 1 + 0j, 1: -1 + 0j}, {0}),
            ({0: 1 + 0j, -1: 0 + 0j, 1: -1 + 0j}, {1}),
            ({0: 1 + 0j, 1: -1 + 0j, -1: 0 + 0j}, {2}),
        ],
    )
    def test_negative_key_maps_to_its_calibration_position(self, states, expected):
        """The emitted label is the centroid's position, wherever the key sits."""
        pp = PostProcessingView.derive(
            _make_system_data(post_process_method=_max_likelihood(states))
        )
        assert pp.channel_to_disallowed_states == {"ch0": expected}

    def test_non_zero_p_min_disallows_the_unmapped_label(self):
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j}, p_min=0.6)
        pp = PostProcessingView.derive(_make_system_data(post_process_method=method))
        assert pp.channel_to_disallowed_states == {"ch0": {UNMAPPED_STATE_LABEL}}

    def test_non_zero_p_min_combines_with_negative_keys(self):
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j, -1: 0 + 0j}, p_min=0.6)
        pp = PostProcessingView.derive(_make_system_data(post_process_method=method))
        assert pp.channel_to_disallowed_states == {"ch0": {UNMAPPED_STATE_LABEL, 2}}

    def test_zero_p_min_leaves_the_unmapped_label_alone(self):
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j}, p_min=0.0)
        pp = PostProcessingView.derive(_make_system_data(post_process_method=method))
        assert pp.channel_to_disallowed_states == {}

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
        pp = PostProcessingView.derive(
            CanonicalSystemData(channels=channels, qubits=qubits)
        )
        assert pp.channel_to_disallowed_states == {"ch0": {1}}

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
        pp = PostProcessingView.derive(
            CanonicalSystemData(channels=channels, qubits=qubits)
        )
        assert pp.channel_to_disallowed_states == {
            "ch0": {1},
            "ch1": {1},
        }
        assert pp.channel_to_discriminate_data == {
            "ch0": DiscriminateData(
                noise_est=1.0, p_min=0.0, state_centroids=(1 + 0j, 0 + 0j)
            ),
            "ch1": DiscriminateData(
                noise_est=1.0, p_min=0.0, state_centroids=(1 + 0j, 0 + 0j)
            ),
        }

    def test_modes_sharing_a_channel_without_calibration_are_ignored(self):
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j, -1: 0 + 0j})
        measure = ModeData(id="q0/measure", channel_id="ch0")
        acquire = ModeData(id="q0/acquire", channel_id="ch0", post_process_method=method)
        qubits = (QubitData(id="q0", index=0, modes=(measure, acquire)),)
        channels = (ChannelData(id="ch0", port_id="port0", frequency=8_800_000_000),)

        pp = PostProcessingView.derive(
            CanonicalSystemData(channels=channels, qubits=qubits)
        )

        assert pp.channel_to_discriminate_data == {
            "ch0": DiscriminateData(
                noise_est=1.0, p_min=0.0, state_centroids=(1 + 0j, -1 + 0j, 0 + 0j)
            )
        }
        # The uncalibrated mode contributes nothing, so positions come from the calibrated
        # mode alone and the negative key still resolves to its own centroid position.
        assert pp.channel_to_disallowed_states == {"ch0": {2}}

    def test_channel_shared_by_uncalibrated_modes_only_is_omitted(self):
        drive = ModeData(id="q0/drive", channel_id="ch0")
        measure = ModeData(id="q0/measure", channel_id="ch0")
        qubits = (QubitData(id="q0", index=0, modes=(drive, measure)),)
        channels = (ChannelData(id="ch0", port_id="port0", frequency=8_800_000_000),)

        pp = PostProcessingView.derive(
            CanonicalSystemData(channels=channels, qubits=qubits)
        )

        assert pp.channel_to_discriminate_data == {}
        assert pp.channel_to_disallowed_states == {}

    def test_modes_sharing_a_channel_with_identical_method_collapse(self):
        method = _max_likelihood({0: 1 + 0j, -1: -0.2 + 0j})
        mode0 = ModeData(id="q0/acquire", channel_id="ch0", post_process_method=method)
        mode1 = ModeData(
            id="q0/readout_acquire", channel_id="ch0", post_process_method=method
        )
        qubits = (QubitData(id="q0", index=0, modes=(mode0, mode1)),)
        channels = (ChannelData(id="ch0", port_id="port0", frequency=8_800_000_000),)

        pp = PostProcessingView.derive(
            CanonicalSystemData(channels=channels, qubits=qubits)
        )

        assert pp.channel_to_discriminate_data == {
            "ch0": DiscriminateData(
                noise_est=1.0, p_min=0.0, state_centroids=(1 + 0j, -0.2 + 0j)
            )
        }

    @pytest.mark.parametrize("linear_first", [True, False])
    def test_linear_map_mode_does_not_mask_a_shared_max_likelihood_mode(self, linear_first):
        """A channel bound by both calibrations resolves to the max-likelihood one.

        Max-likelihood is the only discrimination policy this view can express. A mode
        carrying a linear map must therefore neither claim the channel nor displace the max-
        likelihood mode sharing it, whichever order the modes are declared in.
        """
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j, -1: 0 + 0j})
        linear = ModeData(
            id="q0/acquire",
            channel_id="ch0",
            post_process_method=LinearMapToRealMethodData(),
        )
        max_likelihood = ModeData(
            id="q0/readout_acquire", channel_id="ch0", post_process_method=method
        )
        modes = (linear, max_likelihood) if linear_first else (max_likelihood, linear)
        qubits = (QubitData(id="q0", index=0, modes=modes),)
        channels = (ChannelData(id="ch0", port_id="port0", frequency=8_800_000_000),)

        pp = PostProcessingView.derive(
            CanonicalSystemData(channels=channels, qubits=qubits)
        )

        assert pp.channel_to_discriminate_data == {
            "ch0": DiscriminateData(
                noise_est=1.0, p_min=0.0, state_centroids=(1 + 0j, -1 + 0j, 0 + 0j)
            )
        }
        assert pp.channel_to_disallowed_states == {"ch0": {2}}

    @pytest.mark.parametrize("linear_first", [True, False])
    def test_linear_map_mode_on_another_qubit_does_not_mask_the_shared_channel(
        self, linear_first
    ):
        """Qubits multiplexed onto one channel resolve it the same way modes do."""
        method = _max_likelihood({0: 1 + 0j, -1: 0 + 0j})
        linear_qubit = QubitData(
            id="q0",
            index=0,
            modes=(
                ModeData(
                    id="q0/acquire",
                    channel_id="ch0",
                    post_process_method=LinearMapToRealMethodData(),
                ),
            ),
        )
        max_likelihood_qubit = QubitData(
            id="q1",
            index=1,
            modes=(
                ModeData(id="q1/acquire", channel_id="ch0", post_process_method=method),
            ),
        )
        qubits = (
            (linear_qubit, max_likelihood_qubit)
            if linear_first
            else (max_likelihood_qubit, linear_qubit)
        )
        channels = (ChannelData(id="ch0", port_id="port0", frequency=8_800_000_000),)

        pp = PostProcessingView.derive(
            CanonicalSystemData(channels=channels, qubits=qubits)
        )

        assert pp.channel_to_discriminate_data == {
            "ch0": DiscriminateData(
                noise_est=1.0, p_min=0.0, state_centroids=(1 + 0j, 0 + 0j)
            )
        }
        assert pp.channel_to_disallowed_states == {"ch0": {1}}

    def test_differing_linear_map_modes_sharing_a_channel_do_not_raise(self):
        """Linear maps are not a calibration this view chooses between, so they cannot
        conflict here; the channel is simply left undiscriminated."""
        mode0 = ModeData(
            id="q0/acquire",
            channel_id="ch0",
            post_process_method=LinearMapToRealMethodData(mean_z_map_args=(1 + 0j, 0j)),
        )
        mode1 = ModeData(
            id="q0/readout_acquire",
            channel_id="ch0",
            post_process_method=LinearMapToRealMethodData(
                mean_z_map_args=(0 + 1j, 0.5 + 0j)
            ),
        )
        qubits = (QubitData(id="q0", index=0, modes=(mode0, mode1)),)
        channels = (ChannelData(id="ch0", port_id="port0", frequency=8_800_000_000),)

        pp = PostProcessingView.derive(
            CanonicalSystemData(channels=channels, qubits=qubits)
        )

        assert pp.channel_to_discriminate_data == {}
        assert pp.channel_to_disallowed_states == {}

    def test_modes_sharing_a_channel_with_different_methods_raises(self):
        method0 = _max_likelihood({0: 1 + 0j, -1: -0.2 + 0j})
        method1 = _max_likelihood({0: 0.7 + 0.1j, 1: -0.9 + 0.0j})
        mode0 = ModeData(id="q0/acquire_a", channel_id="ch0", post_process_method=method0)
        mode1 = ModeData(id="q0/acquire_b", channel_id="ch0", post_process_method=method1)
        qubits = (QubitData(id="q0", index=0, modes=(mode0, mode1)),)
        channels = (ChannelData(id="ch0", port_id="port0", frequency=8_800_000_000),)

        with pytest.raises(ValueError, match="different max-likelihood methods"):
            PostProcessingView.derive(CanonicalSystemData(channels=channels, qubits=qubits))

    @pytest.mark.parametrize(
        "transform, offset",
        [
            (((1.0, 0.0), (0.0, 1.0)), (0.0, 0.0)),
            (((0.0, -1.0), (1.0, 0.0)), (0.2, -0.3)),
        ],
    )
    def test_affine_pre_transform_is_rejected(self, transform, offset):
        """A pre-transform must not be silently dropped.

        The centroids of such a method are calibrated in the transformed frame, so
        discriminating without applying the transform would classify in the wrong frame.
        Identity is rejected too: it is indistinguishable here from a calibration whose
        transform simply has not been fitted yet.
        """
        method = MaxLikelihoodMethodData(
            states=((0, MaxLikelihoodDiscriminateParams(location=1 + 0j)),),
            transform=transform,
            offset=offset,
        )
        with pytest.raises(NotImplementedError, match="affine IQ pre-transform"):
            PostProcessingView.derive(_make_system_data(post_process_method=method))

    def test_no_affine_pre_transform_is_accepted(self):
        """The fields default to None, which is every calibration in use today."""
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j})
        assert method.transform is None and method.offset is None
        pp = PostProcessingView.derive(_make_system_data(post_process_method=method))
        assert pp.channel_to_discriminate_data == {
            "ch0": DiscriminateData(
                noise_est=1.0, p_min=0.0, state_centroids=(1 + 0j, -1 + 0j)
            )
        }

    def test_discriminate_data_drops_labels(self):
        """Labels are calibration metadata only; the emitted label is the position."""
        method = MaxLikelihoodMethodData(
            states=(
                (0, MaxLikelihoodDiscriminateParams(location=1 + 0j, label="g")),
                (1, MaxLikelihoodDiscriminateParams(location=-1 + 0j, label="e")),
            )
        )
        pp = PostProcessingView.derive(_make_system_data(post_process_method=method))
        assert pp.channel_to_discriminate_data == {
            "ch0": DiscriminateData(
                noise_est=1.0, p_min=0.0, state_centroids=(1 + 0j, -1 + 0j)
            )
        }


class TestPostProcessingLookup:
    @pytest.fixture
    def post_processing(self) -> PostProcessingView:
        method = _max_likelihood({0: 1 + 0j, 1: -1 + 0j, -1: 0 + 0j})
        return PostProcessingView.derive(
            _make_system_data(channel_id="ch0", post_process_method=method)
        )

    def test_known_channel_returns_disallowed_states(self, post_processing):
        assert post_processing.disallowed_states_for_channel("ch0") == {2}

    def test_unknown_channel_returns_empty_set(self, post_processing):
        assert post_processing.disallowed_states_for_channel("unknown") == set()

    def test_discriminate_data_lookup(self, post_processing):
        entry = post_processing.channel_to_discriminate_data["ch0"]
        assert entry.noise_est == 1.0
        assert entry.p_min == 0.0
        assert entry.state_centroids == (1 + 0j, -1 + 0j, 0 + 0j)

    def test_discriminate_data_lookup_unknown_channel(self, post_processing):
        assert "unknown" not in post_processing.channel_to_discriminate_data
