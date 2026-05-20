# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
"""Tests for runtime post-processing utilities.

These cover mean, linear mapping, discrimination and the granular post-processing pipeline
(Equalise, Discriminate, PostSelect, Demap) used by the runtime to convert acquired IQ data
into final classical outputs.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from qat.ir.instruction_basetypes import AcquireMode, PostProcessType, ProcessAxis
from qat.ir.measure import Demap, Discriminate, Equalise, PostProcessing, PostSelect
from qat.model.post_processing import BG_LABEL, MaxLikelihoodMethod, MLStateMap
from qat.runtime.post_processing import (
    apply_demap_instruction,
    apply_discriminate_instruction,
    apply_equalise,
    apply_post_processing,
    apply_post_select,
    discriminate,
    get_axis_map,
    linear_map_complex_to_real,
    mean,
)


class TestGranularPipeline:
    """Tests for the new granular post-processing instruction functions."""

    _axes = {ProcessAxis.SEQUENCE: -1}

    def test_apply_equalise_identity_is_noop(self):
        """A (2,2) identity matrix with zero offset is a no-op."""
        response = np.array([1.0 + 0.5j, -0.5 + 1.0j, 0.25 - 0.75j])
        instr = Equalise(output_variable="v", transform=np.eye(2), offset=np.zeros(2))
        result, _ = apply_equalise(response, instr, self._axes)
        assert np.allclose(result, response)

    def test_apply_equalise_complex_multiplier_as_2x2_matrix(self):
        """A complex multiplier m = a+jb encoded as [[a,-b],[b,a]] should give Re(m*z) as
        I'."""
        response = np.array([1.0 + 0.5j, -0.5 + 1.0j, 0.25 - 0.75j])
        m = 2.0 + 0.5j
        a, b = m.real, m.imag
        A = np.array([[a, -b], [b, a]], dtype=float)
        instr = Equalise(output_variable="v", transform=A, offset=np.zeros(2))
        result, _ = apply_equalise(response, instr, self._axes)
        assert np.allclose(result.real, np.real(m * response))

    def test_apply_equalise_rejects_complex_multiplier(self):
        """Supplying a complex-valued matrix should raise an error."""
        import numpy as np_mod

        with pytest.raises(
            np_mod.exceptions.ComplexWarning,
            match="Casting complex values to real discards",
        ):
            Equalise(
                output_variable="v",
                transform=np.array([[1.0 + 0.5j, 0j], [0j, 1.0 + 0.5j]]),
            )

    def test_apply_discriminate_threshold(self):
        """Values above threshold → label "0"; at/below → label "1"."""
        response = np.array([1.0, -1.0, 0.5, -0.5])
        instr = Discriminate(output_variable="v", threshold=0.0)
        state_ids, _ = apply_discriminate_instruction(response, instr, self._axes)
        assert list(state_ids) == ["0", "1", "0", "1"]

    def test_apply_post_select_empty_is_noop(self):
        """PostSelect with no disallowed states marks all shots valid."""
        state_ids = np.array(["0", "1", "0", "1"])
        instr = PostSelect(output_variable="v", disallowed_states=[])
        _, mask = apply_post_select(state_ids, instr, self._axes)
        assert all(mask)

    def test_apply_post_select_with_disallowed(self):
        state_ids = np.array(["0", "1", "0", "1"])
        instr = PostSelect(output_variable="v", disallowed_states=["1"])
        _, mask = apply_post_select(state_ids, instr, self._axes)
        assert list(mask) == [True, False, True, False]

    def test_apply_discriminate_ml_returns_state_labels_not_indices(self):
        """ML discrimination emits configured state labels (MLStateMap.label)."""
        states = [
            MLStateMap(label="ten", output_value=0, location=1.0 + 0j),
            MLStateMap(label="thirty", output_value=1, location=-1.0 + 0j),
            MLStateMap(label="twenty", output_value=2, location=0.0 + 1.0j),
        ]
        method = MaxLikelihoodMethod(states=states)
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([0.9 + 0.0j, -0.8 + 0.0j, 0.1 + 0.8j])

        state_ids, _ = apply_discriminate_instruction(response, instr, self._axes)

        assert list(state_ids) == ["ten", "thirty", "twenty"]

    def test_apply_discriminate_ml_single_state(self):
        """With a single state, every shot is assigned to it regardless of location."""
        states = [MLStateMap(label="only", output_value=0, location=0.0 + 0j)]
        method = MaxLikelihoodMethod(states=states)
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([999.0 + 0j, -42.0 + 1j])

        state_ids, _ = apply_discriminate_instruction(response, instr, self._axes)

        assert list(state_ids) == ["only", "only"]

    def test_apply_discriminate_ml_background_state_absorbs_outlier(self):
        """A disallowed state (erasure/pre-selection use-case) filters specific labels."""
        states = [
            MLStateMap(label="0", output_value=0, location=1.0 + 0j),
            MLStateMap(label="1", output_value=1, location=-1.0 + 0j),
            MLStateMap(label="2", output_value=2, location=0.0 + 1j, disallowed=True),
        ]
        method = MaxLikelihoodMethod(states=states)
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([1.0 + 0j, -1.0 + 0j, 0.0 + 1j])

        state_ids, _ = apply_discriminate_instruction(response, instr, self._axes)

        assert list(state_ids) == ["0", "1", "2"]

    def test_apply_discriminate_ml_p_min_zero_no_rejection(self):
        """p_min=0.0 (default) never emits BG_LABEL — even extreme outliers are
        classified."""
        states = [
            MLStateMap(label="0", output_value=0, location=1.0 + 0j),
            MLStateMap(label="1", output_value=1, location=-1.0 + 0j),
        ]
        method = MaxLikelihoodMethod(states=states, p_min=0.0)
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([1.0 + 0j, -1.0 + 0j, 500.0 + 0j])
        state_ids, _ = apply_discriminate_instruction(response, instr, self._axes)
        assert BG_LABEL not in list(state_ids)
        assert list(state_ids) == ["0", "1", "0"]

    def test_apply_discriminate_ml_p_min_rejects_low_confidence_shot(self):
        """Shots with normalised likelihood below p_min are labelled BG_LABEL.

        ``noise_est=0.1`` is used so shots exactly at their centroid have normalised
        likelihood ≈ 1.0 (well above p_min=0.99).  The midpoint 0+0j is equidistant
        from ±1+0j, so each state has normalised likelihood ≈ 0.5, which is below the
        threshold and is therefore labelled BG_LABEL.
        """
        states = [
            MLStateMap(label="0", output_value=0, location=1.0 + 0j),
            MLStateMap(label="1", output_value=1, location=-1.0 + 0j),
        ]
        # noise_est=0.1: shots at their centroid get normalised likelihood ≈ 1.0;
        # the midpoint 0+0j (sq_dist=1 to both states) gets ≈ 0.5, below p_min=0.99.
        method = MaxLikelihoodMethod(states=states, p_min=0.99, noise_est=0.1)
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([1.0 + 0j, -1.0 + 0j, 0.0 + 0j])
        state_ids, _ = apply_discriminate_instruction(response, instr, self._axes)
        assert state_ids[0] == "0"
        assert state_ids[1] == "1"
        assert state_ids[2] == BG_LABEL

    def test_apply_discriminate_ml_p_min_one_rejects_all_multistate(self):
        """p_min=1.0 rejects every shot when K >= 2 since normalised likelihood < 1."""
        states = [
            MLStateMap(label="0", output_value=0, location=1.0 + 0j),
            MLStateMap(label="1", output_value=1, location=-1.0 + 0j),
        ]
        method = MaxLikelihoodMethod(states=states, p_min=1.0)
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([1.0 + 0j, -1.0 + 0j, 0.5 + 0j])
        state_ids, _ = apply_discriminate_instruction(response, instr, self._axes)
        assert all(s == BG_LABEL for s in state_ids)

    def test_apply_discriminate_ml_p_min_single_state_never_bg(self):
        """With one state normalised likelihood is always 1.0 so p_min < 1 never rejects."""
        states = [MLStateMap(label="only", output_value=0, location=0.0 + 0j)]
        method = MaxLikelihoodMethod(states=states, p_min=0.99)
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([999.0 + 0j, -42.0 + 1j])
        state_ids, _ = apply_discriminate_instruction(response, instr, self._axes)
        assert list(state_ids) == ["only", "only"]

    def test_apply_demap_instruction(self):
        state_ids = np.array(["0", "1", "0", "1"])
        instr = Demap(output_variable="v", state_map={"0": 0, "1": 1})
        result, _ = apply_demap_instruction(state_ids, instr, self._axes)
        np.testing.assert_array_equal(result, [0, 1, 0, 1])
        assert result.dtype == int

    def test_apply_demap_instruction_unknown_label_gives_sentinel(self):
        """Labels not present in state_map produce -1 sentinel."""
        state_ids = np.array(["0", "unknown", "1"])
        instr = Demap(output_variable="v", state_map={"0": 0, "1": 1})
        result, _ = apply_demap_instruction(state_ids, instr, self._axes)
        assert result[0] == 0
        assert result[1] == -1
        assert result[2] == 1


class TestApplyPostProcessing:
    def test_invalid_pp_type_raises_not_implemented_error(self):
        pp = PostProcessing(output_variable="test", process_type=PostProcessType.MUL)
        with pytest.raises(
            NotImplementedError, match="Post processing type PostProcessType.MUL"
        ):
            apply_post_processing({}, pp, {})


class TestMean:
    @pytest.mark.parametrize("dims", [0, 1, 2])
    @pytest.mark.parametrize(
        "axes",
        [
            [ProcessAxis.TIME],
            [ProcessAxis.SEQUENCE],
            [ProcessAxis.TIME, ProcessAxis.SEQUENCE],
            [ProcessAxis.SEQUENCE, ProcessAxis.TIME],
        ],
    )
    def test_mean_with_RAW_acquisition(self, dims, axes):
        # create some mock data
        dims = [10 * (i + 1) for i in range(dims)]  # sweeping data
        dims.append(254)  # shots data
        dims.append(52)  # time-series data
        response = np.reshape(np.tile(1.0, (np.prod(dims))), dims)

        axis_map = get_axis_map(AcquireMode.RAW, response)
        response, new_axis_map = mean(response, axis_map, axes)

        if ProcessAxis.SEQUENCE in axes:
            dims.remove(dims[-2])
            assert ProcessAxis.SEQUENCE not in new_axis_map
        if ProcessAxis.TIME in axes:
            dims.remove(dims[-1])
            assert ProcessAxis.TIME not in new_axis_map
        assert list(np.shape(response)) == dims
        assert np.allclose(response, 1.0)

    @pytest.mark.parametrize("dims", [0, 1, 2])
    def test_mean_with_SCOPE_acquisition(self, dims):
        # create some mock data
        dims = [10 * (i + 1) for i in range(dims)]  # sweeping data
        dims.append(52)  # time-series data
        response = np.reshape(np.tile(1.0, (np.prod(dims))), dims)

        axis_map = get_axis_map(AcquireMode.SCOPE, response)
        response, new_axis_map = mean(response, axis_map, ProcessAxis.TIME)

        assert list(np.shape(response)) == dims[:-1]
        assert np.allclose(response, 1.0)
        assert ProcessAxis.TIME not in new_axis_map

    @pytest.mark.parametrize("dims", [0, 1, 2])
    def test_mean_with_INTEGRATOR_acquisition(self, dims):
        # create some mock data
        dims = [10 * (i + 1) for i in range(dims)]  # sweeping data
        dims.append(254)  # shot data
        response = np.reshape(np.tile(1.0, (np.prod(dims))), dims)

        axis_map = get_axis_map(AcquireMode.INTEGRATOR, response)
        response, new_axis_map = mean(response, axis_map, ProcessAxis.SEQUENCE)

        assert list(np.shape(response)) == dims[:-1]
        assert np.allclose(response, 1.0)
        assert ProcessAxis.SEQUENCE not in new_axis_map


@pytest.mark.parametrize("dims", [0, 1, 2])
def test_linear_map_complex_to_real(dims):
    # create some mock data
    dims = [10 * (i + 1) for i in range(dims)]  # sweeping data
    dims.append(254)  # shot data
    response = np.reshape(np.tile(1.0, (np.prod(dims))), dims)

    axis_map = get_axis_map(AcquireMode.INTEGRATOR, response)
    response, new_axis_map = linear_map_complex_to_real(response, axis_map, 5, 1)
    assert new_axis_map == axis_map
    assert np.allclose(response, 6.0)


@pytest.mark.parametrize("dims", [0, 1, 2])
def test_discriminate(dims):
    dims = [10 * (i + 1) for i in range(dims)]  # sweeping data
    dims.append(254)  # shot data
    bits = np.random.rand(*dims) > 0.5
    response = 2.0 * bits

    axis_map = get_axis_map(AcquireMode.INTEGRATOR, response)
    response, new_axis_map = discriminate(response, axis_map, 1.0)
    assert new_axis_map == axis_map
    assert np.allclose(bits, (response == np.ones(dims)))


class TestMaxLikelihoodMethodValidation:
    """Pydantic model-validation tests for MaxLikelihoodMethod."""

    _base_states = [MLStateMap(label="0", output_value=0, location=1.0 + 0j)]

    def test_p_min_below_zero_raises(self):
        """p_min < 0 must be rejected by Pydantic."""
        with pytest.raises(ValidationError, match="p_min"):
            MaxLikelihoodMethod(states=self._base_states, p_min=-0.01)

    def test_p_min_above_one_raises(self):
        """p_min > 1 must be rejected by Pydantic."""
        with pytest.raises(ValidationError, match="p_min"):
            MaxLikelihoodMethod(states=self._base_states, p_min=1.001)

    @pytest.mark.parametrize("p_min", [0.0, 0.5, 1.0])
    def test_p_min_boundary_values_accepted(self, p_min):
        """p_min values 0.0, 0.5, and 1.0 are all valid boundary values."""
        method = MaxLikelihoodMethod(states=self._base_states, p_min=p_min)
        assert method.p_min == p_min

    def test_reserved_bg_label_raises(self):
        """Using BG_LABEL as an MLStateMap label must be rejected."""
        with pytest.raises(ValidationError, match="reserved"):
            MaxLikelihoodMethod(
                states=[MLStateMap(label=BG_LABEL, output_value=0, location=0.0 + 0j)],
                p_min=0.5,
            )
