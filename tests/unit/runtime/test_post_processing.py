# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
"""Tests for runtime post-processing utilities.

These cover mean, linear mapping, discrimination and the granular post-processing pipeline
(Equalise, Discriminate, PostSelect) used by the runtime to convert acquired IQ data
into final classical outputs.

Discriminate now emits integer state keys directly instead of string labels:
- Threshold path: above threshold → 0, at/below → 1.
- ML path: dict key from MaxLikelihoodMethod.states (negative = disallowed).
PostSelect filters shots with negative integer keys.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from qat.ir.instruction_basetypes import AcquireMode, PostProcessType, ProcessAxis
from qat.ir.measure import Discriminate, Equalise, PostProcessing, PostSelect
from qat.model.post_processing import BG_KEY, MaxLikelihoodMethod, MLDiscriminateParams
from qat.runtime.post_processing import (
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
    """Tests for the granular post-processing instruction functions."""

    _axes = {ProcessAxis.SEQUENCE: -1}

    def test_apply_equalise_identity_is_noop(self):
        """A (2,2) identity matrix with zero offset is a no-op."""
        response = np.array([1.0 + 0.5j, -0.5 + 1.0j, 0.25 - 0.75j])
        instr = Equalise(output_variable="v", transform=np.eye(2), offset=np.zeros(2))
        result, _ = apply_equalise(response, instr, self._axes)
        assert np.allclose(result, response)

    def test_apply_equalise_complex_multiplier_as_2x2_matrix(self):
        """A complex multiplier m=a+jb encoded as [[a,-b],[b,a]] gives Re(m*z) as I'."""
        response = np.array([1.0 + 0.5j, -0.5 + 1.0j, 0.25 - 0.75j])
        m = 2.0 + 0.5j
        a, b = m.real, m.imag
        A = np.array([[a, -b], [b, a]], dtype=float)
        instr = Equalise(output_variable="v", transform=A, offset=np.zeros(2))
        result, _ = apply_equalise(response, instr, self._axes)
        assert np.allclose(result.real, np.real(m * response))

    def test_apply_discriminate_threshold_emits_integers(self):
        """Threshold discriminator: values above threshold → 0, at/below → 1."""
        response = np.array([1.0, -1.0, 0.5, -0.5])
        instr = Discriminate(output_variable="v", threshold=0.0)
        state_keys, _ = apply_discriminate_instruction(response, instr, self._axes)
        assert list(state_keys) == [0, 1, 0, 1]

    def test_apply_post_select_filters_negative_keys(self):
        """PostSelect masks shots with negative integer state keys."""
        state_keys = np.array([0, 1, -2, 0, -1])
        instr = PostSelect(output_variable="v")
        _, mask = apply_post_select(state_keys, instr, self._axes)
        assert list(mask) == [True, True, False, True, False]

    def test_apply_post_select_all_allowed_is_noop(self):
        """PostSelect with all non-negative keys marks all shots valid."""
        state_keys = np.array([0, 1, 2, 0])
        instr = PostSelect(output_variable="v")
        _, mask = apply_post_select(state_keys, instr, self._axes)
        assert all(mask)

    def test_apply_post_select_additional_disallowed_masks_matching_keys(self):
        """additional_disallowed masks non-negative keys that appear in the set."""
        state_keys = np.array([0, 1, 2, 0, 1])
        instr = PostSelect(output_variable="v", additional_disallowed={1})
        _, mask = apply_post_select(state_keys, instr, self._axes)
        assert list(mask) == [True, False, True, True, False]

    def test_apply_post_select_additional_disallowed_combined_with_negative_keys(self):
        """additional_disallowed and negative-key masking are both applied (AND logic)."""
        state_keys = np.array([0, 1, -2, 2, 1])
        instr = PostSelect(output_variable="v", additional_disallowed={2})
        _, mask = apply_post_select(state_keys, instr, self._axes)
        # key 1 → allowed, key -2 → negative → masked, key 2 → additional_disallowed → masked
        assert list(mask) == [True, True, False, False, True]

    def test_apply_post_select_empty_additional_disallowed_is_noop(self):
        """An empty additional_disallowed set behaves identically to omitting it."""
        state_keys = np.array([0, 1, -1])
        instr_no_extra = PostSelect(output_variable="v")
        instr_empty_extra = PostSelect(output_variable="v", additional_disallowed=set())
        _, mask_no_extra = apply_post_select(state_keys, instr_no_extra, self._axes)
        _, mask_empty_extra = apply_post_select(state_keys, instr_empty_extra, self._axes)
        assert list(mask_no_extra) == list(mask_empty_extra)

    def test_apply_post_select_additional_disallowed_multiple_keys(self):
        """Multiple keys in additional_disallowed are all masked."""
        state_keys = np.array([0, 1, 2, 3])
        instr = PostSelect(output_variable="v", additional_disallowed={1, 3})
        _, mask = apply_post_select(state_keys, instr, self._axes)
        assert list(mask) == [True, False, True, False]

    def test_apply_discriminate_ml_returns_int_dict_keys(self):
        """ML discrimination emits the integer dict keys from MaxLikelihoodMethod.states."""
        method = MaxLikelihoodMethod(
            states={
                10: MLDiscriminateParams(location=1.0 + 0j),
                30: MLDiscriminateParams(location=-1.0 + 0j),
                20: MLDiscriminateParams(location=0.0 + 1.0j),
            }
        )
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([0.9 + 0.0j, -0.8 + 0.0j, 0.1 + 0.8j])

        state_keys, _ = apply_discriminate_instruction(response, instr, self._axes)

        assert list(state_keys) == [10, 30, 20]

    def test_apply_discriminate_ml_single_state(self):
        """With a single state, every shot is assigned its key regardless of location."""
        method = MaxLikelihoodMethod(states={0: MLDiscriminateParams(location=0.0 + 0j)})
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([999.0 + 0j, -42.0 + 1j])

        state_keys, _ = apply_discriminate_instruction(response, instr, self._axes)

        assert list(state_keys) == [0, 0]

    def test_apply_discriminate_ml_disallowed_negative_key(self):
        """A state with a negative key is classified as disallowed; PostSelect filters
        it."""
        method = MaxLikelihoodMethod(
            states={
                0: MLDiscriminateParams(location=1.0 + 0j),
                1: MLDiscriminateParams(location=-1.0 + 0j),
                -2: MLDiscriminateParams(location=0.0 + 1j),
            }
        )
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([1.0 + 0j, -1.0 + 0j, 0.0 + 1j])

        state_keys, _ = apply_discriminate_instruction(response, instr, self._axes)
        assert list(state_keys) == [0, 1, -2]

        ps_instr = PostSelect(output_variable="v")
        _, mask = apply_post_select(state_keys, ps_instr, self._axes)
        assert list(mask) == [True, True, False]

    def test_apply_discriminate_ml_p_min_zero_no_rejection(self):
        """p_min=0.0 (default) never emits BG_KEY."""
        method = MaxLikelihoodMethod(
            states={
                0: MLDiscriminateParams(location=1.0 + 0j),
                1: MLDiscriminateParams(location=-1.0 + 0j),
            },
            p_min=0.0,
        )
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([1.0 + 0j, -1.0 + 0j, 500.0 + 0j])
        state_keys, _ = apply_discriminate_instruction(response, instr, self._axes)
        assert BG_KEY not in list(state_keys)
        assert list(state_keys) == [0, 1, 0]

    def test_apply_discriminate_ml_p_min_rejects_low_confidence_shot(self):
        """Shots with normalised likelihood below p_min are assigned BG_KEY."""
        method = MaxLikelihoodMethod(
            states={
                0: MLDiscriminateParams(location=1.0 + 0j),
                1: MLDiscriminateParams(location=-1.0 + 0j),
            },
            p_min=0.99,
            noise_est=0.1,
        )
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([1.0 + 0j, -1.0 + 0j, 0.0 + 0j])
        state_keys, _ = apply_discriminate_instruction(response, instr, self._axes)
        assert state_keys[0] == 0
        assert state_keys[1] == 1
        assert state_keys[2] == BG_KEY

    def test_apply_discriminate_ml_p_min_one_rejects_all_multistate(self):
        """p_min=1.0 assigns BG_KEY to every shot when K >= 2."""
        method = MaxLikelihoodMethod(
            states={
                0: MLDiscriminateParams(location=1.0 + 0j),
                1: MLDiscriminateParams(location=-1.0 + 0j),
            },
            p_min=1.0,
        )
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([1.0 + 0j, -1.0 + 0j, 0.5 + 0j])
        state_keys, _ = apply_discriminate_instruction(response, instr, self._axes)
        assert all(k == BG_KEY for k in state_keys)

    def test_apply_discriminate_ml_p_min_single_state_never_bg(self):
        """With one state normalised likelihood is always 1.0 so p_min<1 never rejects."""
        method = MaxLikelihoodMethod(
            states={0: MLDiscriminateParams(location=0.0 + 0j)},
            p_min=0.99,
        )
        instr = Discriminate(output_variable="v", method=method)
        response = np.array([999.0 + 0j, -42.0 + 1j])
        state_keys, _ = apply_discriminate_instruction(response, instr, self._axes)
        assert list(state_keys) == [0, 0]


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

    _base_states = {0: MLDiscriminateParams(location=1.0 + 0j)}

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
