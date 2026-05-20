# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.model.post_processing import MaxLikelihoodMethod, MLStateMap


def _make_states():
    return [
        MLStateMap(label="0", output_value=0.0, location=0 + 0j),
        MLStateMap(label="1", output_value=1.0, location=1 + 0j),
    ]


def _make_transform(shape):
    return np.ones(shape, dtype=float)


class TestMLStateMapNoiseEstValidation:
    def test_default_noise_est_accepted(self):
        method = MaxLikelihoodMethod(states=_make_states())
        assert method.noise_est == 1.0

    def test_explicit_positive_noise_est_accepted(self):
        method = MaxLikelihoodMethod(states=_make_states(), noise_est=1.5)
        assert method.noise_est == 1.5

    def test_zero_noise_est_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            MaxLikelihoodMethod(states=_make_states(), noise_est=0.0)

    def test_negative_noise_est_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            MaxLikelihoodMethod(states=_make_states(), noise_est=-1.0)


class TestValidateStatesNonEmpty:
    def test_non_empty_states_accepted(self):
        method = MaxLikelihoodMethod(states=_make_states())
        assert len(method.states) == 2

    def test_single_state_accepted(self):
        method = MaxLikelihoodMethod(
            states=[MLStateMap(label="0", output_value=0.0, location=0 + 0j)]
        )
        assert len(method.states) == 1

    def test_empty_states_raises(self):
        with pytest.raises(ValueError, match="must define at least one state"):
            MaxLikelihoodMethod(states=[])


class TestValidateTransformAndOffsetBothOrNeither:
    def test_both_none_accepted(self):
        method = MaxLikelihoodMethod(states=_make_states(), transform=None, offset=None)
        assert method.transform is None
        assert method.offset is None

    def test_both_set_accepted(self):
        method = MaxLikelihoodMethod(
            states=_make_states(),
            transform=_make_transform((2, 2)),
            offset=np.zeros(2),
        )
        assert method.transform is not None
        assert method.offset is not None

    def test_transform_without_offset_raises(self):
        with pytest.raises(ValueError, match="both 'transform' and 'offset'"):
            MaxLikelihoodMethod(
                states=_make_states(),
                transform=_make_transform((2, 2)),
                offset=None,
            )

    def test_offset_without_transform_raises(self):
        with pytest.raises(ValueError, match="both 'transform' and 'offset'"):
            MaxLikelihoodMethod(
                states=_make_states(),
                transform=None,
                offset=np.zeros(2),
            )


class TestValidateTransformIs2x2:
    def test_valid_2x2_transform_accepted(self):
        method = MaxLikelihoodMethod(
            states=_make_states(),
            transform=_make_transform((2, 2)),
            offset=np.zeros(2),
        )
        assert method.transform.shape == (2, 2)

    @pytest.mark.parametrize(
        "shape",
        [
            (3, 3),
            (2, 3),
            (1, 2),
            (4,),
        ],
    )
    def test_non_2x2_transform_raises(self, shape):
        with pytest.raises(ValueError, match="'transform' must be a 2x2 matrix"):
            MaxLikelihoodMethod(
                states=_make_states(),
                transform=_make_transform(shape),
                offset=np.zeros(2),
            )

    def test_none_transform_skips_validation(self):
        method = MaxLikelihoodMethod(
            states=_make_states(),
            transform=None,
            offset=None,
        )
        assert method.transform is None


class TestValidateOffsetIsShape2:
    def test_valid_length_2_offset_accepted(self):
        method = MaxLikelihoodMethod(
            states=_make_states(),
            transform=_make_transform((2, 2)),
            offset=np.zeros(2),
        )
        assert method.offset.shape == (2,)

    @pytest.mark.parametrize(
        "shape",
        [
            (3,),
            (1,),
            (2, 2),
        ],
    )
    def test_wrong_shape_offset_raises(self, shape):
        with pytest.raises(ValueError, match="'offset' must be a 1-D vector of length 2"):
            MaxLikelihoodMethod(
                states=_make_states(),
                transform=_make_transform((2, 2)),
                offset=np.zeros(shape),
            )

    def test_none_offset_skips_validation(self):
        method = MaxLikelihoodMethod(
            states=_make_states(),
            transform=None,
            offset=None,
        )
        assert method.offset is None
