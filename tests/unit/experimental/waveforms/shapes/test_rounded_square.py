# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the functions for sampling a Rounded Square waveform shape."""

import numpy as np
import pytest

from qat.experimental.waveforms.shapes.rounded_square import (
    RoundedSquareWaveformShape,
    sample_rounded_square_waveform,
    sample_rounded_square_waveform_derivative,
)
from qat.utils.waveform import RoundedSquareFunction

from tests.unit.experimental.waveforms.utils import estimate_derivative_at_point


class TestSampleRoundedSquareWaveform:
    """Tests the ``sample_rounded_square_waveform`` function."""

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise", [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)]
    )
    def test_waveform_is_symmetric_around_zero(self, fractional_top_width, fractional_rise):
        """Tests that the waveform is symmetric around zero."""

        x = np.linspace(-1, 0, 100)
        y_before = sample_rounded_square_waveform(
            x, fractional_top_width=fractional_top_width, fractional_rise=fractional_rise
        )
        y_after = sample_rounded_square_waveform(
            -x, fractional_top_width=fractional_top_width, fractional_rise=fractional_rise
        )
        assert np.allclose(y_before, y_after, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise", [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)]
    )
    def test_rise_and_fall_are_monotone(self, fractional_top_width, fractional_rise):
        """Tests that the fractional_rise is increasing and the fall is decreasing when
        moving towards the center.

        The erf tails can plateau to machine precision near the edges, so this is checked
        non-strictly.
        """

        x = np.linspace(-1, 1, 100)
        y = sample_rounded_square_waveform(
            x, fractional_top_width=fractional_top_width, fractional_rise=fractional_rise
        )
        rise_mask = x < -fractional_top_width
        fall_mask = x > fractional_top_width
        assert np.all(np.diff(y[rise_mask]) >= 0.0)
        assert np.all(np.diff(y[fall_mask]) <= 0.0)

    def test_larger_rise_gives_smoother_transition(self):
        """Tests that a larger fractional_rise value gives a smoother transition from the
        square region to the edges.

        Tests it by checking that to the left of the center of the fractional_rise, a
        smaller fractional_rise has smaller amplitudes and to the right of the center of the
        fractional_rise, a smaller fractional_rise has larger amplitudes.
        """

        fractional_top_width = 0.5
        x = np.linspace(-1, 0, 100)
        rise_mask_smaller = x < -fractional_top_width
        rise_mask_larger = x > -fractional_top_width

        y_small_rise = sample_rounded_square_waveform(
            x, fractional_top_width=fractional_top_width, fractional_rise=0.05
        )
        y_large_rise = sample_rounded_square_waveform(
            x, fractional_top_width=fractional_top_width, fractional_rise=0.2
        )
        assert np.all(y_large_rise[rise_mask_smaller] >= y_small_rise[rise_mask_smaller])
        assert np.all(y_large_rise[rise_mask_larger] <= y_small_rise[rise_mask_larger])

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise", [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)]
    )
    def test_center_unity(self, fractional_top_width, fractional_rise):
        """Tests that the waveform is one at the center."""
        x = np.array([0.0])
        y = sample_rounded_square_waveform(
            x, fractional_top_width=fractional_top_width, fractional_rise=fractional_rise
        )
        assert np.allclose(y, 1.0, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise", [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)]
    )
    def test_zero_at_edges(self, fractional_top_width, fractional_rise):
        """Tests that the waveform is zero at the edges."""
        x = np.array([-1.0, 1.0])
        y = sample_rounded_square_waveform(
            x, fractional_top_width=fractional_top_width, fractional_rise=fractional_rise
        )
        assert np.allclose(y, 0.0, atol=1e-8)


class TestSampleRoundedSquareWaveformDerivative:
    """Tests the ``sample_rounded_square_waveform_derivative`` function.

    Tests expected high-level properties of derivatives, such as symmetry and peaks. Also
    implements numerical comparisons to ensure the derivatives are approximately equivalent.
    """

    def test_derivative_order_zero_is_equivalent_to_sample_rounded_square_waveform(self):
        """Checks that the derivative of order zero is equivalent to the
        ``sample_rounded_square_waveform`` function."""

        x = np.linspace(-1, 1, 100)
        y_sample = sample_rounded_square_waveform(
            x, fractional_top_width=0.5, fractional_rise=0.1
        )
        y_derivative = sample_rounded_square_waveform_derivative(
            x, order=0, fractional_top_width=0.5, fractional_rise=0.1
        )
        assert np.allclose(y_derivative, y_sample, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise",
        [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)],
    )
    @pytest.mark.parametrize("order", [1, 3])
    def test_odd_order_derivatives_are_antisymmetric(
        self, fractional_top_width, fractional_rise, order
    ):
        """Tests that odd-order derivatives are antisymmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_rounded_square_waveform_derivative(
            x,
            order=order,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
        )
        y_negative = sample_rounded_square_waveform_derivative(
            -x,
            order=order,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
        )
        assert np.allclose(y_positive, -y_negative, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise", [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)]
    )
    @pytest.mark.parametrize("order", [2, 4])
    def test_even_order_derivatives_are_symmetric(
        self, fractional_top_width, fractional_rise, order
    ):
        """Tests that even-order derivatives are symmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_rounded_square_waveform_derivative(
            x,
            order=order,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
        )
        y_negative = sample_rounded_square_waveform_derivative(
            -x,
            order=order,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
        )
        assert np.allclose(y_positive, y_negative, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise", [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)]
    )
    def test_first_derivative_is_positive_on_rise_and_negative_on_fall(
        self, fractional_top_width, fractional_rise
    ):
        """Tests that the first derivative is positive on the fractional_rise and negative
        on the fall."""
        x = np.linspace(-1, 1, 200)
        y_derivative = sample_rounded_square_waveform_derivative(
            x,
            order=1,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
        )
        rise_mask = x < 0
        fall_mask = x > 0
        assert np.all(y_derivative[rise_mask] > 0.0)
        assert np.all(y_derivative[fall_mask] < 0.0)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise",
        [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)],
    )
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_derivative_matches_numerical_estimate(
        self, fractional_top_width, fractional_rise, order
    ):
        """Tests that the derivative matches a numerical estimate of the derivative.

        The waveform is always normalized, so the numerical estimate is computed directly
        from ``sample_rounded_square_waveform``.
        """
        x = np.linspace(-0.5, 0.5, 50)
        y_derivative = sample_rounded_square_waveform_derivative(
            x,
            order=order,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
        )

        def fn(x_val):
            return sample_rounded_square_waveform(
                x_val,
                fractional_top_width=fractional_top_width,
                fractional_rise=fractional_rise,
            )

        numerical_y_derivative = np.asarray(
            [estimate_derivative_at_point(fn, x_i, order=order) for x_i in x]
        )
        assert np.allclose(y_derivative, numerical_y_derivative, atol=1e-3)


class TestParityWithPydanticRoundedSquareWaveform:
    """Tests that the sampling implementation is consistent with the pydantic
    RoundedSquareWaveform class.

    The Rounded Square waveform implements the pydantic ``RoundedSquareWaveform`` class with
    the following parameterisations: ``fractional_rise = 2 * fractional_rise / width`` and
    ``fractional_top_width = std_dev / width``. Note the legacy implementation seems to be inconsistent,
    and is defined between ``[0, T]`` instead of the usual ``[-T/2, T/2]`` domain.
    """

    @pytest.mark.parametrize("width", [2.0, 160e-9])
    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise", [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)]
    )
    def test_sample_rounded_square_waveform_matches_pydantic_class(
        self, fractional_top_width, fractional_rise, width
    ):
        """Tests that the ``sample_rounded_square_waveform`` function matches the pydantic
        RoundedSquareWaveform class."""

        rise_legacy = fractional_rise * width / 2.0
        std_dev = width * fractional_top_width
        scale = width / 2  # Maps x ∈ [-1, 1] to x*scale ∈ [-width/2, width/2]

        x = np.linspace(-1, 1, 101)
        t_array = (
            x + 1.0
        ) * scale  # Shift to legacy RoundedSquareFunction domain [0, width]

        y_function = sample_rounded_square_waveform(
            x, fractional_top_width=fractional_top_width, fractional_rise=fractional_rise
        )
        waveform = RoundedSquareFunction(width=width, rise=rise_legacy, std_dev=std_dev)
        y_class = waveform(t_array)
        y_class = (y_class - y_class[0]) / (np.max(y_class) - y_class[0])
        assert np.allclose(y_function, y_class, atol=1e-8)


class TestRoundedSquareWaveformShapeFromLegacy:
    """Tests legacy classmethod constructor for ``RoundedSquareWaveformShape``."""

    def test_from_legacy(self):
        """Checks fractional_top_width and fractional_rise from legacy RoundedSquareWaveform
        parameters."""
        shape = RoundedSquareWaveformShape.from_legacy(rise=0.02, std_dev=0.1, width=0.4)
        assert np.isclose(shape.fractional_top_width, 0.1 / 0.4)
        assert np.isclose(shape.fractional_rise, 2.0 * 0.02 / 0.4)


@pytest.mark.parametrize("fractional_rise", [0.0, -0.1])
class TestFractionalRiseValidation:
    """Tests that public functions reject non-positive ``fractional_rise`` values."""

    def test_sample_rounded_square_waveform_raises_for_non_positive_rise(
        self, fractional_rise
    ):
        """Checks that ``sample_rounded_square_waveform`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            sample_rounded_square_waveform(np.array([0.0]), fractional_rise=fractional_rise)

    def test_sample_rounded_square_waveform_derivative_raises_for_non_positive_rise(
        self, fractional_rise
    ):
        """Checks that ``sample_rounded_square_waveform_derivative`` raises
        ``ValueError``."""
        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            sample_rounded_square_waveform_derivative(
                np.array([0.0]), fractional_rise=fractional_rise
            )

    def test_rounded_square_waveform_shape_raises_for_non_positive_rise(
        self, fractional_rise
    ):
        """Checks that ``RoundedSquareWaveformShape`` raises ``ValueError`` on
        construction."""
        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            RoundedSquareWaveformShape(fractional_rise=fractional_rise)


class TestRoundedSquareWaveformShapeDelegation:
    """Tests shape methods match direct calls to sampling functions."""

    def test_evaluate_matches_raw_function(self):
        """Checks evaluate output matches sample_rounded_square_waveform."""

        x = np.array([-0.5, 0.5])
        shape = RoundedSquareWaveformShape(fractional_top_width=0.2, fractional_rise=0.3)
        assert np.allclose(
            shape.evaluate(x),
            sample_rounded_square_waveform(
                x, fractional_top_width=0.2, fractional_rise=0.3
            ),
        )

    def test_derivative_matches_raw_function(self):
        """Checks derivative output matches sample_rounded_square_waveform_derivative."""

        x = np.array([-0.5, 0.5])
        shape = RoundedSquareWaveformShape(fractional_top_width=0.2, fractional_rise=0.3)
        assert np.allclose(
            shape.derivative(x, order=1),
            sample_rounded_square_waveform_derivative(
                x, order=1, fractional_top_width=0.2, fractional_rise=0.3
            ),
        )

    def test_from_absolute_generates_correct_fractional_parameters(self):
        """Checks that the from_absolute method generates the correct fractional
        parameters."""

        shape = RoundedSquareWaveformShape.from_absolute(
            width=160e-9, absolute_top_width=80e-9, absolute_rise=16e-9
        )
        assert np.isclose(shape.fractional_rise, 0.1)  # 16ns / 160ns
        assert np.isclose(shape.fractional_top_width, 0.5)  # 80ns / 160ns
