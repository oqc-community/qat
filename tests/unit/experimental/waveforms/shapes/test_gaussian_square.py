# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the functions for sampling a Gaussian Square waveform shape."""

import numpy as np
import pytest

from qat.experimental.waveforms.shapes.exceptions import DerivativeOrderUndefinedError
from qat.experimental.waveforms.shapes.gaussian_square import (
    GaussianSquareWaveformShape,
    sample_gaussian_square_waveform,
    sample_gaussian_square_waveform_derivative,
)
from qat.utils.waveform import GaussianSquareFunction

from tests.unit.experimental.waveforms.utils import estimate_derivative_at_point


class TestSampleGaussianSquareWaveform:
    """Tests the ``sample_gaussian_square_waveform`` function."""

    @pytest.mark.parametrize("fractional_top_width", [-0.1, 1.1])
    def test_out_of_range_fractional_top_width_raises_value_error(
        self, fractional_top_width
    ):
        """Tests that the top-width proportion is constrained to the unit interval."""
        x = np.linspace(-1, 1, 5)

        with pytest.raises(
            ValueError,
            match=("fractional_top_width must satisfy 0 <= fractional_top_width <= 1"),
        ):
            sample_gaussian_square_waveform(x, fractional_top_width=fractional_top_width)

    def test_zero_fractional_rise_raises_value_error(self):
        """Tests that a zero fractional rise is rejected explicitly."""
        x = np.linspace(-1, 1, 5)

        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            sample_gaussian_square_waveform(x, fractional_rise=0.0)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise, regularize",
        [(0.1, 0.1, True), (0.5, 0.15, True), (0.9, 0.05, False), (1.0, 0.2, True)],
    )
    def test_top_width_is_one_and_outside_is_less_than_one(
        self, fractional_top_width, fractional_rise, regularize
    ):
        """Tests that when fractional_top_width is one, the values outside the square region
        are less than one."""
        x = np.linspace(-1, 1, 100)
        y = sample_gaussian_square_waveform(
            x,
            fractional_rise=fractional_rise,
            regularize=regularize,
            fractional_top_width=fractional_top_width,
        )
        if fractional_top_width == 1.0:
            assert np.all(y[x < -fractional_top_width] < 1.0)
            assert np.all(y[x > fractional_top_width] < 1.0)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise, regularize",
        [(0.1, 0.1, True), (0.5, 0.15, True), (0.9, 0.05, False), (1.0, 0.2, True)],
    )
    def test_waveform_is_symmetric_around_zero(
        self, fractional_top_width, fractional_rise, regularize
    ):
        """Tests that the waveform is symmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_gaussian_square_waveform(
            x,
            fractional_rise=fractional_rise,
            regularize=regularize,
            fractional_top_width=fractional_top_width,
        )
        y_negative = sample_gaussian_square_waveform(
            -x,
            fractional_rise=fractional_rise,
            regularize=regularize,
            fractional_top_width=fractional_top_width,
        )
        assert np.allclose(y_positive, y_negative, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise, regularize",
        [(0.1, 0.1, True), (0.5, 0.15, True), (0.9, 0.05, False), (1.0, 0.2, True)],
    )
    def test_rise_and_fall_are_strictly_decreasing_when_moving_towards_edges(
        self, fractional_top_width, fractional_rise, regularize
    ):
        """Tests that the fractional_rise and fall of the waveform are strictly decreasing
        when moving towards the edges."""
        x = np.linspace(-1, 1, 100)
        y = sample_gaussian_square_waveform(
            x,
            fractional_rise=fractional_rise,
            regularize=regularize,
            fractional_top_width=fractional_top_width,
        )
        rise_mask = x < -fractional_top_width
        fall_mask = x > fractional_top_width
        assert np.all(np.diff(y[rise_mask]) > 0)
        assert np.all(np.diff(y[fall_mask]) < 0)

    def test_larger_rise_gives_smoother_transition(self):
        """Tests that a larger fractional_rise value gives a smoother transition from the
        square region to the edges."""
        fractional_top_width = 0.5
        x = np.linspace(-1, 1, 100)
        rise_mask = x < -fractional_top_width
        y_small_rise = sample_gaussian_square_waveform(
            x,
            fractional_rise=0.05,
            regularize=False,
            fractional_top_width=fractional_top_width,
        )
        y_large_rise = sample_gaussian_square_waveform(
            x,
            fractional_rise=0.2,
            regularize=False,
            fractional_top_width=fractional_top_width,
        )
        assert (y_large_rise[rise_mask] > y_small_rise[rise_mask]).all()

    def test_zero_at_edges_when_normalization_is_true(self):
        """Tests that the waveform is zero at the edges when regularize is True."""
        fractional_top_width = 0.5
        x = np.array([-1.0, 1.0])
        y = sample_gaussian_square_waveform(
            x,
            fractional_rise=0.1,
            regularize=True,
            fractional_top_width=fractional_top_width,
        )
        assert np.allclose(y, 0.0, atol=1e-8)


class TestSampleGaussianWaveformDerivative:
    """Tests the ``sample_gaussian_square_waveform_derivative`` function.

    Tests expected high-level properties of derivatives, such as symmetry and peaks. Also
    implements numerical comparisons to ensure the derivatives are approximately equivalent.
    """

    @pytest.mark.parametrize("fractional_top_width", [-0.1, 1.1])
    def test_out_of_range_fractional_top_width_raises_value_error(
        self, fractional_top_width
    ):
        """Tests that derivative sampling rejects invalid top-width proportions."""
        x = np.linspace(-1, 1, 5)

        with pytest.raises(
            ValueError,
            match=("fractional_top_width must satisfy 0 <= fractional_top_width <= 1"),
        ):
            sample_gaussian_square_waveform_derivative(
                x, fractional_top_width=fractional_top_width
            )

    def test_zero_fractional_rise_raises_value_error(self):
        """Tests that derivative sampling rejects a zero fractional rise."""
        x = np.linspace(-1, 1, 5)

        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            sample_gaussian_square_waveform_derivative(x, fractional_rise=0.0)

    def test_orders_higher_than_one_raise_a_derivative_order_undefined_error(self):
        """Tests that orders higher than one raise a DerivativeOrderUndefinedError."""
        x = np.linspace(-1, 1, 100)
        with pytest.raises(
            DerivativeOrderUndefinedError,
            match=(
                r"The derivative of order 2 is not mathematically defined for waveform "
                r"shape 'Gaussian-Square'\."
            ),
        ):
            sample_gaussian_square_waveform_derivative(x, order=2)

    def test_derivative_order_zero_is_equivalent_to_sample_gaussian_square_waveform(self):
        """Checks that the derivative of order zero is equivalent to the
        ``sample_gaussian_square_waveform`` function."""

        x = np.linspace(-1, 1, 100)
        y = sample_gaussian_square_waveform(
            x, fractional_rise=0.1, regularize=True, fractional_top_width=0.5
        )
        y_derivative = sample_gaussian_square_waveform_derivative(
            x, order=0, fractional_rise=0.1, regularize=True, fractional_top_width=0.5
        )
        assert np.allclose(y, y_derivative, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise, regularize",
        [(0.1, 0.1, True), (0.5, 0.15, True), (0.9, 0.05, False), (1.0, 0.2, True)],
    )
    def test_derivative_is_zero_within_top_region(
        self, fractional_top_width, fractional_rise, regularize
    ):
        """Tests that the derivative is zero within the top region of the waveform."""
        x = np.linspace(-1, 1, 100)
        y_derivative = sample_gaussian_square_waveform_derivative(
            x,
            order=1,
            fractional_rise=fractional_rise,
            regularize=regularize,
            fractional_top_width=fractional_top_width,
        )
        top_region_mask = (-fractional_top_width < np.abs(x)) & (
            np.abs(x) < fractional_top_width
        )
        assert np.allclose(y_derivative[top_region_mask], 0.0, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise, regularize",
        [(0.1, 0.1, True), (0.5, 0.15, True), (0.9, 0.05, False), (1.0, 0.2, True)],
    )
    def test_derivative_is_none_zero_outside_top_region(
        self, fractional_top_width, fractional_rise, regularize
    ):
        """Tests that the derivative is non-zero outside the top region of the waveform.

        Specifically, test its positive on the fractional_rise and negative on the fall.
        """
        x = np.linspace(-1, 1, 100)
        y_derivative = sample_gaussian_square_waveform_derivative(
            x,
            order=1,
            fractional_rise=fractional_rise,
            regularize=regularize,
            fractional_top_width=fractional_top_width,
        )
        rise_mask = x < -fractional_top_width
        fall_mask = x > fractional_top_width
        assert np.all(y_derivative[rise_mask] > 0.0)
        assert np.all(y_derivative[fall_mask] < 0.0)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise, regularize",
        [(0.1, 0.1, True), (0.5, 0.15, True), (0.9, 0.05, False), (1.0, 0.2, True)],
    )
    def test_derivative_matches_numerical_estimate(
        self, fractional_top_width, fractional_rise, regularize
    ):
        """Tests that the derivative matches a numerical estimate of the derivative."""
        x = np.linspace(-1, 1, 100)
        y_derivative = sample_gaussian_square_waveform_derivative(
            x,
            order=1,
            fractional_rise=fractional_rise,
            regularize=regularize,
            fractional_top_width=fractional_top_width,
        )

        def fn(x):
            return sample_gaussian_square_waveform(
                x,
                fractional_rise=fractional_rise,
                regularize=regularize,
                fractional_top_width=fractional_top_width,
            )

        numerical_y_derivative = [
            estimate_derivative_at_point(fn, x_i, order=1) for x_i in x
        ]
        assert np.allclose(y_derivative, numerical_y_derivative, atol=1e-3)


class TestParityWithPydanticGaussianSquareWaveform:
    """Tests that the sampling implementation is consistent with the pydantic
    GaussianSquareWaveform class.

    The Gaussian Square waveform implements the legacy ``GaussianSquareWaveform`` under the
    parameterisations ``fractional_rise = 2 * std_dev / width``, ``regularize = zero_at_edges``, and
    ``fractional_top_width = square_width / width``.
    """

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise, regularize",
        [(0.1, 0.1, True), (0.5, 0.15, True), (0.9, 0.05, False)],
    )
    @pytest.mark.parametrize("width", [2.0, 160e-9])
    def test_sample_gaussian_square_waveform_matches_pydantic_class(
        self, fractional_top_width, fractional_rise, regularize, width
    ):
        """Tests that the ``sample_gaussian_square_waveform`` function matches the pydantic
        GaussianSquareWaveform class."""

        x = np.linspace(-1, 1, 100)

        std_dev = fractional_rise * width / 2.0
        zero_at_edges = regularize
        square_width = fractional_top_width * width
        scale = width / 2  # We're mapping from [-1, 1] to [-width / 2, width / 2]
        t_array = x * scale

        y_function = sample_gaussian_square_waveform(
            x,
            fractional_rise=fractional_rise,
            regularize=regularize,
            fractional_top_width=fractional_top_width,
        )
        waveform = GaussianSquareFunction(
            std_dev=std_dev,
            zero_at_edges=zero_at_edges,
            square_width=square_width,
            width=width,
            amp=1.0,
        )
        y_class = waveform(t_array)
        assert np.allclose(y_function, y_class, atol=1e-8)


class TestGaussianSquareWaveformShapeFromLegacy:
    """Tests legacy classmethod constructor for ``GaussianSquareWaveformShape``."""

    def test_from_legacy(self):
        """Checks fractional_rise, regularize, and fractional_top_width from legacy
        GaussianSquareWaveform."""
        shape = GaussianSquareWaveformShape.from_legacy(
            std_dev=0.01, width=0.2, zero_at_edges=True, square_width=0.06
        )
        assert np.isclose(shape.fractional_rise, 2.0 * 0.01 / 0.2)
        assert shape.regularize is True
        assert np.isclose(shape.fractional_top_width, 0.06 / 0.2)

    def test_from_legacy_defaults(self):
        """Checks default zero_at_edges=False."""
        shape = GaussianSquareWaveformShape.from_legacy(
            std_dev=0.02, width=0.4, square_width=0.2
        )
        assert shape.regularize is False


class TestFractionalRiseValidation:
    """Tests that public functions reject non-positive ``fractional_rise`` values."""

    @pytest.mark.parametrize("fractional_rise", [0.0, -0.1, -1.0])
    def test_sample_gaussian_square_waveform_raises_for_non_positive_rise(
        self, fractional_rise
    ):
        """Checks that ``sample_gaussian_square_waveform`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            sample_gaussian_square_waveform(
                np.array([0.0]), fractional_rise=fractional_rise
            )

    @pytest.mark.parametrize("fractional_rise", [0.0, -0.1, -1.0])
    def test_sample_gaussian_square_waveform_derivative_raises_for_non_positive_rise(
        self, fractional_rise
    ):
        """Checks that ``sample_gaussian_square_waveform_derivative`` raises
        ``ValueError``."""
        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            sample_gaussian_square_waveform_derivative(
                np.array([0.0]), fractional_rise=fractional_rise
            )

    @pytest.mark.parametrize("fractional_rise", [0.0, -0.1, -1.0])
    def test_gaussian_square_waveform_shape_raises_for_non_positive_rise(
        self, fractional_rise
    ):
        """Checks that ``GaussianSquareWaveformShape`` raises ``ValueError`` on
        construction."""
        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            GaussianSquareWaveformShape(fractional_rise=fractional_rise)


class TestGaussianSquareWaveformShapeDelegation:
    """Tests shape methods match direct calls to sampling functions."""

    def test_evaluate_rejects_invalid_fractional_top_width(self):
        """Checks that the shape wrapper surfaces top-width validation."""

        with pytest.raises(
            ValueError,
            match=("fractional_top_width must satisfy 0 <= fractional_top_width <= 1"),
        ):
            GaussianSquareWaveformShape(fractional_top_width=1.1)

    def test_evaluate_rejects_zero_fractional_rise(self):
        """Checks that the shape wrapper surfaces rise validation."""

        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            GaussianSquareWaveformShape(fractional_rise=0.0)

    def test_evaluate_matches_raw_function(self):
        """Checks evaluate output matches sample_gaussian_square_waveform."""

        x = np.array([-0.5, 0.5])
        shape = GaussianSquareWaveformShape(
            fractional_rise=0.2, regularize=True, fractional_top_width=0.3
        )
        assert np.allclose(
            shape.evaluate(x),
            sample_gaussian_square_waveform(
                x,
                fractional_rise=0.2,
                regularize=True,
                fractional_top_width=0.3,
            ),
        )

    def test_derivative_matches_raw_function(self):
        """Checks derivative output matches sample_gaussian_square_waveform_derivative."""

        x = np.array([-0.5, 0.5])
        shape = GaussianSquareWaveformShape(
            fractional_rise=0.2, regularize=True, fractional_top_width=0.3
        )
        assert np.allclose(
            shape.derivative(x, order=1),
            sample_gaussian_square_waveform_derivative(
                x,
                order=1,
                fractional_rise=0.2,
                regularize=True,
                fractional_top_width=0.3,
            ),
        )

    def test_from_absolute_generates_correct_fractional_parameters(self):
        """Checks that the from_absolute method generates the correct fractional
        parameters."""

        shape = GaussianSquareWaveformShape.from_absolute(
            width=160e-9, absolute_top_width=80e-9, absolute_rise=16e-9, regularize=True
        )
        assert np.isclose(shape.fractional_rise, 0.1)  # 16ns / 160ns
        assert shape.regularize is True
        assert np.isclose(shape.fractional_top_width, 0.5)  # 80ns / 160ns
