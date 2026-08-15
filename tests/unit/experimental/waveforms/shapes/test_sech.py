# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the functions for sampling a Sech waveform shape."""

import numpy as np
import pytest

from qat.experimental.waveforms.shapes.exceptions import DerivativeOrderNotImplementedError
from qat.experimental.waveforms.shapes.sech import (
    _MAX_COSH_ARG,
    SechWaveformShape,
    sample_sech_waveform,
    sample_sech_waveform_derivative,
)
from qat.utils.waveform import SechFunction

from tests.unit.experimental.waveforms.utils import estimate_derivative_at_point


class TestSampleSechWaveform:
    """Tests the ``sample_sech_waveform`` function."""

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.25, False)])
    def test_waveform_is_symmetric_around_zero(self, fractional_breadth, regularize):
        """Checks the waveform is symmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_sech_waveform(
            x, fractional_breadth=fractional_breadth, regularize=regularize
        )
        y_negative = sample_sech_waveform(
            -x, fractional_breadth=fractional_breadth, regularize=regularize
        )
        assert np.allclose(y_positive, y_negative)

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.25, False)])
    def test_waveform_is_one_at_center(self, fractional_breadth, regularize):
        """Checks the waveform is one at x=0."""
        x = np.array([0.0])
        y = sample_sech_waveform(
            x, fractional_breadth=fractional_breadth, regularize=regularize
        )
        assert np.isclose(y[0], 1.0)

    @pytest.mark.parametrize("fractional_breadth", [1.0, 0.25])
    def test_zero_at_edges_when_normalization_is_true(self, fractional_breadth):
        """Checks the waveform is zero at the edges when regularize is True."""
        x = np.array([-1.0, 1.0])
        y = sample_sech_waveform(x, fractional_breadth=fractional_breadth, regularize=True)
        assert np.allclose(y, 0.0, atol=1e-8)

    @pytest.mark.parametrize("fractional_breadth", [1.0, 0.25])
    def test_values_are_positive_when_not_normalized(self, fractional_breadth):
        """Checks the unnormalized waveform stays positive on the full domain."""
        x = np.linspace(-1, 1, 100)
        y = sample_sech_waveform(x, fractional_breadth=fractional_breadth, regularize=False)
        assert np.all(y > 0.0)

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.25, False)])
    def test_rise_and_fall_are_monotone_when_moving_towards_edges(
        self, fractional_breadth, regularize
    ):
        """Checks the fractional_rise is increasing and the fall is decreasing when moving
        towards the edges."""
        x = np.linspace(-1, 1, 100)
        y = sample_sech_waveform(
            x, fractional_breadth=fractional_breadth, regularize=regularize
        )
        rise_mask = x < 0
        fall_mask = x > 0
        assert np.all(np.diff(y[rise_mask]) >= 0.0)
        assert np.all(np.diff(y[fall_mask]) <= 0.0)

    @pytest.mark.parametrize("regularize", [True, False])
    def test_above_max_cosh_arg_is_clipped(self, regularize):
        """Checks that values of x/fractional_breadth above _MAX_COSH_ARG are clipped.

        In the past, arguments such as this have raised overflow errors.
        """
        fractional_breadth = 0.98 / _MAX_COSH_ARG
        x = np.asarray([-1.0, 1.0])
        y = sample_sech_waveform(
            x, fractional_breadth=fractional_breadth, regularize=regularize
        )
        assert np.allclose(y, 0.0, atol=1e-12)


class TestSampleSechWaveformDerivative:
    """Tests the ``sample_sech_waveform_derivative`` function.

    Tests expected high-level properties of derivatives, such as symmetry and sign. Also
    includes numerical comparisons to ensure derivatives are approximately equivalent.
    """

    def test_order_higher_than_two_raises_derivative_order_undefined_error(self):
        """Checks that order greater than two raises DerivativeOrderNotImplementedError."""
        x = np.linspace(-1, 1, 100)
        with pytest.raises(
            DerivativeOrderNotImplementedError,
            match=(
                r"The derivative of order 3 exists for waveform shape 'Sech' but is not "
                r"implemented\."
            ),
        ):
            sample_sech_waveform_derivative(x, order=3)

    def test_derivative_order_zero_is_equivalent_to_sample_sech_waveform(self):
        """Checks that order-zero derivative equals the waveform samples."""
        x = np.linspace(-1, 1, 100)
        y_sample = sample_sech_waveform(x, fractional_breadth=0.5, regularize=True)
        y_derivative = sample_sech_waveform_derivative(
            x, order=0, fractional_breadth=0.5, regularize=True
        )
        assert np.allclose(y_sample, y_derivative, atol=1e-8)

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.25, False)])
    def test_first_order_derivative_is_antisymmetric(self, fractional_breadth, regularize):
        """Checks the first derivative is antisymmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_sech_waveform_derivative(
            x, order=1, fractional_breadth=fractional_breadth, regularize=regularize
        )
        y_negative = sample_sech_waveform_derivative(
            -x, order=1, fractional_breadth=fractional_breadth, regularize=regularize
        )
        assert np.allclose(y_positive, -y_negative, atol=1e-8)

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.25, False)])
    def test_second_order_derivative_is_symmetric(self, fractional_breadth, regularize):
        """Checks the second derivative is symmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_sech_waveform_derivative(
            x, order=2, fractional_breadth=fractional_breadth, regularize=regularize
        )
        y_negative = sample_sech_waveform_derivative(
            -x, order=2, fractional_breadth=fractional_breadth, regularize=regularize
        )
        assert np.allclose(y_positive, y_negative, atol=1e-8)

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.25, False)])
    def test_first_derivative_is_positive_on_rise_and_negative_on_fall(
        self, fractional_breadth, regularize
    ):
        """Checks first derivative is positive on x<0 and negative on x>0."""
        x = np.linspace(-1, 1, 200)
        y_derivative = sample_sech_waveform_derivative(
            x, order=1, fractional_breadth=fractional_breadth, regularize=regularize
        )
        rise_mask = x < 0
        fall_mask = x > 0
        assert np.all(y_derivative[rise_mask] > 0.0)
        assert np.all(y_derivative[fall_mask] < 0.0)

    @pytest.mark.parametrize("fractional_breadth", [1.0, 0.25])
    def test_normalization_scales_derivatives_consistently(self, fractional_breadth):
        """Checks regularize applies only a scale factor to derivatives."""
        x = np.asarray(
            [
                -0.5 * fractional_breadth,
                -0.25 * fractional_breadth,
                0.25 * fractional_breadth,
                0.5 * fractional_breadth,
            ]
        )
        scales = []
        for order in [1, 2]:
            y_normalized = sample_sech_waveform_derivative(
                x, fractional_breadth=fractional_breadth, order=order, regularize=True
            )
            y_unnormalized = sample_sech_waveform_derivative(
                x, fractional_breadth=fractional_breadth, order=order, regularize=False
            )
            scale = y_normalized / y_unnormalized
            assert np.allclose(scale, scale[0])
            scales.append(scale[0])
        assert np.allclose(scales, scales[0])

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.25, False)])
    @pytest.mark.parametrize("order", [1, 2])
    def test_derivative_matches_numerical_estimate(
        self, fractional_breadth, regularize, order
    ):
        """Checks derivatives match finite-difference estimates."""
        x = np.linspace(-0.5, 0.5, 50)
        y_derivative = sample_sech_waveform_derivative(
            x, fractional_breadth=fractional_breadth, regularize=regularize, order=order
        )

        def fn(x_val):
            return sample_sech_waveform(
                x_val, fractional_breadth=fractional_breadth, regularize=regularize
            )

        numerical_y_derivative = np.asarray(
            [estimate_derivative_at_point(fn, x_i, order=order) for x_i in x]
        )
        assert np.allclose(y_derivative, numerical_y_derivative, atol=1e-3)


class TestParityWithPydanticSechWaveform:
    """Tests sampling consistency with the pydantic Sech waveform class.

    The Sech waveform implements the legacy ``SechWaveform`` under the parameterisation
    ``fractional_breadth = 2 * std_dev / width`` and ``regularize = False``.
    """

    @pytest.mark.parametrize("fractional_breadth", [1.0, 0.25])
    @pytest.mark.parametrize("width", [2.0, 160e-9])
    def test_samples_match_when_not_normalized(self, fractional_breadth, width):
        """Checks unnormalized samples match the pydantic class."""
        std_dev = fractional_breadth * width / 2.0

        x = np.linspace(-1, 1, 100)
        scale = width / 2  # Maps x in [-1, 1] to t in [-width/2, width/2]
        t_array = x * scale

        y_function = sample_sech_waveform(
            x, fractional_breadth=fractional_breadth, regularize=False
        )
        waveform = SechFunction(std_dev=std_dev, amp=1.0)
        y_class = waveform(t_array)
        assert np.allclose(y_function, y_class, atol=1e-8)


class TestSechWaveformShapeFromLegacy:
    """Tests legacy classmethod constructor for ``SechWaveformShape``."""

    def test_from_legacy(self):
        """Checks fractional_breadth and regularize from legacy SechWaveform parameters."""
        shape = SechWaveformShape.from_legacy(std_dev=0.04, width=0.4, zero_at_edges=True)
        assert np.isclose(shape.fractional_breadth, 2.0 * 0.04 / 0.4)
        assert shape.regularize is True

    def test_from_legacy_default_edges(self):
        """Checks zero_at_edges defaults to False."""
        shape = SechWaveformShape.from_legacy(std_dev=0.04, width=0.4)
        assert shape.regularize is False


class TestFractionalBreadthValidation:
    """Tests that public functions reject negative ``fractional_breadth`` values."""

    @pytest.mark.parametrize("fractional_breadth", [0.0, -0.1, -1.0])
    def test_sample_sech_waveform_raises_for_non_positive_breadth(self, fractional_breadth):
        """Checks that ``sample_sech_waveform`` raises ``ValueError``."""
        with pytest.raises(
            ValueError, match="fractional_breadth must be greater than zero"
        ):
            sample_sech_waveform(np.array([0.0]), fractional_breadth=fractional_breadth)

    @pytest.mark.parametrize("fractional_breadth", [0.0, -0.1, -1.0])
    def test_sample_sech_waveform_derivative_raises_for_non_positive_breadth(
        self, fractional_breadth
    ):
        """Checks that ``sample_sech_waveform_derivative`` raises ``ValueError``."""
        with pytest.raises(
            ValueError, match="fractional_breadth must be greater than zero"
        ):
            sample_sech_waveform_derivative(
                np.array([0.0]), fractional_breadth=fractional_breadth
            )

    @pytest.mark.parametrize("fractional_breadth", [0.0, -0.1, -1.0])
    def test_sech_waveform_shape_raises_for_non_positive_breadth(self, fractional_breadth):
        """Checks that ``SechWaveformShape`` raises ``ValueError`` on construction."""
        with pytest.raises(
            ValueError, match="fractional_breadth must be greater than zero"
        ):
            SechWaveformShape(fractional_breadth=fractional_breadth)


class TestSechWaveformShapeDelegation:
    """Tests shape methods match direct calls to sampling functions."""

    def test_evaluate_matches_raw_function(self):
        """Checks evaluate output matches sample_sech_waveform."""

        x = np.array([-0.5, 0.5])
        shape = SechWaveformShape(fractional_breadth=0.4, regularize=True)
        assert np.allclose(
            shape.evaluate(x),
            sample_sech_waveform(x, fractional_breadth=0.4, regularize=True),
        )

    def test_derivative_matches_raw_function(self):
        """Checks derivative output matches sample_sech_waveform_derivative."""

        x = np.array([-0.5, 0.5])
        shape = SechWaveformShape(fractional_breadth=0.4, regularize=True)
        assert np.allclose(
            shape.derivative(x, order=1),
            sample_sech_waveform_derivative(
                x, order=1, fractional_breadth=0.4, regularize=True
            ),
        )

    def test_from_absolute_generates_correct_fractional_parameters(self):
        """Checks that the from_absolute method generates the correct fractional
        parameters."""

        shape = SechWaveformShape.from_absolute(
            width=160e-9, absolute_breadth=16e-9, regularize=True
        )
        assert np.isclose(shape.fractional_breadth, 0.1)  # 16ns / 160ns
        assert shape.regularize is True
