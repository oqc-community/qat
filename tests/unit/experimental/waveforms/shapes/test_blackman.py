# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the functions for sampling a Blackman waveform shape."""

import numpy as np
import pytest

from qat.experimental.waveforms.shapes.blackman import (
    BlackmanWaveformShape,
    sample_blackman_waveform,
    sample_blackman_waveform_derivative,
)
from qat.utils.waveform import BlackmanFunction

from tests.unit.experimental.waveforms.utils import estimate_derivative_at_point


class TestSampleBlackmanWaveform:
    """Tests the ``sample_blackman_waveform`` function."""

    def test_waveform_is_symmetric_around_zero(self):
        """Checks the waveform is symmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_blackman_waveform(x)
        y_negative = sample_blackman_waveform(-x)
        assert np.allclose(y_positive, y_negative, atol=1e-10)

    def test_waveform_has_equal_values_at_both_edges(self):
        """Checks both edges share the same small non-zero pedestal value.

        The exact Blackman coefficients (7938/18608, 9240/18608, 1430/18608) satisfy
        A0 + A1 + A2 = 1 but A0 - A1 + A2 ≈ 0.00688, so the window is not exactly
        zero at its edges.  Both edges must share the same pedestal value.
        """
        y_left = sample_blackman_waveform(np.array([-1.0]))
        y_right = sample_blackman_waveform(np.array([1.0]))
        assert np.isclose(y_left[0], y_right[0], atol=1e-10)

    def test_waveform_is_one_at_center(self):
        """Checks the waveform reaches one at x=0."""
        x = np.array([0.0])
        y = sample_blackman_waveform(x)
        assert np.isclose(y[0], 1.0, atol=1e-10)

    def test_waveform_is_non_negative_on_domain(self):
        """Checks the waveform is non-negative on [-1, 1]."""
        x = np.linspace(-1, 1, 1000)
        y = sample_blackman_waveform(x)
        assert np.all(y >= -1e-12)


class TestSampleBlackmanWaveformDerivative:
    """Tests the ``sample_blackman_waveform_derivative`` function."""

    def test_derivative_order_zero_is_equivalent_to_sample_blackman_waveform(self):
        """Checks that order-zero derivative equals waveform samples."""
        x = np.linspace(-1, 1, 100)
        y_sample = sample_blackman_waveform(x)
        y_derivative = sample_blackman_waveform_derivative(x, order=0)
        assert np.allclose(y_sample, y_derivative, atol=1e-10)

    @pytest.mark.parametrize("order", [1, 3])
    def test_odd_order_derivatives_are_antisymmetric(self, order):
        """Checks odd-order derivatives are antisymmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_blackman_waveform_derivative(x, order=order)
        y_negative = sample_blackman_waveform_derivative(-x, order=order)
        assert np.allclose(y_positive, -y_negative, atol=1e-10)

    @pytest.mark.parametrize("order", [2, 4])
    def test_even_order_derivatives_are_symmetric(self, order):
        """Checks even-order derivatives are symmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_blackman_waveform_derivative(x, order=order)
        y_negative = sample_blackman_waveform_derivative(-x, order=order)
        assert np.allclose(y_positive, y_negative, atol=1e-10)

    def test_first_derivative_is_zero_at_center_and_edges(self):
        """Checks first derivative vanishes at x=-1, x=0, and x=+1."""
        x = np.array([-1.0, 0.0, 1.0])
        y = sample_blackman_waveform_derivative(x, order=1)
        assert np.allclose(y, 0.0, atol=1e-10)

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_derivative_matches_numerical_estimate(self, order):
        """Checks derivatives match finite-difference estimates."""
        x = np.linspace(-0.5, 0.5, 50)
        y_derivative = sample_blackman_waveform_derivative(x, order=order)

        def fn(x_val):
            return sample_blackman_waveform(x_val)

        numerical_y_derivative = np.asarray(
            [estimate_derivative_at_point(fn, x_i, order=order) for x_i in x]
        )
        assert np.allclose(y_derivative, numerical_y_derivative, atol=1e-3 * (np.pi**order))


class TestParityWithPydanticBlackmanWaveform:
    """Tests sampling consistency with the pydantic Blackman waveform class."""

    @pytest.mark.parametrize("width", [2.0, 160e-9])
    def test_samples_match_pydantic_class(self, width):
        """Checks waveform samples match ``BlackmanFunction``."""
        x = np.linspace(-1, 1, 100)
        scale = width / 2
        t_array = x * scale

        y_function = sample_blackman_waveform(x)
        waveform = BlackmanFunction(width=width, amp=1.0)
        y_class = waveform(t_array)
        assert np.allclose(y_function, y_class, atol=1e-10)

    @pytest.mark.parametrize("width", [2.0, 160e-9])
    def test_first_derivative_matches_pydantic_class(self, width):
        """Checks first derivative matches ``BlackmanFunction.derivative``.

        ``sample_blackman_waveform_derivative`` differentiates with respect to x,
        while ``BlackmanFunction.derivative`` differentiates with respect to time.
        These are related by ``dx/dt = 2/width``.
        """
        x = np.linspace(-1, 1, 100)
        scale = width / 2
        t_array = x * scale

        y_function_derivative = sample_blackman_waveform_derivative(x, order=1) / scale
        waveform = BlackmanFunction(width=width, amp=1.0)
        y_class_derivative = waveform.derivative(t_array)
        # cos(pi/2) carries ~1e-16 floating-point error; after dividing by scale
        # (which is as small as width/2 = 80 ns) the boundary roundoff can reach ~1e-8.
        assert np.allclose(y_function_derivative, y_class_derivative, atol=1e-7)


class TestBlackmanWaveformShapeDelegation:
    """Tests shape methods match direct calls to sampling functions."""

    def test_evaluate_matches_raw_function(self):
        """Checks evaluate output matches sample_blackman_waveform."""

        x = np.array([-0.5, 0.5])
        shape = BlackmanWaveformShape()
        assert np.allclose(shape.evaluate(x), sample_blackman_waveform(x))

    def test_derivative_matches_raw_function(self):
        """Checks derivative output matches sample_blackman_waveform_derivative."""

        x = np.array([-0.5, 0.5])
        shape = BlackmanWaveformShape()
        assert np.allclose(
            shape.derivative(x, order=1),
            sample_blackman_waveform_derivative(x, order=1),
        )
