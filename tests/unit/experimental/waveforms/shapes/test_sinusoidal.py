# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the functions for sampling a Sinusoidal waveform shape."""

import numpy as np
import pytest

from qat.experimental.waveforms.shapes.sinusoidal import (
    SinusoidalWaveformShape,
    sample_sinusoidal_waveform,
    sample_sinusoidal_waveform_derivative,
    sample_sinusoidal_waveform_derivative_from_frequency,
    sample_sinusoidal_waveform_from_frequency,
)
from qat.utils.waveform import Cos, Sin

from tests.unit.experimental.waveforms.utils import estimate_derivative_at_point


class TestSampleSinusoidalWaveform:
    """Tests the ``sample_sinusoidal_waveform`` function."""

    @pytest.mark.parametrize("number_of_periods", [0.5, 1.0, 2.0])
    def test_waveform_matches_sine_formula(self, number_of_periods):
        """Checks the waveform matches the analytic sine formula."""
        x = np.linspace(-1, 1, 100)
        y = sample_sinusoidal_waveform(x, number_of_periods=number_of_periods)
        expected = np.sin(2 * np.pi * number_of_periods * x)
        assert np.allclose(y, expected, atol=1e-10)

    @pytest.mark.parametrize("internal_phase", [0.0, np.pi / 4, np.pi / 2])
    def test_internal_phase_shifts_waveform(self, internal_phase):
        """Checks that internal_phase shifts the waveform correctly."""
        x = np.linspace(-1, 1, 100)
        y = sample_sinusoidal_waveform(
            x, number_of_periods=0.5, internal_phase=internal_phase
        )
        expected = np.sin(2 * np.pi * 0.5 * x + internal_phase)
        assert np.allclose(y, expected, atol=1e-10)

    def test_default_parameters_produce_full_period_sine(self):
        """Checks defaults give a full-period sine: 0 at x=-1, 0 at x=0, 0 at x=+1."""
        x = np.array([-1.0, 0.0, 1.0])
        y = sample_sinusoidal_waveform(x)
        assert np.allclose(y, [0.0, 0.0, 0.0], atol=1e-10)

    def test_cosine_via_internal_phase(self):
        """Checks that internal_phase=pi/2 gives a cosine waveform."""
        x = np.linspace(-1, 1, 100)
        y = sample_sinusoidal_waveform(x, number_of_periods=0.5, internal_phase=np.pi / 2)
        expected = np.cos(2 * np.pi * 0.5 * x)
        assert np.allclose(y, expected, atol=1e-10)

    @pytest.mark.parametrize("number_of_periods", [0.5, 1.0])
    def test_antisymmetric_for_zero_phase(self, number_of_periods):
        """Checks that a zero-phase sinusoidal is antisymmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_sinusoidal_waveform(
            x, number_of_periods=number_of_periods, internal_phase=0.0
        )
        y_negative = sample_sinusoidal_waveform(
            -x, number_of_periods=number_of_periods, internal_phase=0.0
        )
        assert np.allclose(y_positive, -y_negative, atol=1e-10)


class TestSampleSinusoidalWaveformDerivative:
    """Tests the ``sample_sinusoidal_waveform_derivative`` function."""

    def test_order_zero_matches_sample_sinusoidal_waveform(self):
        """Checks order-zero derivative equals the waveform samples."""
        x = np.linspace(-1, 1, 100)
        y_sample = sample_sinusoidal_waveform(x, number_of_periods=0.5)
        y_deriv = sample_sinusoidal_waveform_derivative(x, 0, number_of_periods=0.5)
        assert np.allclose(y_sample, y_deriv, atol=1e-10)

    @pytest.mark.parametrize("number_of_periods", [0.5, 1.0])
    def test_first_derivative_matches_cosine(self, number_of_periods):
        """Checks first derivative equals 2*pi*N*cos(2*pi*N*x)."""
        x = np.linspace(-1, 1, 100)
        y = sample_sinusoidal_waveform_derivative(x, 1, number_of_periods=number_of_periods)
        expected = (2 * np.pi * number_of_periods) * np.cos(
            2 * np.pi * number_of_periods * x
        )
        assert np.allclose(y, expected, atol=1e-10)

    @pytest.mark.parametrize("order", [1, 3, 5])
    def test_odd_order_derivatives_are_symmetric(self, order):
        """Checks odd-order derivatives are symmetric (even functions) around zero.

        The n-th derivative of sin(2*pi*N*x) with odd n is proportional to cos(2*pi*N*x),
        which is an even function: f(-x) == f(x).
        """
        x = np.linspace(0, 1, 100)
        y_positive = sample_sinusoidal_waveform_derivative(
            x, order, number_of_periods=0.5, internal_phase=0.0
        )
        y_negative = sample_sinusoidal_waveform_derivative(
            -x, order, number_of_periods=0.5, internal_phase=0.0
        )
        assert np.allclose(y_positive, y_negative, atol=1e-10)

    @pytest.mark.parametrize("order", [2, 4])
    def test_even_order_derivatives_are_antisymmetric(self, order):
        """Checks even-order derivatives are antisymmetric (odd functions) around zero.

        The n-th derivative of sin(2*pi*N*x) with even n is proportional to sin(2*pi*N*x),
        which is an odd function: f(-x) == -f(x).
        """
        x = np.linspace(0, 1, 100)
        y_positive = sample_sinusoidal_waveform_derivative(
            x, order, number_of_periods=0.5, internal_phase=0.0
        )
        y_negative = sample_sinusoidal_waveform_derivative(
            -x, order, number_of_periods=0.5, internal_phase=0.0
        )
        assert np.allclose(y_positive, -y_negative, atol=1e-10)

    @pytest.mark.parametrize("number_of_periods", [0.5, 1.0])
    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_derivative_matches_numerical_estimate(self, number_of_periods, order):
        """Checks derivatives match finite-difference estimates."""
        x = np.linspace(-0.5, 0.5, 50)
        y_deriv = sample_sinusoidal_waveform_derivative(
            x, order, number_of_periods=number_of_periods
        )

        def fn(x_val):
            return sample_sinusoidal_waveform(x_val, number_of_periods=number_of_periods)

        numerical = np.asarray(
            [estimate_derivative_at_point(fn, x_i, order=order) for x_i in x]
        )
        # Scale tolerance by derivative amplitude: finite-difference roundoff is
        # proportional to the function's stencil values, not the derivative magnitude,
        # so absolute errors grow with (2*pi*N)^order.
        scale = (2 * np.pi * number_of_periods) ** order
        assert np.allclose(y_deriv, numerical, atol=1e-3 * scale)

    @pytest.mark.parametrize("order", [1, 3])
    def test_each_derivative_scales_by_2_pi_n(self, order):
        """Checks each derivative order multiplies by an extra 2*pi*N factor.

        Verified at x = 1/(8*N) where the fundamental phase 2*pi*N*x = pi/4, so |sin(phase)|
        = |cos(phase)| = sqrt(2)/2 for all derivative orders. This makes the amplitude ratio
        exactly 2*pi*N regardless of sign.
        """
        n = 0.5
        # At x = 1/(8*N), the argument 2*pi*N*x = pi/4, so |sin| = |cos| = sqrt(2)/2.
        # Consecutive derivatives therefore have |ratio| = 2*pi*N exactly.
        x = np.array([1.0 / (8 * n)])
        y_lower = sample_sinusoidal_waveform_derivative(x, order - 1, number_of_periods=n)
        y_upper = sample_sinusoidal_waveform_derivative(x, order, number_of_periods=n)
        ratio = float(np.abs(y_upper[0]) / np.abs(y_lower[0]))
        assert np.isclose(ratio, 2 * np.pi * n, rtol=1e-10)


class TestSampleSinusoidalWaveformFromFrequency:
    """Tests the ``sample_sinusoidal_waveform_from_frequency`` function."""

    @pytest.mark.parametrize("frequency, width", [(5e6, 160e-9), (1.0, 2.0)])
    def test_matches_sample_sinusoidal_waveform(self, frequency, width):
        """Checks that the from-frequency variant equals the standard call with
        number_of_periods = frequency * width."""
        x = np.linspace(-1, 1, 100)
        y_freq = sample_sinusoidal_waveform_from_frequency(
            x, frequency=frequency, width=width
        )
        y_periods = sample_sinusoidal_waveform(x, number_of_periods=frequency * width)
        assert np.allclose(y_freq, y_periods, atol=1e-10)

    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("frequency, width", [(5e6, 160e-9), (1.0, 2.0)])
    def test_derivative_from_frequency_matches_standard(self, order, frequency, width):
        """Checks the derivative from-frequency variant equals the standard call."""
        x = np.linspace(-1, 1, 100)
        y_freq = sample_sinusoidal_waveform_derivative_from_frequency(
            x, order, frequency=frequency, width=width
        )
        y_periods = sample_sinusoidal_waveform_derivative(
            x, order, number_of_periods=frequency * width
        )
        assert np.allclose(y_freq, y_periods, atol=1e-10)


class TestParityWithPydanticSinusoidalWaveform:
    """Tests sampling consistency with the pydantic Sin and Cos waveform classes.

    The Sinusoidal waveform implements the legacy ``Sin`` under the parameterisation
    ``frequency = number_of_periods / width * 2`` with ``t = x * width / 2``.
    Using ``internal_phase = pi/2`` and the same parameterisation gives ``Cos``.
    """

    @pytest.mark.parametrize("number_of_periods", [0.5, 1.0])
    @pytest.mark.parametrize("width", [2.0, 160e-9])
    def test_sine_samples_match_pydantic_sin_class(self, number_of_periods, width):
        """Checks ``sample_sinusoidal_waveform`` matches the legacy ``Sin`` class."""
        frequency = number_of_periods / (width / 2)

        x = np.linspace(-1, 1, 100)
        scale = width / 2  # Maps x in [-1, 1] to t in [-width/2, width/2]
        t_array = x * scale

        y_function = sample_sinusoidal_waveform(
            x, number_of_periods=number_of_periods, internal_phase=0.0
        )
        waveform = Sin(frequency=frequency, internal_phase=0.0, amp=1.0)
        y_class = waveform(t_array)
        assert np.allclose(y_function, y_class, atol=1e-10)

    @pytest.mark.parametrize("number_of_periods", [0.5, 1.0])
    @pytest.mark.parametrize("width", [2.0, 160e-9])
    def test_cosine_samples_match_pydantic_cos_class(self, number_of_periods, width):
        """Checks ``sample_sinusoidal_waveform`` with ``internal_phase=pi/2`` matches the
        legacy ``Cos`` class."""
        frequency = number_of_periods / (width / 2)

        x = np.linspace(-1, 1, 100)
        scale = width / 2
        t_array = x * scale

        y_function = sample_sinusoidal_waveform(
            x, number_of_periods=number_of_periods, internal_phase=np.pi / 2
        )
        waveform = Cos(frequency=frequency, internal_phase=0.0, amp=1.0)
        y_class = waveform(t_array)
        assert np.allclose(y_function, y_class, atol=1e-10)


class TestSinusoidalWaveformShapeFromFrequency:
    """Tests legacy classmethod constructor for ``SinusoidalWaveformShape``."""

    def test_from_frequency(self):
        """Checks number_of_periods from frequency and width."""
        shape = SinusoidalWaveformShape.from_frequency(frequency=5e6, width=160e-9)
        assert np.isclose(shape.number_of_periods, 5e6 * 160e-9)
        assert np.isclose(shape.internal_phase, 0.0)

    def test_from_frequency_with_phase(self):
        """Checks internal_phase is forwarded."""
        shape = SinusoidalWaveformShape.from_frequency(
            frequency=1.0, width=2.0, internal_phase=0.5
        )
        assert np.isclose(shape.number_of_periods, 2.0)
        assert np.isclose(shape.internal_phase, 0.5)


class TestSinusoidalWaveformShapeDelegation:
    """Tests shape methods match direct calls to sampling functions."""

    def test_evaluate_matches_raw_function(self):
        """Checks evaluate output matches sample_sinusoidal_waveform."""

        x = np.array([-0.5, 0.5])
        shape = SinusoidalWaveformShape(number_of_periods=1.5, internal_phase=0.7)
        assert np.allclose(
            shape.evaluate(x),
            sample_sinusoidal_waveform(x, number_of_periods=1.5, internal_phase=0.7),
        )

    def test_derivative_matches_raw_function(self):
        """Checks derivative output matches sample_sinusoidal_waveform_derivative."""

        x = np.array([-0.5, 0.5])
        shape = SinusoidalWaveformShape(number_of_periods=1.5, internal_phase=0.7)
        assert np.allclose(
            shape.derivative(x, order=1),
            sample_sinusoidal_waveform_derivative(
                x,
                order=1,
                number_of_periods=1.5,
                internal_phase=0.7,
            ),
        )
