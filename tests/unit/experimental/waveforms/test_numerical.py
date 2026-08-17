# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the numerical implementations of waveform derivatives used in evaluation
pathways."""

import numpy as np
import pytest

from qat.experimental.waveforms.numerical import numerical_derivative


class TestNumericalDerivative:
    """Tests that numerical derivative behaves in an expected manner."""

    def test_different_length_x_and_y_raises_value_error(self):
        """Tests that a ValueError is raised when x and y have different lengths."""
        x = np.asarray([0, 1, 2])
        y = np.asarray([0, 1])
        with pytest.raises(ValueError, match="y and x must have the same length"):
            numerical_derivative(y, x)

    def test_fewer_than_two_points_raises_value_error(self):
        """Tests that a ValueError is raised when x and y have fewer than 2 points."""
        x = np.asarray([0])
        y = np.asarray([0])
        with pytest.raises(ValueError, match="y and x must have at least 2 points"):
            numerical_derivative(y, x)

    def test_derivative_evaluates_with_two_points(self):
        """Tests that the numerical derivative evaluates correctly with two points."""
        x = np.asarray([0, 1])
        y = np.asarray([0, 1])
        derivative = numerical_derivative(y, x)
        expected_derivative = np.asarray([1, 1])
        assert np.allclose(derivative, expected_derivative)

    def test_derivative_of_linear_function_is_constant(self):
        """Tests that the numerical derivative of a linear function is constant."""
        x = np.linspace(-1, 1, 100)
        y = 2 * x + 3
        derivative = numerical_derivative(y, x)
        expected_derivative = np.full_like(x, 2)
        assert np.allclose(derivative, expected_derivative)

    def test_derivative_gives_expected_value(self):
        """Builds a quadratic function and tests that the numerical derivative roughly
        matches the analytical expectation."""

        mean_errors = []
        for num_points in [10, 100, 1000, 10000]:
            x = np.linspace(-1, 1, num_points + 1)
            y = x**2
            expected_derivative = 2 * x
            derivative = numerical_derivative(y, x)
            assert len(derivative) == len(x)

            errors = np.sqrt((derivative - expected_derivative) ** 2)
            mean_errors.append(np.mean(errors))
            if len(mean_errors) > 1:
                assert mean_errors[-2] > 10 * mean_errors[-1], (
                    "Error expected to decrease at an order beyond linear."
                )
