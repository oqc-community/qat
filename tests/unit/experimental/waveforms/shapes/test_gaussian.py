# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the functions for sampling a Gaussian waveform shape."""

import numpy as np
import pytest

from qat.experimental.waveforms.shapes.gaussian import (
    GaussianWaveformShape,
    sample_gaussian_waveform,
    sample_gaussian_waveform_derivative,
)
from qat.utils.waveform import (
    GaussianFunction,
    GaussianZeroEdgeFunction,
    SofterGaussianFunction,
)

from tests.unit.experimental.waveforms.utils import estimate_derivative_at_point


class TestSampleGaussianWaveform:
    """Tests the ``sample_gaussian_waveform`` function."""

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.1, False)])
    def test_gaussian_is_one_at_zero(self, fractional_breadth, regularize):
        """Checks the expected peak has value one."""
        x = np.array([0.0])
        y = sample_gaussian_waveform(
            x, fractional_breadth=fractional_breadth, regularize=regularize
        )
        assert np.isclose(y[0], 1.0)

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.1, False)])
    def test_gaussian_is_symmetric_around_zero(self, fractional_breadth, regularize):
        """Creates an array from [0, 1] and equivalently [0, -1], and checks that the values
        are equal."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_gaussian_waveform(
            x, fractional_breadth=fractional_breadth, regularize=regularize
        )
        y_negative = sample_gaussian_waveform(
            -x, fractional_breadth=fractional_breadth, regularize=regularize
        )
        assert np.allclose(y_positive, y_negative)

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.1, False)])
    def test_gaussian_is_strictly_decreasing_away_from_zero(
        self, fractional_breadth, regularize
    ):
        """Checks that the values are decreasing away from zero."""
        x = np.linspace(0, 1, 100)
        y = sample_gaussian_waveform(
            x, fractional_breadth=fractional_breadth, regularize=regularize
        )
        assert np.all(np.diff(y) < 0)

    @pytest.mark.parametrize("fractional_breadth", [1.0, 0.1])
    def test_zero_at_edges_when_normalization_is_true(self, fractional_breadth):
        """Checks that the values at the edges are zero when regularize is True."""
        x = np.array([-1.0, 1.0])
        y = sample_gaussian_waveform(
            x, fractional_breadth=fractional_breadth, regularize=True
        )
        assert np.allclose(y, 0.0)


class TestSampleGaussianWaveformDerivative:
    """Tests the ``sample_gaussian_waveform_derivative`` function.

    Tests expected high-level properties of derivatives, such as symmetry and peaks. Also
    implements numerical comparisons to ensure the derivatives are approximately equivalent.
    """

    def test_derivative_order_zero_is_equivalent_to_sample_gaussian_waveform(self):
        """Checks that the derivative of order zero is equivalent to the
        ``sample_gaussian_waveform`` function."""

        x = np.linspace(-1, 1, 100)
        fractional_breadth = 0.5
        regularize = True
        y_derivative = sample_gaussian_waveform_derivative(
            x, fractional_breadth=fractional_breadth, order=0, regularize=regularize
        )
        y_sample = sample_gaussian_waveform(
            x, fractional_breadth=fractional_breadth, regularize=regularize
        )
        assert np.allclose(y_derivative, y_sample)

    def test_first_order_derivative_is_zero_at_zero(self):
        """Checks that the first-order derivative is zero at ``x=0``, which is true because
        that's where the maximum occurs."""
        x = np.array([0.0])
        fractional_breadth = 0.5
        regularize = True
        y_derivative = sample_gaussian_waveform_derivative(
            x, fractional_breadth=fractional_breadth, order=1, regularize=regularize
        )
        assert np.isclose(y_derivative[0], 0.0)

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.1, False)])
    @pytest.mark.parametrize("order", [1, 3])
    def test_odd_order_derivatives_are_antisymmetric(
        self, fractional_breadth, regularize, order
    ):
        """Checks that odd-order derivatives are antisymmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_gaussian_waveform_derivative(
            x, fractional_breadth=fractional_breadth, order=order, regularize=regularize
        )
        y_negative = sample_gaussian_waveform_derivative(
            -x, fractional_breadth=fractional_breadth, order=order, regularize=regularize
        )
        assert np.allclose(y_positive, -y_negative)

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.1, False)])
    @pytest.mark.parametrize("order", [2, 4])
    def test_even_order_derivatives_are_symmetric(
        self, fractional_breadth, regularize, order
    ):
        """Checks that even-order derivatives are symmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_gaussian_waveform_derivative(
            x, fractional_breadth=fractional_breadth, order=order, regularize=regularize
        )
        y_negative = sample_gaussian_waveform_derivative(
            -x, fractional_breadth=fractional_breadth, order=order, regularize=regularize
        )
        assert np.allclose(y_positive, y_negative)

    @pytest.mark.parametrize("fractional_breadth", [1.0, 0.1])
    def test_normalization_scales_orders(self, fractional_breadth):
        """Checks that regularize scales the derivatives correctly.

        The shift value should not be applied to derivatives, and a way to verify that is to
        check the ratios between normalized and unnormalized derivatives are the same for
        different orders.
        """
        # Choose values to avoid extremely small values that lead to division by zero
        x = np.asarray(
            [
                -0.5 * fractional_breadth,
                -0.25 * fractional_breadth,
                0.25 * fractional_breadth,
                0.5 * fractional_breadth,
            ]
        )
        orders = [1, 2, 3]
        scales = []
        for order in orders:
            y_normalized = sample_gaussian_waveform_derivative(
                x, fractional_breadth=fractional_breadth, order=order, regularize=True
            )
            y_unnormalized = sample_gaussian_waveform_derivative(
                x, fractional_breadth=fractional_breadth, order=order, regularize=False
            )
            scale = y_normalized / y_unnormalized
            assert np.allclose(scale, scale[0])
            scales.append(scale[0])
        assert np.allclose(scales, scales[0])

    @pytest.mark.parametrize("fractional_breadth, regularize", [(1.0, True), (0.1, False)])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_derivative_matches_finite_difference_estimate(
        self, fractional_breadth, regularize, order
    ):
        """Checks that the derivative matches a finite-difference estimate."""

        xs = np.linspace(-0.1, 0.1, 100)
        derivatives = sample_gaussian_waveform_derivative(
            xs, fractional_breadth=fractional_breadth, order=order, regularize=regularize
        )

        def fn(x):
            return sample_gaussian_waveform(
                x, fractional_breadth=fractional_breadth, regularize=regularize
            )

        estimated_derivatives = np.asarray(
            [estimate_derivative_at_point(fn, x, order=order) for x in xs]
        )
        assert np.allclose(derivatives, estimated_derivatives, atol=1e-3)


@pytest.mark.parametrize("fractional_breadth", [1.0, 0.1])
@pytest.mark.parametrize("width", [2.0, 160e-9])
class TestParityWithPydanticGaussianWaveform:
    """Tests the sampling implementation is consistent with the pydantic GaussianWaveform
    class.

    Parity should be achieved with ``fractional_breadth = sqrt(2) * fractional_rise`` and ``regularize = False``.
    """

    def test_samples_match(self, fractional_breadth, width):
        """Checks that the samples match."""

        fractional_rise = fractional_breadth / np.sqrt(2)
        x = np.linspace(-1, 1, 100)
        y_sample = sample_gaussian_waveform(
            x, fractional_breadth=fractional_breadth, regularize=False
        )

        waveform = GaussianFunction(width=width, rise=fractional_rise, amp=1.0)
        t_array = x * width / 2
        pydantic_sample = waveform(t_array)
        assert np.allclose(y_sample, pydantic_sample)

    def test_derivative_matches(self, fractional_breadth, width):
        """Checks that the derivatives match.

        We can't compare the derivatives at face value; the pydantic implementation
        calculates the derivative with respect to time, whereas the sampling implementation
        calculates the derivative with respect to the dimensionless variable ``x``.
        Therefore, the chain rule tells us the two are related by a factor of
        ``dx/dt = 2 / width``.
        """

        fractional_rise = fractional_breadth / np.sqrt(2)
        x = np.linspace(-1, 1, 100)
        scale = width / 2  # We're mapping from [-1, 1] to [0, width]
        t_array = x * scale

        y_sample_derivative = (
            sample_gaussian_waveform_derivative(
                x, fractional_breadth=fractional_breadth, order=1, regularize=False
            )
            / scale
        )

        waveform = GaussianFunction(width=width, rise=fractional_rise, amp=1.0)
        pydantic_sample_derivative = waveform.derivative(t_array)
        assert np.allclose(y_sample_derivative, pydantic_sample_derivative)


class TestParityWithPydanticGaussianZeroEdgeWaveform:
    """Tests the sampling implementation is consistent with the pydantic
    GaussianZeroEdgeWaveform class.

    Parity should be achieved with ``fractional_breadth = 2*std_dev / width`` and
    ``regularize = zero_at_edges``.

    No derivative is implemented in the pydantic class, so we can't test that.
    """

    @pytest.mark.parametrize("fractional_breadth", [1.0, 0.1])
    @pytest.mark.parametrize("regularize", [True, False])
    @pytest.mark.parametrize("width", [2.0, 160e-9])
    def test_samples_match(self, fractional_breadth, regularize, width):
        """Checks that the samples match."""

        zero_at_edges = regularize
        std_dev = fractional_breadth * width / 2

        x = np.linspace(-1, 1, 100)

        y_sample = sample_gaussian_waveform(
            x, fractional_breadth=fractional_breadth, regularize=zero_at_edges
        )
        waveform = GaussianZeroEdgeFunction(
            width=width, std_dev=std_dev, amp=1.0, zero_at_edges=zero_at_edges
        )
        t_array = x * width / 2
        pydantic_sample = waveform(t_array)
        assert np.allclose(y_sample, pydantic_sample)


@pytest.mark.parametrize("fractional_breadth", [1.0, 0.1])
@pytest.mark.parametrize("width", [2.0, 160e-9])
class TestParityWithPydanticSofterGaussianWaveform:
    """Tests the sampling implementation is consistent with the pydantic
    SofterGaussianWaveform class.

    Parity should be achieved with ``fractional_breadth = sqrt(2) * fractional_rise`` and ``regularize = True``.
    The pydantic implementation does regularize at whatever boundary is provided (not
    strictly ``width / 2``), so this test enforces the boundary is ``width / 2`` for a
    fair comparison. This means we don't get complete parity with the legacy implementation,
    but at the price of a more structurally sound definition of the waveform shape.

    Similarly, the sampling implementation calculates the max at zero, whereas the pydantic
    implementation calculates the max at any point, so if the center isn't sampled, it may
    give slightly different results.

    The derivative for pydantic is implemented with a weak numerical implementation, and
    comparison is hard due to that.
    """

    def test_samples_match(self, fractional_breadth, width):
        """Checks that the samples match.

        X values are intentionally sampled at -1, 0 and 1.
        """

        fractional_rise = fractional_breadth / np.sqrt(2)
        x = np.linspace(-1, 1, 101)
        y_sample = sample_gaussian_waveform(
            x, fractional_breadth=fractional_breadth, regularize=True
        )

        waveform = SofterGaussianFunction(width=width, rise=fractional_rise, amp=1.0)
        t_array = x * width / 2
        pydantic_sample = waveform(t_array)
        assert np.allclose(y_sample, pydantic_sample)


class TestGaussianWaveformShapeFromLegacy:
    """Tests legacy classmethod constructors for ``GaussianWaveformShape``."""

    def test_from_gaussian_waveform(self):
        """Checks fractional_breadth and regularize from legacy GaussianWaveform."""
        shape = GaussianWaveformShape.from_gaussian_waveform(rise=0.1)
        assert np.isclose(shape.fractional_breadth, np.sqrt(2.0) * 0.1)
        assert shape.regularize is False

    def test_from_softer_gaussian_waveform(self):
        """Checks fractional_breadth and regularize from legacy SofterGaussianWaveform."""
        shape = GaussianWaveformShape.from_softer_gaussian_waveform(rise=0.1)
        assert np.isclose(shape.fractional_breadth, np.sqrt(2.0) * 0.1)
        assert shape.regularize is True

    def test_from_gaussian_zero_edge_waveform(self):
        """Checks fractional_breadth and regularize from legacy GaussianZeroEdgeWaveform."""
        shape = GaussianWaveformShape.from_gaussian_zero_edge_waveform(
            std_dev=0.02, width=0.2, zero_at_edges=True
        )
        assert np.isclose(shape.fractional_breadth, 2.0 * 0.02 / 0.2)
        assert shape.regularize is True

    def test_from_gaussian_zero_edge_waveform_default_edges(self):
        """Checks zero_at_edges defaults to True."""
        shape = GaussianWaveformShape.from_gaussian_zero_edge_waveform(
            std_dev=0.01, width=0.1
        )
        assert shape.regularize is True


class TestFractionalBreadthValidation:
    """Tests that public functions reject negative ``fractional_breadth`` values."""

    @pytest.mark.parametrize("fractional_breadth", [0.0, -0.1, -1.0])
    def test_sample_gaussian_waveform_raises_for_non_positive_breadth(
        self, fractional_breadth
    ):
        """Checks that ``sample_gaussian_waveform`` raises ``ValueError``."""
        with pytest.raises(
            ValueError, match="fractional_breadth must be greater than zero"
        ):
            sample_gaussian_waveform(np.array([0.0]), fractional_breadth=fractional_breadth)

    @pytest.mark.parametrize("fractional_breadth", [0.0, -0.1, -1.0])
    def test_sample_gaussian_waveform_derivative_raises_for_non_positive_breadth(
        self, fractional_breadth
    ):
        """Checks that ``sample_gaussian_waveform_derivative`` raises ``ValueError``."""
        with pytest.raises(
            ValueError, match="fractional_breadth must be greater than zero"
        ):
            sample_gaussian_waveform_derivative(
                np.array([0.0]), fractional_breadth=fractional_breadth
            )

    @pytest.mark.parametrize("fractional_breadth", [0.0, -0.1, -1.0])
    def test_gaussian_waveform_shape_raises_for_non_positive_breadth(
        self, fractional_breadth
    ):
        """Checks that ``GaussianWaveformShape`` raises ``ValueError`` on construction."""
        with pytest.raises(
            ValueError, match="fractional_breadth must be greater than zero"
        ):
            GaussianWaveformShape(fractional_breadth=fractional_breadth)


class TestGaussianWaveformShapeDelegation:
    """Tests shape methods match direct calls to sampling functions."""

    def test_evaluate_matches_raw_function(self):
        """Checks evaluate output matches sample_gaussian_waveform."""

        x = np.array([-0.5, 0.5])
        shape = GaussianWaveformShape(fractional_breadth=0.2, regularize=True)
        assert np.allclose(
            shape.evaluate(x),
            sample_gaussian_waveform(x, fractional_breadth=0.2, regularize=True),
        )

    def test_derivative_matches_raw_function(self):
        """Checks derivative output matches sample_gaussian_waveform_derivative."""

        x = np.array([-0.5, 0.5])
        shape = GaussianWaveformShape(fractional_breadth=0.2, regularize=True)
        assert np.allclose(
            shape.derivative(x, order=1),
            sample_gaussian_waveform_derivative(
                x, order=1, fractional_breadth=0.2, regularize=True
            ),
        )

    def test_from_absolute_generates_correct_fractional_parameters(self):
        """Checks that the from_absolute method generates the correct fractional
        parameters."""

        shape = GaussianWaveformShape.from_absolute(
            width=160e-9, absolute_breadth=16e-9, regularize=True
        )
        assert np.isclose(shape.fractional_breadth, 0.1)  # 16ns / 160ns
        assert shape.regularize is True
