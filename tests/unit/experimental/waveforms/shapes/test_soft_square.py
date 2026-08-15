# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the functions for sampling a Soft Square waveform shape."""

import numpy as np
import pytest

from qat.experimental.waveforms.shapes.exceptions import DerivativeOrderNotImplementedError
from qat.experimental.waveforms.shapes.soft_square import (
    SoftSquareWaveformShape,
    sample_soft_square_waveform,
    sample_soft_square_waveform_derivative,
)
from qat.utils.waveform import (
    ExtraSoftSquareFunction,
    SofterSquareFunction,
    SoftSquareFunction,
)

from tests.unit.experimental.waveforms.utils import estimate_derivative_at_point


class TestSampleSoftSquareWaveform:
    """Tests the ``sample_soft_square_waveform`` function."""

    def test_fractional_rise_zero_raises_value_error(self):
        """Tests that a zero fractional rise is rejected explicitly."""
        x = np.linspace(-1, 1, 5)

        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            sample_soft_square_waveform(x, fractional_rise=0.0)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise, regularize",
        [(0.1, 0.1, True), (0.5, 0.15, True), (0.9, 0.05, False)],
    )
    def test_waveform_is_symmetric_around_zero(
        self, fractional_top_width, fractional_rise, regularize
    ):
        """Tests that the waveform is symmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_soft_square_waveform(
            x,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
            regularize=regularize,
        )
        y_negative = sample_soft_square_waveform(
            -x,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
            regularize=regularize,
        )
        assert np.allclose(y_positive, y_negative, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise, regularize",
        [(0.1, 0.1, True), (0.5, 0.15, True), (0.9, 0.05, False)],
    )
    def test_rise_and_fall_are_strictly_monotone(
        self, fractional_top_width, fractional_rise, regularize
    ):
        """Tests that the fractional_rise is strictly increasing and the fall is strictly
        decreasing when moving towards the center."""
        x = np.linspace(-1, 1, 200)
        y = sample_soft_square_waveform(
            x,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
            regularize=regularize,
        )
        rise_mask = x < 0
        fall_mask = x > 0
        assert np.all(np.diff(y[rise_mask]) > 0)
        assert np.all(np.diff(y[fall_mask]) < 0)

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

        y_small_rise = sample_soft_square_waveform(
            x,
            fractional_top_width=fractional_top_width,
            fractional_rise=0.05,
            regularize=False,
        )
        y_large_rise = sample_soft_square_waveform(
            x,
            fractional_top_width=fractional_top_width,
            fractional_rise=0.2,
            regularize=False,
        )
        assert np.all(y_large_rise[rise_mask_smaller] > y_small_rise[rise_mask_smaller])
        assert np.all(y_large_rise[rise_mask_larger] < y_small_rise[rise_mask_larger])

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise",
        [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)],
    )
    def test_normalize_true_makes_center_unity(self, fractional_top_width, fractional_rise):
        """Tests that the waveform is one at the center when regularize is True."""
        x = np.array([0.0])
        y = sample_soft_square_waveform(
            x,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
            regularize=True,
        )
        assert np.isclose(y[0], 1.0)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise",
        [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)],
    )
    def test_zero_at_edges_when_normalize_is_true(
        self, fractional_top_width, fractional_rise
    ):
        """Tests that the waveform is zero at the edges when regularize is True."""
        x = np.array([-1.0, 1.0])
        y = sample_soft_square_waveform(
            x,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
            regularize=True,
        )
        assert np.allclose(y, 0.0, atol=1e-8)


class TestSampleSoftSquareWaveformDerivative:
    """Tests the ``sample_soft_square_waveform_derivative`` function.

    Tests expected high-level properties of derivatives, such as symmetry and peaks. Also
    implements numerical comparisons to ensure the derivatives are approximately equivalent.
    """

    def test_zero_fractional_rise_raises_value_error(self):
        """Tests that derivative sampling rejects a zero fractional rise."""
        x = np.linspace(-1, 1, 5)

        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            sample_soft_square_waveform_derivative(x, fractional_rise=0.0)

    def test_order_higher_than_two_raises_derivative_order_not_implemented(self):
        """Tests that orders higher than two raise a DerivativeOrderNotImplementedError."""
        x = np.linspace(-1, 1, 100)
        with pytest.raises(
            DerivativeOrderNotImplementedError,
            match=(
                r"The derivative of order 3 exists for waveform shape 'Soft Square' but "
                r"is not implemented\."
            ),
        ):
            sample_soft_square_waveform_derivative(x, order=3)

    def test_derivative_order_zero_is_equivalent_to_sample_soft_square_waveform(self):
        """Checks that the derivative of order zero is equivalent to the
        ``sample_soft_square_waveform`` function."""
        x = np.linspace(-1, 1, 100)
        y = sample_soft_square_waveform(x, fractional_top_width=0.5, fractional_rise=0.1)
        y_derivative = sample_soft_square_waveform_derivative(
            x, order=0, fractional_top_width=0.5, fractional_rise=0.1
        )
        assert np.allclose(y, y_derivative, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise",
        [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)],
    )
    def test_first_order_derivatives_are_antisymmetric(
        self, fractional_top_width, fractional_rise
    ):
        """Tests that the first-order derivative is antisymmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_soft_square_waveform_derivative(
            x,
            order=1,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
        )
        y_negative = sample_soft_square_waveform_derivative(
            -x,
            order=1,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
        )
        assert np.allclose(y_positive, -y_negative, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise",
        [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)],
    )
    def test_second_order_derivatives_are_symmetric(
        self, fractional_top_width, fractional_rise
    ):
        """Tests that even-order derivatives are symmetric around zero."""
        x = np.linspace(0, 1, 100)
        y_positive = sample_soft_square_waveform_derivative(
            x,
            order=2,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
        )
        y_negative = sample_soft_square_waveform_derivative(
            -x,
            order=2,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
        )
        assert np.allclose(y_positive, y_negative, atol=1e-8)

    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise",
        [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)],
    )
    def test_first_derivative_is_positive_on_rise_and_negative_on_fall(
        self, fractional_top_width, fractional_rise
    ):
        """Tests that the first derivative is positive on the fractional_rise and negative
        on the fall."""
        x = np.linspace(-1, 1, 200)
        y_derivative = sample_soft_square_waveform_derivative(
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
    @pytest.mark.parametrize("order", [1, 2])
    def test_derivative_matches_numerical_estimate(
        self, fractional_top_width, fractional_rise, order
    ):
        """Tests that the derivative matches a numerical estimate of the derivative."""
        x = np.linspace(-0.5, 0.5, 50)
        y_derivative = sample_soft_square_waveform_derivative(
            x,
            order=order,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
            regularize=True,
        )

        def fn(x_val):
            return sample_soft_square_waveform(
                x_val,
                fractional_top_width=fractional_top_width,
                fractional_rise=fractional_rise,
                regularize=True,
            )

        numerical_y_derivative = np.asarray(
            [estimate_derivative_at_point(fn, x_i, order=order) for x_i in x]
        )
        assert np.allclose(y_derivative, numerical_y_derivative, atol=1e-3)


class TestParityWithPydanticSoftSquareWaveform:
    """Tests that the sampling implementation is consistent with the pydantic
    SoftSquareWaveform class.

    The Soft Square waveform implements the legacy ``SoftSquareWaveform`` under the
    parameterisations ``fractional_top_width = 1 - fractional_rise / width``, ``fractional_rise = 2 * fractional_rise / width``,
    ``regularize = False``.
    """

    @pytest.mark.parametrize("width", [2.0, 160e-9])
    @pytest.mark.parametrize("fractional_top_width", [0.1, 0.5, 0.9])
    def test_sample_soft_square_waveform_matches_pydantic_class(
        self, fractional_top_width, width
    ):
        """Tests that the ``sample_soft_square_waveform`` function matches the pydantic
        SoftSquareFunction class."""

        rise_pydantic = width * (1.0 - fractional_top_width)
        fractional_rise = 2.0 * rise_pydantic / width
        scale = width / 2  # Maps x ∈ [-1, 1] to t ∈ [-width/2, width/2]

        x = np.linspace(-1, 1, 101)
        t_array = x * scale

        y_function = sample_soft_square_waveform(
            x, fractional_top_width=fractional_top_width, fractional_rise=fractional_rise
        )
        waveform = SoftSquareFunction(width=width, rise=rise_pydantic)
        y_class = waveform(t_array)
        assert np.allclose(y_function, y_class, atol=1e-8)


class TestParityWithPydanticSofterSquareWaveform:
    """Tests that the sampling implementation is consistent with the pydantic
    SofterSquareWaveform class.

    The Softer Square waveform implements the legacy ``SofterSquareWaveform`` under the
    parameterisations ``fractional_top_width = (std_dev - 2 * fractional_rise) / width``,
    ``fractional_rise = 2 * fractional_rise / width``, ``regularize = True``.
    """

    @pytest.mark.parametrize("width", [2.0, 160e-9])
    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise", [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)]
    )
    def test_sample_softer_square_waveform_matches_pydantic_class(
        self, width, fractional_top_width, fractional_rise
    ):
        """Tests that the ``sample_soft_square_waveform`` function with ``regularize=True``
        matches the pydantic SofterSquareFunction class."""

        # You can determine these by solving the equations written above
        rise_legacy = fractional_rise * width / 2.0
        std_dev = width * (fractional_top_width + fractional_rise)
        scale = width / 2  # Maps x ∈ [-1, 1] to t ∈ [-width/2, width/2]

        # Use 101 points so that the boundary (x=±1) and center (x=0) are both sampled
        # exactly; the legacy class normalizes via min-max over the array, so these
        # extremes must be present.
        x = np.linspace(-1, 1, 101)
        t_array = x * scale

        y_function = sample_soft_square_waveform(
            x,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
            regularize=True,
        )
        waveform = SofterSquareFunction(std_dev=std_dev, rise=rise_legacy)
        y_class = waveform(t_array)
        assert np.allclose(y_function, y_class, atol=1e-8)


class TestParityWithPydanticExtraSoftSquareWaveform:
    """Tests that the sampling implementation is consistent with the pydantic
    ExtraSoftSquareWaveform class.

    The Extra Soft Square waveform implements the legacy ``ExtraSoftSquareWaveform`` under
    the parameterisations ``fractional_top_width = (std_dev - 4 * fractional_rise) / width``,
    ``fractional_rise = 2 * fractional_rise / width``, ``regularize = True``.
    """

    @pytest.mark.parametrize("width", [2.0, 160e-9])
    @pytest.mark.parametrize(
        "fractional_top_width, fractional_rise", [(0.1, 0.1), (0.5, 0.15), (0.9, 0.05)]
    )
    def test_sample_extra_soft_square_waveform_matches_pydantic_class(
        self, width, fractional_top_width, fractional_rise
    ):
        """Tests that the ``sample_soft_square_waveform`` function with ``regularize=True``
        matches the pydantic ExtraSoftSquareFunction class."""

        rise_legacy = fractional_rise * width / 2.0
        std_dev = width * (fractional_top_width + 2 * fractional_rise)
        scale = width / 2  # Maps x ∈ [-1, 1] to t ∈ [-width/2, width/2]

        # Use 101 points so that the boundary (x=±1) and center (x=0) are both sampled
        # exactly; the legacy class normalizes via min-max over the array, so these
        # extremes must be present.
        x = np.linspace(-1, 1, 101)
        t_array = x * scale

        y_function = sample_soft_square_waveform(
            x,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
            regularize=True,
        )
        waveform = ExtraSoftSquareFunction(std_dev=std_dev, rise=rise_legacy)
        y_class = waveform(t_array)
        assert np.allclose(y_function, y_class, atol=1e-8)


class TestSoftSquareWaveformShapeFromLegacy:
    """Tests legacy classmethod constructors for ``SoftSquareWaveformShape``."""

    def test_from_soft_square_waveform(self):
        """Checks parameters from legacy SoftSquareWaveform."""
        shape = SoftSquareWaveformShape.from_soft_square_waveform(rise=0.04, width=0.4)
        assert np.isclose(shape.fractional_top_width, 1.0 - 0.04 / 0.4)
        assert np.isclose(shape.fractional_rise, 2.0 * 0.04 / 0.4)
        assert shape.regularize is False

    def test_from_softer_square_waveform(self):
        """Checks parameters from legacy SofterSquareWaveform."""
        shape = SoftSquareWaveformShape.from_softer_square_waveform(
            std_dev=0.12, rise=0.04, width=0.4
        )
        assert np.isclose(shape.fractional_top_width, (0.12 - 2.0 * 0.04) / 0.4)
        assert np.isclose(shape.fractional_rise, 2.0 * 0.04 / 0.4)
        assert shape.regularize is True

    def test_from_extra_soft_square_waveform(self):
        """Checks parameters from legacy ExtraSoftSquareWaveform."""
        shape = SoftSquareWaveformShape.from_extra_soft_square_waveform(
            std_dev=0.20, rise=0.04, width=0.4
        )
        assert np.isclose(shape.fractional_top_width, (0.20 - 4.0 * 0.04) / 0.4)
        assert np.isclose(shape.fractional_rise, 2.0 * 0.04 / 0.4)
        assert shape.regularize is True


class TestFractionalRiseValidation:
    """Tests that public functions reject non-positive ``fractional_rise`` values."""

    @pytest.mark.parametrize("fractional_rise", [0.0, -0.1, -1.0])
    def test_sample_soft_square_waveform_raises_for_non_positive_rise(
        self, fractional_rise
    ):
        """Checks that ``sample_soft_square_waveform`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            sample_soft_square_waveform(np.array([0.0]), fractional_rise=fractional_rise)

    @pytest.mark.parametrize("fractional_rise", [0.0, -0.1, -1.0])
    def test_sample_soft_square_waveform_derivative_raises_for_non_positive_rise(
        self, fractional_rise
    ):
        """Checks that ``sample_soft_square_waveform_derivative`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            sample_soft_square_waveform_derivative(
                np.array([0.0]), fractional_rise=fractional_rise
            )

    @pytest.mark.parametrize("fractional_rise", [0.0, -0.1, -1.0])
    def test_soft_square_waveform_shape_raises_for_non_positive_rise(self, fractional_rise):
        """Checks that ``SoftSquareWaveformShape`` raises ``ValueError`` on construction."""
        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            SoftSquareWaveformShape(fractional_rise=fractional_rise)


class TestSoftSquareWaveformShapeDelegation:
    """Tests shape methods match direct calls to sampling functions."""

    def test_evaluate_rejects_zero_fractional_rise(self):
        """Checks that the shape wrapper surfaces fractional rise validation."""

        with pytest.raises(ValueError, match="fractional_rise must be greater than zero"):
            SoftSquareWaveformShape(fractional_rise=0.0)

    def test_evaluate_matches_raw_function(self):
        """Checks evaluate output matches sample_soft_square_waveform."""

        x = np.array([-0.5, 0.5])
        shape = SoftSquareWaveformShape(
            fractional_top_width=0.4, fractional_rise=0.2, regularize=True
        )
        assert np.allclose(
            shape.evaluate(x),
            sample_soft_square_waveform(
                x, fractional_top_width=0.4, fractional_rise=0.2, regularize=True
            ),
        )

    def test_derivative_matches_raw_function(self):
        """Checks derivative output matches sample_soft_square_waveform_derivative."""

        x = np.array([-0.5, 0.5])
        shape = SoftSquareWaveformShape(
            fractional_top_width=0.4, fractional_rise=0.2, regularize=True
        )
        assert np.allclose(
            shape.derivative(x, order=1),
            sample_soft_square_waveform_derivative(
                x, order=1, fractional_top_width=0.4, fractional_rise=0.2, regularize=True
            ),
        )

    def test_from_absolute_generates_correct_fractional_parameters(self):
        """Checks that the from_absolute method generates the correct fractional
        parameters."""

        shape = SoftSquareWaveformShape.from_absolute(
            width=160e-9, absolute_top_width=80e-9, absolute_rise=16e-9, regularize=True
        )
        assert np.isclose(shape.fractional_rise, 0.1)  # 16ns / 160ns
        assert shape.regularize is True
        assert np.isclose(shape.fractional_top_width, 0.5)  # 80ns / 160ns
