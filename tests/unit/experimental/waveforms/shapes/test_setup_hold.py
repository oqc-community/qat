# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the functions for sampling a Setup Hold waveform shape."""

import numpy as np
import pytest

from qat.experimental.waveforms.shapes.exceptions import DerivativeOrderUndefinedError
from qat.experimental.waveforms.shapes.setup_hold import (
    SetupHoldWaveformShape,
    sample_setup_hold_waveform,
    sample_setup_hold_waveform_derivative,
)
from qat.utils.waveform import SetupHoldFunction


class TestSampleSetupHoldWaveform:
    """Tests the ``sample_setup_hold_waveform`` function."""

    @pytest.mark.parametrize("setup, rise_location", [(0.1, 0.1), (0.4, 0.35), (0.9, 0.8)])
    def test_waveform_matches_piecewise_definition(self, setup, rise_location):
        """Checks the setup segment then hold segment piecewise definition."""
        x = np.linspace(-1, 1, 101)
        boundary = 2 * rise_location - 1
        expected = np.where(x < boundary, setup, 1.0)

        y = sample_setup_hold_waveform(x, setup=setup, rise_location=rise_location)
        assert np.allclose(y, expected, atol=1e-8)

    @pytest.mark.parametrize("rise_location", [0.1, 0.35, 0.8])
    def test_setup_equal_one_returns_ones(self, rise_location):
        """Checks that setup=1.0 produces a constant-one waveform."""
        x = np.linspace(-1, 1, 101)
        y = sample_setup_hold_waveform(x, setup=1.0, rise_location=rise_location)
        assert np.allclose(y, 1.0, atol=1e-8)

    def test_rise_zero_raises(self):
        """Checks rise_location=0 raises ValueError."""
        x = np.linspace(-1, 1, 101)
        with pytest.raises(ValueError, match="rise_location must be greater than zero"):
            sample_setup_hold_waveform(x, setup=0.2, rise_location=0.0)

    def test_rise_one_sets_all_but_last_sample_to_setup(self):
        """Checks rise_location=1 sets all x<1 samples to setup and x=1 to one."""
        x = np.linspace(-1, 1, 101)
        y = sample_setup_hold_waveform(x, setup=0.2, rise_location=1.0)
        assert np.allclose(y[:-1], 0.2, atol=1e-8)
        assert np.isclose(y[-1], 1.0)


class TestSampleSetupHoldWaveformDerivative:
    """Tests the ``sample_setup_hold_waveform_derivative`` function."""

    def test_derivative_order_zero_matches_waveform(self):
        """Checks that derivative order 0 returns the same waveform as the base function."""
        x = np.linspace(-1, 1, 101)
        setup = 0.2
        rise_location = 0.3

        y_waveform = sample_setup_hold_waveform(x, setup=setup, rise_location=rise_location)
        y_derivative = sample_setup_hold_waveform_derivative(
            x, setup=setup, rise_location=rise_location, order=0
        )
        assert np.allclose(y_waveform, y_derivative, atol=1e-8)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_derivative_raises_for_any_order(self, order):
        """Checks derivative is undefined for this discontinuous waveform."""
        x = np.linspace(-1, 1, 101)
        with pytest.raises(
            DerivativeOrderUndefinedError,
            match=(
                rf"The derivative of order {order} is not mathematically defined for "
                r"waveform shape 'Setup Hold'\."
            ),
        ):
            sample_setup_hold_waveform_derivative(
                x, setup=0.2, rise_location=0.3, order=order
            )


class TestParityWithPydanticSetupHoldWaveform:
    """Tests sampling consistency with the pydantic Setup Hold waveform class.

    The Setup Hold waveform implements the legacy ``SetupHoldWaveform`` under
    ``setup = amp_setup / amp`` and ``rise_location = rise_location / width``.
    """

    @pytest.mark.parametrize("width", [2.0, 160e-9])
    @pytest.mark.parametrize("setup, rise_location", [(0.1, 0.1), (0.4, 0.35), (0.9, 0.8)])
    def test_sample_setup_hold_waveform_matches_pydantic_class(
        self, setup, rise_location, width
    ):
        """Checks ``sample_setup_hold_waveform`` matches ``SetupHoldFunction``."""

        rise_legacy = rise_location * width
        amp = 1.0
        amp_setup = setup

        x = np.linspace(-1, 1, 101)
        scale = width / 2  # Maps x in [-1, 1] to t in [-width/2, width/2]
        t_array = x * scale

        y_function = sample_setup_hold_waveform(x, setup=setup, rise_location=rise_location)
        waveform = SetupHoldFunction(
            width=width, rise=rise_legacy, amp_setup=amp_setup, amp=amp
        )
        y_class = waveform(t_array)
        assert np.allclose(y_function, y_class, atol=1e-8)


class TestSetupHoldWaveformShapeFromLegacy:
    """Tests legacy classmethod constructor for ``SetupHoldWaveformShape``."""

    def test_from_legacy(self):
        """Checks setup and rise_location from legacy SetupHoldWaveform parameters."""
        shape = SetupHoldWaveformShape.from_legacy(
            amp_setup=0.2, amp=1.0, rise=16e-9, width=160e-9
        )
        assert np.isclose(shape.setup, 0.2 / 1.0)
        assert np.isclose(shape.rise_location, 16e-9 / 160e-9)


class TestFractionalRiseValidation:
    """Tests that public functions reject non-positive ``fractional_rise`` values."""

    @pytest.mark.parametrize("rise_location", [0.0, -0.1, -1.0])
    def test_sample_setup_hold_waveform_raises_for_non_positive_rise(self, rise_location):
        """Checks that ``sample_setup_hold_waveform`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="rise_location must be greater than zero"):
            sample_setup_hold_waveform(np.array([0.0]), rise_location=rise_location)

    @pytest.mark.parametrize("rise_location", [0.0, -0.1, -1.0])
    def test_sample_setup_hold_waveform_derivative_raises_for_non_positive_rise(
        self, rise_location
    ):
        """Checks that ``sample_setup_hold_waveform_derivative`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="rise_location must be greater than zero"):
            sample_setup_hold_waveform_derivative(
                np.array([0.0]), rise_location=rise_location
            )

    @pytest.mark.parametrize("rise_location", [0.0, -0.1, -1.0])
    def test_setup_hold_waveform_shape_raises_for_non_positive_rise(self, rise_location):
        """Checks that ``SetupHoldWaveformShape`` raises ``ValueError`` on construction."""
        with pytest.raises(ValueError, match="rise_location must be greater than zero"):
            SetupHoldWaveformShape(rise_location=rise_location)


class TestSetupHoldWaveformShapeDelegation:
    """Tests shape methods match direct calls to sampling functions."""

    def test_evaluate_matches_raw_function(self):
        """Checks evaluate output matches sample_setup_hold_waveform."""

        x = np.array([-0.5, 0.5])
        shape = SetupHoldWaveformShape(setup=0.2, rise_location=0.6)
        assert np.allclose(
            shape.evaluate(x),
            sample_setup_hold_waveform(x, setup=0.2, rise_location=0.6),
        )

    def test_derivative_raises_error(self):
        """Checks derivative raises an error for this non-differentiable shape."""

        x = np.array([-0.5, 0.5])
        shape = SetupHoldWaveformShape(setup=0.2, rise_location=0.6)

        with pytest.raises(
            DerivativeOrderUndefinedError,
            match=(
                r"The derivative of order 1 is not mathematically defined for waveform "
                r"shape 'Setup Hold'\."
            ),
        ):
            shape.derivative(x, order=1)

    def test_from_absolute_creates_correct_fractional(self):
        """Checks that from_absolute generates the correct fractional parameters."""

        shape = SetupHoldWaveformShape.from_absolute(
            width=160e-9, amp=0.5, absolute_rise=16e-9, absolute_amp_setup=0.2
        )
        assert np.isclose(shape.rise_location, 0.1)  # 16ns / 160ns
        assert np.isclose(shape.setup, 0.4)  # 0.2 / 0.5
