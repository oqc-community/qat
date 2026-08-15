# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the functions for sampling a square waveform shape."""

import numpy as np

from qat.experimental.waveforms.shapes.square import (
    SquareWaveformShape,
    sample_square_waveform,
    sample_square_waveform_derivative,
)


class TestSampleSquareWaveform:
    """Tests the ``sample_square_waveform`` function."""

    def test_sample_square_waveform_returns_ones(self):
        """Tests that the ``sample_square_waveform`` function returns an array of ones."""

        x = np.linspace(-1, 1, 100)
        result = sample_square_waveform(x)
        assert (result == 1).all()
        assert len(result) == len(x)


class TestSampleSquareWaveformDerivative:
    """Tests the ``sample_square_waveform_derivative`` function."""

    def test_sample_square_waveform_derivative_returns_zeros_for_nonzero_order(self):
        """Tests that the ``sample_square_waveform_derivative`` function returns an array of
        zeros for non-zero derivative orders."""

        x = np.linspace(-1, 1, 100)
        for order in range(1, 5):
            result = sample_square_waveform_derivative(x, order=order)
            assert (result == 0).all()
            assert len(result) == len(x)

    def test_sample_square_waveform_derivative_returns_ones_for_zero_order(self):
        """Tests that the ``sample_square_waveform_derivative`` function returns an array of
        ones for zero derivative order."""

        x = np.linspace(-1, 1, 100)
        result = sample_square_waveform_derivative(x, order=0)
        assert (result == 1).all()
        assert len(result) == len(x)


class TestSquareWaveformShapeDelegation:
    """Tests shape methods match direct calls to sampling functions."""

    def test_evaluate_matches_raw_function(self):
        """Checks evaluate output matches sample_square_waveform."""

        x = np.array([-0.5, 0.5])
        shape = SquareWaveformShape()
        assert np.allclose(shape.evaluate(x), sample_square_waveform(x))

    def test_derivative_matches_raw_function(self):
        """Checks derivative output matches sample_square_waveform_derivative."""

        x = np.array([-0.5, 0.5])
        shape = SquareWaveformShape()
        assert np.allclose(
            shape.derivative(x, order=1),
            sample_square_waveform_derivative(x, order=1),
        )
