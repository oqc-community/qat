# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements the sampling definitions for a Square waveform shape.

The shape for Square waveforms is defined to be unity for all values, and is not
parameterised by any value. Its derivative with respect to time is zero at all orders.
"""

import numpy as np
from numpy.typing import NDArray

from qat.experimental.waveforms.shapes.base import (
    WaveformShape,
    derivative_definition,
    shape_definition,
)


@shape_definition(sample_parameter="x")
def sample_square_waveform(
    x: NDArray[np.floating] | list[float],
) -> NDArray[np.complexfloating]:
    """Samples a square waveform shape.

    A square waveform is defined as unity for all values.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    """
    return np.ones_like(x, dtype=np.complex128)


@derivative_definition(sample_parameter="x", order_parameter="order")
def sample_square_waveform_derivative(
    x: NDArray[np.floating] | list[float], order: int = 1
) -> NDArray[np.complexfloating]:
    """Samples the derivative of any order of a square waveform shape.

    A square waveform is defined as unity for all values, so its derivative for all orders
    is zero.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param order: The order of the derivative to sample. Default is ``1``.
    """
    if order == 0:
        return sample_square_waveform(x)
    return np.zeros_like(x, dtype=np.complex128)


class SquareWaveformShape(WaveformShape):
    """Waveform-shape wrapper for square sampling functions."""

    def evaluate(
        self, x: NDArray[np.floating] | list[float]
    ) -> NDArray[np.complexfloating]:
        """Evaluates the square waveform shape at the sample points."""

        return sample_square_waveform(x)

    def derivative(
        self, x: NDArray[np.floating] | list[float], order: int = 1
    ) -> NDArray[np.complexfloating]:
        """Evaluates the derivative of the square waveform shape."""

        return sample_square_waveform_derivative(x, order)
