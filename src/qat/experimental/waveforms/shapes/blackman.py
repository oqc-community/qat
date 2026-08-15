# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements the sampling definitions for a Blackman-window waveform shape.

The Blackman waveform is a three-term cosine-sum window function. Mapping the standard
discrete-window index variable onto :math:`x \\in [-1, 1]` yields

.. math::

    f(x) = A_0 - A_1 \\cos(\\pi(x + 1)) + A_2 \\cos(2\\pi(x + 1)),

where the "exact Blackman" coefficients are

.. math::

    A_0 = \\frac{7938}{18608}, \\quad
    A_1 = \\frac{9240}{18608}, \\quad
    A_2 = \\frac{1430}{18608}.

The waveform is bell-shaped, rising from approximately zero at :math:`x = -1`, peaking at
:math:`x = 0`, and returning to approximately zero at :math:`x = +1`.
"""

import numpy as np
from numpy.typing import NDArray

from qat.experimental.waveforms.shapes.base import (
    WaveformShape,
    derivative_definition,
    shape_definition,
)

_A0 = 7938.0 / 18608.0
_A1 = 9240.0 / 18608.0
_A2 = 1430.0 / 18608.0


@shape_definition(sample_parameter="x")
def sample_blackman_waveform(
    x: NDArray[np.floating] | list[float],
) -> NDArray[np.complexfloating]:
    """Samples a Blackman waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :returns: The sampled Blackman waveform values as a complex array.
    """

    arguments = 2 * np.pi * (x / 2 + 0.5)
    return (_A0 - _A1 * np.cos(arguments) + _A2 * np.cos(2 * arguments)).astype(
        np.complex128
    )


@derivative_definition(sample_parameter="x", order_parameter="order")
def sample_blackman_waveform_derivative(
    x: NDArray[np.floating] | list[float], order: int = 1
) -> NDArray[np.complexfloating]:
    """Samples the derivative of a Blackman waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param order: The order of the derivative to sample. Default is ``1``.
    :returns: The sampled Blackman derivative values as a complex array.
    """

    if order == 0:
        return sample_blackman_waveform(x)

    arguments = 2 * np.pi * (x / 2 + 0.5)
    # d^n/dx^n [cos(k*pi*(x+1))] = k^n * pi^n * cos(k*pi*(x+1) + n*pi/2)
    # so f^(n)(x) = pi^n * (-A1 * cos(args + n*pi/2) + 2^n * A2 * cos(2*args + n*pi/2))
    phase = order * np.pi / 2
    prefactor = np.pi**order
    return (
        prefactor
        * (
            -_A1 * np.cos(arguments + phase)
            + (2**order) * _A2 * np.cos(2 * arguments + phase)
        )
    ).astype(np.complex128)


class BlackmanWaveformShape(WaveformShape):
    """Waveform-shape wrapper for Blackman sampling functions."""

    def evaluate(
        self, x: NDArray[np.floating] | list[float]
    ) -> NDArray[np.complexfloating]:
        """Evaluates the Blackman waveform shape at the sample points."""

        return sample_blackman_waveform(x)

    def derivative(
        self, x: NDArray[np.floating] | list[float], order: int = 1
    ) -> NDArray[np.complexfloating]:
        """Evaluates the derivative of the Blackman waveform shape."""

        return sample_blackman_waveform_derivative(x, order)
