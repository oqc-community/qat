# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements the sampling definitions for a Sinusoidal waveform shape.

The Sinusoidal waveform is defined by a standard sine function, which is parameterised by
``number_of_periods``, a dimensionless parameter that creates a waveform with that
specified number of periods (it needs not be an integer). It is also parameterised by the
parameter ``internal_phase``, which is specified in radians and sets the starting phase of
the waveform shape. By default, the Sinusoidal function is defined with
``number_of_periods = 1/2``, and ``internal_phase = 0``. To have a cosine-like waveform,
use ``internal_phase = pi/2``.

The Sinusoidal waveform is defined as

.. math::

    f(x) = \\text{sin}(2\\pi Nx + \\theta),

where :math:`N` is the number of periods, and :math:`\\theta` is the internal phase in
radians.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qat.experimental.waveforms.shapes.base import (
    WaveformShape,
    derivative_definition,
    shape_definition,
)


@derivative_definition(sample_parameter="x", order_parameter="order")
def sample_sinusoidal_waveform_derivative(
    x: NDArray[np.floating] | list[float],
    order: int = 1,
    *,
    number_of_periods: float = 1 / 2,
    internal_phase: float = 0.0,
) -> NDArray[np.complexfloating]:
    """Samples the derivative of a Sinusoidal waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param number_of_periods: The number of periods of the Sinusoidal function, default is
        ``1/2``.
    :param internal_phase: The internal phase of the Sinusoidal function in radians, default
        is ``0``.
    :param order: The order of the derivative to sample. Default is ``1``.
    """

    prefactor = (2 * np.pi * number_of_periods) ** order
    phase_shift = order * np.pi / 2

    return (
        prefactor * np.sin(2 * np.pi * number_of_periods * x + internal_phase + phase_shift)
    ).astype(np.complex128)


@shape_definition(sample_parameter="x")
def sample_sinusoidal_waveform(
    x: NDArray[np.floating] | list[float],
    *,
    number_of_periods: float = 1 / 2,
    internal_phase: float = 0.0,
) -> NDArray[np.complexfloating]:
    """Samples a Sinusoidal waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param number_of_periods: The number of periods of the Sinusoidal function, default is
        ``1/2``.
    :param internal_phase: The internal phase of the Sinusoidal function in radians, default
        is ``0``.
    """

    return sample_sinusoidal_waveform_derivative(
        x, number_of_periods=number_of_periods, internal_phase=internal_phase, order=0
    )


@derivative_definition(sample_parameter="x", order_parameter="order")
def sample_sinusoidal_waveform_derivative_from_frequency(
    x: NDArray[np.floating] | list[float],
    order: int = 1,
    *,
    frequency: float = 0.5,
    width: float = 1.0,
    internal_phase: float = 0.0,
) -> NDArray[np.complexfloating]:
    """Samples the derivative of a Sinusoidal waveform shape from frequency and width.

    This helper supports legacy parameterisation with ``number_of_periods = frequency *
    width``.
    """

    number_of_periods = frequency * width
    return sample_sinusoidal_waveform_derivative(
        x, number_of_periods=number_of_periods, internal_phase=internal_phase, order=order
    )


@shape_definition(sample_parameter="x")
def sample_sinusoidal_waveform_from_frequency(
    x: NDArray[np.floating] | list[float],
    *,
    frequency: float = 0.5,
    width: float = 1.0,
    internal_phase: float = 0.0,
) -> NDArray[np.complexfloating]:
    """Samples a Sinusoidal waveform shape from frequency and width.

    This helper supports legacy parameterisation with ``number_of_periods = frequency *
    width``.
    """

    number_of_periods = frequency * width
    return sample_sinusoidal_waveform(
        x, number_of_periods=number_of_periods, internal_phase=internal_phase
    )


@dataclass(frozen=True)
class SinusoidalWaveformShape(WaveformShape):
    """Waveform-shape wrapper for sinusoidal sampling functions."""

    number_of_periods: float = 1 / 2
    internal_phase: float = 0.0

    def evaluate(
        self, x: NDArray[np.floating] | list[float]
    ) -> NDArray[np.complexfloating]:
        """Evaluates the sinusoidal waveform shape at the sample points."""

        return sample_sinusoidal_waveform(
            x,
            number_of_periods=self.number_of_periods,
            internal_phase=self.internal_phase,
        )

    def derivative(
        self, x: NDArray[np.floating] | list[float], order: int = 1
    ) -> NDArray[np.complexfloating]:
        """Evaluates the derivative of the sinusoidal waveform shape."""

        return sample_sinusoidal_waveform_derivative(
            x,
            order,
            number_of_periods=self.number_of_periods,
            internal_phase=self.internal_phase,
        )

    @classmethod
    def from_frequency(
        cls, frequency: float, width: float, internal_phase: float = 0.0
    ) -> "SinusoidalWaveformShape":
        """Constructs from a physical frequency and waveform width.

        The number of periods is calculated as ``frequency * width``.

        :param frequency: The waveform frequency.
        :param width: The waveform width (same units as the reciprocal of ``frequency``).
        :param internal_phase: The starting phase in radians. Default ``0.0``.
        """
        return cls(number_of_periods=frequency * width, internal_phase=internal_phase)
