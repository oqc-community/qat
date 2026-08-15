# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements the sampling definitions for a Rounded Square waveform shape.

The Rounded Square waveform shape is implemented by ERF functions that has the consequence
of being roughly square in the center ``fractional_top_width`` region, which is the
"top width", with fractional_rise and fall edges at the tails that have a sharpness
determined by the ``fractional_rise`` parameter. Note that the function isn't actually
square in the center, but has the appearance of being so. The function is actually
continuously differentiable.

The Rounded Square waveform is defined by the function

.. math::

    f(x) = \\frac{1}{N}\\left[
        \\text{erf}\\left(\\frac{x + w_t}{r}\\right)
        - \\text{erf}\\left(\\frac{x - w_t}{r}\\right)
        - C
    \\right],

where :math:`w_t` is ``fractional_top_width`` and :math:`r` is ``fractional_rise``. The
shift :math:`C` and regularize constant :math:`N` are

.. math::

    C = \\text{erf}\\left(\\frac{1 + w_t}{r}\\right)
        - \\text{erf}\\left(\\frac{1 - w_t}{r}\\right),

and

.. math::

    N = \\left[
        \\text{erf}\\left(\\frac{w_t}{r}\\right)
        - \\text{erf}\\left(-\\frac{w_t}{r}\\right)
    \\right] - C.

This implements the legacy ``RoundedSquareWaveform`` shape with the following
parameterisations: ``fractional_rise = 2 * fractional_rise / width``,
``fractional_top_width = std_dev / width``.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf, eval_hermite

from qat.experimental.waveforms.shapes.base import (
    WaveformShape,
    derivative_definition,
    shape_definition,
)
from qat.experimental.waveforms.shapes.validators import validate_fractional_rise


def _shift_and_rescale_values(
    fractional_top_width: float, fractional_rise: float
) -> tuple[float, float]:
    """Returns the shift and rescale values for a Rounded Square waveform shape.

    :param fractional_top_width: The "top width" of the Rounded Square function.
    :param fractional_rise: The sharpness of the fractional_rise and fall edges of the
        Rounded Square function.
    """
    boundary = erf((1 + fractional_top_width) / fractional_rise) - erf(
        (1 - fractional_top_width) / fractional_rise
    )
    center = erf(fractional_top_width / fractional_rise) - erf(
        -fractional_top_width / fractional_rise
    )
    scale = 1 / (center - boundary)
    return boundary, scale


@shape_definition(sample_parameter="x")
def sample_rounded_square_waveform(
    x: NDArray[np.floating] | list[float],
    *,
    fractional_top_width: float = 0.5,
    fractional_rise: float = 0.1,
) -> NDArray[np.complexfloating]:
    """Samples a Rounded Square waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_top_width: The "top width" of the Rounded Square function, default is
        ``0.5``.
    :param fractional_rise: The sharpness of the fractional_rise and fall edges of the
        Rounded Square function, default is ``0.1``.
    :return: The sampled Rounded Square waveform as a complex-valued array.
    """

    validate_fractional_rise(fractional_rise)

    # Centers the two ERFs around -fractional_top_width and +fractional_top_width
    # respectively.
    forwards_erf = erf((x + fractional_top_width) / fractional_rise)
    backwards_erf = -erf((x - fractional_top_width) / fractional_rise)

    # Shift to the domain [0, 1]
    shift, scale = _shift_and_rescale_values(fractional_top_width, fractional_rise)
    return scale * (forwards_erf + backwards_erf - shift).astype(np.complex128)


@derivative_definition(sample_parameter="x", order_parameter="order")
def sample_rounded_square_waveform_derivative(
    x: NDArray[np.floating] | list[float],
    order: int = 1,
    *,
    fractional_top_width: float = 0.5,
    fractional_rise: float = 0.1,
) -> NDArray[np.complexfloating]:
    """Samples the derivative of a Rounded Square waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_top_width: The "top width" of the Rounded Square function, default is
        ``0.5``.
    :param fractional_rise: The sharpness of the fractional_rise and fall edges of the
        Rounded Square function, default is ``0.1``.
    :param order: The order of the derivative to sample, default is ``1``.
    :return: The sampled derivative of the Rounded Square waveform as a complex-valued
        array.
    """

    validate_fractional_rise(fractional_rise)

    if order == 0:
        return sample_rounded_square_waveform(
            x, fractional_top_width=fractional_top_width, fractional_rise=fractional_rise
        )

    _, scale = _shift_and_rescale_values(fractional_top_width, fractional_rise)
    prefactor = scale * 2 / (np.sqrt(np.pi) * fractional_rise**order) * (-1) ** (order - 1)

    hermite_forward = eval_hermite(order - 1, (x + fractional_top_width) / fractional_rise)
    hermite_backward = eval_hermite(order - 1, (x - fractional_top_width) / fractional_rise)
    return prefactor * (
        hermite_forward * np.exp(-((x + fractional_top_width) ** 2) / fractional_rise**2)
        - hermite_backward * np.exp(-((x - fractional_top_width) ** 2) / fractional_rise**2)
    ).astype(np.complex128)


@dataclass(frozen=True)
class RoundedSquareWaveformShape(WaveformShape):
    """Waveform-shape wrapper for rounded-square sampling functions."""

    fractional_top_width: float = 0.5
    fractional_rise: float = 0.1

    def __post_init__(self) -> None:
        validate_fractional_rise(self.fractional_rise)

    def evaluate(
        self, x: NDArray[np.floating] | list[float]
    ) -> NDArray[np.complexfloating]:
        """Evaluates the rounded-square waveform shape at the sample points."""

        return sample_rounded_square_waveform(
            x,
            fractional_top_width=self.fractional_top_width,
            fractional_rise=self.fractional_rise,
        )

    def derivative(
        self, x: NDArray[np.floating] | list[float], order: int = 1
    ) -> NDArray[np.complexfloating]:
        """Evaluates the derivative of the rounded-square waveform shape."""

        return sample_rounded_square_waveform_derivative(
            x,
            order,
            fractional_top_width=self.fractional_top_width,
            fractional_rise=self.fractional_rise,
        )

    @classmethod
    def from_absolute(
        cls, width: float, absolute_top_width: float, absolute_rise: float
    ) -> "RoundedSquareWaveformShape":
        """Constructs from absolute parameters.

        :param width: The waveform width.
        :param absolute_top_width: The "top width" of the Rounded Square function.
        :param absolute_rise: The sharpness of the fractional_rise and fall edges of the
            Rounded Square function.
        """
        return cls(
            fractional_top_width=absolute_top_width / width,
            fractional_rise=absolute_rise / width,
        )

    @classmethod
    def from_legacy(
        cls, rise: float, std_dev: float, width: float
    ) -> "RoundedSquareWaveformShape":
        """Constructs from legacy ``RoundedSquareWaveform`` parameters.

        :param rise: The ``rise`` parameter from the legacy implementation.
        :param std_dev: The ``std_dev`` parameter from the legacy implementation.
        :param width: The waveform width.
        """
        return cls(
            fractional_top_width=std_dev / width,
            fractional_rise=2 * rise / width,
        )
