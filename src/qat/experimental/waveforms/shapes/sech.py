# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements the sampling definitions for a Sech waveform shape.

The Sech waveform is defined by a hyperbolic secant function, which is parameterised by
``fractional_breadth``, a dimensionless parameter that sets the fractional_breadth of the
waveform; decreasing ``fractional_breadth`` makes the waveform more narrow, while increasing
``fractional_breadth`` makes the waveform more broad. The Sech waveform is also
parameterised by the parameter ``regularize``, which if set to ``True``, applies a simple
shift and rescaling to the Sech function so that it has zero value at the edges.

The waveform is defined as

.. math::

    f(x) = \\frac{1}{N} \\left(\\text{sech}\\left(\\frac{x}{\\sigma}\\right) - C \\right),

where :math:`\\sigma` sets the ``fractional_breadth`` of the Sech function. :math:`C` is the
shift value, and :math:`N` is the rescale (``regularize``) value, which are defined as

.. math::

    C = \\text{sech}\\left(\\frac{1}{\\sigma}\\right),

and :math:`N = 1 - C`. If ``regularize = False``, then :math:`C = 0` and :math:`N = 1`.

This implements the legacy ``SechWaveform`` under the parameterisations
``fractional_breadth = 2 * std_dev / width``, and ``regularize = zero_at_edges``.
"""

import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qat.experimental.waveforms.shapes.base import (
    WaveformShape,
    derivative_definition,
    shape_definition,
)
from qat.experimental.waveforms.shapes.exceptions import DerivativeOrderNotImplementedError
from qat.experimental.waveforms.shapes.validators import validate_fractional_breadth

# This function has historically caused overflows when extreme values of ``fractional_breadth`` are
# used, so this guard is used to prevent that from happening.
_MAX_COSH_ARG = np.arccosh(0.99 * sys.float_info.max)


def _scale_and_clip_values(
    x: NDArray[np.floating], fractional_breadth: float
) -> NDArray[np.floating]:
    """Scales and clips the values for a Sech waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_breadth: The standard deviation of the Sech function.
    """
    x_scaled = x / fractional_breadth
    x_scaled = np.clip(x_scaled, -_MAX_COSH_ARG, _MAX_COSH_ARG)
    return x_scaled


def _shift_and_rescale_values(fractional_breadth: float) -> tuple[float, float]:
    """Returns the shift and rescale values for a Sech waveform shape.

    :param fractional_breadth: The standard deviation of the Sech function.
    """
    if fractional_breadth < 1 / _MAX_COSH_ARG:
        return 0.0, 1.0
    boundary = 1 / np.cosh(1 / fractional_breadth)
    scale = 1 / (1 - boundary)
    return boundary, scale


@shape_definition(sample_parameter="x")
def sample_sech_waveform(
    x: NDArray[np.floating] | list[float],
    *,
    fractional_breadth: float = 1.0 / 3.0,
    regularize: bool = False,
) -> NDArray[np.complexfloating]:
    """Samples a Sech waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_breadth: The standard deviation of the Sech function, default is ``1/3``.
    :param regularize: If ``True``, applies a shift and rescaling so that the waveform is
        zero at the edges. Default is ``False``.
    """

    validate_fractional_breadth(fractional_breadth)

    x_scaled = _scale_and_clip_values(x, fractional_breadth)
    y = 1 / np.cosh(x_scaled)

    # If fractional_breadth is very small, the waveform rapidly approaches zero so we can
    # just treat it as zero
    if regularize:
        boundary, scale = _shift_and_rescale_values(fractional_breadth)
        y = (y - boundary) * scale
    return y.astype(np.complex128)


def _first_derivative_sech(
    x: NDArray[np.floating], fractional_breadth: float, regularize: bool = False
) -> NDArray[np.complexfloating]:
    """Returns the first derivative of a Sech waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_breadth: The standard deviation of the Sech function.
    """
    x_scaled = _scale_and_clip_values(x, fractional_breadth)
    dy = -np.tanh(x_scaled) / (fractional_breadth * np.cosh(x_scaled))
    if regularize:
        _, scale = _shift_and_rescale_values(fractional_breadth)
        dy = dy * scale
    return dy.astype(np.complex128)


def _second_derivative_sech(
    x: NDArray[np.floating], fractional_breadth: float, regularize: bool = False
) -> NDArray[np.complexfloating]:
    """Returns the second derivative of a Sech waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_breadth: The standard deviation of the Sech function.
    """
    x_scaled = _scale_and_clip_values(x, fractional_breadth)
    sech_x = 1 / np.cosh(x_scaled)
    tanh_x = np.tanh(x_scaled)
    d2y = (1 / fractional_breadth**2) * (sech_x * tanh_x**2 - sech_x**3)
    if regularize:
        _, scale = _shift_and_rescale_values(fractional_breadth)
        d2y = d2y * scale
    return d2y.astype(np.complex128)


@derivative_definition(sample_parameter="x", order_parameter="order")
def sample_sech_waveform_derivative(
    x: NDArray[np.floating] | list[float],
    order: int = 1,
    *,
    fractional_breadth: float = 1.0 / 3.0,
    regularize: bool = False,
) -> NDArray[np.complexfloating]:
    """Samples the derivative of a Sech waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_breadth: The standard deviation of the Sech function, default is
        ``1/3``.
    :param regularize: If ``True``, applies a shift and rescaling so that the waveform is
        zero at the edges. Default is ``False``.
    :param order: The order of the derivative to sample. Default is ``1``.
    """

    validate_fractional_breadth(fractional_breadth)

    if order == 0:
        return sample_sech_waveform(
            x, fractional_breadth=fractional_breadth, regularize=regularize
        )

    if order == 1:
        return _first_derivative_sech(
            x, fractional_breadth=fractional_breadth, regularize=regularize
        )

    if order == 2:
        return _second_derivative_sech(
            x, fractional_breadth=fractional_breadth, regularize=regularize
        )

    raise DerivativeOrderNotImplementedError("Sech", order)


@dataclass(frozen=True)
class SechWaveformShape(WaveformShape):
    """Waveform-shape wrapper for sech sampling functions."""

    fractional_breadth: float = 1.0 / 3.0
    regularize: bool = False

    def __post_init__(self) -> None:
        validate_fractional_breadth(self.fractional_breadth)

    def evaluate(
        self, x: NDArray[np.floating] | list[float]
    ) -> NDArray[np.complexfloating]:
        """Evaluates the sech waveform shape at the sample points."""

        return sample_sech_waveform(
            x,
            fractional_breadth=self.fractional_breadth,
            regularize=self.regularize,
        )

    def derivative(
        self, x: NDArray[np.floating] | list[float], order: int = 1
    ) -> NDArray[np.complexfloating]:
        """Evaluates the derivative of the sech waveform shape."""

        return sample_sech_waveform_derivative(
            x,
            order,
            fractional_breadth=self.fractional_breadth,
            regularize=self.regularize,
        )

    @classmethod
    def from_absolute(
        cls, width: float, absolute_breadth: float, regularize: bool = False
    ) -> "SechWaveformShape":
        """Constructs from absolute parameters.

        :param width: The waveform width.
        :param absolute_breadth: The absolute fractional_breadth of the Sech function.
        :param regularize: Whether the waveform is zero at the edges. Default ``False``.
        """
        return cls(fractional_breadth=absolute_breadth / width, regularize=regularize)

    @classmethod
    def from_legacy(
        cls, std_dev: float, width: float, zero_at_edges: bool = False
    ) -> "SechWaveformShape":
        """Constructs from legacy ``SechWaveform`` parameters.

        :param std_dev: The ``std_dev`` parameter from the legacy implementation.
        :param width: The waveform width.
        :param zero_at_edges: Whether the waveform is zero at the edges. Default ``False``.
        """
        return cls(fractional_breadth=2.0 * std_dev / width, regularize=zero_at_edges)
