# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements the sampling definitions for a Gaussian waveform shape.

The Gaussian waveform is defined as a standard Gaussian function, with optional
normalisation so it can be zero at the boundary of the waveform. The parameters for the
waveform are

* ``fractional_breadth``: The standard deviation of the Gaussian function, which controls
  the fractional_breadth of the waveform. A larger value of ``fractional_breadth`` results
  in a broader waveform, and a smaller value results in a narrower peak. The default value
  is ``sqrt(2)/3``, which coincides with the legacy implementation (which is ``1/3`` under
  that parameterisation).
* ``regularize``: If ``True``, the waveform is normalised so that it has value zero at
  the edges of the waveform. Equivalent to ``zero_at_edges`` in the legacy implementation.
  It is ``False`` by default.

The Gaussian waveform is defined as

.. math::

    f(x) = \\frac{1}{N}\\left[\\text{exp}\\left(-\\frac{x^2}{2\\sigma^2}\\right) - C\\right],

where ``fractional_breadth`` is the standard deviation. ``C`` is the shift value, and ``N``
is the regularization value, where ``N`` and ``C`` are one and zero respectively if
``regularize = False``, and are otherwise defined as

.. math::

    C = \\text{exp}\\left(-\\frac{1}{2\\sigma^2}\\right),

and :math:`N = 1 - C`.

This implements the legacy ``GaussianWaveform``, ``GaussianZeroEdgeWaveform``,
``DragGaussianWaveform``, and the ``SofterGaussianWaveform`` with the following
parameterisations:

* ``GaussianWaveform``: ``fractional_breadth = sqrt(2) * fractional_rise``,
  ``regularize = False``,
* ``GaussianZeroEdgeWaveform``: ``fractional_breadth = 2 * std_dev / width``,
  ``regularize = zero_at_edges``,
* ``DragGaussianWaveform``: ``fractional_breadth = std_dev``, ``regularize = zero_at_edges``,
  where DRAG is implemented away from the waveform definition, using the derivatives,
* ``SofterGaussianWaveform``: ``fractional_breadth = sqrt(2) * fractional_rise``,
  ``regularize = True``.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_hermite

from qat.experimental.waveforms.shapes.base import (
    WaveformShape,
    derivative_definition,
    shape_definition,
)
from qat.experimental.waveforms.shapes.validators import validate_fractional_breadth


def _shift_and_rescale_values(fractional_breadth: float) -> tuple[float, float]:
    """Returns the shift and rescale values for a Gaussian waveform shape."""

    boundary = np.exp(-1 / (2 * fractional_breadth**2))
    scale = 1 / (1 - boundary)
    return boundary, scale


@derivative_definition(sample_parameter="x", order_parameter="order")
def sample_gaussian_waveform_derivative(
    x: NDArray[np.floating] | list[float],
    order: int = 1,
    *,
    fractional_breadth: float = np.sqrt(2.0) / 3.0,
    regularize: bool = False,
) -> NDArray[np.complexfloating]:
    """Samples the derivative of a Gaussian waveform shape.

    The derivative is calculated using the Hermite polynomial of order ``order``, and the
    Gaussian function. Note this is the Physicist's Hermite polynomial, and not the
    Probabilist's Hermite polynomial.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_breadth: The standard deviation of the Gaussian function, default is
        ``sqrt(2)/3``, which coincides with the legacy implementation under the new
        definition.
    :param regularize: If ``True``, applies a shift and rescaling so that the waveform is
        zero at the edges. Default is ``False``.
    :param order: The order of the derivative to sample. Default is ``1``.
    """

    validate_fractional_breadth(fractional_breadth)

    sigma = fractional_breadth * np.sqrt(2)
    x_scaled = x / sigma
    hermite_poly = eval_hermite(order, x_scaled)
    prefactor = (-1 / sigma) ** order

    y = prefactor * hermite_poly * np.exp(-(x_scaled**2))
    if regularize:
        shift, scale = _shift_and_rescale_values(fractional_breadth)
        if order == 0:
            y = y - shift
        y = y * scale
    return y.astype(np.complex128)


@shape_definition(sample_parameter="x")
def sample_gaussian_waveform(
    x: NDArray[np.floating] | list[float],
    *,
    fractional_breadth: float = np.sqrt(2.0) / 3.0,
    regularize: bool = False,
) -> NDArray[np.complexfloating]:
    """Samples a Gaussian waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_breadth: The standard deviation of the Gaussian function, default is
        ``sqrt(2)/3``.
    :param regularize: If ``True``, applies a shift and rescaling so that the waveform is
        zero at the edges. Default is ``False``.
    """

    return sample_gaussian_waveform_derivative(
        x, fractional_breadth=fractional_breadth, regularize=regularize, order=0
    )


@dataclass(frozen=True)
class GaussianWaveformShape(WaveformShape):
    """Waveform-shape wrapper for Gaussian sampling functions."""

    fractional_breadth: float = np.sqrt(2.0) / 3.0
    regularize: bool = False

    def __post_init__(self) -> None:
        validate_fractional_breadth(self.fractional_breadth)

    def evaluate(
        self, x: NDArray[np.floating] | list[float]
    ) -> NDArray[np.complexfloating]:
        """Evaluates the Gaussian waveform shape at the sample points."""

        return sample_gaussian_waveform(
            x,
            fractional_breadth=self.fractional_breadth,
            regularize=self.regularize,
        )

    def derivative(
        self, x: NDArray[np.floating] | list[float], order: int = 1
    ) -> NDArray[np.complexfloating]:
        """Evaluates the derivative of the Gaussian waveform shape."""

        return sample_gaussian_waveform_derivative(
            x,
            order,
            fractional_breadth=self.fractional_breadth,
            regularize=self.regularize,
        )

    @classmethod
    def from_absolute(
        cls, width: float, absolute_breadth: float, regularize: bool = False
    ) -> "GaussianWaveformShape":
        """Constructs from absolute parameters.

        :param width: The waveform width.
        :param absolute_breadth: The absolute fractional_breadth of the Gaussian waveform.
        :param regularize: If ``True``, applies a shift and rescaling so that the waveform is
            zero at the edges with maximum component one. Default is ``False``.
        """
        fractional_breadth = absolute_breadth / width
        return cls(fractional_breadth=fractional_breadth, regularize=regularize)

    @classmethod
    def from_gaussian_waveform(cls, rise: float) -> "GaussianWaveformShape":
        """Constructs from legacy ``GaussianWaveform`` parameters.

        :param rise: The ``rise`` parameter from the legacy implementation.
        """
        return cls(fractional_breadth=np.sqrt(2.0) * rise, regularize=False)

    @classmethod
    def from_softer_gaussian_waveform(cls, rise: float) -> "GaussianWaveformShape":
        """Constructs from legacy ``SofterGaussianWaveform`` parameters.

        :param rise: The ``rise`` parameter from the legacy implementation.
        """
        return cls(fractional_breadth=np.sqrt(2.0) * rise, regularize=True)

    @classmethod
    def from_gaussian_zero_edge_waveform(
        cls, std_dev: float, width: float, zero_at_edges: bool = True
    ) -> "GaussianWaveformShape":
        """Constructs from legacy ``GaussianZeroEdgeWaveform`` / ``DragGaussianWaveform``
        parameters.

        :param std_dev: The ``std_dev`` parameter from the legacy implementation.
        :param width: The waveform width.
        :param zero_at_edges: Whether the waveform is zero at the edges. Default ``True``.
        """
        return cls(fractional_breadth=2.0 * std_dev / width, regularize=zero_at_edges)
