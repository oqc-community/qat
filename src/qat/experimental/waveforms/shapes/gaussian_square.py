# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements the sampling definitions for a Gaussian-Square waveform shape.

The Gaussian Square waveform shape has a fractional_rise and fall at the edges that is
defined by a Gaussian function, connected by a square bridge in between. The proportion of
the waveform that is square is parameterised by the parameter ``fractional_top_width``,
which is a dimensionless parameter between 0 and 1; if ``fractional_top_width = 0``, then
the waveform is a pure Gaussian function, and if ``fractional_top_width = 1``, then the
waveform is a pure square function, and otherwise it linearly interpolates between the two.

The Gaussian fractional_rise and fall components of the waveform are parameterised by
``fractional_rise``, which is expressed as a fraction of the full normalised waveform domain
``[-1, 1]``. To evaluate the Gaussian, the fractional_rise interval
``[-1, -fractional_top_width]`` and the fall interval ``[fractional_top_width, 1]`` are each
remapped to ``[-1, 0]`` and ``[0, 1]`` respectively via

.. math::

    x_{\\text{rise}} = \\frac{x + w_{t}}{1 - w_{t}}, \\qquad x_{\\text{fall}} = \\frac{x - w_{t}}{1 - w_{t}},

and the Gaussian is sampled with effective parameter :math:`r / (1 - w_{t})`. This ensures
that ``fractional_rise`` retains the same meaning regardless of ``fractional_top_width``;
changing ``fractional_top_width`` does not implicitly rescale the Gaussian edge. The
``regularize`` parameter is applied to the Gaussian edge components independently of
``fractional_top_width``.

If ``regularize`` is used, the Gaussian edge components are shifted and rescaled so that
they are zero at the waveform edges. Since the edges ``x = ±1`` map to
:math:`x_{\\text{rise/fall}} = \\pm 1` after remapping, the edge value of the
Gaussian is

.. math::

    C = \\text{exp}\\left(-\\frac{1}{2}\\frac{(1 - w_{t})^2}{r^2}\\right),

and the regularize factor is :math:`N = 1 - C`. If ``regularize = False``, then
:math:`C = 0` and :math:`N = 1`. The Gaussian Square waveform is then

.. math::

    f(x) = \\begin{cases}
        \\frac{1}{N}\\left[\\text{exp}\\left(-\\frac{1}{2}\\frac{(x + w_{t})^2}{r^2}\\right) - C\\right] & \\text{if } x < -w_{t} \\\\
        1 & \\text{if } -w_{t} \\leq x \\leq w_{t} \\\\
        \\frac{1}{N}\\left[\\text{exp}\\left(-\\frac{1}{2}\\frac{(x - w_{t})^2}{r^2}\\right) - C\\right] & \\text{if } x > w_{t}
    \\end{cases},

where :math:`w_{t}` is ``fractional_top_width`` and :math:`r` is ``fractional_rise``.

Since the Gaussian Square waveform is differentiable, the first derivative of the waveform
can also be sampled. However, it is not continuously differentiable, so further orders
cannot be sampled.

The Gaussian Square waveform implements the legacy ``GaussianSquareWaveform`` under the
parameterisations ``fractional_rise = 2 * std_dev / width``,
``regularize = zero_at_edges``, and ``fractional_top_width = square_width / width``.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from qat.experimental.waveforms.shapes.base import (
    WaveformShape,
    derivative_definition,
    shape_definition,
)
from qat.experimental.waveforms.shapes.exceptions import DerivativeOrderUndefinedError
from qat.experimental.waveforms.shapes.gaussian import (
    sample_gaussian_waveform,
    sample_gaussian_waveform_derivative,
)
from qat.experimental.waveforms.shapes.validators import (
    validate_fractional_rise,
    validate_fractional_top_width,
)


class _EdgeRegions(NamedTuple):
    """Container for edge-region masks, scaled coordinates, and edge width."""

    rise_mask: NDArray[np.bool_]
    fall_mask: NDArray[np.bool_]
    scaled_rise_values: NDArray[np.floating]
    scaled_fall_values: NDArray[np.floating]
    edge_width: float


def _edge_regions(x: NDArray[np.floating], fractional_top_width: float) -> _EdgeRegions:
    """Returns edge masks, rescaled edge coordinates, and the rescaled fractional_rise
    parameter.

    The fractional_rise and fall intervals ``[-1, -fractional_top_width]`` and
    ``[fractional_top_width, 1]`` are remapped to ``[-1, 0]`` and ``[0, 1]`` by dividing
    local offsets by ``edge_width = 1 - fractional_top_width``. The ``fractional_rise``
    parameter is also divided by ``edge_width`` before being passed to the Gaussian sampler,
    so its meaning on the full domain is preserved.
    """

    rise_mask = x < -fractional_top_width
    fall_mask = x > fractional_top_width
    edge_width = 1.0 - fractional_top_width

    if edge_width == 0:
        return _EdgeRegions(rise_mask, fall_mask, x[:0], x[:0], 0.0)

    scaled_rise_values = (x[rise_mask] + fractional_top_width) / edge_width
    scaled_fall_values = (x[fall_mask] - fractional_top_width) / edge_width
    return _EdgeRegions(
        rise_mask,
        fall_mask,
        scaled_rise_values,
        scaled_fall_values,
        edge_width,
    )


@shape_definition(sample_parameter="x")
def sample_gaussian_square_waveform(
    x: NDArray[np.floating] | list[float],
    *,
    fractional_rise: float = np.sqrt(2.0) / 3.0,
    regularize: bool = False,
    fractional_top_width: float = 0.5,
) -> NDArray[np.complexfloating]:
    """Samples a Gaussian-Square waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_rise: The standard deviation of the Gaussian edge profile on the
        normalised waveform domain, default is ``sqrt(2)/3``.
    :param regularize: If ``True``, applies a shift and rescaling so that the waveform is
        zero at the edges. Default is ``False``.
    :param fractional_top_width: The proportion of the waveform that is square, between 0
        and 1. Default is ``0.5``.
    """

    validate_fractional_top_width(fractional_top_width)
    validate_fractional_rise(fractional_rise)

    edge_regions = _edge_regions(x, fractional_top_width)

    y = np.ones_like(x, dtype=np.complex128)
    if edge_regions.edge_width == 0:
        return y

    scaled_rise = fractional_rise / edge_regions.edge_width
    y[edge_regions.rise_mask] = sample_gaussian_waveform(
        edge_regions.scaled_rise_values,
        fractional_breadth=scaled_rise,
        regularize=regularize,
    )
    y[edge_regions.fall_mask] = sample_gaussian_waveform(
        edge_regions.scaled_fall_values,
        fractional_breadth=scaled_rise,
        regularize=regularize,
    )
    return y


@derivative_definition(sample_parameter="x", order_parameter="order")
def sample_gaussian_square_waveform_derivative(
    x: NDArray[np.floating] | list[float],
    order: int = 1,
    *,
    fractional_rise: float = np.sqrt(2.0) / 3.0,
    regularize: bool = False,
    fractional_top_width: float = 0.5,
) -> NDArray[np.complexfloating]:
    """Samples the derivative of a Gaussian-Square waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_rise: The standard deviation of the Gaussian edge profile on the
        normalised waveform domain, default is ``sqrt(2)/3``.
    :param regularize: If ``True``, applies a shift and rescaling so that the waveform is
        zero at the edges. Default is ``False``.
    :param fractional_top_width: The proportion of the waveform that is square, between 0
        and 1. Default is ``0.5``.
    :param order: The order of the derivative to sample. Default is ``1``.
    """

    validate_fractional_top_width(fractional_top_width)
    validate_fractional_rise(fractional_rise)

    if order == 0:
        return sample_gaussian_square_waveform(
            x,
            fractional_rise=fractional_rise,
            regularize=regularize,
            fractional_top_width=fractional_top_width,
        )

    if order > 1:
        raise DerivativeOrderUndefinedError("Gaussian-Square", order)

    edge_regions = _edge_regions(x, fractional_top_width)

    y = np.zeros_like(x, dtype=np.complex128)
    if edge_regions.edge_width == 0:
        return y

    scaled_rise = fractional_rise / edge_regions.edge_width
    # Chain rule: d/dx f(x/a) = f'(x/a) / a, applied for each derivative order.
    chain_factor = edge_regions.edge_width**order
    y[edge_regions.rise_mask] = (
        sample_gaussian_waveform_derivative(
            edge_regions.scaled_rise_values,
            fractional_breadth=scaled_rise,
            regularize=regularize,
            order=order,
        )
        / chain_factor
    )
    y[edge_regions.fall_mask] = (
        sample_gaussian_waveform_derivative(
            edge_regions.scaled_fall_values,
            fractional_breadth=scaled_rise,
            regularize=regularize,
            order=order,
        )
        / chain_factor
    )
    return y


@dataclass(frozen=True)
class GaussianSquareWaveformShape(WaveformShape):
    """Waveform-shape wrapper for Gaussian-square sampling functions.

    :ivar fractional_top_width: The proportion of the waveform that is square, between 0
        and 1. Default is ``0.5``.
    :ivar fractional_rise: The standard deviation of the Gaussian edge profile on the
        normalised waveform domain, default is ``sqrt(2)/3``.
    :ivar regularize: If ``True``, applies a shift and rescaling so that the waveform is
        zero at the edges. Default is ``False``.
    """

    fractional_top_width: float = 0.5
    fractional_rise: float = np.sqrt(2.0) / 3.0
    regularize: bool = False

    def __post_init__(self) -> None:
        validate_fractional_top_width(self.fractional_top_width)
        validate_fractional_rise(self.fractional_rise)

    def evaluate(
        self, x: NDArray[np.floating] | list[float]
    ) -> NDArray[np.complexfloating]:
        """Evaluates the Gaussian-square waveform shape at the sample points."""

        return sample_gaussian_square_waveform(
            x,
            fractional_rise=self.fractional_rise,
            regularize=self.regularize,
            fractional_top_width=self.fractional_top_width,
        )

    def derivative(
        self, x: NDArray[np.floating] | list[float], order: int = 1
    ) -> NDArray[np.complexfloating]:
        """Evaluates the derivative of the Gaussian-square waveform shape."""

        return sample_gaussian_square_waveform_derivative(
            x,
            order,
            fractional_rise=self.fractional_rise,
            regularize=self.regularize,
            fractional_top_width=self.fractional_top_width,
        )

    @classmethod
    def from_absolute(
        cls,
        width: float,
        absolute_top_width: float,
        absolute_rise: float,
        regularize: bool = True,
    ) -> "GaussianSquareWaveformShape":
        """Constructs from absolute waveform parameters.

        :param width: The waveform width.
        :param absolute_top_width: The width of the flat-top square region.
        :param absolute_rise: The standard deviation of the Gaussian edge profile.
        :param regularize: Whether the waveform is zero at the edges. Default ``True``.
        """
        return cls(
            fractional_rise=absolute_rise / width,
            regularize=regularize,
            fractional_top_width=absolute_top_width / width,
        )

    @classmethod
    def from_legacy(
        cls,
        std_dev: float,
        width: float,
        square_width: float,
        zero_at_edges: bool = False,
    ) -> "GaussianSquareWaveformShape":
        """Constructs from legacy ``GaussianSquareWaveform`` parameters.

        :param std_dev: The ``std_dev`` parameter from the legacy implementation.
        :param width: The waveform width.
        :param square_width: The width of the flat-top square region.
        :param zero_at_edges: Whether the waveform is zero at the edges. Default ``False``.
        """
        return cls(
            fractional_rise=2.0 * std_dev / width,
            regularize=zero_at_edges,
            fractional_top_width=square_width / width,
        )
