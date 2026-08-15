# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements the sampling definitions for a Soft Square waveform shape.

This waveform has a fractional_rise and fall at the edges of the waveform, with a roughly
square region in the center. It is implemented by two hyperbolic tangent (tanh) functions in
opposite directions.

The first tanh is centered at ``-fractional_top_width`` and second is centered at
``fractional_top_width``, where ``fractional_top_width`` is a dimensionless parameter
between 0 and 1 that defined the width of the roughly square region in the center of the
waveform. The sharpness of the fractional_rise and fall edges are defined by the parameter
``fractional_rise``, which is a dimensionless parameter that defines the sharpness of the
fractional_rise and fall. Optionally, the ``regularize`` parameter can be set to ``True`` so
that the waveform is zero at the edges and unity at the center.

The waveform has the formula

.. math::

    f(x) = \\frac{1}{N}\\left[
        \\tanh\\left(\\frac{x + w_t}{r}\\right)
        - \\tanh\\left(\\frac{x - w_t}{r}\\right)
        - C
    \\right]

where :math:`w_t` is ``fractional_top_width`` and :math:`r` is ``fractional_rise``.
:math:`C` is the shift value, and :math:`N` is the rescale (regularize) value, which are
zero and one respectively if ``regularize`` is set to ``False``, and otherwise

.. math::

    C = \\left[
        \\tanh\\left(\\frac{1 + w_t}{r}\\right) - \\tanh\\left(\\frac{1 - w_t}{r}\\right)
    \\right]

.. math::

    N = \\tanh\\left(\\frac{w_t}{r}\\right) - \\tanh\\left(-\\frac{w_t}{r}\\right) - C.

This waveform implements the ``SoftSquareWaveform``, the ``SofterSquareWaveform``, and the
``ExtraSoftSquareWaveform`` shapes in the legacy implementation, under the following
parameterisations:

* ``SoftSquareWaveform``: ``fractional_top_width = 1 - fractional_rise / width``,
  ``fractional_rise = 2 * fractional_rise / width``, ``regularize = False``.
* ``SofterSquareWaveform``: ``fractional_top_width = (std_dev - 2 * fractional_rise) / width``,
  ``fractional_rise = 2 * fractional_rise / width``, ``regularize = True``.
* ``ExtraSoftSquareWaveform``: ``fractional_top_width = (std_dev - 4 * fractional_rise) / width``,
  ``fractional_rise = 2 * fractional_rise / width``, ``regularize = True``.

You can see the ``fractional_top_width`` parameterisation effectively becoming smaller as we
make the squares softer in this parameterisation.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qat.experimental.waveforms.shapes.base import (
    WaveformShape,
    derivative_definition,
    shape_definition,
)
from qat.experimental.waveforms.shapes.exceptions import DerivativeOrderNotImplementedError
from qat.experimental.waveforms.shapes.validators import validate_fractional_rise


def _shift_and_rescale_values(
    fractional_top_width: float, fractional_rise: float
) -> tuple[float, float]:
    """Returns the shift and rescale values for a Soft Square waveform shape."""

    boundary = 0.5 * (
        np.tanh((1 + fractional_top_width) / fractional_rise)
        - np.tanh((1 - fractional_top_width) / fractional_rise)
    )
    max_value = 0.5 * (
        np.tanh((fractional_top_width) / fractional_rise)
        - np.tanh((-fractional_top_width) / fractional_rise)
    )
    scale = 1 / (max_value - boundary)
    return boundary, scale


@shape_definition(sample_parameter="x")
def sample_soft_square_waveform(
    x: NDArray[np.floating] | list[float],
    *,
    fractional_top_width: float = 0.5,
    fractional_rise: float = 0.1,
    regularize: bool = False,
) -> NDArray[np.complexfloating]:
    """Samples a Soft Square waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_top_width: The width of the roughly square region in the center of the
        waveform, default is ``0.5``.
    :param fractional_rise: The sharpness of the fractional_rise and fall edges of the
        waveform, default is ``0.1``.
    :param regularize: Whether to regularize the waveform to be zero at the edges and unity
        at the center, default is ``False``.
    :return: The sampled Soft Square waveform as a complex-valued array.
    """

    validate_fractional_rise(fractional_rise)

    # Centers the two tanh functions around -fractional_top_width and +fractional_top_width
    # respectively.
    forwards_tanh = np.tanh((x + fractional_top_width) / fractional_rise)
    backwards_tanh = -np.tanh((x - fractional_top_width) / fractional_rise)
    y = 0.5 * (forwards_tanh + backwards_tanh)

    # Shift to the domain [0, 1]
    if regularize:
        shift, scale = _shift_and_rescale_values(fractional_top_width, fractional_rise)
        y = scale * (y - shift)
    return y.astype(np.complex128)


def _first_derivative_soft_square(
    x: NDArray[np.floating] | list[float],
    *,
    fractional_top_width: float = 0.5,
    fractional_rise: float = 0.1,
    regularize: bool = False,
) -> NDArray[np.complexfloating]:
    """Samples the first derivative of a Soft Square waveform shape."""

    raw = (
        0.5
        / fractional_rise
        * (
            1 / (np.cosh((x + fractional_top_width) / fractional_rise) ** 2)
            - 1 / (np.cosh((x - fractional_top_width) / fractional_rise) ** 2)
        )
    ).astype(np.complex128)
    if regularize:
        _, scale = _shift_and_rescale_values(fractional_top_width, fractional_rise)
        return scale * raw
    return raw


def _second_derivative_soft_square(
    x: NDArray[np.floating] | list[float],
    *,
    fractional_top_width: float = 0.5,
    fractional_rise: float = 0.1,
    regularize: bool = False,
) -> NDArray[np.complexfloating]:
    """Samples the second derivative of a Soft Square waveform shape."""

    raw = (
        1
        / (fractional_rise**2)
        * (
            -1
            * np.tanh((x + fractional_top_width) / fractional_rise)
            / (np.cosh((x + fractional_top_width) / fractional_rise) ** 2)
            + np.tanh((x - fractional_top_width) / fractional_rise)
            / (np.cosh((x - fractional_top_width) / fractional_rise) ** 2)
        )
    ).astype(np.complex128)
    if regularize:
        _, scale = _shift_and_rescale_values(fractional_top_width, fractional_rise)
        return scale * raw
    return raw


@derivative_definition(sample_parameter="x", order_parameter="order")
def sample_soft_square_waveform_derivative(
    x: NDArray[np.floating] | list[float],
    order: int = 1,
    *,
    fractional_top_width: float = 0.5,
    fractional_rise: float = 0.1,
    regularize: bool = False,
) -> NDArray[np.complexfloating]:
    """Samples the derivative of a Soft Square waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param fractional_top_width: The width of the roughly square region in the center of the
        waveform, default is ``0.5``.
    :param fractional_rise: The sharpness of the fractional_rise and fall edges of the
        waveform, default is ``0.1``.
    :param regularize: Whether to apply the same regularize scale as the waveform, default
        is ``False``.
    :param order: The order of the derivative to sample, default is ``1``.
    """

    if order == 0:
        return sample_soft_square_waveform(
            x,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
            regularize=regularize,
        )

    validate_fractional_rise(fractional_rise)
    if order == 1:
        return _first_derivative_soft_square(
            x,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
            regularize=regularize,
        )
    if order == 2:
        return _second_derivative_soft_square(
            x,
            fractional_top_width=fractional_top_width,
            fractional_rise=fractional_rise,
            regularize=regularize,
        )
    raise DerivativeOrderNotImplementedError("Soft Square", order)


@dataclass(frozen=True)
class SoftSquareWaveformShape(WaveformShape):
    """Waveform-shape wrapper for soft-square sampling functions."""

    fractional_top_width: float = 0.5
    fractional_rise: float = 0.1
    regularize: bool = False

    def __post_init__(self) -> None:
        validate_fractional_rise(self.fractional_rise)

    def evaluate(
        self, x: NDArray[np.floating] | list[float]
    ) -> NDArray[np.complexfloating]:
        """Evaluates the soft-square waveform shape at the sample points."""

        return sample_soft_square_waveform(
            x,
            fractional_top_width=self.fractional_top_width,
            fractional_rise=self.fractional_rise,
            regularize=self.regularize,
        )

    def derivative(
        self, x: NDArray[np.floating] | list[float], order: int = 1
    ) -> NDArray[np.complexfloating]:
        """Evaluates the derivative of the soft-square waveform shape."""

        return sample_soft_square_waveform_derivative(
            x,
            order,
            fractional_top_width=self.fractional_top_width,
            fractional_rise=self.fractional_rise,
            regularize=self.regularize,
        )

    @classmethod
    def from_absolute(
        cls,
        width: float,
        absolute_top_width: float,
        absolute_rise: float,
        regularize: bool = False,
    ) -> "SoftSquareWaveformShape":
        """Constructs from absolute parameters.

        :param width: The waveform width.
        :param absolute_top_width: The "top width" of the Soft Square function.
        :param absolute_rise: The sharpness of the fractional_rise and fall edges of the
            Soft Square function.
        :param regularize: If ``True``, applies a shift and rescaling so that the waveform is
            zero at the edges with maximum component one. Default is ``False``.
        """
        return cls(
            fractional_top_width=absolute_top_width / width,
            fractional_rise=absolute_rise / width,
            regularize=regularize,
        )

    @classmethod
    def from_soft_square_waveform(
        cls, rise: float, width: float
    ) -> "SoftSquareWaveformShape":
        """Constructs from legacy ``SoftSquareWaveform`` parameters.

        :param rise: The ``rise`` parameter from the legacy implementation.
        :param width: The waveform width.
        """
        return cls(
            fractional_top_width=1.0 - rise / width,
            fractional_rise=2.0 * rise / width,
            regularize=False,
        )

    @classmethod
    def from_softer_square_waveform(
        cls, std_dev: float, rise: float, width: float
    ) -> "SoftSquareWaveformShape":
        """Constructs from legacy ``SofterSquareWaveform`` parameters.

        :param std_dev: The ``std_dev`` parameter from the legacy implementation.
        :param rise: The ``rise`` parameter from the legacy ``SofterSquareWaveform`` implementation.
        :param width: The waveform width.
        """
        return cls(
            fractional_top_width=(std_dev - 2.0 * rise) / width,
            fractional_rise=2.0 * rise / width,
            regularize=True,
        )

    @classmethod
    def from_extra_soft_square_waveform(
        cls, std_dev: float, rise: float, width: float
    ) -> "SoftSquareWaveformShape":
        """Constructs from legacy ``ExtraSoftSquareWaveform`` parameters.

        :param std_dev: The ``std_dev`` parameter from the legacy implementation.
        :param rise: The ``rise`` parameter from the legacy implementation.
        :param width: The waveform width.
        """
        return cls(
            fractional_top_width=(std_dev - 4.0 * rise) / width,
            fractional_rise=2.0 * rise / width,
            regularize=True,
        )
