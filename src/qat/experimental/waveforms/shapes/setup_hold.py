# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements the sampling definitions for a Setup Hold waveform shape.

The Setup Hold waveform is piecewise constant: a short initial setup segment with amplitude
``setup`` followed by a hold segment with amplitude ``1``. Sampling is performed on the
normalised domain ``x in [-1, 1]``.

Using the normalised setup duration ``rise_location`` (fraction of total waveform width in
``[0, 1]``), the transition boundary in ``x`` is

.. math::

    x_{b} = 2r - 1

where :math:`r` is the normalised setup duration. The waveform is then defined as

.. math::

    f(x) = \\begin{cases}
        A_{s} & \\text{if } x < x_b \\
        1 & \\text{if } x \\ge x_b
    \\end{cases},

where :math:`A_{s}` is the setup amplitude.

This implements the legacy ``SetupHoldWaveform`` under the parameterisations
``setup = amp_setup / amp`` and ``rise_location = rise_location / width``.

Since this waveform has a jump discontinuity at ``x_b``, classical derivatives are not
well-defined at the transition point and are therefore treated as undefined.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qat.experimental.waveforms.shapes.base import (
    WaveformShape,
    derivative_definition,
    shape_definition,
)
from qat.experimental.waveforms.shapes.exceptions import DerivativeOrderUndefinedError
from qat.experimental.waveforms.shapes.validators import validate_rise_location


@shape_definition(sample_parameter="x")
def sample_setup_hold_waveform(
    x: NDArray[np.floating] | list[float],
    *,
    setup: float = 0.1,
    rise_location: float = 0.1,
) -> NDArray[np.complexfloating]:
    """Samples a Setup Hold waveform shape.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param setup: Amplitude of the initial setup segment relative to the hold segment.
        Default is ``0.1``.
    :param rise_location: Fraction of the waveform duration occupied by the setup segment.
        Default is ``0.1``.
    :return: The sampled Setup Hold waveform as a complex-valued array.

    The setup-to-hold transition occurs at ``x_b = 2 * rise_location - 1``.
    """

    validate_rise_location(rise_location)

    boundary = 2 * rise_location - 1
    times_before_mask = x < boundary

    amps = np.ones_like(x, dtype=np.complex128)
    amps[times_before_mask] *= setup
    return amps


@derivative_definition(sample_parameter="x", order_parameter="order")
def sample_setup_hold_waveform_derivative(
    x: NDArray[np.floating] | list[float],
    order: int = 1,
    *,
    setup: float = 0.1,
    rise_location: float = 0.1,
) -> NDArray[np.complexfloating]:
    """Samples the derivative of a Setup Hold waveform shape.

    The derivative of a Setup Hold waveform shape is not defined at the transition point,
    and is therefore treated as undefined.

    :param x: The list of values in the range [-1, 1] to sample the waveform for.
    :param setup: Amplitude of the initial setup segment relative to the hold segment.
    :param rise_location: Fraction of the waveform duration occupied by the setup segment.
    :param order: The derivative order requested.
    :raises DerivativeOrderUndefinedError: Always for order greater than equal to one, since
        Setup Hold is discontinuous and derivatives are not defined in this sampling API.
    """
    validate_rise_location(rise_location)

    if order == 0:
        return sample_setup_hold_waveform(x, setup=setup, rise_location=rise_location)
    raise DerivativeOrderUndefinedError("Setup Hold", order)


@dataclass(frozen=True)
class SetupHoldWaveformShape(WaveformShape):
    """Waveform-shape wrapper for setup-hold sampling functions."""

    setup: float = 0.1
    rise_location: float = 0.1

    def __post_init__(self) -> None:
        validate_rise_location(self.rise_location)

    def evaluate(
        self, x: NDArray[np.floating] | list[float]
    ) -> NDArray[np.complexfloating]:
        """Evaluates the setup-hold waveform shape at the sample points."""

        return sample_setup_hold_waveform(
            x,
            setup=self.setup,
            rise_location=self.rise_location,
        )

    def derivative(
        self, x: NDArray[np.floating] | list[float], order: int = 1
    ) -> NDArray[np.complexfloating]:
        """Evaluates the derivative of the setup-hold waveform shape."""

        return sample_setup_hold_waveform_derivative(
            x,
            order,
            setup=self.setup,
            rise_location=self.rise_location,
        )

    @classmethod
    def from_absolute(
        cls, width: float, amp: float, absolute_rise: float, absolute_amp_setup: float
    ) -> "SetupHoldWaveformShape":
        """Constructs from legacy ``SetupHoldWaveform`` parameters.

        :param width: The waveform width.
        :param amp: The hold-segment amplitude from the legacy implementation.
        :param absolute_rise: The rise time of the setup-hold transition from the legacy
            implementation.
        :param absolute_amp_setup: The setup-segment amplitude from the legacy
            implementation.
        """
        return cls(setup=absolute_amp_setup / amp, rise_location=absolute_rise / width)

    @classmethod
    def from_legacy(
        cls, amp_setup: float, amp: float, rise: float, width: float
    ) -> "SetupHoldWaveformShape":
        """Constructs from legacy ``SetupHoldWaveform`` parameters.

        :param amp_setup: The setup-segment amplitude from the legacy implementation.
        :param amp: The hold-segment amplitude from the legacy implementation.
        :param rise: The rise time of the setup-hold transition from the legacy
            implementation.
        :param width: The waveform width.
        """
        return cls(setup=amp_setup / amp, rise_location=rise / width)
