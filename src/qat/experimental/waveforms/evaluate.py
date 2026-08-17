# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements the entry point for evaluating waveform shapes, along with their amplitude,
width, phase multiplier and DRAG coefficients."""

from numbers import Number
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from qat.experimental.waveforms.numerical import numerical_derivative
from qat.experimental.waveforms.shapes.base import WaveformShape
from qat.experimental.waveforms.shapes.exceptions import DerivativeOrderNotImplementedError

_PICOSECONDS_AS_SECONDS = 1e-12


class _TimeTuple(NamedTuple):
    """A named tuple to hold the time and scaled time arrays."""

    times: NDArray[np.floating]
    scaled_times: NDArray[np.floating]
    scale: float
    num_points: int


def _sample_times(width: int, sample_time: int) -> _TimeTuple:
    """Returns the discrete times at which a waveform is sampled, given the width of the
    waveform and the sample time.

    The sample times are calculated by splitting the width into :math:`N` equal intervals,
    and sampling at the center of those intervals, i.e. at times
    :math:`t_{n} = (n + 1/2) \\delta t`.
    """

    if width % sample_time != 0:
        raise ValueError(
            f"Width {width} is not an integer multiple of sample time {sample_time}."
        )

    scale = 2 / (width * _PICOSECONDS_AS_SECONDS)
    num_points = width // sample_time
    scaled_times = np.linspace(-1 + 1 / num_points, 1 - 1 / num_points, num_points)
    times = np.linspace(sample_time / 2, width - sample_time / 2, num_points)
    return _TimeTuple(
        times=times, scaled_times=scaled_times, scale=scale, num_points=num_points
    )


def evaluate_waveform(
    *,
    width: int,
    sample_time: int,
    shape: WaveformShape,
    amplitude: float = 1.0,
    phase: float = 0.0,
    drag_coefficients: float | list[float] | None = None,
    allow_numerical_derivative: bool = True,
) -> NDArray[np.complexfloating]:
    """Evaluates the samples of a waveform.

    Requires the width of the waveform to be given, and a sample time which must multiply
    into the width by an integer multiple ``N``. That is,

    .. math:: N = \\frac{t_{d}}{\\delta t}

    must be an integer. The discrete times at which the waveform is sampled at is decided by
    splitting the width into :math:`N` equal intervals, and sampling at the center of those
    intervals, i.e. at times

    .. math::

        t_{n} = \\left(n + \\frac{1}{2}\\right) \\delta t.

    The waveform shape must also be provided. Optionally, an amplitude to scale the waveform
    can be given, and a phase which acts as a global rotation in complex space can also be
    provided. DRAG coefficients can also be provided to implement the DRAG technique up to
    the desired order if the waveform shape supports derivatives up to that order. The full
    waveform is implemented as

    .. math::

        A(t) = A e^{i \\theta}
        \\left( 1 + \\sum_{j=1}^{M} \\beta_{j} (i)**j \\frac{d^{j}}{dt^{j}}\\right)
        S\\left(\\frac{2t}{T} - 1\\right)

    Note that practically, the waveform shapes are implemented as a function of
    :math:`x \\in [-1, 1]`, which we map onto :math:`t \\in [0, T]` for width :math:`T`,so
    evaluating the derivative through that means we have to apply the chain rule.

    Analytical implementation of derivatives takes precedence, but if the derivative is
    known to be mathematically defined but not implemented, then a numerical derivative is
    used instead. If the derivative is mathematically undefined, then a ValueError is
    raised.

    :param width: The width of the waveform in picoseconds
    :param sample_time: The time between samples in picoseconds.
    :param shape: The waveform shape to evaluate.
    :param amplitude: The amplitude to scale the waveform.
    :param phase: The global phase rotation in complex space.
    :param drag_coefficients: The DRAG coefficients to implement the DRAG technique. Use
        0.0, or an empty list to not implement DRAG. Every entry represents the ith order
        of DRAG to implement, starting from order 1. A float only implements first order.
    :param allow_numerical_derivative: Whether to allow numerical derivatives if the
        derivative is mathematically defined but not implemented. If False, a
        DerivativeOrderNotImplementedError is raised instead.
    :raises ValueError: If the width is not an integer multiple of the sample time.
    :raises DerivativeOrderUndefinedError: If a derivative order is mathematically undefined
        for the waveform shape.
    :return: The evaluated waveform samples at the discrete sample times.
    """

    if drag_coefficients is None:
        drag_coefficients = []
    drag_coefficients = (
        [drag_coefficients] if isinstance(drag_coefficients, Number) else drag_coefficients
    )

    times_data = _sample_times(width, sample_time)
    samples = shape.evaluate(times_data.scaled_times)

    last_order_derivatives = samples
    for order, coeff in enumerate(drag_coefficients, start=1):
        try:
            derivative = shape.derivative(times_data.scaled_times, order)
        except DerivativeOrderNotImplementedError:
            if allow_numerical_derivative:
                derivative = numerical_derivative(
                    last_order_derivatives, times_data.scaled_times
                )
            else:
                raise

        samples += coeff * (1j * times_data.scale) ** order * derivative
        last_order_derivatives = derivative

    amplitude_scale = amplitude * np.exp(1j * phase)
    samples *= amplitude_scale
    return samples
