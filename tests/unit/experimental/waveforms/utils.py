# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Utilities for testing waveform shape sampling functions."""

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def estimate_derivative_at_point(
    sample_fn: Callable[[NDArray[np.floating]], NDArray[np.complexfloating]],
    x0: float,
    order: int,
    h: float | None = None,
    stencil_radius: int | None = None,
) -> np.complexfloating:
    """Estimate an arbitrary-order derivative at a point using finite differences.

    :param sample_fn: Sampler that accepts an array of sample points and optional keyword
            arguments.
    :param x0: Point where the derivative is estimated.
    :param order: Derivative order to estimate. Must be greater than or equal to zero.
    :param h: Optional finite-difference step size. If omitted, a heuristic value is used.
    :param stencil_radius: Optional symmetric stencil radius. If omitted, this is chosen as
            ``max(2, order + 1)``.
    :returns: Numerical estimate of the derivative at ``x0``.
    """

    radius = stencil_radius if stencil_radius is not None else max(2, order + 1)
    if radius < 1:
        raise ValueError("stencil_radius must be greater than or equal to one")

    # Symmetric stencil over integer offsets, e.g. [-2, -1, 0, 1, 2].
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    n_points = offsets.size

    if order >= n_points:
        raise ValueError(
            "order must be less than the number of stencil points "
            f"({n_points} for radius={radius})"
        )

    # Solve for finite-difference coefficients c_j such that
    # sum_j c_j * k_j^p = 0 for p != order and order! for p == order.
    a_matrix = np.vstack([offsets**power for power in range(n_points)])
    b_vector = np.zeros(n_points, dtype=np.float64)
    b_vector[order] = math.factorial(order)
    coeffs = np.linalg.solve(a_matrix, b_vector)

    if h is None:
        scale = max(1.0, abs(x0))
        h = (np.finfo(np.float64).eps ** (1.0 / (order + 1))) * scale
    if h <= 0:
        raise ValueError("h must be greater than zero")

    x_values = x0 + h * offsets
    y_values = np.asarray(sample_fn(x_values), dtype=np.complex128)

    return np.sum(coeffs * y_values) / (h**order)
