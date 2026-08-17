# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""This module contains numerical methods for evaluating waveform derivatives, which can be
used to implement DRAG when analytical definitions are not available."""

from numbers import Number

import numpy as np


def numerical_derivative(
    y: np.ndarray | list[Number], x: np.ndarray | list[Number]
) -> np.ndarray:
    """Computes a derivative of a function using numerical methods.

    Currently just calls ``np.gradient`` which uses a second order central difference
    method.

    :param y: The y values of the function to differentiate.
    :param x: The x values of the function to differentiate.
    :return: The derivative of the function at each x value.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if len(y) != len(x):
        raise ValueError(
            f"y and x must have the same length, but got {len(y)} and {len(x)}."
        )
    if len(y) < 2:
        raise ValueError(
            f"y and x must have at least 2 points to compute a derivative, but got {len(y)}."
        )

    return np.gradient(y, x)
