# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Shared parameter validators for waveform shape sampling functions."""


def validate_fractional_breadth(fractional_breadth: float) -> None:
    """Raises ``ValueError`` if ``fractional_breadth`` is not greater than zero.

    :param fractional_breadth: The value to validate.
    """
    if fractional_breadth <= 0:
        raise ValueError("fractional_breadth must be greater than zero.")


def validate_fractional_rise(fractional_rise: float) -> None:
    """Raises ``ValueError`` if ``fractional_rise`` is not greater than zero.

    :param fractional_rise: The value to validate.
    """
    if fractional_rise <= 0:
        raise ValueError("fractional_rise must be greater than zero.")


def validate_rise_location(rise_location: float) -> None:
    """Raises ``ValueError`` if ``rise_location`` is not greater than zero.

    :param rise_location: The value to validate.
    """
    if rise_location <= 0:
        raise ValueError("rise_location must be greater than zero.")


def validate_fractional_top_width(fractional_top_width: float) -> None:
    """Raises ``ValueError`` if ``fractional_top_width`` is not in ``[0, 1]``.

    :param fractional_top_width: The value to validate.
    """
    if not 0.0 <= fractional_top_width <= 1.0:
        raise ValueError(
            "fractional_top_width must satisfy 0 <= fractional_top_width <= 1."
        )
