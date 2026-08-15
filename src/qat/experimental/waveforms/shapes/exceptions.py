# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Custom exceptions for experimental waveform shape sampling, which can be used by
evaluation engines to implement a strategy for calculating derivatives (e.g. for DRAG) where
possible, and to raise an error where not possible."""


class DerivativeOrderUndefinedError(ValueError):
    """Raised when a derivative order is mathematically undefined for a shape."""

    def __init__(self, waveform_shape: str, order: int):
        super().__init__(
            f"The derivative of order {order} is not mathematically defined for "
            f"waveform shape '{waveform_shape}'."
        )


class DerivativeOrderNotImplementedError(NotImplementedError):
    """Raised when a mathematically valid derivative order is not implemented."""

    def __init__(self, waveform_shape: str, order: int):
        super().__init__(
            f"The derivative of order {order} exists for waveform shape "
            f"'{waveform_shape}' but is not implemented."
        )
