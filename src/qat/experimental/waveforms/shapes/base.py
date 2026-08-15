# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Shared decorators and helpers for waveform shape sampling."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from inspect import Signature, signature
from numbers import Integral
from typing import Any, ParamSpec, TypeVar

import numpy as np
from numpy.typing import NDArray

_P = ParamSpec("P")
_T = TypeVar("T")
_DOMAIN_TOLERANCE = 1e-6


def _validate_sample_domain(x: NDArray[np.floating]) -> None:
    """Validates that all sample points lie in the normalized range [-1, 1]."""

    if np.any((x < (-1.0 - _DOMAIN_TOLERANCE)) | (x > (1.0 + _DOMAIN_TOLERANCE))):
        raise ValueError("Waveform sample points x must satisfy -1 <= x <= 1.")


def _resolve_parameter_name(
    fn: Callable[..., Any], fn_signature: Signature, parameter: str | int
) -> str:
    """Resolves a parameter reference into a concrete parameter name."""

    if isinstance(parameter, str):
        if parameter not in fn_signature.parameters:
            raise ValueError(
                f"Configured parameter '{parameter}' does not exist in {fn.__name__}()."
            )
        return parameter

    positional_params = [
        p.name
        for p in fn_signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if parameter < 0 or parameter >= len(positional_params):
        raise ValueError(
            f"Configured positional parameter index {parameter} is out of range for "
            f"{fn.__name__}()."
        )
    return positional_params[parameter]


def shape_definition(
    fn: Callable[_P, _T] | None = None, *, sample_parameter: str | int = "x"
) -> Callable[_P, _T] | Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Coerces and validates normalized sampling points before shape evaluation.

    :param fn: The function to decorate.
    :param sample_parameter: The parameter that contains the sample points to validate. This
        may be a parameter name (for example ``"x"``) or the index of a positional
        parameter.
    """

    def decorator(inner_fn: Callable[_P, _T]) -> Callable[_P, _T]:
        fn_signature = signature(inner_fn)
        sample_parameter_name = _resolve_parameter_name(
            inner_fn, fn_signature, sample_parameter
        )

        @wraps(inner_fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            bound_arguments = fn_signature.bind_partial(*args, **kwargs)
            if sample_parameter_name not in bound_arguments.arguments:
                # If no sample points are provided, just call the function so the standard
                # error is raised.
                return inner_fn(*args, **kwargs)

            x = np.asarray(
                bound_arguments.arguments[sample_parameter_name], dtype=np.float64
            )
            _validate_sample_domain(x)

            bound_arguments.arguments[sample_parameter_name] = x
            return inner_fn(*bound_arguments.args, **bound_arguments.kwargs)

        return wrapper

    if fn is None:
        return decorator
    return decorator(fn)


def derivative_definition(
    fn: Callable[_P, _T] | None = None,
    *,
    sample_parameter: str | int = "x",
    order_parameter: str | int = "order",
) -> Callable[_P, _T] | Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Coerces/validates sampling points and validates derivative order.

    :param fn: The function to decorate.
    :param sample_parameter: The parameter that contains the sample points to validate. This
        may be a parameter name (for example ``"x"``) or the index of a positional
        parameter.
    :param order_parameter: The parameter that contains the derivative order. This may be a
        parameter name (for example ``"order"``) or the index of a positional parameter.
    """

    def decorator(inner_fn: Callable[_P, _T]) -> Callable[_P, _T]:
        fn_signature = signature(inner_fn)
        sample_parameter_name = _resolve_parameter_name(
            inner_fn, fn_signature, sample_parameter
        )
        order_parameter_name = _resolve_parameter_name(
            inner_fn, fn_signature, order_parameter
        )

        @wraps(inner_fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            bound_arguments = fn_signature.bind_partial(*args, **kwargs)
            if sample_parameter_name not in bound_arguments.arguments:
                # If no sample points are provided, just call the function so the standard
                # error is raised.
                return inner_fn(*args, **kwargs)

            x = np.asarray(
                bound_arguments.arguments[sample_parameter_name], dtype=np.float64
            )
            _validate_sample_domain(x)
            bound_arguments.arguments[sample_parameter_name] = x

            order = bound_arguments.arguments.get(order_parameter_name, 1)
            if not isinstance(order, Integral) or isinstance(order, bool) or order < 0:
                raise ValueError("Derivative order must be a non-negative integer.")

            return inner_fn(*bound_arguments.args, **bound_arguments.kwargs)

        return wrapper

    if fn is None:
        return decorator
    return decorator(fn)


class WaveformShape(ABC):
    """Abstract base class for waveform shapes.

    Child classes must implement the `evaluate` and `derivative` methods to provide the
    specific waveform shape and its derivatives.
    """

    @abstractmethod
    def evaluate(self, x: np.ndarray | list[float]) -> NDArray[np.complexfloating]:
        """Evaluates the waveform shape at the given sample points.

        :param x: The list of values in the range [-1, 1] to sample the waveform for.
        """
        pass

    @abstractmethod
    def derivative(
        self,
        x: np.ndarray | list[float],
        order: int = 1,
    ) -> NDArray[np.complexfloating]:
        """Evaluates the derivative of the waveform shape at the given sample points.

        :param x: The list of values in the range [-1, 1] to sample the waveform for.
        :param order: The order of the derivative to evaluate. Default is 1.
        """
        pass

    def __call__(self, x: np.ndarray | list[float]) -> NDArray[np.complexfloating]:
        """Allows the waveform shape to be called directly to evaluate its value or
        derivative.

        :param x: The list of values in the range [-1, 1] to sample the waveform for.
        """
        return self.evaluate(x)
