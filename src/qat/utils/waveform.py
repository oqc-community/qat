# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import abc
import sys
from typing import Type

import numpy as np
from scipy.special import erf

from qat.utils.pydantic import AllowExtraFieldsModel

UPCONVERT_SIGN = 1.0
# use slightly below the maxmiumum allowed float to avoid flaky overflow errors
MAX_COSH_ARG = np.arccosh(0.99 * sys.float_info.max)


class ComplexFunction(AllowExtraFieldsModel, abc.ABC):
    """Function object used to represent Complex 1D functions."""

    _dtype: Type = np.complex128
    dt: float = 0.5e-9

    @abc.abstractmethod
    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Function evaluated in domain described by x.
        """
        pass

    @abc.abstractmethod
    def derivative(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        First order derivative.
        """
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.eval(x)


def _validate_input_array(func):
    """
    Wrapper method to validate the input array.
    """

    def validator(*args, **kwargs):
        max_size = 1e6
        # Check every argument except for any `self`.
        for arg in [val for val in args if not isinstance(val, ComplexFunction)]:
            if not isinstance(arg, np.ndarray):
                raise TypeError(f"Function given {type(arg)}, expecting numpy.ndarray.")
            if arg.size > max_size:
                raise RuntimeError(
                    f"Function given {arg.size} element inputs "
                    f"exceeding maximum {max_size}."
                )
        return func(*args, **kwargs)

    return validator


class NumericFunction(ComplexFunction):
    """
    Base class for functions applying an numerical first derivative.
    """

    @_validate_input_array
    def derivative(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        For a custom wave-pulse or pulse without analytic derivative compute it
        numerically.
        """
        if len(x) < 2:
            return np.zeros_like(y)
        else:
            amplitude_differential = np.diff(y) / np.diff(x)
            # np.diff reduces the array by 1 item when calculating the differential,
            # so we approximate what the last point should look like
            return np.append(
                amplitude_differential,
                2 * amplitude_differential[-1] - amplitude_differential[-2],
            )


class SquareFunction(ComplexFunction):
    """
    Square function of fixed amplitude.
    """

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        return np.ones(shape=x.shape, dtype=self._dtype)

    @_validate_input_array
    def derivative(self, x: np.ndarray, _=None) -> np.ndarray:
        return np.zeros(shape=x.shape, dtype=self._dtype)


class GaussianFunction(ComplexFunction):
    """
    Gaussian function.
    """

    width: float
    rise: float

    @property
    def k(self):
        return self.width * self.rise

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-((x / self.k) ** 2), dtype=self._dtype)

    @_validate_input_array
    def derivative(self, x: np.ndarray, _=None) -> np.ndarray:
        c = -2.0 * x / self.k**2
        return c * self.eval(x)


class GaussianZeroEdgeFunction(ComplexFunction):
    """
    A Gaussian pulse that can be normalized to be zero at the edges.
    """

    std_dev: float
    width: float
    zero_at_edges: bool

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        gauss = np.exp(-0.5 * (x / self.std_dev) ** 2)
        if self.zero_at_edges:
            zae_chunk = self.zero_at_edges * (
                np.exp(-0.5 * ((self.width / 2) / self.std_dev) ** 2)
            )
            gauss = (gauss - zae_chunk) / (1 - zae_chunk)
        return gauss

    def derivative(self, x: np.ndarray, _=None):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` does not have an implementation for `derivative`."
        )


class GaussianSquareFunction(NumericFunction):
    """
    A square pulse with a Gaussian rise and fall at the edges.
    """

    square_width: float
    std_dev: float
    zero_at_edges: float

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        y = np.ones(shape=x.shape, dtype=self._dtype)
        x_rise = x[x < -self.square_width / 2] + (self.square_width / 2)
        x_fall = x[x > self.square_width / 2] - (self.square_width / 2)
        y[x < -self.square_width / 2] = np.exp(-0.5 * (x_rise / self.std_dev) ** 2)
        y[x > self.square_width / 2] = np.exp(-0.5 * (x_fall / self.std_dev) ** 2)
        if self.zero_at_edges:
            y = (y - y[0]) / (1 - y[0])
        return y


class DragGaussianFunction(ComplexFunction):
    """
    Drag Gaussian, tighter on one side and long tail on the other.
    """

    std_dev: float
    beta: float
    zero_at_edges: bool

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        zae_chunk = -self.zero_at_edges * (
            np.exp(-0.5 * (self.std_dev / self.std_dev) ** 2)
        )
        beta_chunk = 1 - 1j * self.beta * x / self.std_dev**2
        coef = beta_chunk / (1 - zae_chunk)
        gauss = np.exp(-0.5 * (x / (2 * self.std_dev)) ** 2)
        return coef * (gauss - zae_chunk)

    def derivative(self, x: np.ndarray, _=None):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` does not have an implementation for `derivative`."
        )


class SechFunction(ComplexFunction):
    """
    Implements a sech pulse defined by sech(x / width). Note that it is not normalized to be
    zero at the edges.
    """

    std_dev: float

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        # Having a narrow width can cause overflows in numpy
        # Restricting the argument such that cosh is within the max float range
        # will overcome this, and has a neglibable effect (as sech(x) outside this
        # range is practically zero).
        x = np.clip(x.real / self.std_dev, -MAX_COSH_ARG, MAX_COSH_ARG)
        return 1 / np.cosh(x)

    def derivative(self, x: np.ndarray, _=None):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` does not have an implementation for `derivative`."
        )


class Sin(ComplexFunction):
    frequency: float
    internal_phase: float

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * x * self.frequency + self.internal_phase)

    def derivative(self, x: np.ndarray, _=None):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` does not have an implementation for `derivative`."
        )


class Cos(ComplexFunction):
    frequency: float
    internal_phase: float

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        return np.cos(2 * np.pi * x * self.frequency + self.internal_phase)

    def derivative(self, x: np.ndarray, _=None):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` does not have an implementation for `derivative`."
        )


class RoundedSquareFunction(ComplexFunction):
    """
    Rounded square.
           ___
         /    \
     ___|      |___
    """

    width: float
    rise: float
    std_dev: float

    def step(self, val):
        return (erf(val) + 1) / 2

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        x1 = (self.width - self.std_dev) / 2
        x2 = (self.width + self.std_dev) / 2
        rescaling = (
            self.step((self.width / 2 - x1) / self.rise)
            + self.step(-(self.width / 2 - x2) / self.rise)
            - 1
        )
        return rescaling * (
            self.step((x - x1) / self.rise) + self.step(-(x - x2) / self.rise) - 1
        )

    def derivative(self, x: np.ndarray, _=None):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` does not have an implementation for `derivative`."
        )


class BlackmanFunction(ComplexFunction):
    width: float

    a0: float = 7938.0 / 18608.0
    a1: float = 9240.0 / 18608.0
    a2: float = 1430.0 / 18608.0

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        nN = 2.0 * np.pi * (x / self.width + 0.5)
        return self.a0 - self.a1 * np.cos(nN) + self.a2 * np.cos(2 * nN)

    @_validate_input_array
    def derivative(self, x: np.ndarray, _=None) -> np.ndarray:
        nN = 2.0 * np.pi * (x / self.width + 0.5)
        c = 2.0 * np.pi / self.width
        return self.a1 * c * np.sin(nN) - self.a2 * 2 * c * np.sin(2 * nN)


class SoftSquareFunction(NumericFunction):
    width: float
    rise: float

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        pulse = 0.5 * (
            np.tanh((x.real + (self.width - self.rise) / 2.0) / self.rise)
            - np.tanh((x.real - (self.width - self.rise) / 2.0) / self.rise)
        )
        return pulse.astype(self._dtype)


class SofterSquareFunction(NumericFunction):
    std_dev: float
    rise: float

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        pulse = np.tanh((x.real + self.std_dev / 2.0 - self.rise) / self.rise) - np.tanh(
            (x.real - self.std_dev / 2.0 + self.rise) / self.rise
        )
        if pulse.any():
            pulse -= np.min(pulse)
            pulse /= np.max(pulse)
        return pulse.astype(self._dtype)


class ExtraSoftSquareFunction(NumericFunction):
    std_dev: float
    rise: float

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        pulse = np.tanh(
            (x.real + self.std_dev / 2.0 - 2.0 * self.rise) / self.rise
        ) - np.tanh((x.real - self.std_dev / 2.0 + 2.0 * self.rise) / self.rise)
        if pulse.any():
            pulse -= np.min(pulse)
            pulse /= np.max(pulse)
        return pulse.astype(self._dtype)


class SofterGaussianFunction(NumericFunction):
    width: float
    rise: float

    @_validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        pulse = GaussianFunction(rise=self.rise, width=self.width).eval(x)
        if pulse.any():
            pulse -= min(pulse)
            pulse /= max(pulse)
        return np.array(pulse, dtype=self._dtype)


class SetupHoldFunction(NumericFunction):
    width: float
    rise: float
    amp_setup: float
    amp: float

    @_validate_input_array
    def eval(self, x):
        arr = np.ones(shape=x.shape, dtype=self._dtype)
        dt = x[1] - x[0]
        high_samples = int(self.rise / dt + 0.5)
        arr[0:high_samples] *= self.amp_setup / self.amp
        return arr
