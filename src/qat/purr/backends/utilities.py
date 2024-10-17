# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import sys
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, pi, sin
from scipy.special import erf

from qat.purr.compiler.devices import PhysicalChannel, PulseChannel, PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CustomPulse,
    ProcessAxis,
    Pulse,
    QuantumInstruction,
    Waveform,
)

UPCONVERT_SIGN = 1.0
MAX_COSH_ARG = np.arccosh(sys.float_info.max)


def remove_axes(original_dims, removed_axis_indices, axis_locations):
    # map original axis index to new axis index
    axis_map = {i: i for i in range(original_dims)}
    for r in removed_axis_indices:
        if r < 0:
            r = original_dims + r
        axis_map[r] = None
        for i in range(r + 1, original_dims):
            if axis_map[i] is not None:
                axis_map[i] -= 1
    axis_negative = {k: v < 0 for k, v in axis_locations.items()}
    new_axis_locations = {
        k: (original_dims + v if v < 0 else v) for k, v in axis_locations.items()
    }
    new_axis_locations = {
        k: axis_map[v] for k, v in new_axis_locations.items() if axis_map[v] is not None
    }
    new_dims = original_dims - len(removed_axis_indices)
    new_axis_locations = {
        k: (v - new_dims) if axis_negative[k] else v for k, v in new_axis_locations.items()
    }
    return new_axis_locations


class ComplexFunction:
    """Function object used to represent Complex 1D functions"""

    _dtype = np.complex128
    dt = 0.5e-9

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Function evaluated in domain described by x
        """
        pass

    def derivative(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        First order derivative
        """
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.eval(x)


def validate_input_array(func):
    @wraps(func)
    def validator(*args, **kwargs):
        max_size = 1e6
        # Check every argument except for any `self`
        for arg in [val for val in args if not isinstance(val, ComplexFunction)]:
            if not isinstance(arg, np.ndarray):
                raise TypeError(f"Function given {type(arg)}, expecting numpy.ndarray")
            if arg.size > max_size:
                raise RuntimeError(
                    f"Function given {arg.size} element inputs "
                    f"exceeding maximum {max_size}"
                )
        return func(*args, **kwargs)

    return validator


class NumericFunction(ComplexFunction):
    """
    Base class for functions applying an numerical first derivative
    """

    @validate_input_array
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
    Square function of fixed amplitude
    """

    @validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        return np.ones(shape=x.shape, dtype=self._dtype)

    @validate_input_array
    def derivative(self, x: np.ndarray, _=None) -> np.ndarray:
        return np.zeros(shape=x.shape, dtype=self._dtype)


class GaussianFunction(ComplexFunction):
    """
    Gaussian function
    """

    def __init__(self, width, rise):
        self._k = width * rise

    @validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-((x / self._k) ** 2), dtype=self._dtype)

    @validate_input_array
    def derivative(self, x: np.ndarray, _=None) -> np.ndarray:
        c = -2.0 * x / self._k**2
        return c * self.eval(x)


class GaussianZeroEdgeFunction(ComplexFunction):
    """
    A Gaussian pulse that can be normalized to be zero at the edges.
    """

    def __init__(self, std_dev: float, width: float, zero_at_edges: bool):
        self.std_dev = std_dev
        self.width = width
        self.zero_at_edges = zero_at_edges

    @validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        gauss = np.exp(-0.5 * (x / self.std_dev) ** 2)
        if self.zero_at_edges:
            zae_chunk = self.zero_at_edges * (
                np.exp(-0.5 * ((self.width / 2) / self.std_dev) ** 2)
            )
            gauss = (gauss - zae_chunk) / (1 - zae_chunk)
        return gauss


class GaussianSquareFunction(NumericFunction):
    """
    A square pulse with a Gaussian rise and fall at the edges.
    """

    def __init__(self, square_width: float, std_dev: float, zero_at_edges: bool):
        self.square_width = square_width
        self.std_dev = std_dev
        self.zero_at_edges = zero_at_edges

    @validate_input_array
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

    def __init__(self, width, beta, zero_at_edges):
        self.width = width
        self.beta = beta
        self.zero_at_edges = zero_at_edges

    def eval(self, x: np.ndarray) -> np.ndarray:
        zae_chunk = -self.zero_at_edges * (np.exp(-0.5 * (self.width / self.width) ** 2))
        beta_chunk = 1 - 1j * self.beta * x / self.width**2
        coef = beta_chunk / (1 - zae_chunk)
        gauss = np.exp(-0.5 * (x / (2 * self.width)) ** 2)
        return coef * (gauss - zae_chunk)


class SechFunction(ComplexFunction):
    """
    Implements a sech pulse defined by sech(x / width). Note that it is not normalized to be
    zero at the edges.
    """

    def __init__(self, width):
        self.width = width

    def eval(self, x: np.ndarray) -> np.ndarray:
        # Having a narrow width can cause overflows in numpy
        # Restricting the argument such that cosh is within the max float range
        # will overcome this, and has a neglibable effect (as sech(x) outside this
        # range is practically zero).
        x = np.clip(x.real / self.width, -MAX_COSH_ARG, MAX_COSH_ARG)
        return 1 / np.cosh(x)


class Sin(ComplexFunction):
    def __init__(self, frequency, internal_phase):
        self.frequency = frequency
        self.internal_phase = internal_phase

    def eval(self, x: np.ndarray) -> np.ndarray:
        return sin(2 * pi * x * self.frequency + self.internal_phase)


class Cos(ComplexFunction):
    def __init__(self, frequency, internal_phase):
        self.frequency = frequency
        self.internal_phase = internal_phase

    def eval(self, x: np.ndarray) -> np.ndarray:
        return cos(2 * pi * x * self.frequency + self.internal_phase)


class RoundedSquareFunction(ComplexFunction):
    """
    Rounded square.
           ___
         /    \
     ___|      |___
    """

    def __init__(self, width, std_dev, rise):
        self.width = width
        self.std_dev = std_dev
        self.rise_time = rise

    def step(self, val):
        return (erf(val) + 1) / 2

    @validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        x1 = (self.width - self.std_dev) / 2
        x2 = (self.width + self.std_dev) / 2
        rescaling = (
            self.step((self.width / 2 - x1) / self.rise_time)
            + self.step(-(self.width / 2 - x2) / self.rise_time)
            - 1
        )
        return rescaling * (
            self.step((x - x1) / self.rise_time) + self.step(-(x - x2) / self.rise_time) - 1
        )


class BlackmanFunction(ComplexFunction):
    def __init__(self, width):
        self._a0 = 7938.0 / 18608.0
        self._a1 = 9240.0 / 18608.0
        self._a2 = 1430.0 / 18608.0
        self._width = width

    @validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        nN = 2.0 * np.pi * (x / self._width + 0.5)
        return self._a0 - self._a1 * np.cos(nN) + self._a2 * np.cos(2 * nN)

    @validate_input_array
    def derivative(self, x: np.ndarray, _=None) -> np.ndarray:
        nN = 2.0 * np.pi * (x / self._width + 0.5)
        c = 2.0 * np.pi / self._width
        return self._a1 * c * np.sin(nN) - self._a2 * 2 * c * np.sin(2 * nN)


class SoftSquareFunction(NumericFunction):
    def __init__(self, width, rise):
        self._width = width
        self._rise = rise

    @validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * (
            np.tanh((x + (self._width - self._rise) / 2.0) / self._rise, dtype=self._dtype)
            - np.tanh(
                (x - (self._width - self._rise) / 2.0) / self._rise, dtype=self._dtype
            )
        )


class SofterSquareFunction(NumericFunction):
    def __init__(self, width, rise):
        self._width = width
        self._rise = rise

    @validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        pulse = np.tanh((x + self._width / 2.0 - self._rise) / self._rise) - np.tanh(
            (x - self._width / 2.0 + self._rise) / self._rise
        )
        if pulse.any():
            pulse -= np.min(pulse)
            pulse /= np.max(pulse)
        return np.array(pulse, dtype=self._dtype)


class ExtraSoftSquareFunction(NumericFunction):
    def __init__(self, width, rise):
        self._width = width
        self._rise = rise

    @validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        pulse = np.tanh((x + self._width / 2.0 - 2.0 * self._rise) / self._rise) - np.tanh(
            (x - self._width / 2.0 + 2.0 * self._rise) / self._rise
        )
        if pulse.any():
            pulse -= np.min(pulse)
            pulse /= np.max(pulse)
        return np.array(pulse, dtype=self._dtype)


class SofterGaussianFunction(NumericFunction):
    def __init__(self, width, rise):
        self._width = width
        self._rise = rise

    @validate_input_array
    def eval(self, x: np.ndarray) -> np.ndarray:
        pulse = GaussianFunction(rise=self._rise, width=self._width).eval(x)
        if pulse.any():
            pulse -= min(pulse)
            pulse /= max(pulse)
        return np.array(pulse, dtype=self._dtype)


class SetupHoldFunction(NumericFunction):
    def __init__(self, width, rise, amp_setup, amp):
        self._width = width
        self._rise = rise
        self._amp_setup = amp_setup
        self._amp = amp

    @validate_input_array
    def eval(self, x):
        arr = np.ones(shape=x.shape, dtype=self._dtype)
        dt = x[1] - x[0]
        high_samples = int(self._rise / dt + 0.5)
        arr[0:high_samples] *= self._amp_setup / self._amp
        return arr


def evaluate_shape(data: Waveform, t, phase_offset=0.0):
    amp = 1.0
    scale_factor = 1.0
    drag = 0.0

    if isinstance(data, Pulse):
        # OQC internal
        if data.shape == PulseShapeType.SQUARE:
            num_func = SquareFunction()
        elif data.shape == PulseShapeType.GAUSSIAN:
            num_func = GaussianFunction(data.width, data.rise)
        elif data.shape == PulseShapeType.SOFT_SQUARE:
            num_func = SoftSquareFunction(data.width, data.rise)
        elif data.shape == PulseShapeType.BLACKMAN:
            num_func = BlackmanFunction(data.width)
        elif data.shape == PulseShapeType.SETUP_HOLD:
            num_func = SetupHoldFunction(data.std_dev, data.rise, data.amp_setup, data.amp)
        elif data.shape == PulseShapeType.SOFTER_SQUARE:
            num_func = SofterSquareFunction(data.std_dev, data.rise)
        elif data.shape == PulseShapeType.EXTRA_SOFT_SQUARE:
            num_func = ExtraSoftSquareFunction(data.std_dev, data.rise)
        elif data.shape == PulseShapeType.SOFTER_GAUSSIAN:
            num_func = SofterGaussianFunction(data.width, data.rise)

        # external
        elif data.shape == PulseShapeType.ROUNDED_SQUARE:
            num_func = RoundedSquareFunction(data.width, data.std_dev, data.rise)
        elif data.shape == PulseShapeType.GAUSSIAN_DRAG:
            num_func = DragGaussianFunction(data.std_dev, data.beta, data.zero_at_edges)
        elif data.shape == PulseShapeType.GAUSSIAN_ZERO_EDGE:
            num_func = GaussianZeroEdgeFunction(
                data.std_dev, data.width, data.zero_at_edges
            )
        elif data.shape == PulseShapeType.SECH:
            num_func = SechFunction(data.std_dev)
        elif data.shape == PulseShapeType.SIN:
            num_func = Sin(data.frequency, data.internal_phase)
        elif data.shape == PulseShapeType.COS:
            num_func = Cos(data.frequency, data.internal_phase)
        elif data.shape == PulseShapeType.GAUSSIAN_SQUARE:
            num_func = GaussianSquareFunction(
                data.square_width, data.std_dev, data.zero_at_edges
            )
        else:
            raise ValueError(f"'{str(data.shape)}' is an unknown pulse shape.")

        amplitude = num_func(t)

        phase_offset = data.phase + phase_offset
        amp = data.amp
        scale_factor = data.scale_factor
        drag = data.drag
    elif isinstance(data, CustomPulse):
        num_func = NumericFunction()
        amplitude = np.array(data.samples, dtype=np.csingle)
    else:
        raise ValueError(f"'{str(data)}' is an unknown pulse type. Can't evaluate shape.")

    buf = scale_factor * amp * np.exp(1.0j * phase_offset) * amplitude
    if not drag == 0.0:
        amplitude_differential = num_func.derivative(t, amplitude)
        if len(amplitude_differential) < len(buf):
            amplitude_differential = np.pad(
                amplitude_differential, (0, len(buf) - len(amplitude_differential)), "edge"
            )
        buf += (
            drag
            * 1.0j
            * amp
            * scale_factor
            * np.exp(1.0j * phase_offset)
            * amplitude_differential
        )

    return buf


def evaluate_pulse_integral(data: Pulse):
    dt = 0.5e-9
    t = np.arange(-data.width / 2.0, data.width / 2.0, dt)

    return np.abs(np.sum(evaluate_shape(data, t)) * dt)


def predict_pulse_amplitude(data: Pulse, target_integrand):
    return data.amp * target_integrand / evaluate_pulse_integral(data)


@dataclass()
class PositionData:
    start: float
    end: float
    instruction: QuantumInstruction


@dataclass
class SimpleAcquire:
    start: int
    samples: int
    output_variable: str
    pulse_channel: PulseChannel
    physical_channel: PhysicalChannel
    mode: AcquireMode
    delay: float
    instruction: Acquire


def plot_buffers(buffers, getter_function):
    fig, ax = plt.subplots(nrows=len(buffers) // 2, ncols=2, squeeze=True)
    for i, (channel_id, buffer) in enumerate(buffers.items()):
        dt = getter_function(channel_id).sample_time
        t = np.arange(0.0, (len(buffer) - 0.5) * dt, dt)
        ax[i // 2, i % 2].plot(t, buffer.real, label="I")
        ax[i // 2, i % 2].plot(t, buffer.imag, label="Q")
        ax[i // 2, i % 2].set_title(f"{channel_id}")
    fig.legend()
    plt.show()


def get_axis_map(mode: AcquireMode, result: np.ndarray):
    axis_map = {}
    if mode == AcquireMode.RAW:
        if result.ndim > 1:
            axis_map = {ProcessAxis.SEQUENCE: -2, ProcessAxis.TIME: -1}
        else:
            axis_map = {ProcessAxis.TIME: -1}
    elif mode == AcquireMode.SCOPE:
        axis_map = {ProcessAxis.TIME: -1}
    elif mode == AcquireMode.INTEGRATOR:
        axis_map = {ProcessAxis.SEQUENCE: -1}
    return axis_map


def software_post_process_down_convert(
    args, axes: List[ProcessAxis], raw: np.ndarray, source_axes: Dict[ProcessAxis, int]
):
    freq, dt = args

    axis = source_axes[axes[0]]
    samples = raw.shape[axis]

    t = np.linspace(0.0, dt * (samples - 1), samples)
    thing = np.exp(-UPCONVERT_SIGN * 2.0j * np.pi * freq * t)

    npaxis = [np.newaxis] * raw.ndim
    npaxis[axis] = slice(None, None, None)
    result = raw * thing[tuple(npaxis)]

    return result, source_axes


def software_post_process_mean(
    target_axes: List[ProcessAxis], raw: np.ndarray, axes: Dict[ProcessAxis, int]
):
    axis_indices = tuple(axes[axis] for axis in axes if axis in target_axes)
    final_axes = remove_axes(raw.ndim, axis_indices, axes)
    final_data = np.mean(raw, axis=axis_indices)
    return final_data, final_axes


def software_post_process_linear_map_complex_to_real(
    args, raw: List[np.ndarray], axes: Dict[ProcessAxis, int]
):
    return np.real(args[0] * raw + args[1]), axes


def software_post_process_discriminate(
    args, raw: Union[np.ndarray, List[np.ndarray]], axes: Dict[ProcessAxis, int]
):
    z_vals = raw[0] if isinstance(raw, list) else raw
    discr = args[0]
    return np.array([-1 if z_val < discr else 1 for z_val in z_vals]), axes
