# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2025 Oxford Quantum Circuits Ltd
from itertools import product

import numpy as np
import pytest

from qat.purr.backends.utilities import (
    BlackmanFunction,
    ExtraSoftSquareFunction,
    GaussianFunction,
    GaussianSquareFunction,
    GaussianZeroEdgeFunction,
    NumericFunction,
    SechFunction,
    SofterSquareFunction,
    SoftSquareFunction,
    SquareFunction,
    evaluate_shape,
)
from qat.purr.compiler.instructions import CustomPulse


@pytest.mark.parametrize("sizes", [1, 2, 5, 7])
def test_square_function(sizes):
    func = SquareFunction()
    y1 = func(np.empty(shape=[sizes]))
    y2 = func.eval(np.empty(shape=[sizes]))
    assert np.array_equal(y1, y2)
    assert np.array_equal(y1, np.ones(shape=[sizes], dtype=np.complex128))


def test_size_limits():
    func = SquareFunction()
    max_size = int(1e6)
    func.eval(np.empty(shape=[max_size]))
    with pytest.raises(RuntimeError):
        func.eval(np.empty(shape=[max_size + 1]))
    func.derivative(np.empty(shape=[max_size]))
    with pytest.raises(RuntimeError):
        func.derivative(np.empty(shape=[max_size + 1]))


@pytest.mark.parametrize("sizes", [1, 2, 5, 7])
def test_square_function_first_derivative(sizes):
    x = np.empty(shape=[sizes])
    assert np.array_equal(
        SquareFunction().derivative(x), np.zeros(shape=[sizes], dtype=np.complex128)
    )
    assert np.array_equal(
        SquareFunction().derivative(x), np.zeros(shape=[sizes], dtype=np.complex128)
    )


def test_gaussian_function():
    x = np.array([-2, -1, 0, 1, 2])
    gaussian = GaussianFunction(rise=1, width=1)
    y = gaussian.eval(x)
    assert y.dtype == np.complex128
    # Known maxima
    assert y[2] == 1 + 0j
    # Test based on even function symmetry
    assert y[0] == y[-1]
    assert y[1] == y[-2]
    assert np.argmax(y) == 2


def test_gaussian_function_first_derivative():
    x = np.array([-2, -1, 0, 1, 2])
    gaussian = GaussianFunction(rise=1, width=1)
    y = gaussian.eval(x)
    y_x = gaussian.derivative(x)
    assert y_x.dtype == np.complex128
    assert np.argmax(y) == 2
    # Known maxima
    assert y_x[2] == 0 + 0j
    # Test based on even function symmetry
    assert y_x[0] == -y_x[-1]
    assert y_x[1] == -y_x[-2]


@pytest.mark.parametrize(
    ["width", "std_dev", "zero_at_edges"],
    product([0.5, 1.0, 1.5], [0.05, 0.1, 1.0], [False, True]),
)
def test_gaussian_zero_edge(width, std_dev, zero_at_edges):
    x = np.linspace(-width / 2, width / 2, 101)
    gaussian_zero = GaussianZeroEdgeFunction(std_dev, width, zero_at_edges)
    y = gaussian_zero(x)

    assert np.isclose(max(y), 1.0)
    assert np.isclose(x[np.argmax(y)], 0.0)
    if zero_at_edges:
        assert np.isclose(y[0], 0.0) and np.isclose(y[-1], 0.0)
    else:
        assert y[0] > 0.0 and y[-1] > 0.0


@pytest.mark.parametrize(
    ["width", "std_dev", "zero_at_edges"],
    product([0.5, 1.0, 1.5], [0.05, 0.1, 1.0], [False, True]),
)
def test_gaussian_square(width, std_dev, zero_at_edges):
    x = np.linspace(-2.0, 2.0, 101)
    gaussian_square = GaussianSquareFunction(width, std_dev, zero_at_edges)
    y = gaussian_square(x)

    # Test the shape looks like we expect it
    square_edge = width / 2 + 1e-8  # add small amount to deal with float errors
    assert all([val < 1.0 for val in y[x < -square_edge]])
    assert all([val < 1.0 for val in y[x > square_edge]])
    assert all([np.isclose(val, 1.0) for val in y[(x > -square_edge) * (x < square_edge)]])

    # Test zero at the edges
    if zero_at_edges:
        assert np.isclose(y[0], 0.0) and np.isclose(y[-1], 0.0)


def test_blackman_function():
    x = np.arange(start=-1, stop=1.5, step=0.5)
    blackman = BlackmanFunction(width=2)
    y = blackman.eval(x)
    # Known maxima
    assert np.isclose(y[2], 1 + 0j, atol=1e-6)
    # Test based on function symmetry
    assert np.isclose(y[0], y[-1], atol=1e-6)
    assert np.isclose(y[1], y[-2], atol=1e-6)
    assert np.argmax(y) == 2


def test_blackman_function_first_derivative():
    x = np.arange(start=-1, stop=1.5, step=0.5)
    blackman = BlackmanFunction(width=2)
    y = blackman.derivative(x)
    # Known maxima
    assert np.isclose(y[2], 0 + 0j, atol=1e-6)
    # Test based on function symmetry
    assert np.isclose(y[0], -y[-1], atol=1e-6)
    assert np.isclose(y[1], -y[-2], atol=1e-6)


@pytest.mark.parametrize("width", [-2.0, -1.0, -0.1, -1e-6, 1e-3, 0.2, 1.2, 10])
def test_sech_function(width):
    # Tests the sech pulse
    x = np.linspace(-1.0, 1.0, 101)
    sech = SechFunction(width)
    y = sech(x)
    max_idx = np.argmax(y)
    assert np.isclose(x[max_idx], 0.0)
    assert all(np.isclose(y[max_idx::-1], y[max_idx:]))


@pytest.mark.parametrize(
    ["func", "width", "rise"],
    product(
        [SoftSquareFunction, SofterSquareFunction, ExtraSoftSquareFunction],
        [0.5, 1.0, 2.0],
        [1e-3, 1e-2, 1e-1],
    ),
)
def test_soft_square_functions(func, width, rise):
    # Tests the properties of soft square functions: maximum, symmetry
    x = np.linspace(-1.0, 1.0, 101)
    f = func(width, rise)
    y = f(x).real
    assert all(y[50] >= y.real)
    assert all(np.isclose(y[50::-1], y[50:]))


@pytest.mark.skip(reason="I don't know what the new results should be.")
def test_numeric_derivative():
    class SomeFunction(NumericFunction):
        def eval(self, x):
            # linear gradient
            return x

    f = SomeFunction()
    d_y = f.derivative(np.arange(start=0, stop=5))
    assert np.allclose(d_y, np.ones(5), atol=1e-6)


def test_custom_pulse_evaluate_shape():
    samples = np.linspace(0, 1, 100, dtype=np.complex64)
    t = np.linspace(0, 1e-6, 100)
    pulse = CustomPulse(None, samples)
    buffer = evaluate_shape(pulse, t)
    assert len(buffer) == len(t)
    assert all(buffer == samples)
