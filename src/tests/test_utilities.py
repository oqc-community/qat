# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import numpy as np
import pytest
from qat.purr.backends.utilities import (
    BlackmanFunction,
    GaussianFunction,
    NumericFunction,
    SquareFunction,
)


@pytest.mark.parametrize("sizes", [1, 2, 5, 7])
def test_square_function(sizes):
    func = SquareFunction()
    y1 = func(np.empty(shape=[sizes]))
    y2 = func.eval(np.empty(shape=[sizes]))
    assert np.array_equal(y1, y2)
    assert np.array_equal(y1, np.ones(shape=[sizes], dtype='cfloat'))


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
        SquareFunction().derivative(x), np.zeros(shape=[sizes], dtype='cfloat')
    )
    assert np.array_equal(
        SquareFunction().derivative(x), np.zeros(shape=[sizes], dtype='cfloat')
    )


def test_gaussian_function():
    x = np.array([-2, -1, 0, 1, 2])
    gaussian = GaussianFunction(rise=1, width=1)
    y = gaussian.eval(x)
    assert y.dtype == 'cfloat'
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
    assert y_x.dtype == 'cfloat'
    assert np.argmax(y) == 2
    # Known maxima
    assert y_x[2] == 0 + 0j
    # Test based on even function symmetry
    assert y_x[0] == -y_x[-1]
    assert y_x[1] == -y_x[-2]


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


@pytest.mark.skip(reason="I don't know what the new results should be.")
def test_numeric_derivative():
    class SomeFunction(NumericFunction):
        def eval(self, x):
            # linear gradient
            return x

    f = SomeFunction()
    d_y = f.derivative(np.arange(start=0, stop=5))
    assert np.allclose(d_y, np.ones(5), atol=1e-6)
