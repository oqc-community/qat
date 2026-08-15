# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the base utility for waveform shape sampling."""

import numpy as np
import pytest

from qat.experimental.waveforms.shapes.base import (
    WaveformShape,
    derivative_definition,
    shape_definition,
)


class TestShapeDefinition:
    """Tests the ``shape_definition`` decorator."""

    def create_decorated_function(self):
        """Creates a decorated function for testing."""

        @shape_definition
        def sample_fn(x, *, param=0.0):
            return np.asarray(x) + param

        return sample_fn

    def test_valid_sample_points_does_not_raise(self):
        """Tests that valid sample points do not raise an error."""

        fn = self.create_decorated_function()
        x = [-1.0, 0.0, 1.0]
        result = fn(x, param=1.0)
        assert (result == [0.0, 1.0, 2.0]).all()

    def test_valid_sample_points_as_kwargs_does_not_raise(self):
        """Tests that valid sample points passed as a keyword argument do not raise an
        error."""

        fn = self.create_decorated_function()
        x = [-1.0, 0.0, 1.0]
        result = fn(x=x, param=1.0)
        assert (result == [0.0, 1.0, 2.0]).all()

    def test_input_is_coerced_to_numpy_array(self):
        """Tests that list input is coerced to a NumPy array before invocation."""

        captured = {}

        @shape_definition
        def sample_fn(x, *, param=0.0):
            captured["type"] = type(x)
            return x + param

        result = sample_fn([-1.0, 0.0, 1.0], param=1.0)
        assert captured["type"] is np.ndarray
        assert (result == [0.0, 1.0, 2.0]).all()

    def test_invalid_sample_points_raises_a_value_error(self):
        """Tests that invalid sample points raise a ValueError."""

        fn = self.create_decorated_function()
        with pytest.raises(
            ValueError, match="Waveform sample points x must satisfy -1 <= x <= 1."
        ):
            fn([-1.5, 0.0, 1.5])

    def test_boundary_points_with_small_floating_deviation_are_accepted(self):
        """Tests that tiny floating-point excursions near +/-1 are tolerated."""

        fn = self.create_decorated_function()
        x = [-1.0 - 1e-8, 0.0, 1.0 + 1e-8]
        result = fn(x, param=0.0)
        assert np.allclose(result, x)

    def test_missing_x_argument_raises_a_type_error(self):
        """Tests that missing the required 'x' argument raises a TypeError."""

        fn = self.create_decorated_function()
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'x'"):
            fn(param=1.0)

    def test_configurable_sample_parameter_by_name(self):
        """Tests selecting the sample parameter by name."""

        @shape_definition(sample_parameter="samples")
        def sample_fn(offset, samples):
            return np.asarray(samples) + offset

        result = sample_fn(1.0, [-1.0, 0.0, 1.0])
        assert (result == [0.0, 1.0, 2.0]).all()

    def test_configurable_sample_parameter_by_index(self):
        """Tests selecting the sample parameter by positional index."""

        @shape_definition(sample_parameter=1)
        def sample_fn(context, samples):
            del context
            return np.asarray(samples)

        result = sample_fn("ignored", [-1.0, 0.0, 1.0])
        assert (result == [-1.0, 0.0, 1.0]).all()

    def test_invalid_sample_parameter_name_raises_a_value_error(self):
        """Tests that an unknown sample-parameter name is rejected."""

        with pytest.raises(
            ValueError, match="Configured parameter 'samples' does not exist"
        ):

            @shape_definition(sample_parameter="samples")
            def sample_fn(x):
                return np.asarray(x)

            sample_fn([-1.0, 0.0, 1.0])

    def test_invalid_sample_parameter_index_raises_a_value_error(self):
        """Tests that an out-of-range sample-parameter index is rejected."""

        with pytest.raises(
            ValueError, match="Configured positional parameter index 1 is out of range"
        ):

            @shape_definition(sample_parameter=1)
            def sample_fn(x):
                return np.asarray(x)

            sample_fn([-1.0, 0.0, 1.0])


class TestDerivativeDefinition:
    """Tests the ``derivative_definition`` decorator."""

    def create_decorated_function(self):
        """Creates a decorated function for testing."""

        @derivative_definition
        def sample_fn(x, order=0, param=0.0):
            return np.asarray(x) + param + order

        return sample_fn

    def test_valid_sample_points_does_not_raise(self):
        """Tests that valid sample points do not raise an error."""

        fn = self.create_decorated_function()
        x = [-1.0, 0.0, 1.0]
        result = fn(x, order=1, param=1.0)
        assert (result == [1.0, 2.0, 3.0]).all()

    def test_valid_sample_points_as_kwargs_does_not_raise(self):
        """Tests that valid sample points passed as a keyword argument do not raise an
        error."""

        fn = self.create_decorated_function()
        x = [-1.0, 0.0, 1.0]
        result = fn(x=x, order=1, param=1.0)
        assert (result == [1.0, 2.0, 3.0]).all()

    def test_input_is_coerced_to_numpy_array(self):
        """Tests that list input is coerced to a NumPy array before invocation."""

        captured = {}

        @derivative_definition
        def sample_fn(x, order=0, param=0.0):
            captured["type"] = type(x)
            return x + param + order

        result = sample_fn([-1.0, 0.0, 1.0], order=1, param=1.0)
        assert captured["type"] is np.ndarray
        assert (result == [1.0, 2.0, 3.0]).all()

    def test_valid_sample_points_and_order_as_args_does_not_raise(self):
        """Tests that valid sample points and order passed as positional arguments do not
        raise an error."""

        fn = self.create_decorated_function()
        x = [-1.0, 0.0, 1.0]
        result = fn(x, 2, param=1.0)
        assert (result == [2.0, 3.0, 4.0]).all()

    def test_default_order_does_not_raise(self):
        """Tests that the default order does not raise an error."""

        fn = self.create_decorated_function()
        x = [-1.0, 0.0, 1.0]
        result = fn(x, param=1.0)
        assert (result == [0.0, 1.0, 2.0]).all()

    def test_invalid_sample_points_raises_a_value_error(self):
        """Tests that invalid sample points raise a ValueError."""

        fn = self.create_decorated_function()
        with pytest.raises(
            ValueError, match="Waveform sample points x must satisfy -1 <= x <= 1."
        ):
            fn(
                [-1.5, 0.0, 1.5],
            )

    def test_missing_x_argument_raises_a_type_error(self):
        """Tests that missing the required 'x' argument raises a TypeError."""

        fn = self.create_decorated_function()
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'x'"):
            fn(order=1, param=1.0)

    def test_negative_int_order_raises_a_value_error(self):
        """Tests that a negative integer order raises a ValueError."""

        fn = self.create_decorated_function()
        with pytest.raises(
            ValueError, match="Derivative order must be a non-negative integer."
        ):
            fn([-1.0, 0.0, 1.0], order=-1)

    def test_non_integer_order_raises_a_value_error(self):
        """Tests that a non-integer order raises a ValueError."""

        fn = self.create_decorated_function()
        with pytest.raises(
            ValueError, match="Derivative order must be a non-negative integer."
        ):
            fn([-1.0, 0.0, 1.0], order=1.5)

    @pytest.mark.parametrize("order", [np.int32(1), np.int64(2), np.uint8(3)])
    def test_numpy_integer_orders_are_accepted(self, order):
        """Tests that NumPy integer scalar types are accepted for derivative order."""

        fn = self.create_decorated_function()
        result = fn([-1.0, 0.0, 1.0], order=order)
        assert (result == np.asarray([-1.0, 0.0, 1.0]) + order).all()

    @pytest.mark.parametrize("order", [True, np.bool_(True)])
    def test_boolean_order_raises_a_value_error(self, order):
        """Tests that boolean derivative orders are rejected."""

        fn = self.create_decorated_function()
        with pytest.raises(
            ValueError, match="Derivative order must be a non-negative integer."
        ):
            fn([-1.0, 0.0, 1.0], order=order)

    def test_configurable_sample_parameter_by_name(self):
        """Tests selecting the sample parameter by name."""

        @derivative_definition(sample_parameter="samples")
        def sample_fn(scale, samples, order=0):
            return scale * np.asarray(samples) + order

        result = sample_fn(2.0, [-1.0, 0.0, 1.0], order=1)
        assert (result == [-1.0, 1.0, 3.0]).all()

    def test_configurable_sample_and_order_parameter_by_index(self):
        """Tests selecting sample and order parameters by positional index."""

        @derivative_definition(sample_parameter=1, order_parameter=2)
        def sample_fn(context, samples, order=0):
            del context
            return np.asarray(samples) + order

        result = sample_fn("ignored", [-1.0, 0.0, 1.0], 2)
        assert (result == [1.0, 2.0, 3.0]).all()


class TestWaveformShape:
    """Tests the ``WaveformShape`` abstract base class."""

    class _MockWaveformShape(WaveformShape):
        """A mock implementation of the WaveformShape abstract base class for testing."""

        def evaluate(self, x: np.ndarray | list[float]):
            return np.asarray(x) ** 2

        def derivative(self, x: np.ndarray | list[float], order: int = 1):
            if order == 0:
                return self.evaluate(x)
            elif order == 1:
                return 2 * np.asarray(x)
            else:
                return np.zeros_like(x)

    def test_call_calls_evaluate_method(self):
        """Tests that calling the waveform shape instance calls the evaluate method."""

        waveform = self._MockWaveformShape()
        x = np.array([-1.0, 0.0, 1.0])
        result = waveform(x)
        expected = waveform.evaluate(x)
        assert np.allclose(result, expected)

    def test_abstract_evaluate_placeholder_is_executable(self):
        """Tests the abstract placeholder body for evaluate."""

        assert WaveformShape.evaluate(None, np.asarray([0.0])) is None

    def test_abstract_derivative_placeholder_is_executable(self):
        """Tests the abstract placeholder body for derivative."""

        assert WaveformShape.derivative(None, np.asarray([0.0]), order=1) is None
