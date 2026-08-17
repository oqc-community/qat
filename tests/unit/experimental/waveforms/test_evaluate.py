# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the evaluation pathway for waveforms."""

from dataclasses import dataclass

import numpy as np
import pytest

from qat.experimental.waveforms.evaluate import evaluate_waveform
from qat.experimental.waveforms.shapes.base import WaveformShape
from qat.experimental.waveforms.shapes.exceptions import (
    DerivativeOrderNotImplementedError,
    DerivativeOrderUndefinedError,
)
from qat.experimental.waveforms.shapes.gaussian import GaussianWaveformShape
from qat.experimental.waveforms.shapes.soft_square import SoftSquareWaveformShape
from qat.ir.waveforms import (
    ExtraSoftSquareWaveform,
    GaussianWaveform,
    GaussianZeroEdgeWaveform,
    sample_waveform,
)


@dataclass
class _PowerFunctionWaveform(WaveformShape):
    """A simple waveform that is a power function of time."""

    power: int

    def evaluate(self, times: np.ndarray) -> np.ndarray:
        """Evaluates the waveform at the given times."""
        return (times**self.power).astype(np.complex128)

    def derivative(self, times: np.ndarray, order: int) -> np.ndarray:
        """Evaluates the derivative of the waveform at the given times."""
        if order > self.power:
            return np.zeros_like(times).astype(np.complex128)
        coeff = 1
        for i in range(order):
            coeff *= self.power - i
        return (coeff * times ** (self.power - order)).astype(np.complex128)


@dataclass
class _PowerFunctionWithoutDerivative(_PowerFunctionWaveform):
    """A simple waveform that is a power function of time, but does not implement the
    derivative method."""

    order_for_error: int

    def derivative(self, times: np.ndarray, order: int) -> np.ndarray:
        """Raises a DerivativeOrderNotImplementedError."""
        if order == self.order_for_error:
            raise DerivativeOrderNotImplementedError("power", order)
        return super().derivative(times, order)


@dataclass
class _WaveformWithUndefinedDerivative(WaveformShape):
    """A waveform that has an undefined derivative at a given order."""

    undefined_order: int

    def evaluate(self, times: np.ndarray) -> np.ndarray:
        """Evaluates the waveform at the given times."""
        return np.ones_like(times).astype(np.complex128)

    def derivative(self, times: np.ndarray, order: int) -> np.ndarray:
        """Raises a ValueError if the derivative order is undefined."""
        if order == self.undefined_order:
            raise DerivativeOrderUndefinedError(self.__class__.__name__, order)
        return np.zeros_like(times).astype(np.complex128)


class TestEvaluateWaveformAnalyticalPathway:
    """Tests the evaluation pathway for waveforms."""

    @pytest.mark.parametrize(
        "amplitude, phase, time, sample_time",
        [(1.0, 0.0, 100, 10), (0.5, np.pi / 4, 50, 2)],
    )
    def test_prefactor_is_calculated_correctly(self, amplitude, phase, time, sample_time):
        """Checks that the prefactor is calculated correctly."""

        power_function = _PowerFunctionWaveform(power=0)
        evaluated = evaluate_waveform(
            width=time,
            sample_time=sample_time,
            shape=power_function,
            amplitude=amplitude,
            phase=phase,
        )
        assert np.allclose(evaluated, amplitude * np.exp(1j * phase))
        assert len(evaluated) == time // sample_time

    @pytest.mark.parametrize("width, sample_time", [(100, 1), (50, 2)])
    def test_times_are_sampled_at_center_of_bins(self, width, sample_time):
        """It's expected that the times are sampled at the center of the bins.

        We can test this by making the function just return the x values and checking those.
        """

        num_samples = width // sample_time
        spacing = 2 * sample_time / width

        expected_times = np.linspace(-1 + spacing / 2, 1 - spacing / 2, num_samples)

        linear = _PowerFunctionWaveform(power=1)
        evaluated = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=linear,
        )
        assert np.allclose(evaluated, expected_times)

    def test_first_order_drag_is_calculated_correctly(self):
        """Tests that the first order DRAG term is calculated correctly for a linear
        waveform shape."""

        width = 80000
        sample_time = 1000
        drag = 1e-8

        linear = _PowerFunctionWaveform(power=1)
        evaluated = evaluate_waveform(
            width=width, sample_time=sample_time, shape=linear, drag_coefficients=drag
        )
        assert not np.any(np.isclose(evaluated.imag, 0)), (
            "DRAG should add imaginary components."
        )
        assert np.allclose(evaluated.imag[0], evaluated.imag), (
            "The imaginary components should be constant for a linear waveform."
        )
        evaluated_without_drag = evaluate_waveform(
            width=width, sample_time=sample_time, shape=linear, drag_coefficients=0.0
        )
        assert np.allclose(evaluated_without_drag, evaluated_without_drag.real), (
            "Real components shouldn't be affected by DRAG."
        )

        # Check the chain rule is implemented correctly
        scale = 2 / (width * 1e-12)
        assert np.isclose(evaluated.imag[0], drag * scale)

        # Check it scales with amplitude and phase
        with_amplitude_and_phase = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=linear,
            amplitude=2.0,
            phase=np.pi / 4,
            drag_coefficients=drag,
        )
        assert np.allclose(
            with_amplitude_and_phase, evaluated * 2.0 * np.exp(1j * np.pi / 4)
        ), "DRAG should scale with amplitude and phase."

    def test_second_order_drag_is_calculated_correctly(self):
        """Tests that the second order DRAG term is calculated correctly for a quadratic
        waveform shape."""

        width = 80000
        sample_time = 1000
        drag_first = 1e-8
        drag_second = 1e-16

        quadratic = _PowerFunctionWaveform(power=2)
        without_drag = evaluate_waveform(
            width=width, sample_time=sample_time, shape=quadratic, drag_coefficients=0.0
        )
        with_first_order_drag = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=quadratic,
            drag_coefficients=drag_first,
        )
        with_second_order_drag = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=quadratic,
            drag_coefficients=[drag_first, drag_second],
        )

        assert np.allclose(without_drag, with_first_order_drag.real), (
            "Real components shouldn't be affected by DRAG."
        )
        assert np.allclose(with_first_order_drag.imag, with_second_order_drag.imag), (
            "The second order DRAG term should add additional imaginary components."
        )
        assert not np.any(np.isclose(with_second_order_drag.real, without_drag.real)), (
            "The second order DRAG term should affect the real components."
        )

        diff = with_second_order_drag.real - with_first_order_drag.real
        assert np.allclose(diff[0], diff), (
            "Addition from second order DRAG should be constant for a quadratic waveform."
        )

        evaluated = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=quadratic,
            drag_coefficients=[drag_first, drag_second],
        )
        assert not np.any(np.isclose(evaluated.imag, 0)), (
            "DRAG should add imaginary components."
        )
        assert not np.allclose(evaluated.imag[0], evaluated.imag), (
            "The imaginary components should not be constant for a quadratic waveform."
        )

    def test_numpy_array_drag_coefficients_matches_list(self):
        """Tests that passing drag_coefficients as a numpy array gives the same result as
        passing an equivalent Python list, verifying that the truthiness-based check is not
        used (a zero-valued numpy scalar or array would raise with ``or []``)."""

        width = 80000
        sample_time = 1000
        drag_first = 1e-8
        drag_second = 1e-16

        quadratic = _PowerFunctionWaveform(power=2)
        with_list = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=quadratic,
            drag_coefficients=[drag_first, drag_second],
        )
        with_array = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=quadratic,
            drag_coefficients=np.array([drag_first, drag_second]),
        )
        assert np.allclose(with_list, with_array), (
            "numpy array drag_coefficients should produce the same result as a list."
        )


class TestEvaluateWaveformNumericalPathway:
    """Tests the evaluation pathway for waveforms when numerical derivatives are used."""

    def test_numerical_derivative_matches_analytical_derivative(self):
        """Tests that the numerical derivative matches the analytical derivative for a
        waveform shape that implements the derivative method."""

        width = 80000
        sample_time = 1000
        drag_first = 1e-8

        evaluated_analytical = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=_PowerFunctionWaveform(power=2),
            drag_coefficients=drag_first,
        )
        evaluated_numerical = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=_PowerFunctionWithoutDerivative(power=2, order_for_error=1),
            drag_coefficients=drag_first,
        )

        ratio_error = np.abs(np.abs(evaluated_analytical / evaluated_numerical) - 1)

        # Outliers at edges can have larger errors
        assert np.all(ratio_error < 1e-2), (
            "Numerical derivative should match analytical derivative within a tolerance."
        )
        assert np.any(ratio_error > 1e-6), (
            "Numerical derivative should not match analytical derivative too closely, as "
            "this would indicate that the numerical derivative is not being used."
        )
        assert np.mean(ratio_error) < 1e-4, (
            "Numerical derivative should match analytical derivative within a tolerance."
        )

    def test_drag_with_analytical_derivative_up_to_order_and_then_numerical(self):
        """Tests that the evaluation pathway can use analytical derivatives up to a given
        order, and then use numerical derivatives for higher orders."""

        width = 80000
        sample_time = 1000
        drag_first = 1e-8
        drag_second = 1e-16

        evaluated_analytical = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=_PowerFunctionWaveform(power=2),
            drag_coefficients=[drag_first, drag_second],
        )
        evaluated_numerical = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=_PowerFunctionWithoutDerivative(power=2, order_for_error=2),
            drag_coefficients=[drag_first, drag_second],
        )

        # First derivative is exact (analytical), second derivative is numerical but done on
        # the linear first order derivative, so should be exact as well. So the two should
        # match.
        assert np.allclose(evaluated_analytical, evaluated_numerical), (
            "Numerical derivative should match analytical derivative."
        )

    def test_subsequent_orders_with_numerical_derivative(self):
        """Tests that the numerical derivative can be used for many orders of DRAG."""

        width = 100
        sample_time = 1
        drag_first = 1e-8
        drag_second = 1e-16

        evaluated_analytical = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=_PowerFunctionWaveform(power=2),
            drag_coefficients=[drag_first, drag_second],
        )
        evaluated_numerical = evaluate_waveform(
            width=width,
            sample_time=sample_time,
            shape=_PowerFunctionWithoutDerivative(power=2, order_for_error=2),
            drag_coefficients=[drag_first, drag_second],
        )

        assert np.allclose(evaluated_analytical, evaluated_numerical), (
            "Numerical derivative should match analytical derivative."
        )


class TestEvaluateWaveformExceptions:
    """Tests that the evaluation pathway raises exceptions in the expected manner."""

    def test_width_not_integer_multiple_of_sample_time_raises_value_error(self):
        """Tests that a ValueError is raised when the width is not an integer multiple of
        the sample time."""

        width = 100
        sample_time = 3
        linear = _PowerFunctionWaveform(power=1)
        with pytest.raises(
            ValueError, match="Width 100 is not an integer multiple of sample time 3."
        ):
            evaluate_waveform(width=width, sample_time=sample_time, shape=linear)

    def test_allow_numerical_derivative_false_raises_not_implemented_error(self):
        """Tests that setting allow_numerical_derivative=False raises
        DerivativeOrderNotImplementedError when the derivative is not analytically
        implemented, rather than falling back to numerical differentiation."""

        width = 80000
        sample_time = 1000
        shape = _PowerFunctionWithoutDerivative(power=2, order_for_error=1)

        with pytest.raises(DerivativeOrderNotImplementedError):
            evaluate_waveform(
                width=width,
                sample_time=sample_time,
                shape=shape,
                drag_coefficients=1e-8,
                allow_numerical_derivative=False,
            )

    def test_undefined_derivative_order_raises_derivative_order_undefined_error(self):
        """Tests that DerivativeOrderUndefinedError propagates when the derivative order is
        mathematically undefined for the waveform shape."""

        width = 80000
        sample_time = 1000
        undefined_2 = _WaveformWithUndefinedDerivative(undefined_order=2)

        # Shouldn't raise — order 1 is defined
        evaluate_waveform(
            width=width, sample_time=sample_time, shape=undefined_2, drag_coefficients=1e-8
        )

        # Should raise — order 2 is undefined
        with pytest.raises(DerivativeOrderUndefinedError):
            evaluate_waveform(
                width=width,
                sample_time=sample_time,
                shape=undefined_2,
                drag_coefficients=[1e-8, 1e-16],
            )

        # Should also raise — order 2 is still hit
        with pytest.raises(DerivativeOrderUndefinedError):
            evaluate_waveform(
                width=width,
                sample_time=sample_time,
                shape=undefined_2,
                drag_coefficients=[1e-8, 1e-16, 1e-24],
            )


class TestParityWithLegacy:
    """Tests parity against the legacy evaluation pathway."""

    @pytest.mark.parametrize(
        "drag_ratio, error_tolerance", [(0.0, 1e-4), (0.01, 1e-3), (0.1, 1e-2)]
    )
    def test_with_extra_soft_square_waveform(self, drag_ratio, error_tolerance):
        """Tests that the new evaluation pathway matches the legacy evaluation pathway for
        the ExtraSoftSquareWaveform.

        The waveforms wont match up exactly due to the following reasons:

        * The legacy implementation has a non-closed form representation to make it
          "zero at the edges" (not treating the edges as 0 and width).
        * The legacy implementation has a weak numerical implementation of derivatives
          which implements a first order finite differences, but actually calculates it
          at t + delta_t / 2 instead of t.

        The error will be more substantial with larger values of DRAG, so we increase the
        tolerance.
        """

        width_ps = 160000
        width = width_ps * 1e-12
        std_dev = 80e-9
        rise = 10e-9
        sample_time_ps = 1000
        sample_time = sample_time_ps * 1e-12
        drag = drag_ratio * width / 2
        amp = 0.454
        phase = np.pi / 4

        legacy_waveform = ExtraSoftSquareWaveform(
            width=width, std_dev=std_dev, rise=rise, drag=drag, amp=amp, phase=phase
        )
        legacy_evaluated = sample_waveform(
            waveform=legacy_waveform, sample_time=sample_time
        ).samples

        new_waveform = SoftSquareWaveformShape.from_extra_soft_square_waveform(
            std_dev, rise, width
        )
        new_evaluated = evaluate_waveform(
            width=width_ps,
            sample_time=sample_time_ps,
            shape=new_waveform,
            amplitude=amp,
            phase=phase,
            drag_coefficients=drag,
        )

        diff = np.abs(legacy_evaluated - new_evaluated)
        assert np.all(diff < error_tolerance), (
            "New evaluation pathway should match legacy evaluation pathway within a "
            "tolerance."
        )

    @pytest.mark.parametrize("drag_ratio", [0.0, 0.01, 0.1])
    def test_with_gaussian_waveform(self, drag_ratio):
        """Tests that the new evaluation pathway matches the legacy evaluation pathway for
        the GaussianWaveform.

        The legacy has the first derivative implemented analytically, so we can test that
        exactly.
        """

        width_ps = 160000
        width = width_ps * 1e-12
        rise = 1 / 3
        sample_time_ps = 1000
        sample_time = sample_time_ps * 1e-12
        drag = drag_ratio * width / 2
        amp = 0.454
        phase = np.pi / 4

        legacy_waveform = GaussianWaveform(
            width=width, rise=rise, drag=drag, amp=amp, phase=phase
        )
        legacy_evaluated = sample_waveform(
            waveform=legacy_waveform, sample_time=sample_time
        ).samples

        new_waveform = GaussianWaveformShape.from_gaussian_waveform(rise)
        new_evaluated = evaluate_waveform(
            width=width_ps,
            sample_time=sample_time_ps,
            shape=new_waveform,
            amplitude=amp,
            phase=phase,
            drag_coefficients=drag,
        )

        assert np.allclose(legacy_evaluated, new_evaluated), (
            "New evaluation pathway should match legacy evaluation pathway."
        )

    def test_with_gaussian_zero_edge_waveform(self):
        """Tests that the new evaluation pathway matches the legacy evaluation pathway for
        the GaussianZeroEdgeWaveform.

        Derivatives are not implemented in the legacy pathway, so we can only test that the
        evaluation matches for the case of no DRAG.
        """

        width_ps = 160000
        width = width_ps * 1e-12
        std_dev = 80e-9
        sample_time_ps = 1000
        sample_time = sample_time_ps * 1e-12
        amp = 0.454
        phase = np.pi / 4

        legacy_waveform = GaussianZeroEdgeWaveform(
            width=width, std_dev=std_dev, amp=amp, phase=phase, zero_at_edges=True
        )
        legacy_evaluated = sample_waveform(
            waveform=legacy_waveform, sample_time=sample_time
        ).samples

        new_waveform = GaussianWaveformShape.from_gaussian_zero_edge_waveform(
            std_dev, width, True
        )
        new_evaluated = evaluate_waveform(
            width=width_ps,
            sample_time=sample_time_ps,
            shape=new_waveform,
            amplitude=amp,
            phase=phase,
        )

        assert np.allclose(legacy_evaluated, new_evaluated), (
            "New evaluation pathway should match legacy evaluation pathway."
        )
