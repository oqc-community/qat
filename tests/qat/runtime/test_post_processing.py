# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.ir.measure import AcquireMode, PostProcessing, PostProcessType, ProcessAxis
from qat.runtime.post_processing import (
    UPCONVERT_SIGN,
    apply_post_processing,
    discriminate,
    down_convert,
    get_axis_map,
    linear_map_complex_to_real,
    mean,
)


class TestApplyPostProcessing:

    def test_invalid_pp_type_raises_not_implemented_error(self):
        pp = PostProcessing(output_variable="test", process=PostProcessType.MUL)
        with pytest.raises(NotImplementedError):
            apply_post_processing({}, pp, {})

    @pytest.mark.parametrize(
        "axes",
        [
            ProcessAxis.SEQUENCE,
            [ProcessAxis.SEQUENCE, ProcessAxis.TIME],
            [ProcessAxis.TIME, ProcessAxis.SEQUENCE],
        ],
    )
    def test_down_convert_on_wrong_axes_raises_value_error(self, axes):
        pp = PostProcessing(
            output_variable="test", process=PostProcessType.DOWN_CONVERT, axes=axes
        )
        with pytest.raises(ValueError):
            apply_post_processing({}, pp, {})


@pytest.mark.parametrize("dims", [0, 1, 2])
@pytest.mark.parametrize("mode", [AcquireMode.RAW, AcquireMode.SCOPE])
def test_down_convert(dims, mode):
    # create up-scaled time-series data
    samples = 52
    dt = 8e-8
    freq = 2.54e9
    ts = np.linspace(0, dt * (samples - 1), samples)
    response = np.exp(UPCONVERT_SIGN * 2.0j * np.pi * freq * ts)

    # assemble the mock data
    dims = [10 * (i + 1) for i in range(dims)]  # sweeping data
    if mode == AcquireMode.RAW:
        dims.append(254)  # shots data
    dims.append(52)  # time-series data
    response = np.reshape(np.tile(response, (np.prod(dims[0:-1]))), dims)
    assert list(np.shape(response)) == dims

    # apply down conversion
    axis_map = get_axis_map(mode, response)
    response, axes = down_convert(response, axis_map, freq, dt)
    assert list(np.shape(response)) == dims
    assert np.allclose(response, 1.0)
    assert axes == axis_map


class TestMean:
    @pytest.mark.parametrize("dims", [0, 1, 2])
    @pytest.mark.parametrize(
        "axes",
        [
            [ProcessAxis.TIME],
            [ProcessAxis.SEQUENCE],
            [ProcessAxis.TIME, ProcessAxis.SEQUENCE],
            [ProcessAxis.SEQUENCE, ProcessAxis.TIME],
        ],
    )
    def test_mean_with_RAW_acquisition(self, dims, axes):
        # create some mock data
        dims = [10 * (i + 1) for i in range(dims)]  # sweeping data
        dims.append(254)  # shots data
        dims.append(52)  # time-series data
        response = np.reshape(np.tile(1.0, (np.prod(dims))), dims)

        axis_map = get_axis_map(AcquireMode.RAW, response)
        response, new_axis_map = mean(response, axis_map, axes)

        if ProcessAxis.SEQUENCE in axes:
            dims.remove(dims[-2])
            assert ProcessAxis.SEQUENCE not in new_axis_map
        if ProcessAxis.TIME in axes:
            dims.remove(dims[-1])
            assert ProcessAxis.TIME not in new_axis_map
        assert list(np.shape(response)) == dims
        assert np.allclose(response, 1.0)

    @pytest.mark.parametrize("dims", [0, 1, 2])
    def test_mean_with_SCOPE_acquisition(self, dims):
        # create some mock data
        dims = [10 * (i + 1) for i in range(dims)]  # sweeping data
        dims.append(52)  # time-series data
        response = np.reshape(np.tile(1.0, (np.prod(dims))), dims)

        axis_map = get_axis_map(AcquireMode.SCOPE, response)
        response, new_axis_map = mean(response, axis_map, ProcessAxis.TIME)

        assert list(np.shape(response)) == dims[:-1]
        assert np.allclose(response, 1.0)
        assert ProcessAxis.TIME not in new_axis_map

    @pytest.mark.parametrize("dims", [0, 1, 2])
    def test_mean_with_INTEGRATOR_acquisition(self, dims):
        # create some mock data
        dims = [10 * (i + 1) for i in range(dims)]  # sweeping data
        dims.append(254)  # shot data
        response = np.reshape(np.tile(1.0, (np.prod(dims))), dims)

        axis_map = get_axis_map(AcquireMode.INTEGRATOR, response)
        response, new_axis_map = mean(response, axis_map, ProcessAxis.SEQUENCE)

        assert list(np.shape(response)) == dims[:-1]
        assert np.allclose(response, 1.0)
        assert ProcessAxis.SEQUENCE not in new_axis_map


@pytest.mark.parametrize("dims", [0, 1, 2])
def test_linear_map_complex_to_real(dims):
    # create some mock data
    dims = [10 * (i + 1) for i in range(dims)]  # sweeping data
    dims.append(254)  # shot data
    response = np.reshape(np.tile(1.0, (np.prod(dims))), dims)

    axis_map = get_axis_map(AcquireMode.INTEGRATOR, response)
    response, new_axis_map = linear_map_complex_to_real(response, axis_map, 5, 1)
    assert new_axis_map == axis_map
    assert np.allclose(response, 6.0)


@pytest.mark.parametrize("dims", [0, 1, 2])
def test_discriminate(dims):
    dims = [10 * (i + 1) for i in range(dims)]  # sweeping data
    dims.append(254)  # shot data
    bits = np.random.rand(*dims) > 0.5
    response = 2.0 * bits

    axis_map = get_axis_map(AcquireMode.INTEGRATOR, response)
    response, new_axis_map = discriminate(response, axis_map, 1.0)
    assert new_axis_map == axis_map
    assert np.allclose(bits, (response == np.ones(dims)))
