# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from numbers import Number

import numpy as np

from qat.ir.measure import PostProcessing
from qat.purr.compiler.instructions import AcquireMode, PostProcessType, ProcessAxis

# TODO: move to QAT config?
UPCONVERT_SIGN = 1.0


def get_axis_map(mode: AcquireMode, response: np.ndarray) -> dict[ProcessAxis, int]:
    """
    Given the acquisition mode, determine what each axis corresponds to.

    The way the results are returned are defined by the acquisition mode: this could be
    averaged over shots, averaged over time, or neither. We must determine how to unpack
    the results.

    :param mode: The acquisition mode.
    :param response: The response returned by the target machine.
    :returns: A dictionary containing the axis index for each type of :class:`ProcessAxis`.
    """

    match mode:
        case AcquireMode.SCOPE:
            return {ProcessAxis.TIME: -1}
        case AcquireMode.INTEGRATOR:
            return {ProcessAxis.SEQUENCE: -1}
        case AcquireMode.RAW:
            if response.ndim > 1:
                return {ProcessAxis.SEQUENCE: -2, ProcessAxis.TIME: -1}
            else:
                return {ProcessAxis.TIME: -1}
    raise NotImplementedError(f"The acquisition mode {mode} is not supported.")


def apply_post_processing(
    response: np.ndarray, post_processing: PostProcessing, axes: dict[ProcessAxis, int]
) -> tuple[np.ndarray, dict[ProcessAxis, int]]:
    """
    Applies software post processing to the results.

    Uses the information in the :class:`PostProcessing` instruction to determine what method
    to  apply.

    :param response: Readout results from an execution engine.
    :param post_processing: The post processing instruction.
    :param axes: A dictionary containing which axes contain the shots and which contain time
        series.
    :returns: The processed results as an array and the axis map.
    """

    match post_processing.process_type:
        case PostProcessType.MEAN:
            return mean(response, axes, post_processing.axes)
        case PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL:
            return linear_map_complex_to_real(response, axes, *post_processing.args)
        case PostProcessType.DISCRIMINATE:
            return discriminate(response, axes, post_processing.args[0])

    raise NotImplementedError(
        f"Post processing type {post_processing.process_type} not implemented."
    )


def mean(
    response: np.ndarray,
    axes: dict[ProcessAxis, int],
    target_axes: ProcessAxis | list[ProcessAxis],
):
    """
    Calculates the mean over the given axes.

    :param response: Readout results from an execution engine.
    :param axes: A dictionary containing which axes contain the shots and which contain time
        series.
    :param target_axes: Which axis or axes should the mean be done over?
    :returns: The processed results as an array and the axis map.
    """

    target_axes = target_axes if isinstance(target_axes, list) else [target_axes]
    axis_indices = tuple(axes[axis] for axis in axes if axis in target_axes)
    final_axes = _remove_axes(response.ndim, axis_indices, axes)
    final_data = np.mean(response, axis=axis_indices)
    return final_data, final_axes


def linear_map_complex_to_real(
    response: list[np.ndarray],
    axes: dict[ProcessAxis, int],
    multiplier: Number,
    constant: Number,
):
    """
    Maps complex values onto a real z-projection using a provided linear mapping.

    :param np.ndarray response: Readout results from an execution engine.
    :param axes: A dictionary containing which axes contain the shots and which contain time
        series.
    :type axes: dict[ProcessAxis, Int]
    :param numbers.Number multiplier: Coeffecient for the linear map.
    :param numbers.Number constant: Constant for the linear map.
    :returns: The processed results as an array and the axis map.
    """

    return np.real(multiplier * response + constant), axes


def discriminate(response: np.ndarray, axes: dict[ProcessAxis, int], threshold: float):
    """
    Discriminates a real value to a classical bit by comparison to a supplied
    discrimination threshold.

    :param np.ndarray response: Readout results from an execution engine.
    :param axes: A dictionary containing which axes contain the shots and which contain time
        series.
    :type axes: dict[ProcessAxis, Int]
    :param float threshold: The supplied discrimination threshold.
    :returns: The processed results as an array and the axis map.
    """
    return 2 * (response > threshold) - 1, axes


def _remove_axes(original_dims, removed_axis_indices, axis_locations):
    """
    Extracted from `purr/backends/utilities.py`.

    Returns the new axis map after axes have been removed, e.g., from calculating a mean.
    """

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
