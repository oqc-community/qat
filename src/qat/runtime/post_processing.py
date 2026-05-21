# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
"""Runtime post-processing utilities.

This module provides helpers to map raw IQ readout arrays into processed
forms expected by higher-level code.  It implements the supported legacy
``PostProcessType`` operations (mean, linear map, discriminate), plus the
granular post-processing pipeline functions (:func:`apply_equalise`,
:func:`apply_discriminate_instruction`, :func:`apply_post_select`,
:func:`apply_demap_instruction`) and small helpers for axis bookkeeping.

For a full description of the granular pipeline and how each step fits together,
see the :ref:`post_processing_pipeline` guide.

The legacy ``LINEAR_MAP_COMPLEX_TO_REAL`` path is preserved unchanged for
backward compatibility.
"""

from numbers import Number

import numpy as np

from qat.ir.instruction_basetypes import AcquireMode, PostProcessType, ProcessAxis
from qat.ir.measure import Demap, Discriminate, Equalise, PostProcessing, PostSelect
from qat.model.post_processing import BG_LABEL, MaxLikelihoodMethod

# TODO: move to QAT config?
UPCONVERT_SIGN = 1.0


def get_axis_map(mode: AcquireMode, response: np.ndarray) -> dict[ProcessAxis, int]:
    """Given the acquisition mode, determine what each axis corresponds to.

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
    """Applies software post processing to the results.

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
        case _:
            raise NotImplementedError(
                f"Post processing type {post_processing.process_type} is not implemented."
            )


def mean(
    response: np.ndarray,
    axes: dict[ProcessAxis, int],
    target_axes: ProcessAxis | list[ProcessAxis],
):
    """Calculates the mean over the given axes.

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
    """Maps complex values onto a real z-projection using a provided linear mapping.

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
    """Discriminates a real value to a classical bit by comparison to a supplied
    discrimination threshold.

    :param np.ndarray response: Readout results from an execution engine.
    :param axes: A dictionary containing which axes contain the shots and which contain time
        series.
    :type axes: dict[ProcessAxis, Int]
    :param float threshold: The supplied discrimination threshold.
    :returns: The processed results as an array and the axis map.
    """
    return 2 * (response > threshold) - 1, axes


def apply_equalise(
    response: np.ndarray,
    instr: Equalise,
    axes: dict[ProcessAxis, int],
) -> tuple[np.ndarray, dict[ProcessAxis, int]]:
    """Runtime implementation of :class:`qat.ir.measure.Equalise`.

    Applies the affine ``[I', Q'] = A @ [I, Q] + b`` transform to the
    complex readout array and returns the result as a complex array
    ``I' + j Q'``.

    :param response: Complex IQ readout array (arbitrary batch shape).
    :param instr: The :class:`Equalise` instruction carrying ``transform`` and
        ``offset``.
    :param axes: Current axis map.
    :returns: Complex-valued transformed data and unchanged axis map.
    :raises ValueError: If ``instr.transform`` is not shape ``(2, 2)`` or
        ``instr.offset`` is not shape ``(2,)``.
    """
    A = np.asarray(instr.transform, dtype=float)
    b_iq = np.asarray(instr.offset, dtype=float)

    if A.shape != (2, 2):
        raise ValueError(
            f"Equalise transform must be shape (2, 2) for single-channel operation; "
            f"got {A.shape}."
        )
    if b_iq.shape != (2,):
        raise ValueError(
            f"Equalise offset must be shape (2,) for single-channel operation; "
            f"got {b_iq.shape}."
        )

    # Represent each complex sample as a real [I, Q] vector, apply the affine
    # transform, then repack into a complex scalar.
    iq = np.stack([response.real, response.imag], axis=-1)  # (..., 2)
    iq_out = iq @ A.T + b_iq  # (..., 2)
    return iq_out[..., 0] + 1j * iq_out[..., 1], axes


def apply_discriminate_instruction(
    response: np.ndarray,
    instr: Discriminate,
    axes: dict[ProcessAxis, int],
) -> tuple[np.ndarray, dict[ProcessAxis, int]]:
    """Runtime implementation of :class:`qat.ir.measure.Discriminate`.

    Discriminate equalised values to string state labels using a
    :class:`Discriminate` instruction.

    For the threshold path (``instr.threshold is not None``) values above the
    threshold map to ``"0"`` and values at or below map to ``"1"``.

    For the ML path (``instr.method`` is set) normalised Gaussian likelihoods
    are computed using a single global ``noise_est``; the state with the highest
    likelihood wins and its ``label`` string is returned.  Likelihoods are
    computed in log-domain with log-sum-exp stabilisation to avoid underflow.
    When ``p_min > 0``, shots below the minimum confidence threshold are
    labelled :data:`~qat.model.post_processing.BG_LABEL`.

    :param response: Equalised (real or complex) readout array.
    :param instr: The :class:`Discriminate` instruction.
    :param axes: Current axis map.
    :returns: String state-label array and unchanged axis map.
    """
    if instr.threshold is not None:
        state_labels = np.where(np.real(response) > instr.threshold, "0", "1")
        return state_labels, axes

    # ML path — compute normalised likelihoods via log-sum-exp stabilisation.
    method: MaxLikelihoodMethod = instr.method  # type: ignore[assignment]
    if len(method.states) == 0:
        raise ValueError("MaxLikelihoodMethod must define at least one state.")

    locations = np.array([s.location for s in method.states], dtype=np.complex128)
    orig_shape = response.shape
    flat = response.flatten()  # shape (N,)

    # log_p[n, k] = -|z_n - loc_k|^2 / (2 * noise_est)  where noise_est is the noise power (variance σ²)
    sq_dist = np.abs(flat[:, np.newaxis] - locations[np.newaxis, :]) ** 2  # (N, K)
    log_p = -sq_dist / (2.0 * method.noise_est)  # (N, K)

    best_idx = np.argmax(log_p, axis=1)  # (N,)
    state_label_arr = np.array([s.label for s in method.states])
    classified = state_label_arr[best_idx]

    # p_min check: compute normalised likelihoods and reject low-confidence shots.
    if method.p_min > 0.0:
        # Log-sum-exp stabilisation: subtract row-wise max before exp to prevent
        # underflow when all log-likelihoods are very negative (extreme outliers).
        log_p_stable = log_p - log_p.max(axis=1, keepdims=True)  # (N, K)
        likelihoods = np.exp(log_p_stable)  # (N, K)
        norm_likelihoods = likelihoods / likelihoods.sum(axis=1, keepdims=True)  # (N, K)
        best_p = norm_likelihoods[np.arange(len(flat)), best_idx]  # (N,)
        classified = np.where(best_p >= method.p_min, classified, BG_LABEL)

    return classified.reshape(orig_shape), axes


def apply_post_select(
    state_ids: np.ndarray,
    instr: PostSelect,
    axes: dict[ProcessAxis, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Runtime implementation of :class:`qat.ir.measure.PostSelect`.

    Build a per-shot validity mask from a :class:`PostSelect` instruction.

    Shots whose state label appears in ``instr.disallowed_states`` are marked
    ``False`` in the returned validity mask. When ``disallowed_states`` is empty
    every shot is valid.

    :param state_ids: String state-label array produced by a :class:`Discriminate` step.
    :param instr: The :class:`PostSelect` instruction.
    :param axes: Current axis map (not used; present for interface consistency).
    :returns: A tuple of ``(state_ids, validity_mask)`` where ``validity_mask``
        is a boolean ndarray with the same shape as ``state_ids``.
    """
    disallowed = list(instr.disallowed_states)
    validity_mask = ~np.isin(state_ids, disallowed)
    return state_ids, validity_mask


def apply_demap_instruction(
    state_ids: np.ndarray,
    instr: Demap,
    axes: dict[ProcessAxis, int],
) -> tuple[np.ndarray, dict[ProcessAxis, int]]:
    """Runtime implementation of :class:`qat.ir.measure.Demap`.

    Demap string state labels to integer output values using a :class:`Demap` instruction.

    Labels that do not appear in ``instr.state_map`` (including the background label
    :data:`~qat.model.post_processing.BG_LABEL` emitted by a ``p_min`` threshold) are
    mapped to the sentinel value ``-1``.  When a :class:`~qat.ir.measure.PostSelect`
    instruction precedes ``Demap`` in the post-processing chain, the subsequent
    :func:`~qat.runtime.passes.transform._build_and_apply_global_mask` step will remove
    these sentinel shots from the final results.

    :param state_ids: String state-label array produced by a :class:`Discriminate` step.
    :param instr: The :class:`Demap` instruction carrying the ``state_map`` dict.
    :param axes: Current axis map.
    :returns: Demapped int values and unchanged axis map.
    """
    state_ids = np.asarray(state_ids)
    lookup = instr.state_map
    if state_ids.size == 0:
        return np.empty(0, dtype=int), axes
    mapped = np.vectorize(lambda k: lookup.get(k, -1))(state_ids).astype(int)
    return mapped, axes


def _remove_axes(original_dims, removed_axis_indices, axis_locations):
    """Extracted from `purr/backends/utilities.py`.

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
