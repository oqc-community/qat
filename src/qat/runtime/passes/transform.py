# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
"""Runtime transform passes for result formatting and post-processing.

This module contains transform passes that operate on acquired results and the
executable package to shape, post-process and format results returned by the
runtime. Passes are implemented as subclasses of :class:`TransformPass` and are
composed by the runtime pipeline.
"""

from typing import Any

import numpy as np
from compiler_config.config import (
    CompilerConfig,
    ErrorMitigationConfig,
    InlineResultsProcessing,
    ResultsFormatting,
)

from qat.backend.qblox.acquisition import Acquisition
from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.executables import Executable
from qat.ir.instruction_basetypes import AcquireMode, PostProcessType, ProcessAxis
from qat.ir.instructions import Variable
from qat.ir.measure import Demap, Discriminate, Equalise, PostProcessing, PostSelect
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.target_data import TargetData
from qat.purr.compiler.error_mitigation.readout_mitigation import get_readout_mitigation
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.passes.analysis import (
    DiscriminateResult,
    EqualiseResult,
    PostSelectionResult,
)
from qat.runtime.passes.purr.analysis import IndexMappingResult
from qat.runtime.post_processing import (
    apply_demap_instruction,
    apply_discriminate_instruction,
    apply_equalise,
    apply_post_processing,
    apply_post_select,
    get_axis_map,
)
from qat.runtime.results_processing import (
    binary_average,
    binary_count,
    label_count,
    numpy_array_to_list,
)


def _retained_shots(res_mgr: ResultManager | None, package: Executable) -> int:
    """Return the effective shot count for count-based result formatting.

    Uses ``shots_retained`` from :class:`PostSelectionResult` when post-selection has
    filtered shots, otherwise falls back to the total shots in the package.

    :param res_mgr: The results manager to inspect.
    :param package: The executable whose ``shots`` field is the fallback.
    :returns: Effective shot count to use as the denominator.
    """
    if res_mgr is not None and res_mgr.check_for_type(PostSelectionResult):
        return res_mgr.lookup_by_type(PostSelectionResult).shots_retained
    return package.shots or TargetData().default_shots


def _results_format_is_raw(compiler_config: CompilerConfig) -> bool:
    """Return ``True`` when the compiler config requests raw (non-binary-count) output.

    ``raw()`` sets ``InlineResultsProcessing.Raw`` on the format object.  Because
    ``binary_count()`` also sets ``Raw`` internally, we must additionally check that
    ``BinaryCount`` is *not* requested.

    :param compiler_config: The compiler configuration to inspect.
    :returns: ``True`` if the user asked for raw output and not a count.
    """
    fmt = compiler_config.results_format
    if fmt is None:
        return False
    has_raw = (
        isinstance(fmt, InlineResultsProcessing) and InlineResultsProcessing.Raw in fmt
    ) or (
        hasattr(fmt, "format")
        and fmt.format is not None
        and InlineResultsProcessing.Raw in fmt
    )
    has_binary_count = ResultsFormatting.BinaryCount in (
        fmt.transforms if hasattr(fmt, "transforms") else fmt
    )
    return has_raw and not has_binary_count


def _build_and_apply_global_mask(
    results: dict[str, np.ndarray],
    per_output_masks: dict[str, np.ndarray],
    res_mgr: ResultManager | None,
    *,
    sequence_axes: dict[str, int] | None = None,
    intermediates: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """AND per-output validity masks, compress all result arrays, and record metadata.

    Builds a global boolean shot mask by ANDing all entries in ``per_output_masks``,
    applies it to every array in ``results`` via :func:`numpy.compress`, and stores a
    :class:`PostSelectionResult` in ``res_mgr``.

    When ``sequence_axes`` is provided each key maps an output variable to the axis index
    that represents the shot/sequence dimension. If an output variable is absent from
    ``sequence_axes``, or if ``sequence_axes`` is ``None``, axis 0 is used.

    When ``intermediates`` is provided it must be a dict mapping an arbitrary key (e.g.
    ``"equalise"``, ``"discriminate"``) to a ``{var: array}`` dict.  The global mask is
    applied to each intermediate array along axis 0 (shot axis) so that callers can store
    them in ``res_mgr`` with consistent shapes.

    :param results: Mutable dict of output arrays to filter in-place.
    :param per_output_masks: Boolean validity masks, one per post-selected output.
    :param res_mgr: Optional :class:`ResultManager` to receive the
        :class:`PostSelectionResult`.
    :param sequence_axes: Optional mapping from output name to sequence axis index.
    :param intermediates: Optional dict of stage-name → {var: array} to mask in-place.
    :returns: The (mutated) ``results`` dict with filtered arrays.
    """
    if not per_output_masks:
        return results

    masks = list(per_output_masks.values())
    if any(m.ndim != 1 for m in masks):
        raise NotImplementedError(
            "Post-selection is only supported for 1-D shot arrays (no sweep dims)."
        )
    if len({len(m) for m in masks}) > 1:
        raise ValueError(
            "Cannot apply global post-selection: per-output validity masks have "
            "different lengths. All acquires in a single executable must have the "
            "same number of shots."
        )

    global_mask = masks[0].copy()
    for m in masks[1:]:
        global_mask &= m

    shots_requested = int(global_mask.shape[0])
    shots_retained = int(global_mask.sum())

    for name, data in results.items():
        if not isinstance(data, np.ndarray) or data.ndim < 1:
            continue
        if sequence_axes is not None and name not in sequence_axes:
            # Caller has provided an explicit axis map; outputs absent from it carry no
            # sequence dimension and must not be masked.
            continue
        axis = (sequence_axes or {}).get(name, 0) % data.ndim
        if data.shape[axis] != shots_requested:
            raise ValueError(
                f"Cannot apply global post-selection mask: output '{name}' has "
                f"axis-{axis} length {data.shape[axis]}, expected {shots_requested}."
            )
        results[name] = np.compress(global_mask, data, axis=axis)

    # Apply mask to each intermediate dict (shot axis is always 0 for 1-D shot arrays).
    if intermediates:
        for stage_arrays in intermediates.values():
            for name in list(stage_arrays):
                arr = stage_arrays[name]
                if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                    stage_arrays[name] = np.compress(global_mask, arr, axis=0)

    if res_mgr is not None:
        res_mgr.add(
            PostSelectionResult(
                shots_requested=shots_requested,
                shots_retained=shots_retained,
                global_mask=global_mask,
            )
        )

    return results


class AcquisitionPostprocessing(TransformPass):
    """Uses the post-processing instructions from the executable package to process the
    results from the engine.

    The target machine will return the results in a format that depends on the specified
    :class:`AcquireMode`. However, it is often the case results need to be returned in an
    explicit format, e.g., as discriminated bits. To achieve this, extra software
    post-processing is needed.

    The post-processing that appears here is the same as the post-processing
    responsibilities taken on by the :class:`QuantumExecutionEngine` in
    :mod:`qat.purr.compiler.execution`.
    """

    def __init__(self, target_data: TargetData | None = None):
        """
        :param target_data: The target data is needed to know the sample time of the
            acquired results.
        """
        self.sample_time = (
            target_data.RESONATOR_DATA.sample_time if target_data is not None else None
        )

    def run(
        self,
        acquisitions: dict[str, Any],
        res_mgr: ResultManager | None = None,
        *args,
        package: Executable,
        **kwargs,
    ):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param res_mgr: The results manager used to record post-selection metadata and
            granular pipeline intermediate outputs.
        :param package: The executable program containing the results-processing
            information should be passed as a keyword argument.
        """

        # Per-output validity masks produced by PostSelect steps.
        # Each entry is a boolean ndarray of shape (n_shots,).
        per_output_masks: dict[str, np.ndarray] = {}
        # Per-output final axis maps after all post-processing steps.
        final_response_axes: dict[str, dict[ProcessAxis, int]] = {}

        # Intermediate outputs for the granular pipeline stages (captured per-variable).
        equalise_outputs: dict[str, np.ndarray] = {}
        discriminate_outputs: dict[str, np.ndarray] = {}

        for output_variable, acquire in package.acquires.items():
            response = acquisitions[output_variable]

            # Starting from all axes, we iterate through each post-processing, keeping
            # track of what axes remain as we go.
            response_axes = get_axis_map(acquire.mode, response)

            # Determine whether this acquire uses the granular pipeline at all.
            is_granular = any(
                isinstance(pp, Equalise | Discriminate | Demap)
                for pp in acquire.post_processing
            )

            for pp in acquire.post_processing:
                match pp:
                    case Equalise():
                        response, response_axes = apply_equalise(
                            response, pp, response_axes
                        )
                        equalise_outputs[output_variable] = response
                    case Discriminate():
                        # If no Equalise was emitted, stash the pre-discriminate IQ as the
                        # equalise output so raw() has something to return.
                        if output_variable not in equalise_outputs and is_granular:
                            equalise_outputs[output_variable] = response
                        response, response_axes = apply_discriminate_instruction(
                            response, pp, response_axes
                        )
                        discriminate_outputs[output_variable] = response
                    case PostSelect():
                        response, validity_mask = apply_post_select(
                            response, pp, response_axes
                        )
                        per_output_masks[output_variable] = validity_mask
                    case Demap():
                        response, response_axes = apply_demap_instruction(
                            response, pp, response_axes
                        )
                    case _:
                        response, response_axes = apply_post_processing(
                            response, pp, response_axes
                        )
            acquisitions[output_variable] = response
            final_response_axes[output_variable] = response_axes

        # Collect granular intermediates for post-selection masking.
        intermediates: dict[str, dict[str, np.ndarray]] = {}
        if equalise_outputs:
            intermediates["equalise"] = equalise_outputs
        if discriminate_outputs:
            intermediates["discriminate"] = discriminate_outputs

        # Derive per-output sequence axes, then apply the global post-selection mask.
        sequence_axes = {
            var: axes[ProcessAxis.SEQUENCE] % len(acquisitions[var].shape)
            for var, axes in final_response_axes.items()
            if ProcessAxis.SEQUENCE in axes
            and isinstance(acquisitions.get(var), np.ndarray)
            and acquisitions[var].ndim >= 1
        }
        result = _build_and_apply_global_mask(
            acquisitions,
            per_output_masks,
            res_mgr,
            sequence_axes=sequence_axes,
            intermediates=intermediates,
        )

        # Store masked intermediates in res_mgr so ResultTransform can access them.
        if res_mgr is not None:
            if equalise_outputs:
                res_mgr.add(EqualiseResult(outputs=dict(equalise_outputs)))
            if discriminate_outputs:
                res_mgr.add(DiscriminateResult(outputs=dict(discriminate_outputs)))

        return result


class QBloxAcquisitionPostProcessing(TransformPass):
    """Post-process QBlox acquisition playback into output arrays.

    This pass operates on the combined QBlox playback structure, extracts scope or
    integrator data, applies any requested post-processing steps and shapes the results
    according to the acquisition definitions in the executable package.
    """

    def run(
        self,
        playback: dict[str, dict[str, Acquisition]],
        res_mgr: ResultManager | None = None,
        *args,
        package: Executable,
        **kwargs,
    ):
        """Now that the combined playback is ready, we can compute and process results as
        required by customers.

        This requires loop nest information as well as post-processing and array shaping
        requirements.

        :param playback: The combined QBlox playback structure.
        :param res_mgr: The results manager used to record post-selection metadata.
        :param package: The executable program containing the results-processing information
            should be passed as a keyword argument.
        """

        results = {}
        # Per-output validity masks produced by PostSelect steps.
        # Each entry is a boolean ndarray of shape (n_shots,).
        per_output_masks: dict[str, np.ndarray] = {}

        # Intermediate outputs for the granular pipeline stages (captured per-variable).
        equalise_outputs: dict[str, np.ndarray] = {}
        discriminate_outputs: dict[str, np.ndarray] = {}

        for acquisitions in playback.values():
            # TODO - Support multiple acquires target (unicity by name only is not wise)
            acquires = package.acquires
            for name, acquisition in acquisitions.items():
                scope_data = acquisition.acquisition.scope
                integ_data = acquisition.acquisition.bins.integration
                thrld_data = acquisition.acquisition.bins.threshold

                # TODO - COMPILER-860 - Safer and more flexible acquisition addressing
                acquire_data = acquires[name]
                if acquire_data.mode in [AcquireMode.SCOPE, AcquireMode.RAW]:
                    response = scope_data.path0.data + 1j * scope_data.path1.data
                elif acquire_data.mode == AcquireMode.INTEGRATOR:
                    response = integ_data.path0 + 1j * integ_data.path1
                else:
                    raise ValueError(f"Unrecognised acquire mode {acquire_data.mode}")

                post_procs = acquire_data.post_processing
                is_granular = any(
                    isinstance(pp, Equalise | Discriminate | Demap) for pp in post_procs
                )
                # QBlox post-processing operates on a single sweep-point at a time so the
                # axis map starts empty — axes are not tracked across sweep iterations here.
                response_axes: dict[ProcessAxis, int] = {}
                for pp in post_procs:
                    if isinstance(pp, Discriminate):
                        if name not in equalise_outputs and is_granular:
                            equalise_outputs[name] = response
                        # Granular Discriminate: use software helper so that
                        # threshold/method and any preceding Equalise step are respected.
                        response, response_axes = apply_discriminate_instruction(
                            response, pp, response_axes
                        )
                        discriminate_outputs[name] = response
                    elif (
                        isinstance(pp, PostProcessing)
                        and pp.process_type == PostProcessType.DISCRIMINATE
                    ):
                        # Legacy DISCRIMINATE uses the QBlox hardware-thresholded output
                        # rather than the current IQ response.
                        # f : {0, 1} ----> {-1, 1}  via  1 - 2x
                        response = 1 - 2 * thrld_data
                    elif isinstance(pp, Equalise | PostSelect | Demap):
                        # Granular instructions in the QBlox path use the software helpers
                        if isinstance(pp, Equalise):
                            response, response_axes = apply_equalise(
                                response, pp, response_axes
                            )
                            equalise_outputs[name] = response
                        elif isinstance(pp, PostSelect):
                            response, validity_mask = apply_post_select(
                                response, pp, response_axes
                            )
                            per_output_masks[name] = validity_mask
                        elif isinstance(pp, Demap):
                            response, response_axes = apply_demap_instruction(
                                response, pp, response_axes
                            )
                    else:
                        response, response_axes = apply_post_processing(
                            response, pp, response_axes
                        )

                response = response.reshape(acquire_data.shape)

                # TODO - COMPILER-860 - Safer and more flexible acquisition addressing
                if name in results:
                    raise ValueError(f"Key {name} already exists")
                results[name] = response

        # Collect granular intermediates for post-selection masking.
        intermediates: dict[str, dict[str, np.ndarray]] = {}
        if equalise_outputs:
            intermediates["equalise"] = equalise_outputs
        if discriminate_outputs:
            intermediates["discriminate"] = discriminate_outputs

        # Derive per-output sequence axes: only INTEGRATOR acquires carry a shot
        # dimension (axis 0). SCOPE/RAW outputs are time-series without a shot axis
        # and must not be masked.
        sequence_axes = {
            name: 0
            for name in results
            if name in package.acquires
            and package.acquires[name].mode == AcquireMode.INTEGRATOR
        }

        result = _build_and_apply_global_mask(
            results,
            per_output_masks,
            res_mgr,
            sequence_axes=sequence_axes,
            intermediates=intermediates,
        )

        # Store masked intermediates in res_mgr so ResultTransform can access them.
        if res_mgr is not None:
            if equalise_outputs:
                res_mgr.add(EqualiseResult(outputs=dict(equalise_outputs)))
            if discriminate_outputs:
                res_mgr.add(DiscriminateResult(outputs=dict(discriminate_outputs)))

        return result


class InlineResultsProcessingTransform(TransformPass):
    """Uses :class:`InlineResultsProcessing` instructions from the executable package to
    format the acquired results in the desired format.

    **Legacy vs granular acquires**

    All acquires that go through :meth:`~qat.ir.instruction_builder.QuantumInstructionBuilder.measure_with_granular_post_processing`
    now produce granular ``Equalise`` → ``Discriminate`` → ``Demap`` instructions and
    appear in :class:`~qat.runtime.passes.analysis.DiscriminateResult`.  This includes
    qubits with :class:`~qat.model.post_processing.LinearMapToRealMethod`,
    :class:`~qat.model.post_processing.MaxLikelihoodMethod`, and legacy
    ``mean_z_map_args`` qubits.  Pre-selection acquires (``presel_*``) use a shorter
    chain (``Equalise`` → ``Discriminate`` → ``PostSelect``, no ``Demap``) and are
    for internal runtime use only — they drive the validity mask but are never returned
    to the user.

    **Per-variable routing**

    Routing decisions are made per output variable (by looking up the variable in
    ``DiscriminateResult.outputs``) rather than with a single execution-wide flag.
    This correctly handles executables that mix legacy and granular acquires.

    **``numpy_array_to_list``** is skipped for granular variables so ``ResultTransform``
    can access the numpy arrays when routing ``QuantumResultsFormat`` results.

    **``binary_average``** (triggered by ``InlineResultsProcessing.Binary``/``Program``)
    is suppressed for granular variables when a top-level ``QuantumResultsFormat``
    (``binary()`` or ``raw()``) is active — ``ResultTransform`` owns routing in that
    case and needs the raw per-shot int array from ``Demap``.  For all other cases
    (no top-level format, or ``binary_count()``, or legacy acquires) ``binary_average``
    runs as before.
    """

    def run(
        self,
        acquisitions: dict[str, Any],
        res_mgr: ResultManager | None = None,
        *args,
        package: Executable,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param res_mgr: Optional results manager; used to detect granular pipeline usage.
        :param package: The executable program containing the results-processing
            information should be passed as a keyword argument.
        :param compiler_config: Optional compiler configuration; when provided, a
            top-level ``QuantumResultsFormat`` requesting per-shot output (``binary()``
            or ``raw()``) suppresses the per-acquire ``binary_average`` reduction for
            granular acquires so that ``ResultTransform`` can handle routing.
        """
        # Build the set of output variables that actually went through the granular
        # Discriminate step.  Using a per-variable check (rather than a single flag)
        # ensures that legacy Program-mode acquires in the same executable still receive
        # numpy_array_to_list / binary_average even when other acquires used the granular
        # pipeline.
        granular_vars: set[str] = set()
        if res_mgr is not None and res_mgr.check_for_type(DiscriminateResult):
            granular_vars = set(res_mgr.lookup_by_type(DiscriminateResult).outputs.keys())

        # For binary_average suppression: we only skip reduction for ML granular acquires
        # (multi-state, where majority-vote is meaningless).  LinearMapToRealMethod and
        # legacy granular acquires produce binary {0, 1} Demap outputs where
        # binary_average correctly majority-votes to a single 0/1 scalar.
        # Note: pre-selection acquires (presel_*) have results_processing=Raw so
        # binary_average is never triggered for them regardless.
        ml_granular_vars: set[str] = (
            {
                var
                for var in granular_vars
                if any(
                    isinstance(pp, Discriminate) and pp.method is not None
                    for pp in package.acquires[var].post_processing
                )
            }
            if granular_vars
            else set()
        )

        # When a top-level QuantumResultsFormat (binary() / raw()) is active and there
        # are MaxLikelihood granular acquires, ResultTransform handles the per-shot
        # routing.  In that case binary_average must be suppressed for those variables
        # so the per-shot int array from Demap reaches ResultTransform intact.
        # LinearMapToRealMethod / legacy granular acquires produce binary {0, 1} Demap
        # outputs where binary_average correctly majority-votes to a single scalar.
        # BinaryCount is not suppressed — it counts string labels directly.
        fmt = compiler_config.results_format if compiler_config is not None else None
        suppress_binary_avg_for_granular = (
            fmt is not None
            and bool(ml_granular_vars)
            and ResultsFormatting.BinaryCount not in fmt
        )

        for output_variable in acquisitions:
            target_values = acquisitions[output_variable]

            rp = package.acquires[output_variable].results_processing
            if rp is None:
                rp = InlineResultsProcessing.Experiment

            if InlineResultsProcessing.Raw in rp and InlineResultsProcessing.Binary in rp:
                raise ValueError(
                    "Raw and Binary processing attempted to be applied to "
                    f"{output_variable}. Only one should be selected."
                )

            is_granular = output_variable in granular_vars
            is_ml_granular = output_variable in ml_granular_vars

            # Strip numpy arrays if we're set to do so.
            # For granular-pipeline acquires, keep as numpy arrays so ResultTransform
            # can process them correctly.
            if InlineResultsProcessing.NumpyArrays not in rp and not is_granular:
                target_values = numpy_array_to_list(target_values)

            # InlineResultsProcessing.Binary is a per-acquire compilation flag (typically
            # set by the QASM frontend for each measure instruction).  It requests a
            # majority-vote reduction of the per-shot array to a single 0/1 value.
            # Skip it only for MaxLikelihood granular acquires when a top-level
            # QuantumResultsFormat (binary() / raw()) is active — ResultTransform owns
            # routing in those cases and needs the per-shot int array from Demap.
            # LinearMapToRealMethod granular acquires produce {0, 1} where binary_average
            # correctly majority-votes to a single scalar.
            if InlineResultsProcessing.Binary in rp and not (
                is_ml_granular and suppress_binary_avg_for_granular
            ):
                target_values = binary_average(target_values)

            acquisitions[output_variable] = target_values

        return acquisitions


class AssignResultsTransform(TransformPass):
    """Processes :class:`Assign` instructions.

    As assigns are classical instructions they are not processed as a part of the quantum
    execution (right now). Read through the results dictionary and perform the assigns
    directly, return the results.

    Extracted from purr.compiler.execution.QuantumExecutionEngine._process_assigns.
    """

    def run(self, acquisitions: dict[str, Any], *args, package: Executable, **kwargs):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param package: The executable program containing the results-processing
            information should be passed as a keyword argument.
        """
        assigns = dict(acquisitions)
        for assign in package.assigns:
            assigns[assign.name] = self._recurse_arrays(assigns, assign.value)
        return {key: assigns[key] for key in package.returns}

    @staticmethod
    def _recurse_arrays(results_map: dict[str, Any], value: Any) -> Any:
        """Recurse through assignment lists and fetch values in sequence.

        :param results_map: The current results dictionary to index into.
        :param value: The assignment value, which may be a nested list, tuple, string
            key, or :class:`~qat.ir.instructions.Variable`.
        :returns: The resolved value.
        """
        if isinstance(value, list):
            return [
                AssignResultsTransform._recurse_arrays(results_map, val) for val in value
            ]
        elif isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(
                    f"Expected a 2-element (variable, index) tuple; got {len(value)}"
                    f" elements: {value!r}"
                )
            return results_map[value[0]][value[1]]
        elif isinstance(value, str):
            return results_map[value]
        elif isinstance(value, Variable):
            return results_map[value.name]
        else:
            return value


class ResultTransform(TransformPass):
    """Transform the raw results into the format that we've been asked to provide.

    This pass applies ``ResultsFormatting`` flags (``BinaryCount``, ``SquashBinaryResultArrays``,
    ``DynamicStructureReturn``) to shape final results.

    **Granular pipeline routing**

    When the granular post-processing pipeline (``Equalise`` → ``Discriminate`` →
    ``PostSelect`` → ``Demap``) has been used for end-of-circuit measurements,
    intermediate outputs are stored as
    :class:`~qat.runtime.passes.analysis.EqualiseResult` and
    :class:`~qat.runtime.passes.analysis.DiscriminateResult` in ``res_mgr``.
    ``ResultTransform`` reads these to implement the requested format:

    - **``raw()``**: Returns the complex IQ arrays from
      :class:`~qat.runtime.passes.analysis.EqualiseResult` (post-mask). For legacy
      acquires without an ``Equalise`` step, falls back to the mapped float arrays.
    - **``binary()``**: Returns the per-shot int output-value arrays from ``Demap``
      (i.e. ``acquisitions`` as-is — one int per retained shot).
    - **``binary_count()``**: Calls :func:`~qat.runtime.results_processing.label_count`
      on the string labels from
      :class:`~qat.runtime.passes.analysis.DiscriminateResult`, giving a
      ``{label: count}`` dict that correctly handles multi-state classifiers.

    For legacy acquires (no ``DiscriminateResult`` present) all three modes fall back to
    the existing ``binary_count`` / ``binary_average`` logic, with ``shots_retained``
    used as the denominator when post-selection is active.

    Extracted from :meth:`qat.purr.compiler.runtime.QuantumRuntime._transform_results`.
    """

    def run(
        self,
        acquisitions: dict[str, Any],
        res_mgr: ResultManager | None = None,
        *args,
        compiler_config: CompilerConfig,
        package: Executable,
        **kwargs,
    ):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param res_mgr: The results manager is used to look up granular pipeline
            intermediates and post-selection metadata.
        :param compiler_config: The compiler config is needed to know how to process the
            results, and should be provided as a keyword argument.
        """
        format_flags = (
            compiler_config.results_format or ResultsFormatting.DynamicStructureReturn
        )

        if len(acquisitions) == 0:
            return []

        # Build per-variable granularity sets from the result manager.  Using per-variable
        # checks (rather than execution-wide flags) correctly handles executables that mix
        # legacy PostProcessing acquires with granular Equalise/Discriminate acquires.
        disc_labels: dict[str, np.ndarray] = {}
        eq_outputs: dict[str, np.ndarray] = {}
        if res_mgr is not None and res_mgr.check_for_type(DiscriminateResult):
            disc_labels = res_mgr.lookup_by_type(DiscriminateResult).outputs
        if res_mgr is not None and res_mgr.check_for_type(EqualiseResult):
            eq_outputs = res_mgr.lookup_by_type(EqualiseResult).outputs

        if ResultsFormatting.BinaryCount in format_flags:
            repeats = _retained_shots(res_mgr, package)
            if repeats == 0:
                # All shots were filtered by post-selection — return empty counts.
                acquisitions = {key: {} for key in acquisitions}
            elif disc_labels:
                # ML granular acquires: count string state labels directly — works for any
                # number of states without float conversion.
                # LinearMapToRealMethod / legacy granular acquires produce binary
                # {0, 1} int Demap outputs; use binary_count on those ints so that
                # multi-qubit register keys ("00", "01", "10", "11") are assembled
                # correctly by AssignResultsTransform.
                ml_label_vars = {
                    var
                    for var in disc_labels
                    if var not in package.acquires
                    or any(
                        isinstance(pp, Discriminate) and pp.method is not None
                        for pp in package.acquires[var].post_processing
                    )
                }
                # After AssignResultsTransform the output keys may be register names
                # (e.g. "c") rather than per-acquire variable names (e.g. "c[0]_0").
                # Build a mapping from each output key to the set of acquire-variable
                # leaf names that feed it, so that we can check whether any underlying
                # acquire is an ML discriminate.
                assign_value: dict[str, Any] = {a.name: a.value for a in package.assigns}
                # For a key that IS a direct acquire variable the leaf set is {key}.
                # For a register key assembled by AssignResultsTransform it is the
                # acquire variables referenced in the assign expression.
                ml_output_keys: set[str] = set()
                for key in acquisitions:
                    if key in ml_label_vars or (
                        key in assign_value
                        and ResultTransform._assign_has_granular_var(
                            assign_value[key],
                            ml_label_vars,  # type: ignore[arg-type]
                        )
                    ):
                        ml_output_keys.add(key)

                acquisitions = {
                    key: label_count(disc_labels[key])
                    if key in ml_output_keys and key in disc_labels
                    # Register key assembled from ML acquire variables: reconstruct
                    # per-shot bitstrings from disc_labels then count them.
                    else ResultTransform._label_count_register(
                        assign_value[key], disc_labels
                    )
                    if key in ml_output_keys
                    else binary_count(val, repeats)
                    for key, val in acquisitions.items()
                }
            else:
                # Legacy PostProcessing path: convert floats → binary then count.
                acquisitions = {
                    key: binary_count(val, repeats) for key, val in acquisitions.items()
                }

        elif eq_outputs and _results_format_is_raw(compiler_config):
            # raw() with granular pipeline → return equalised complex IQ arrays, but
            # only for variables that actually went through the granular Equalise step.
            # Legacy acquire variables are left unchanged (already list-converted upstream).
            # Convert ndarray IQ outputs to plain lists so they are JSON-serialisable,
            # matching the contract of raw() in the legacy pipeline.
            acquisitions = {
                key: numpy_array_to_list(eq_outputs[key]) if key in eq_outputs else val
                for key, val in acquisitions.items()
            }
            # After AssignResultsTransform, outputs may be register names (e.g. "c")
            # whose assign payload references acquire vars (e.g. "c[0]_0"). Rebuild
            # those register entries from equalise-stage arrays only when at least one
            # of the referenced variables is granular.
            for assign in package.assigns:
                if assign.name in acquisitions and self._assign_has_granular_var(
                    assign.value, eq_outputs
                ):
                    acquisitions[assign.name] = self._resolve_from_equalise(
                        assign.value, eq_outputs
                    )
        # binary() with granular pipeline: acquisitions hold the per-shot int/float
        # output values from Demap — convert to lists for JSON serialisability.
        # Legacy binary() is handled by InlineResultsProcessingTransform upstream.
        elif disc_labels and not _results_format_is_raw(compiler_config):
            # Convert any remaining numpy arrays to plain lists.
            acquisitions = {
                key: numpy_array_to_list(val) if isinstance(val, np.ndarray) else val
                for key, val in acquisitions.items()
            }

        if ResultsFormatting.SquashBinaryResultArrays in format_flags:
            acquisitions = {
                key: self._squash_binary(val) for key, val in acquisitions.items()
            }

        # Dynamic structure return is an ease-of-use flag to strip things that you know
        # your use-case won't use, such as variable names and nested lists.
        if ResultsFormatting.DynamicStructureReturn in format_flags:
            acquisitions = self._simplify_results(acquisitions)
        return acquisitions

    @staticmethod
    def _simplify_results(simplify_target: dict[str, Any]) -> Any:
        """Simplify results by stripping generated variable names.

        To facilitate backwards compatibility and being able to run low-level experiments
        alongside quantum programs we make some assumptions based upon form of the results.

        If all results have default variable names then the user didn't care about value
        assignment or this was a low-level experiment — in both cases, it means we can throw
        away the names and simply return the results in the order they were defined in the
        instructions.

        If we only have one result after this, just return that list directly instead, as
        it's probably just a single experiment.

        :param simplify_target: Demapping of output variable name to result value.
        :returns: Simplified result — either the original dict, a single value, or a list.
        """
        if all(ResultTransform._is_generated_name(k) for k in simplify_target):
            if len(simplify_target) == 1:
                return list(simplify_target.values())[0]
            else:
                squashed_results = list(simplify_target.values())
                if all(isinstance(val, np.ndarray) for val in squashed_results):
                    return np.array(squashed_results)
                return squashed_results
        return simplify_target

    @staticmethod
    def _assign_has_granular_var(value: Any, eq_outputs: dict[str, np.ndarray]) -> bool:
        """Return ``True`` if any leaf variable in ``value`` is a granular acquire output.

        :param value: The assign value, which may be a nested list, tuple index reference,
            variable name string, or :class:`~qat.ir.instructions.Variable`.
        :param eq_outputs: Demapping of output variable name to equalise-stage array.
        :returns: ``True`` if at least one referenced acquire variable is granular.
        """
        if isinstance(value, list):
            return any(
                ResultTransform._assign_has_granular_var(v, eq_outputs) for v in value
            )
        if isinstance(value, tuple):
            return ResultTransform._assign_has_granular_var(value[0], eq_outputs)
        if isinstance(value, str):
            return value in eq_outputs
        if isinstance(value, Variable):
            return value.name in eq_outputs
        return False

    @staticmethod
    def _resolve_from_equalise(value: Any, eq_outputs: dict[str, np.ndarray]) -> Any:
        """Resolve assign payloads using equalise outputs where possible.

        Arrays from the equalise stage are converted to plain Python lists so that the
        final results remain JSON-serialisable (matching the ``raw()`` contract).

        :param value: The assign value, which may be a nested list, tuple index
            reference, variable name string, or :class:`~qat.ir.instructions.Variable`.
        :param eq_outputs: Demapping of output variable name to equalise-stage array.
        :returns: The resolved value with equalise arrays substituted where applicable.
        """
        if isinstance(value, list):
            return [ResultTransform._resolve_from_equalise(v, eq_outputs) for v in value]
        if isinstance(value, tuple):
            source = ResultTransform._resolve_from_equalise(value[0], eq_outputs)
            if isinstance(source, list | np.ndarray):
                return source[value[1]]
            return source
        if isinstance(value, str) and value in eq_outputs:
            return numpy_array_to_list(eq_outputs[value])
        if isinstance(value, Variable) and value.name in eq_outputs:
            return numpy_array_to_list(eq_outputs[value.name])
        return value

    @staticmethod
    def _label_count_register(
        assign_value: Any, disc_labels: dict[str, np.ndarray]
    ) -> dict[str, int]:
        """Count bitstring occurrences for a register assembled from ML acquire vars.

        When ``AssignResultsTransform`` has replaced a register key (e.g. ``"c"``) by
        assembling per-qubit ML label arrays into a list, the original per-acquire
        variable arrays are still available in ``disc_labels``.  This method walks the
        assign expression to collect the per-qubit label arrays in order, concatenates
        them per-shot into bitstrings (e.g. ``"01"``), and counts the occurrences.

        :param assign_value: The assign expression for the register key, e.g.
            ``["c[0]_0", "c[1]_0"]``.
        :param disc_labels: Per-acquire-variable discriminated label arrays.
        :returns: ``{bitstring: count}`` dictionary, e.g. ``{"00": 5, "01": 3}``.
        """

        def collect_labels(value: Any) -> list[np.ndarray]:
            if isinstance(value, list):
                result = []
                for v in value:
                    result.extend(collect_labels(v))
                return result
            if isinstance(value, tuple):
                return collect_labels(value[0])
            name = value.name if isinstance(value, Variable) else value
            if isinstance(name, str) and name in disc_labels:
                return [disc_labels[name]]
            return []

        per_qubit = collect_labels(assign_value)
        if not per_qubit:
            return {}
        # Build per-shot bitstrings by concatenating corresponding labels.
        n_shots = len(per_qubit[0])
        counts: dict[str, int] = {}
        for i in range(n_shots):
            bitstring = "".join(str(arr[i]) for arr in per_qubit)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    @staticmethod
    def _squash_binary(value: Any) -> str | None:
        """Squash a binary result array into a concatenated string.

        :param value: An integer or an iterable of integers representing binary bits.
        :returns: String representation of the binary value(s).
        """
        if isinstance(value, int):
            return str(value)
        elif all(isinstance(val, int) for val in value):
            return "".join([str(val) for val in value])

    @staticmethod
    def _is_generated_name(name: str) -> bool:
        """Check if a name is a generated name."""
        return name.startswith("generated_name_")


class ErrorMitigation(TransformPass):
    """Applies readout error mitigation to the results.

    Extracted from :meth:`qat.purr.compiler.runtime.QuantumRuntime._apply_error_mitigation`.
    """

    def __init__(self, hardware_model: QuantumHardwareModel | PhysicalHardwareModel):
        """
        :param hardware_model: The hardware model contains the error mitigation properties.
        """
        self.hardware_model = hardware_model

    def run(
        self,
        acquisitions: dict[str, Any],
        res_mgr: ResultManager,
        *args,
        compiler_config: CompilerConfig,
        **kwargs,
    ):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param res_mgr: The results manager is needed to look up the qubit-to-variable
            mapping.
        :param compiler_config: The compiler config is needed to know how to apply error
            mitigaiton, and should be provided as a keyword argument.
        """
        error_mitigation = compiler_config.error_mitigation

        if error_mitigation is None or error_mitigation == ErrorMitigationConfig.Empty:
            return acquisitions

        mapping = res_mgr.lookup_by_type(IndexMappingResult).mapping

        # TODO: add support for multiple registers
        # TODO: reconsider results length
        if len(acquisitions) > 1:
            raise ValueError(
                "Cannot have multiple registers in conjunction with readout error "
                "mitigation."
            )

        for mitigator in get_readout_mitigation(error_mitigation):
            new_acquisitions = mitigator.apply_error_mitigation(
                acquisitions, mapping, self.hardware_model
            )
            acquisitions[mitigator.name] = new_acquisitions
        return acquisitions
