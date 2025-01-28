# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from typing import List

import numpy as np
from compiler_config.config import (
    CompilerConfig,
    ErrorMitigationConfig,
    InlineResultsProcessing,
    ResultsFormatting,
)

from qat.ir.pass_base import QatIR, TransformPass
from qat.ir.result_base import ResultManager
from qat.purr.compiler.error_mitigation.readout_mitigation import get_readout_mitigation
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import IndexAccessor, Variable, is_generated_name
from qat.runtime.analysis_passes import IndexMappingResult
from qat.runtime.executables import Executable
from qat.runtime.post_processing import apply_post_processing, get_axis_map
from qat.runtime.results_processing import binary_average, binary_count, numpy_array_to_list


class PostProcessingTransform(TransformPass):

    def run(self, results: QatIR, *args, package: Executable, **kwargs):
        """
        Uses the post-processing instructions from the executable package to process the
        results from the engine.

        The backend will return the results in a format that depends on the specified
        `AquireMode`. However, it is often the case results need to be returned in an explicit
        format, e.g., as discriminated bits. To achieve this, extra software post-processing
        is needed.

        The post-processing that appears here is the same as the post-processing
        responsibilities taken on by the `QuantumExecutionEngine` in the `purr` stack.

        :param results: Results to be processed.
        :type results: QatIR
        :param package: The executable program containing post-processing information.
        :type package: Executable

        TODO: Change argument from QatIR with changes to the pass manager (maybe its own
        object?)
        """

        results = results.value
        for acquire in package.acquires:
            response = results[acquire.output_variable]

            # Starting from all axes, we iterate through each post-processing, keeping track
            # of what axes remain as we go
            response_axes = get_axis_map(acquire.mode, response)
            for pp in package.post_processing.get(acquire.output_variable, []):
                response, response_axes = apply_post_processing(response, pp, response_axes)
            results[acquire.output_variable] = response


class InlineResultsProcessingTransform(TransformPass):

    def run(self, results: QatIR, *args, package: Executable, **kwargs):
        """
        Uses `InlineResultsProcessing` instructions from the executable package to format the
        acquired results in the desired format.

        :param results: Results to be processed.
        :type results: QatIR
        :param package: The executable program containing the results-processing information.
        :type package: Executable

        TODO: change argument type from QatIR
        TODO: clean up imported utility
        """

        results = results.value
        for output_variable in results:
            target_values = results[output_variable]

            # TODO: ResultProcessing sanitisation and validation passes
            rp = package.results_processing.get(
                output_variable, InlineResultsProcessing.Experiment
            )

            if InlineResultsProcessing.Raw in rp and InlineResultsProcessing.Binary in rp:
                raise ValueError(
                    f"Raw and Binary processing attempted to be applied to {output_variable}. "
                    "Only one should be selected."
                )

            # Strip numpy arrays if we're set to do so.
            if InlineResultsProcessing.NumpyArrays not in rp:
                target_values = numpy_array_to_list(target_values)

            # Transform to various formats if required.
            if InlineResultsProcessing.Binary in rp:
                target_values = binary_average(target_values)

            results[output_variable] = target_values


class AssignResultsTransform(TransformPass):

    def run(self, results: QatIR, *args, package: Executable, **kwargs):
        """
        Extracted from `purr.compiler.execution.QuantumExecutionEngine._process_assigns`.

        As assigns are classical instructions they are not processed as a part of the
        quantum execution (right now).
        Read through the results dictionary and perform the assigns directly, return the
        results.

        :param results: Results to be processed.
        :type results: QatIR
        :param package: The executable program containing the results-processing information.
        :type package: Executable

        TODO: change argument type from QatIR
        TODO: refactor
        """

        def recurse_arrays(results_map, value):
            """Recurse through assignment lists and fetch values in sequence."""
            if isinstance(value, List):
                return [recurse_arrays(results_map, val) for val in value]
            elif isinstance(value, Variable):
                if value.name not in results_map:
                    raise ValueError(
                        f"Attempt to assign variable that doesn't exist {value.name}."
                    )

                if isinstance(value, IndexAccessor):
                    return results_map[value.name][value.index]
                else:
                    return results_map[value.name]
            else:
                return value

        res = results.value
        assigns = dict(res)
        for assign in package.assigns:
            assigns[assign.name] = recurse_arrays(assigns, assign.value)
        results.value = {key: assigns[key] for key in package.returns}


class ResultTransform(TransformPass):
    """Extracted from legacy QuantumRuntime._transform_results()."""

    def run(
        self,
        ir: QatIR,
        *args,
        compiler_config: CompilerConfig,
        **kwargs,
    ):
        """
        Transform the raw results into the format that we've been asked to provide. Look
        at individual transformation documentation for descriptions on what they do.
        """
        # TODO: Consider the suggested implementation of a results type.

        format_flags = (
            compiler_config.results_format or ResultsFormatting.DynamicStructureReturn
        )
        repeats = compiler_config.repeats or 1000

        results = ir.value

        if len(results) == 0:
            ir.value = []
            return

        def simplify_results(simplify_target):
            """
            To facilitate backwards compatability and being able to run low-level
            experiments alongside quantum programs we make some assumptions based upon
            form of the results.

            If all results have default variable names then the user didn't care about
            value assignment or this was a low-level experiment - in both cases, it
            means we can throw away the names and simply return the results in the order
            they were defined in the instructions.

            If we only have one result after this, just return that list directly
            instead, as it's probably just a single experiment.
            """
            if all([is_generated_name(k) for k in simplify_target.keys()]):
                if len(simplify_target) == 1:
                    return list(simplify_target.values())[0]
                else:
                    squashed_results = list(simplify_target.values())
                    if all(isinstance(val, np.ndarray) for val in squashed_results):
                        return np.array(squashed_results)
                    return squashed_results
            else:
                return simplify_target

        if ResultsFormatting.BinaryCount in format_flags:
            results = {key: binary_count(val, repeats) for key, val in results.items()}

        def squash_binary(value):
            if isinstance(value, int):
                return str(value)
            elif all(isinstance(val, int) for val in value):
                return "".join([str(val) for val in value])

        if ResultsFormatting.SquashBinaryResultArrays in format_flags:
            results = {key: squash_binary(val) for key, val in results.items()}

        # Dynamic structure return is an ease-of-use flag to strip things that you know
        # your use-case won't use, such as variable names and nested lists.
        if ResultsFormatting.DynamicStructureReturn in format_flags:
            results = simplify_results(results)

        ir.value = results


class ErrorMitigation(TransformPass):
    """Extracted from legacy QuantumRuntime._apply_error_mitigation()."""

    def __init__(self, hardware_model: QuantumHardwareModel):
        self.hardware_model = hardware_model

    def run(
        self,
        ir: QatIR,
        res_mgr: ResultManager,
        *args,
        compiler_config: CompilerConfig,
        **kwargs,
    ):

        error_mitigation = compiler_config.error_mitigation

        if error_mitigation is None or error_mitigation == ErrorMitigationConfig.Empty:
            return

        mapping = res_mgr.lookup_by_type(IndexMappingResult).mapping
        results = ir.value

        # TODO: add support for multiple registers
        # TODO: reconsider results length
        if len(results) > 1:
            raise ValueError(
                "Cannot have multiple registers in conjunction with readout error mitigation."
            )

        for mitigator in get_readout_mitigation(error_mitigation):
            new_result = mitigator.apply_error_mitigation(
                results, mapping, self.hardware_model
            )
            results[mitigator.name] = new_result
        ir.value = results  # TODO: new results object
