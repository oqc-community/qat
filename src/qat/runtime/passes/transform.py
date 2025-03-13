# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from typing import Dict, List

import numpy as np
from compiler_config.config import (
    CompilerConfig,
    ErrorMitigationConfig,
    InlineResultsProcessing,
    ResultsFormatting,
)

from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.purr.compiler.error_mitigation.readout_mitigation import get_readout_mitigation
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import is_generated_name
from qat.runtime.executables import Executable
from qat.runtime.passes.analysis import IndexMappingResult
from qat.runtime.post_processing import apply_post_processing, get_axis_map
from qat.runtime.results_processing import binary_average, binary_count, numpy_array_to_list


class PostProcessingTransform(TransformPass):
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

    def run(self, acquisitions: Dict[str, any], *args, package: Executable, **kwargs):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param package: The executable program containing the results-processing
            information should be passed as a keyword argument.
        """

        for acquire in package.acquires:
            response = acquisitions[acquire.output_variable]

            # Starting from all axes, we iterate through each post-processing, keeping track
            # of what axes remain as we go
            response_axes = get_axis_map(acquire.mode, response)
            for pp in package.post_processing.get(acquire.output_variable, []):
                response, response_axes = apply_post_processing(response, pp, response_axes)
            acquisitions[acquire.output_variable] = response

        return acquisitions


class InlineResultsProcessingTransform(TransformPass):
    """Uses :class:`InlineResultsProcessing` instructions from the executable package to
    format the acquired results in the desired format.
    """

    def run(self, acquisitions: Dict[str, any], *args, package: Executable, **kwargs):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param package: The executable program containing the results-processing
            information should be passed as a keyword argument.
        """
        # TODO: clean up imported utility

        for output_variable in acquisitions:
            target_values = acquisitions[output_variable]

            # TODO: ResultProcessing sanitisation and validation passes
            rp = package.results_processing.get(
                output_variable, InlineResultsProcessing.Experiment
            )

            if InlineResultsProcessing.Raw in rp and InlineResultsProcessing.Binary in rp:
                raise ValueError(
                    "Raw and Binary processing attempted to be applied to "
                    f"{output_variable}. Only one should be selected."
                )

            # Strip numpy arrays if we're set to do so.
            if InlineResultsProcessing.NumpyArrays not in rp:
                target_values = numpy_array_to_list(target_values)

            # Transform to various formats if required.
            if InlineResultsProcessing.Binary in rp:
                target_values = binary_average(target_values)

            acquisitions[output_variable] = target_values

        return acquisitions


class AssignResultsTransform(TransformPass):
    """Processes :class:`Assign` instructions.

    As assigns are classical instructions they are not processed as a part of the quantum
    execution (right now). Read through the results dictionary and perform the assigns
    directly, return the results.

    Extracted from :meth:`purr.compiler.execution.QuantumExecutionEngine._process_assigns`.
    """

    def run(self, acquisitions: Dict[str, any], *args, package: Executable, **kwargs):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param package: The executable program containing the results-processing
            information should be passed as a keyword argument.
        """

        # TODO: refactor
        def recurse_arrays(results_map, value):
            """Recurse through assignment lists and fetch values in sequence."""
            if isinstance(value, List):
                return [recurse_arrays(results_map, val) for val in value]
            elif isinstance(value, tuple):
                return results_map[value[0]][value[1]]
            elif isinstance(value, str):
                return results_map[value]
            else:
                return value

        assigns = dict(acquisitions)
        for assign in package.assigns:
            assigns[assign.name] = recurse_arrays(assigns, assign.value)
        return {key: assigns[key] for key in package.returns}


class ResultTransform(TransformPass):
    """Transform the raw results into the format that we've been asked to provide. Look at
    individual transformation documentation for descriptions on what they do.

    Extracted from :meth:`qat.purr.compiler.runtime.QuantumRuntime._transform_results`.
    """

    def run(
        self, acquisitions: Dict[str, any], *args, compiler_config: CompilerConfig, **kwargs
    ):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param compiler_config: The compiler config is needed to know how to process the
            results, and should be provided as a keyword argument.
        """
        # TODO: should the needed information in the compiler config be processed (in a
        # compilation pass) so that its not here. My opinion is that the compiler config
        # shouldn't make it to runtime.

        format_flags = (
            compiler_config.results_format or ResultsFormatting.DynamicStructureReturn
        )
        repeats = compiler_config.repeats or 1000

        if len(acquisitions) == 0:
            return []

        def simplify_results(simplify_target):
            """To facilitate backwards compatability and being able to run low-level
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
            acquisitions = {
                key: binary_count(val, repeats) for key, val in acquisitions.items()
            }

        def squash_binary(value):
            if isinstance(value, int):
                return str(value)
            elif all(isinstance(val, int) for val in value):
                return "".join([str(val) for val in value])

        if ResultsFormatting.SquashBinaryResultArrays in format_flags:
            acquisitions = {key: squash_binary(val) for key, val in acquisitions.items()}

        # Dynamic structure return is an ease-of-use flag to strip things that you know
        # your use-case won't use, such as variable names and nested lists.
        if ResultsFormatting.DynamicStructureReturn in format_flags:
            acquisitions = simplify_results(acquisitions)
        return acquisitions


class ErrorMitigation(TransformPass):
    """Applies readout error mitigation to the results.

    Extracted from :meth:`qat.purr.compiler.runtime.QuantumRuntime._apply_error_mitigation`.
    """

    def __init__(self, hardware_model: QuantumHardwareModel):
        """
        :param hardware_model: The hardware model contains the error mitigation properties.
        """
        self.hardware_model = hardware_model

    def run(
        self,
        acquisitions: Dict[str, any],
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
