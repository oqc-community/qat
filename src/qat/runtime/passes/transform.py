# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

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
from qat.ir.instructions import Variable
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.target_data import TargetData
from qat.purr.backends.utilities import software_post_process_linear_map_complex_to_real
from qat.purr.compiler.error_mitigation.readout_mitigation import get_readout_mitigation
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import AcquireMode, PostProcessType
from qat.runtime.passes.purr.analysis import IndexMappingResult
from qat.runtime.post_processing import apply_post_processing, get_axis_map
from qat.runtime.results_processing import binary_average, binary_count, numpy_array_to_list


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

    def run(self, acquisitions: dict[str, any], *args, package: Executable, **kwargs):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param package: The executable program containing the results-processing
            information should be passed as a keyword argument.
        """

        for output_variable, acquire in package.acquires.items():
            response = acquisitions[output_variable]

            # Starting from all axes, we iterate through each post-processing, keeping track
            # of what axes remain as we go
            response_axes = get_axis_map(acquire.mode, response)
            for pp in acquire.post_processing:
                response, response_axes = apply_post_processing(response, pp, response_axes)
            acquisitions[output_variable] = response

        return acquisitions


class QBloxAcquisitionPostProcessing(TransformPass):
    def run(
        self,
        playback: dict[str, dict[str, Acquisition]],
        *args,
        package: Executable,
        **kwargs,
    ):
        """
        Now that the combined playback is ready, we can compute and process results as required
        by customers. This requires loop nest information as well as post-processing and array shaping
        requirements.
        """

        results = {}
        for pulse_channel_id, acquisitions in playback.items():
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

                post_procs, axes = acquire_data.post_processing, {}
                for pp in post_procs:
                    if pp.process_type == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL:
                        response, _ = software_post_process_linear_map_complex_to_real(
                            pp.args, response, axes
                        )
                    elif pp.process_type == PostProcessType.DISCRIMINATE:
                        # f : {0, 1} ----> {-1, 1}
                        #       x    |---> 1 - 2x
                        response = 1 - 2 * thrld_data

                response = response.reshape(acquire_data.shape)

                # TODO - COMPILER-860 - Safer and more flexible acquisition addressing
                if name in results:
                    raise ValueError(f"Key {name} already exists")
                results[name] = response

        return results


class InlineResultsProcessingTransform(TransformPass):
    """Uses :class:`InlineResultsProcessing` instructions from the executable package to
    format the acquired results in the desired format.
    """

    def run(self, acquisitions: dict[str, any], *args, package: Executable, **kwargs):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param package: The executable program containing the results-processing
            information should be passed as a keyword argument.
        """
        # TODO: clean up imported utility

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

    def run(self, acquisitions: dict[str, any], *args, package: Executable, **kwargs):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param package: The executable program containing the results-processing
            information should be passed as a keyword argument.
        """

        def recurse_arrays(results_map, value):
            """Recurse through assignment lists and fetch values in sequence."""
            if isinstance(value, list):
                return [recurse_arrays(results_map, val) for val in value]
            elif isinstance(value, tuple):
                return results_map[value[0]][value[1]]
            elif isinstance(value, str):
                return results_map[value]
            elif isinstance(value, Variable):
                return results_map[value.name]
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
        self,
        acquisitions: dict[str, any],
        *args,
        compiler_config: CompilerConfig,
        package: Executable,
        **kwargs,
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
        repeats = package.shots or TargetData.default().default_shots

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
            if all([self._is_generated_name(k) for k in simplify_target.keys()]):
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
        acquisitions: dict[str, any],
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
