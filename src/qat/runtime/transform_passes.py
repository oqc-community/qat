from typing import Dict

import numpy
from compiler_config.config import CompilerConfig, ErrorMitigationConfig, ResultsFormatting

from qat.ir.pass_base import QatIR, TransformPass
from qat.ir.result_base import ResultManager
from qat.purr.compiler.error_mitigation.readout_mitigation import get_readout_mitigation
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import is_generated_name
from qat.purr.compiler.runtime import _binary_count


class ResultTransform(TransformPass):
    """Extracted from legacy QuantumRuntime._transform_results()."""

    def __init__(self, compiler_config: CompilerConfig):
        # TODO: System defaults instead of hardcoded values
        self.format_flags = (
            compiler_config.results_format or ResultsFormatting.DynamicStructureReturn
        )
        self.repeats = compiler_config.repeats or 1000

    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        """
        Transform the raw results into the format that we've been asked to provide. Look
        at individual transformation documentation for descriptions on what they do.
        """
        # TODO: Consider the suggested implementation of a results type.
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
                    if all(isinstance(val, numpy.ndarray) for val in squashed_results):
                        return numpy.array(squashed_results)
                    return squashed_results
            else:
                return simplify_target

        if ResultsFormatting.BinaryCount in self.format_flags:
            results = {
                key: _binary_count(val, self.repeats) for key, val in results.items()
            }

        def squash_binary(value):
            if isinstance(value, int):
                return str(value)
            elif all(isinstance(val, int) for val in value):
                return "".join([str(val) for val in value])

        if ResultsFormatting.SquashBinaryResultArrays in self.format_flags:
            results = {key: squash_binary(val) for key, val in results.items()}

        # Dynamic structure return is an ease-of-use flag to strip things that you know
        # your use-case won't use, such as variable names and nested lists.
        if ResultsFormatting.DynamicStructureReturn in self.format_flags:
            results = simplify_results(results)

        ir.value = results


class ErrorMitigation(TransformPass):
    """Extracted from legacy QuantumRuntime._apply_error_mitigation()."""

    def __init__(
        self, hardware_model: QuantumHardwareModel, compiler_config: CompilerConfig
    ):
        self.hardware_model = hardware_model
        self.error_mitigation = compiler_config.error_mitigation

    def run(
        self, ir: QatIR, res_mgr: ResultManager, *args, mapping: Dict[str, str], **kwargs
    ):
        if (
            self.error_mitigation is None
            or self.error_mitigation == ErrorMitigationConfig.Empty
        ):
            return

        results = ir.value

        # TODO: add support for multiple registers
        # TODO: reconsider results length
        if len(results) > 1:
            raise ValueError(
                "Cannot have multiple registers in conjunction with readout error mitigation."
            )

        for mitigator in get_readout_mitigation(self.error_mitigation):
            new_result = mitigator.apply_error_mitigation(
                results, mapping, self.hardware_model
            )
            results[mitigator.name] = new_result
        ir.value = results  # TODO: new results object
