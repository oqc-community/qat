# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import re

import numpy as np
from compiler_config.config import CompilerConfig

from qat.core.pass_base import TransformPass
from qat.purr.backends.qiskit_simulator import QiskitBuilder
from qat.purr.compiler.error_mitigation.readout_mitigation import get_readout_mitigation
from qat.purr.compiler.instructions import Assign, is_generated_name


class QiskitErrorMitigation(TransformPass):
    """Implements readout error mitigation for legacy Qiskit engines.

    Because the legacy Qiskit engine returns results in a format that is not consistent with
    other legacy engines (or refactored ones), it requires its own implementation. The
    refactored engine will be written so that the output of the results are formatted
    in a way that the runtime will expect to receive them. This is just a work around to
    support legacy engines in the new pipelines API.
    """

    # the pattern assign variables should follow
    index_pattern = re.compile(r"(.*)\[(?P<clbit_index>[0-9]+)\]_(?P<qubit_index>[0-9]+)")

    def run(
        self,
        acquisitions: dict[str, any],
        *args,
        compiler_config: CompilerConfig,
        package: QiskitBuilder,
        **kwargs,
    ) -> dict[str, any]:
        """
        :param acquisitions: Acquisition data returned from the Qiskit simulator.
        :param compiler_config: The compiler config contains the error mitigation
            configuration.
        :param package: The Qiskit instruction builder.
        :raises ValueError: Multiple registers are not allowed with error mitigation.
        :return: The transformed result acquisitions.
        """
        error_mitigation = compiler_config.error_mitigation
        if error_mitigation is None:
            return acquisitions

        if len(acquisitions) > 1:
            raise ValueError(
                "Cannot have multiple registers in conjunction with readout error "
                "mitigation."
            )

        mapping = self.classical_to_quantum_mapping(package)
        for mitigator in get_readout_mitigation(error_mitigation):
            new_result = mitigator.apply_error_mitigation(acquisitions, mapping, self.model)
            acquisitions[mitigator.name] = new_result
        return acquisitions

    def classical_to_quantum_mapping(self, package: QiskitBuilder) -> dict[str, int]:
        """Generates a mapping between classical register indices and qubit indices.

        This would probably be an analysis pass for refactored code, but since its only used
        here, its just included within this transformation pass.

        :param package: The Qiskit instruction builder.
        :raises ValueError: If the assign variables is not in the expected format, an error
            is thrown.
        :return: Returns the mapping as a dictionary.
        """
        mapping = {}
        for instruction in package.instructions:
            if not isinstance(instruction, Assign):
                continue
            for value in instruction.value:
                result = self.index_pattern.match(value.name)
                if result is None:
                    raise ValueError(
                        "Could not extract cl register index from "
                        f"{instruction.output_variable}."
                    )
                mapping[result.group("clbit_index")] = int(result.group("qubit_index"))
        return mapping


class QiskitStripMetadata(TransformPass):
    """Detects if Qiskit is returning meta data associated with the simulation, and trims
    it away if so."""

    def run(self, acquisitions: dict[str, any], *args, **kwargs):
        """
        :param acquisitions: Acquisition data returned from the Qiskit simulator.
        :return: The acquisition data with metadata stripped away (if found).
        """
        if isinstance(acquisitions, tuple):
            return acquisitions[0]
        else:
            return acquisitions


class QiskitSimplifyResults(TransformPass):
    """Strips away randomly generated names.

    This is the same as the `simplify_results` function in :class:`ResultsTransform`.
    """

    def run(self, acquisitions: dict[str, any], *args, **kwargs):
        """
        :param acquisitions: Acquisition data returned from the Qiskit simulator.
        :return: The acquisition data simplified.
        """
        if all([is_generated_name(k) for k in acquisitions.keys()]):
            if len(acquisitions) == 1:
                acquisitions = list(acquisitions.values())[0]
            else:
                squashed_results = list(acquisitions.values())
                if all(isinstance(val, np.ndarray) for val in squashed_results):
                    acquisitions = np.array(squashed_results)
                else:
                    acquisitions = squashed_results
        return acquisitions
