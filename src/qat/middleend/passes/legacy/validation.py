# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from compiler_config.config import CompilerConfig, QuantumResultsFormat, ResultsFormatting

from qat.core.pass_base import ValidationPass
from qat.purr.backends.qiskit_simulator import QiskitBuilder
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class QiskitResultsFormatValidation(ValidationPass):
    """Validates the results format contains `BinaryCount`, and throws a warning if not."""

    def run(self, ir: QiskitBuilder, *args, compiler_config: CompilerConfig, **kwargs):
        """
        :param ir: The Qiskit instruction builder.
        :param compiler_config: The compiler config contains the results format.
        :return: The instruction builder, unaltered.
        """
        results_format = compiler_config.results_format
        format_flags = (
            results_format.transforms
            if isinstance(results_format, QuantumResultsFormat)
            else results_format
        )
        if format_flags == None or not ResultsFormatting.BinaryCount in format_flags:
            log.warning(
                "The results formatting `BinaryCount` was not found in the formatting "
                "flags. Please note that the Qiskit runtime only currently supports "
                "results returned as a binary count."
            )
        return ir
