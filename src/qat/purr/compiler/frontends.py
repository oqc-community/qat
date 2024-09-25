# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import abc
import os
import tempfile
from typing import Tuple

import regex
from compiler_config.config import CompilerConfig, Languages, get_optimizer_config

from qat.purr.backends.calibrations.remote import find_calibration
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.metrics import CompilationMetrics
from qat.purr.compiler.optimisers import DefaultOptimizers
from qat.purr.compiler.runtime import execute_instructions, get_builder, get_model
from qat.purr.integrations.qasm import get_qasm_parser
from qat.purr.integrations.qir import QIRParser
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration

log = get_default_logger()


def _get_file_contents(file_path):
    """Get Program contents from a file."""
    with open(file_path) as ifile:
        return ifile.read()


path_regex = regex.compile(r"^.+\.(qasm|ll|bc)$")


class LanguageFrontend(abc.ABC):
    def _build_instructions(
        self,
        quantum_builder: InstructionBuilder,
        hardware,
        compiler_config: CompilerConfig,
    ):
        instructions = (
            get_builder(hardware)
            .repeat(compiler_config.repeats, compiler_config.repetition_period)
            .add(quantum_builder)
        )
        return instructions

    def _execute(
        self,
        hardware,
        compiler_config: CompilerConfig,
        instructions,
        *args,
        **kwargs,
    ):
        calibrations = [
            find_calibration(arg) for arg in compiler_config.active_calibrations
        ]

        return execute_instructions(
            hardware,
            instructions,
            compiler_config,
            calibrations,
        )

    def _default_common_args(self, hardware=None, compiler_config=None):
        hardware = hardware or get_default_RTCS_hardware()
        compiler_config = compiler_config or CompilerConfig()
        return hardware, compiler_config

    @abc.abstractmethod
    def parse(
        self, program_str: str, hardware, compiler_config: CompilerConfig
    ) -> Tuple[InstructionBuilder, CompilationMetrics]: ...

    @abc.abstractmethod
    def execute(
        self,
        instructions: InstructionBuilder,
        hardware,
        compiler_config: CompilerConfig,
        *args,
        **kwargs,
    ): ...


class QIRFrontend(LanguageFrontend):
    def _parse_from_file(
        self, path_or_str: str, hardware=None, compiler_config: CompilerConfig = None
    ):
        hardware, compiler_config = self._default_common_args(hardware, compiler_config)

        metrics = CompilationMetrics()
        metrics.enable(compiler_config.metrics)

        parser = QIRParser(hardware)
        if compiler_config.optimizations is None:
            compiler_config.optimizations = get_optimizer_config(Languages.QIR)

        if compiler_config.results_format.format is not None:
            parser.results_format = compiler_config.results_format.format

        quantum_builder = parser.parse(path_or_str)
        return self._build_instructions(quantum_builder, hardware, compiler_config), metrics

    def parse(
        self, path_or_str: str, hardware=None, compiler_config: CompilerConfig = None
    ):
        # Parse from file
        if not os.path.exists(path_or_str):
            suffix = ".bc" if isinstance(path_or_str, bytes) else ".ll"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
                if suffix == ".ll":
                    fp.write(path_or_str.encode())
                else:
                    fp.write(path_or_str)
                fp.close()
                try:
                    return self._parse_from_file(fp.name, hardware, compiler_config)
                finally:
                    os.remove(fp.name)
        return self._parse_from_file(path_or_str, hardware, compiler_config)

    def execute(
        self,
        instructions: InstructionBuilder,
        hardware=None,
        compiler_config: CompilerConfig = None,
        *args,
        **kwargs,
    ):
        hardware, compiler_config = self._default_common_args(hardware, compiler_config)

        with log_duration("Execution completed, took {} seconds."):
            return self._execute(hardware, compiler_config, instructions, *args, **kwargs)

    def parse_and_execute(
        self,
        qir_file: str,
        hardware=None,
        compiler_config: CompilerConfig = None,
        *args,
        **kwargs,
    ):
        instructions, parse_metrics = self.parse(qir_file, hardware, compiler_config)
        result, execution_metrics = self.execute(
            instructions, hardware, compiler_config, *args, **kwargs
        )
        execution_metrics.merge(parse_metrics)
        return result, execution_metrics


class QASMFrontend(LanguageFrontend):
    def parse(
        self, path_or_str: str, hardware=None, compiler_config: CompilerConfig = None
    ):
        # Parse from contents
        qasm_string = path_or_str
        if os.path.isfile(path_or_str):
            qasm_string = _get_file_contents(path_or_str)

        hardware, compiler_config = self._default_common_args(hardware, compiler_config)

        metrics = CompilationMetrics()
        metrics.enable(compiler_config.metrics)

        parser = get_qasm_parser(qasm_string)
        if compiler_config.optimizations is None:
            compiler_config.optimizations = get_optimizer_config(parser.parser_language())

        with log_duration("Compilation completed, took {} seconds."):
            log.info(
                f"Processing QASM with {str(parser)} as parser and {str(hardware)} "
                "as hardware."
            )

            qasm_string = DefaultOptimizers(metrics).optimize_qasm(
                qasm_string, get_model(hardware), compiler_config.optimizations
            )

            if compiler_config.results_format.format is not None:
                parser.results_format = compiler_config.results_format.format

            quantum_builder = parser.parse(get_builder(hardware), qasm_string)
            return (
                self._build_instructions(quantum_builder, hardware, compiler_config),
                metrics,
            )

    def execute(
        self,
        instructions: InstructionBuilder,
        hardware=None,
        compiler_config: CompilerConfig = None,
        *args,
        **kwargs,
    ):
        hardware, compiler_config = self._default_common_args(hardware, compiler_config)

        with log_duration("Execution completed, took {} seconds."):
            return self._execute(hardware, compiler_config, instructions, *args, **kwargs)

    def parse_and_execute(
        self,
        qasm_string: str,
        hardware=None,
        compiler_config: CompilerConfig = None,
        *args,
        **kwargs,
    ):
        """
        Execute a qasm string against a particular piece of hardware. Initializes a
        default qubit simulator if no hardware provided.
        """
        instructions, parse_metrics = self.parse(qasm_string, hardware, compiler_config)
        result, execution_metrics = self.execute(
            instructions, hardware, compiler_config, *args, **kwargs
        )
        parse_metrics.merge(execution_metrics)
        return result, parse_metrics
