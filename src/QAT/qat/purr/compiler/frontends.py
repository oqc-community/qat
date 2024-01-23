# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import abc
import os
import tempfile
from pathlib import Path
from typing import Tuple, Union

import regex
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import CompilerConfig, Languages, get_optimizer_config, default_language_options
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.metrics import CompilationMetrics
from qat.purr.compiler.optimisers import DefaultOptimizers
from qat.purr.compiler.runtime import (
    execute_instructions_via_config,
    get_builder,
    get_model,
)
from qat.purr.integrations.qasm import get_qasm_parser
from qat.purr.integrations.qir import get_qir_executor
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration

log = get_default_logger()


def _get_file_contents(file_path):
    """Get Program contents from a file."""
    with open(file_path) as ifile:
        return ifile.read()


path_regex = regex.compile("^.+\.(qasm|ll|bc)$")

contents_match_pattern = regex.compile(
    '(OPENQASM [0-9]*(.0)?;|defcalgrammar "[a-zA-Z ]+";)|(@__quantum__qis)'
)


class LanguageFrontend(abc.ABC):
    @abc.abstractmethod
    def execute(
        self,
        instructions: InstructionBuilder,
        hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
        compiler_config: CompilerConfig,
    ):
        ...

    @abc.abstractmethod
    def parse_and_execute(
        self, file_or_str: str, hardware, compiler_config: CompilerConfig
    ):
        ...

    @abc.abstractmethod
    def parse(
        self, program_str: str, hardware, compiler_config: CompilerConfig
    ) -> Tuple[InstructionBuilder, CompilationMetrics]:
        ...


class QIRFrontend(LanguageFrontend):
    def execute(
        self,
        instructions: InstructionBuilder,
        hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
        compiler_config: CompilerConfig,
    ):
        raise NotImplementedError(
            "QIR needs to be executed via a graph, not via this API."
        )

    def parse_and_execute(
        self, file_or_str: str, hardware, compiler_config: CompilerConfig
    ):
        # Parsing and execution are done at the same time with early versions of QIR.
        return self.parse(file_or_str, hardware, compiler_config)

    def _parse_from_file(
        self, qir_file: str, hardware, compiler_config: CompilerConfig
    ):
        metrics = CompilationMetrics()
        metrics.initialize(compiler_config.metrics)

        default_language_options(Languages.QIR, compiler_config)

        model = get_model(hardware)

        with log_duration("QIR parsing completed, took {} seconds."):
            executor = get_qir_executor(model, compiler_config)
            return executor.run(qir_file), metrics

    def parse(self, path_or_str: str, hardware, compiler_config: CompilerConfig):
        # Parse from file
        if os.path.exists(path_or_str):
            return self._parse_from_file(path_or_str, hardware, compiler_config)

        with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as fp:
            fp.write(path_or_str.encode())
            fp.close()
            try:
                return self._parse_from_file(fp.name, hardware, compiler_config)
            finally:
                os.remove(fp.name)


class QASMFrontend(LanguageFrontend):
    def execute(
        self,
        instructions: InstructionBuilder,
        hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
        compiler_config: CompilerConfig,
    ):
        return execute_instructions_via_config(hardware, instructions, compiler_config)

    def parse_and_execute(
        self, file_or_str: str, hardware, compiler_config: CompilerConfig
    ):
        instructions, parse_metrics = self.parse(file_or_str, hardware, compiler_config)
        result, execution_metrics = self.execute(
            instructions, hardware, compiler_config
        )
        execution_metrics.merge(parse_metrics)
        return result, execution_metrics

    def parse(self, path_or_str: str, hardware, compiler_config: CompilerConfig):
        # Parse from contents
        qasm_string = path_or_str
        if os.path.isfile(path_or_str):
            qasm_string = _get_file_contents(path_or_str)

        metrics = CompilationMetrics()
        metrics.initialize(compiler_config.metrics)

        parser = get_qasm_parser(qasm_string)
        default_language_options(parser.parser_language(), compiler_config)

        with log_duration("Compilation completed, took {} seconds."):
            log.info(
                f"Processing QASM with {str(parser)} as parser and {str(hardware)} as hardware."
            )

            qasm_string = DefaultOptimizers(metrics).optimize_qasm(
                qasm_string, get_model(hardware), compiler_config.optimizations
            )

            if compiler_config.results_format.format is not None:
                parser.results_format = compiler_config.results_format.format

            quantum_builder = parser.parse(get_builder(hardware), qasm_string)
            return quantum_builder, metrics


def fetch_frontend(path_or_str: str) -> LanguageFrontend:
    if path_regex.match(path_or_str) is not None:
        if not os.path.exists(path_or_str):
            raise ValueError(f"Path {path_or_str} does not exist.")

        path = Path(path_or_str)
        if path.suffix == ".qasm":
            return QASMFrontend()
        elif path.suffix in (".bc", ".ll"):
            return QIRFrontend()
        else:
            raise ValueError(f"File with extension {path.suffix} unrecognized.")

    results = regex.search(contents_match_pattern, path_or_str)
    if results is not None:
        if results.captures(1):
            return QASMFrontend()
        if results.captures(3):
            return QIRFrontend()
    else:
        raise ValueError("Cannot establish a LanguageFrontend based on contents")
