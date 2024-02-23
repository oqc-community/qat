# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import abc
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import regex

from qat.features import is_rasqal_enabled
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import CompilerConfig, Languages, default_language_options
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
from qat.purr.integrations.qir import QIRParser
from qat.purr.integrations.rasqal import create_runtime
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration

log = get_default_logger()


def _get_file_contents(file_path):
    """Get Program contents from a file."""
    with open(file_path) as ifile:
        return ifile.read()


class LanguageFrontend(abc.ABC):
    @abc.abstractmethod
    def execute(
        self,
        instructions: Union[InstructionBuilder, str, bytes],
        hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
        compiler_config: CompilerConfig,
        args: Optional[List[Any]],
    ): ...

    @abc.abstractmethod
    def parse_and_execute(
        self,
        file_or_str: str,
        hardware,
        compiler_config: CompilerConfig,
        args: Optional[List[Any]],
    ): ...

    @abc.abstractmethod
    def parse(
        self, program_str: str, hardware, compiler_config: CompilerConfig
    ) -> Tuple[InstructionBuilder, CompilationMetrics]: ...


class QIRFrontend(LanguageFrontend):
    def execute(
        self,
        path_or_contents: Union[str, bytes],
        hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
        compiler_config: CompilerConfig,
        args: Optional[List[Any]] = None,
    ):
        if is_rasqal_enabled():
            runtime = create_runtime(get_model(hardware))
            if is_path(path_or_contents):
                result = runtime.run(path_or_contents, args)
            elif isinstance(path_or_contents, str):
                result = runtime.run_ll(path_or_contents, args)
            elif isinstance(path_or_contents, bytes):
                result = runtime.run_bitcode(path_or_contents, args)
            else:
                raise TypeError(
                    f"Tried to execute QIR file of invalid type: {type(path_or_contents)}"
                )

            # TODO: Metrics for the runtime in general are very different, even if we still have some
            #  from the engine. Either way, currently not propagated.
            metrics = CompilationMetrics()
            return result, metrics
        else:
            if is_path(path_or_contents):
                builder, parse_metrics = self._parse_legacy(
                    path_or_contents, hardware, compiler_config
                )
            elif isinstance(path_or_contents, (str, bytes)):
                suffix = ".bc" if isinstance(path_or_contents, bytes) else ".ll"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
                    if suffix == ".ll":
                        fp.write(path_or_contents.encode())
                    else:
                        fp.write(path_or_contents)
                    fp.close()
                    try:
                        builder, parse_metrics = self._parse_legacy(
                            fp.name, hardware, compiler_config
                        )
                    finally:
                        os.remove(fp.name)
            else:
                raise TypeError(
                    f"Tried to execute QIR file of invalid type: {type(path_or_contents)}"
                )

            result, execution_metrics = execute_instructions_via_config(
                hardware, builder, compiler_config
            )
            execution_metrics.merge(parse_metrics)
            return result, execution_metrics

    def parse_and_execute(
        self,
        path_or_str: Union[str, bytes],
        hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
        compiler_config: CompilerConfig,
        args: Optional[List[Any]] = None,
    ):
        return self.execute(path_or_str, hardware, compiler_config, args)

    def _parse_legacy(
        self,
        qir_file: str,
        hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
        compiler_config: CompilerConfig,
    ):
        metrics = CompilationMetrics()
        metrics.initialize(compiler_config.metrics)

        parser = QIRParser(hardware)
        default_language_options(Languages.QIR, compiler_config)

        if compiler_config.results_format.format is not None:
            parser.results_format = compiler_config.results_format.format

        return parser.parse(qir_file), metrics

    def parse(
        self,
        path_or_contents: str,
        hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
        compiler_config: CompilerConfig,
    ):
        # Parsing doesn't happen here, but if we're a file we want the contents of the
        # file to send over.
        if is_path(path_or_contents):
            return _get_file_contents(path_or_contents)

        return path_or_contents


class QASMFrontend(LanguageFrontend):
    def execute(
        self,
        instructions: InstructionBuilder,
        hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
        compiler_config: CompilerConfig,
        args: Optional[List[Any]] = None,
    ):
        return execute_instructions_via_config(hardware, instructions, compiler_config)

    def parse_and_execute(
        self,
        file_or_str: str,
        hardware,
        compiler_config: CompilerConfig,
        args: Optional[List[Any]] = None,
    ):
        instructions, parse_metrics = self.parse(file_or_str, hardware, compiler_config)
        result, execution_metrics = self.execute(
            instructions, hardware, compiler_config, args
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


def is_path(potential_path):
    return isinstance(potential_path, str) and path_regex.match(potential_path) is not None


path_regex = regex.compile("^.+\.(qasm|ll|bc)$")


contents_match_pattern = regex.compile(
    '(OPENQASM [0-9]*(.0)?;|defcalgrammar "[a-zA-Z ]+";)|(@__quantum__qis)'
)


def fetch_frontend(path_or_str: Union[str, bytes]) -> LanguageFrontend:
    if is_path(path_or_str):
        if not os.path.exists(path_or_str):
            raise ValueError(f"Path {path_or_str} does not exist.")

        path = Path(path_or_str)
        if path.suffix == ".qasm":
            return QASMFrontend()
        elif path.suffix in (".bc", ".ll"):
            return QIRFrontend()
        else:
            raise ValueError(f"File with extension {path.suffix} unrecognized.")

    if isinstance(path_or_str, bytes):
        return QIRFrontend()

    results = regex.search(contents_match_pattern, path_or_str)
    if results is not None:
        if results.captures(1):
            return QASMFrontend()
        if results.captures(3):
            return QIRFrontend()

    raise ValueError("Cannot establish a LanguageFrontend based on contents")
