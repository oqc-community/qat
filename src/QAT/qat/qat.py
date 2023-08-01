# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import os.path
import typing
from pathlib import Path
from typing import Union

import regex
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import CompilerConfig
from qat.purr.compiler.frontends import LanguageFrontend, QASMFrontend, QIRFrontend
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.metrics import CompilationMetrics
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()

# Credible input are str (contents or file path) or pre-build instruction builder
QATInput = Union[str, InstructionBuilder]


def _return_or_build(ingest: QATInput, build_func: typing.Callable, **kwargs):
    is_source = isinstance(ingest, str)
    is_instructions = isinstance(ingest, InstructionBuilder)
    if (not is_instructions) and (not is_source):
        raise ValueError(f"No compiler support for inputs of type {str(type(ingest))}")
    return build_func(ingest, **kwargs) if is_source else ingest


def execute(
    qat_input: QATInput,
    hardware: QuantumHardwareModel = None,
    compiler_config: CompilerConfig = None
):
    """ Execute file path or code blob. """
    results, _ = execute_with_metrics(qat_input, hardware, compiler_config)
    return results


contents_match_pattern = regex.compile(
    "(OPENQASM [0-9]*(.0)?;|defcalgrammar \"[a-zA-Z ]+\";)|(@__quantum__qis)"
)

path_regex = regex.compile('^.+\.(qasm|ll|bc)$')


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


def execute_with_metrics(
    path_or_str: str,
    hardware: QuantumHardwareModel = None,
    compiler_config: CompilerConfig = None
):
    """ Execute file path or code blob. """

    frontend: LanguageFrontend = fetch_frontend(path_or_str)
    return _execute_with_metrics(frontend, path_or_str, hardware, compiler_config)


def execute_qir(qat_input: QATInput, hardware=None, compiler_config: CompilerConfig = None):
    results, _ = execute_qir_with_metrics(qat_input, hardware, compiler_config)
    return results


def execute_qir_with_metrics(
    qat_input: QATInput, hardware=None, compiler_config: CompilerConfig = None
):
    frontend = QIRFrontend()
    return _execute_with_metrics(frontend, qat_input, hardware, compiler_config)


def execute_qasm(
    qat_input: QATInput, hardware=None, compiler_config: CompilerConfig = None
):
    results, _ = execute_qasm_with_metrics(qat_input, hardware, compiler_config)
    return results


def execute_qasm_with_metrics(
    qat_input: QATInput, hardware=None, compiler_config: CompilerConfig = None
):
    frontend = QASMFrontend()
    return _execute_with_metrics(frontend, qat_input, hardware, compiler_config)


def _execute_with_metrics(
    frontend: LanguageFrontend,
    qat_input: QATInput,
    hardware=None,
    compiler_config: CompilerConfig = None
):
    metrics = CompilationMetrics()
    if compiler_config is not None:
        metrics.enable(compiler_config.metrics, overwrite=True)

    instructions, build_metrics = _return_or_build(
        qat_input, frontend.parse, hardware=hardware, compiler_config=compiler_config
    )
    metrics.merge(build_metrics)

    results, execution_metrics = \
        frontend.execute(instructions, hardware, compiler_config)
    metrics.merge(execution_metrics)

    return results, metrics.as_dict()
