# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import os.path
import typing
from pathlib import Path
from typing import Union

import regex
from compiler_config.config import CompilerConfig

import qat.purr.compiler.experimental.frontends as experimental_frontends
import qat.purr.compiler.frontends as core_frontends
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.frontends import LanguageFrontend
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.metrics import CompilationMetrics
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()

# Credible input are str (contents or file path) or pre-build instruction builder
QATInput = Union[str, bytes, InstructionBuilder]


def _return_or_build(qat_input: QATInput, build_func: typing.Callable, **kwargs):
    if isinstance(qat_input, (str, bytes)):
        return build_func(qat_input, **kwargs)

    if isinstance(qat_input, InstructionBuilder):
        return qat_input, CompilationMetrics()

    raise TypeError(f"No compiler support for inputs of type {str(type(qat_input))}")


def execute(
    qat_input: QATInput,
    hardware: QuantumHardwareModel = None,
    compiler_config: CompilerConfig = None,
):
    """Execute file path or code blob."""
    results, _ = execute_with_metrics(qat_input, hardware, compiler_config)
    return results


contents_match_pattern = regex.compile(
    '(OPENQASM [0-9]*(.0)?;|defcalgrammar "[a-zA-Z ]+";)|(@__quantum__qis)'
)

path_regex = regex.compile(r"^.+\.(qasm|ll|bc)$")


def fetch_frontend(
    path_or_str: Union[str, bytes],
    use_experimental: bool = False,
) -> LanguageFrontend:
    frontend_mod = core_frontends
    if use_experimental:
        frontend_mod = experimental_frontends
    if isinstance(path_or_str, bytes) and path_or_str.startswith(b"BC"):
        return frontend_mod.QIRFrontend()

    if path_regex.match(path_or_str) is not None:
        if not os.path.exists(path_or_str):
            raise ValueError(f"Path {path_or_str} does not exist.")

        path = Path(path_or_str)
        if path.suffix == ".qasm":
            return frontend_mod.QASMFrontend()
        elif path.suffix in (".bc", ".ll"):
            return frontend_mod.QIRFrontend()

        raise ValueError(f"File with extension {path.suffix} unrecognized.")

    results = regex.search(contents_match_pattern, path_or_str)
    if results is not None:
        if results.captures(1):
            return frontend_mod.QASMFrontend()
        if results.captures(3):
            return frontend_mod.QIRFrontend()
    else:
        raise ValueError("Cannot establish a LanguageFrontend based on contents")


def execute_with_metrics(
    path_or_str: str,
    hardware: QuantumHardwareModel = None,
    compiler_config: CompilerConfig = None,
):
    """Execute file path or code blob."""

    frontend: LanguageFrontend = fetch_frontend(path_or_str)
    return _execute_with_metrics(frontend, path_or_str, hardware, compiler_config)


def execute_qir(qat_input: QATInput, hardware=None, compiler_config: CompilerConfig = None):
    results, _ = execute_qir_with_metrics(qat_input, hardware, compiler_config)
    return results


def execute_qir_with_metrics(
    qat_input: QATInput, hardware=None, compiler_config: CompilerConfig = None
):
    frontend = core_frontends.QIRFrontend()
    return _execute_with_metrics(frontend, qat_input, hardware, compiler_config)


def execute_qasm(
    qat_input: QATInput, hardware=None, compiler_config: CompilerConfig = None
):
    results, _ = execute_qasm_with_metrics(qat_input, hardware, compiler_config)
    return results


def execute_qasm_with_metrics(
    qat_input: QATInput, hardware=None, compiler_config: CompilerConfig = None
):
    frontend = core_frontends.QASMFrontend()
    return _execute_with_metrics(frontend, qat_input, hardware, compiler_config)


def _execute_with_metrics(
    frontend: LanguageFrontend,
    qat_input: QATInput,
    hardware=None,
    compiler_config: CompilerConfig = None,
):
    metrics = CompilationMetrics()
    if compiler_config is not None:
        metrics.enable(compiler_config.metrics, overwrite=True)

    instructions, build_metrics = _return_or_build(
        qat_input, frontend.parse, hardware=hardware, compiler_config=compiler_config
    )
    metrics.merge(build_metrics)

    results, execution_metrics = frontend.execute(instructions, hardware, compiler_config)
    metrics.merge(execution_metrics)

    return results, metrics.as_dict()
