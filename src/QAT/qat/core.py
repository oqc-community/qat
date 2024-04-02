# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from typing import Any, List, Optional, Union

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import CompilerConfig
from qat.purr.compiler.frontends import (
    LanguageFrontend,
    QASMFrontend,
    QIRFrontend,
    fetch_frontend,
)
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.runtime import execute_instructions_via_config
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()

# Credible input are str (contents or file path) or pre-build instruction builder
QATInput = Union[str, bytes, InstructionBuilder]


def _default_arguments(hardware: QuantumHardwareModel, config: CompilerConfig):
    """Centralized place for config/hardware defaulting for various generalized execution pathways."""
    return hardware or get_default_echo_hardware(), config or CompilerConfig()


def execute(
    qat_input: QATInput,
    hardware: QuantumHardwareModel = None,
    compiler_config: CompilerConfig = None,
    args: List[Any] = None,
):
    """Execute file path or code blob."""
    results, _ = execute_with_metrics(qat_input, hardware, compiler_config, args)
    return results


def execute_with_metrics(
    incoming: QATInput,
    hardware: QuantumHardwareModel = None,
    config: CompilerConfig = None,
    args: List[Any] = None,
):
    hardware, config = _default_arguments(hardware, config)
    """ Execute file path or code blob. """
    if isinstance(incoming, (str, bytes)):
        frontend: LanguageFrontend = fetch_frontend(incoming)
        return _parse_and_execute(frontend, incoming, hardware, config, args)
    elif isinstance(incoming, InstructionBuilder):
        results, metrics = execute_instructions_via_config(
            hardware, incoming.instructions, config
        )
        return results, metrics.as_dict()

    raise TypeError(f"No compiler support for inputs of type {str(type(incoming))}")


def execute_qir(
    qir_file: str,
    hardware=None,
    compiler_config: CompilerConfig = None,
    args: List[Any] = None,
):
    results, _ = execute_qir_with_metrics(qir_file, hardware, compiler_config, args)
    return results


def execute_qir_with_metrics(
    qir_file: str,
    hardware=None,
    compiler_config: CompilerConfig = None,
    args: List[Any] = None,
):
    return _parse_and_execute(QIRFrontend(), qir_file, hardware, compiler_config, args)


def execute_qasm(
    qat_input: str,
    hardware=None,
    compiler_config: CompilerConfig = None,
    args: List[Any] = None,
):
    results, _ = execute_qasm_with_metrics(qat_input, hardware, compiler_config, args)
    return results


def execute_qasm_with_metrics(
    qat_input: str,
    hardware=None,
    compiler_config: CompilerConfig = None,
    args: List[Any] = None,
):
    return _parse_and_execute(QASMFrontend(), qat_input, hardware, compiler_config, args)


def _parse_and_execute(
    frontend: LanguageFrontend,
    str_or_path: str,
    hardware,
    config: CompilerConfig = None,
    args: Optional[List[Any]] = None,
):
    hardware, config = _default_arguments(hardware, config)
    results, metrics = frontend.parse_and_execute(str_or_path, hardware, config, args)
    return results, metrics.as_dict()
