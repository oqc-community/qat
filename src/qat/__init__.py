# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from importlib.metadata import version
from warnings import warn

from compiler_config.config import CompilerConfig as CompilerConfig

from qat.core.config.configure import get_config
from qat.core.config.session import QatSessionConfig as QatSessionConfig
from qat.core.qat import QAT as QAT
from qat.pipelines.pipeline import Pipeline as Pipeline
from qat.purr.qat import execute as execute
from qat.purr.qat import execute_qasm as execute_qasm
from qat.purr.qat import execute_qasm_with_metrics as execute_qasm_with_metrics
from qat.purr.qat import execute_qir as execute_qir
from qat.purr.qat import execute_qir_with_metrics as execute_qir_with_metrics
from qat.purr.qat import execute_with_metrics as execute_with_metrics
from qat.purr.qat import fetch_frontend as fetch_frontend
from qat.purr.qatconfig import QatConfig

qatconfig: QatConfig = get_config()

# TODO: Remove version checking, COMPILER-658
if version("qiskit") < "2.0.0":
    warn(
        "Support for Qiskit <2.0.0 is deprecated in QAT and Qiskit >=2.0.0 will soon "
        "be required. Please update your local Qiskit dependency to >=2.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
