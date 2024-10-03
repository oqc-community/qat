# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from compiler_config.config import (
    MetricsType,
    OptimizationConfig,
    Qiskit,
    QiskitOptimizations,
    Tket,
    TketOptimizations,
)
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.transpiler import TranspilerError

from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.metrics import MetricsMixin
from qat.purr.integrations.tket import run_tket_optimizations
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration

log = get_default_logger()


class DefaultOptimizers(MetricsMixin):
    def __init__(self, metrics=None):
        super().__init__()
        self.compilation_metrics = metrics

    def optimize_qasm(
        self,
        qasm_string,
        hardware: QuantumHardwareModel,
        optimizations: OptimizationConfig,
    ):
        """Run all available optimizers on this QASM program."""
        with log_duration("QASM optimization took {} seconds."):
            if (
                isinstance(optimizations, Tket)
                and optimizations.tket_optimizations != TketOptimizations.Empty
            ):
                qasm_string = run_tket_optimizations(
                    qasm_string, optimizations.tket_optimizations, hardware
                )

            # TODO: [QK] Spend time looking at qiskit optimization and seeing if it's
            #   worth keeping around.
            if (
                isinstance(optimizations, Qiskit)
                and optimizations.qiskit_optimizations != QiskitOptimizations.Empty
            ):
                qasm_string = self.run_qiskit_optimization(
                    qasm_string, optimizations.qiskit_optimizations
                )

            self.record_metric(MetricsType.OptimizedCircuit, qasm_string)
            return qasm_string

    def run_qiskit_optimization(self, qasm_string, level):
        """
        TODO: [QK] Current setup is unlikely to provide much benefit, refine settings
            before using.
        """
        if level is not None:
            try:
                optimized_circuits = transpile(
                    QuantumCircuit.from_qasm_str(qasm_string),
                    basis_gates=["u1", "u2", "u3", "cx"],
                    optimization_level=level,
                )
                qasm_string = qasm2.dumps(optimized_circuits)
            except TranspilerError as ex:
                log.warning(f"Qiskit transpile pass failed. {str(ex)}")

        return qasm_string
