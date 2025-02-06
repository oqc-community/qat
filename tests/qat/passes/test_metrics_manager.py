# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest
from compiler_config.config import MetricsType

from qat.passes.metrics_base import MetricsManager


@pytest.mark.parametrize("flag", [*[mt for mt in MetricsType], MetricsType.Default])
def test_metrics_recording(flag):
    met_mgr = MetricsManager(flag)

    met_mgr.record_metric(MetricsType.OptimizedCircuit, "Some qasm string")
    met_mgr.record_metric(MetricsType.OptimizedInstructionCount, 42)

    if MetricsType.OptimizedCircuit in flag:
        assert met_mgr.optimized_circuit == "Some qasm string"
    else:
        assert met_mgr.optimized_circuit == None

    if MetricsType.OptimizedInstructionCount in flag:
        assert met_mgr.optimized_instruction_count == 42
    else:
        assert met_mgr.optimized_instruction_count == None
