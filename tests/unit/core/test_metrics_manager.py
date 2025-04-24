# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest
from compiler_config.config import MetricsType

from qat.core.metrics_base import MetricsManager


class TestMetricsManager:
    @pytest.mark.parametrize("flag", [*[mt for mt in MetricsType], MetricsType.Default])
    def test_metrics_recording(self, flag):
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

    @pytest.mark.parametrize(
        "records",
        [
            {
                MetricsType.OptimizedInstructionCount: 20,
            },
            {
                MetricsType.OptimizedCircuit: "Circuit",
            },
        ],
    )
    def test_merge_metrics_managers(self, records):
        met_mgr_1 = MetricsManager(MetricsType.Default)
        met_mgr_2 = MetricsManager(MetricsType.Default)

        met_mgr_1.optimized_circuit = "Original circuit"
        met_mgr_1.optimized_instruction_count = 42

        for metric, value in records.items():
            met_mgr_2.record_metric(metric, value)

        met_mgr_2.merge(met_mgr_1)

        for metric in [MetricsType.OptimizedCircuit, MetricsType.OptimizedInstructionCount]:
            if metric in records:
                assert met_mgr_2.get_metric(metric) == records[metric]
            else:
                assert met_mgr_2.get_metric(metric) == met_mgr_1.get_metric(metric)
