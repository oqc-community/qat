# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd
import pytest
from compiler_config.config import MetricsType

from qat.core.metrics_base import MetricsManager


class TestMetricsManager:
    @pytest.mark.parametrize("flag", [*list(MetricsType), MetricsType.Experimental])
    def test_metrics_recording(self, flag):
        met_mgr = MetricsManager(flag)

        met_mgr.record_metric(MetricsType.OptimizedCircuit, "Some qasm string")
        met_mgr.record_metric(MetricsType.OptimizedInstructionCount, 42)

        if MetricsType.OptimizedCircuit in flag:
            assert met_mgr.optimized_circuit == "Some qasm string"
        else:
            assert met_mgr.optimized_circuit is None

        if MetricsType.OptimizedInstructionCount in flag:
            assert met_mgr.optimized_instruction_count == 42
        else:
            assert met_mgr.optimized_instruction_count is None

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
        met_mgr_1 = MetricsManager(MetricsType.Experimental)
        met_mgr_2 = MetricsManager(MetricsType.Experimental)

        met_mgr_1.optimized_circuit = "Original circuit"
        met_mgr_1.optimized_instruction_count = 42

        for metric, value in records.items():
            met_mgr_2.record_metric(metric, value)

        met_mgr_ret = met_mgr_2.merge(met_mgr_1)

        # Assert mgr.merge(other) returns mgr
        assert met_mgr_2 is met_mgr_ret

        for metric in [MetricsType.OptimizedCircuit, MetricsType.OptimizedInstructionCount]:
            if metric in records:
                assert met_mgr_2.get_metric(metric) == records[metric]
            else:
                assert met_mgr_2.get_metric(metric) == met_mgr_1.get_metric(metric)

    @pytest.mark.parametrize(
        ("overwrite", "circuit_enabled"),
        [
            pytest.param(False, True, id="extend"),
            pytest.param(True, False, id="overwrite"),
        ],
    )
    def test_enable_metrics(self, overwrite, circuit_enabled):
        met_mgr = MetricsManager(MetricsType.OptimizedCircuit)

        met_mgr.enable(MetricsType.OptimizedInstructionCount, overwrite=overwrite)

        assert met_mgr.are_enabled(MetricsType.OptimizedCircuit) is circuit_enabled
        assert met_mgr.are_enabled(MetricsType.OptimizedInstructionCount)

    def test_enable_ignores_none(self):
        met_mgr = MetricsManager(MetricsType.OptimizedCircuit)

        met_mgr.enable(None)

        assert met_mgr.enabled_metrics == MetricsType.OptimizedCircuit

    def test_enable_metrics_defaults_none_to_experimental(self):
        met_mgr = MetricsManager(None)

        met_mgr.enable_metrics(None)

        assert met_mgr.enabled_metrics == MetricsType.Experimental

    def test_disabled_manager_does_not_record_metrics(self):
        met_mgr = MetricsManager(None)

        met_mgr.record_metric(MetricsType.OptimizedCircuit, "ignored")

        assert not met_mgr.are_enabled(MetricsType.OptimizedCircuit)
        assert met_mgr.optimized_circuit is None

    def test_as_dict_excludes_enabled_metrics(self):
        met_mgr = MetricsManager(MetricsType.OptimizedCircuit)
        met_mgr.record_metric(MetricsType.OptimizedCircuit, "OPENQASM 2.0;")

        assert met_mgr.as_dict() == {
            "optimized_circuit": "OPENQASM 2.0;",
            "optimized_instruction_count": None,
            "physical_qubit_indices": None,
        }

    def test_merge_rejects_incompatible_type(self):
        with pytest.raises(TypeError, match="other must be of type MetricsManager"):
            MetricsManager().merge(object())
