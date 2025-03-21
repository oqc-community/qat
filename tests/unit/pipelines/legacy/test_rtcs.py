# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from compiler_config.config import CompilerConfig, QuantumResultsFormat

from qat import QAT
from qat.pipelines.legacy.rtcs import legacy_rtcs2

from tests.unit.utils.qasm_qir import get_qasm2


class TestRTCSPipelines:
    """Tests legacy RTCS pipelines.

    The tests here are not extensive: the RTCS engine is already tested in the `purr`
    package. The purpose here it just to test that it integrates smoothly in the pipelines,
    and test the integration points that have some friction (e.g. "optimize" function in the
    legacy engine becomes a pass in the pipeline).
    """

    def execute_bell_state(self, config=None):
        qasm_str = get_qasm2("ghz_2.qasm")
        results, _ = QAT().run(qasm_str, compiler_config=config, pipeline=legacy_rtcs2)
        return results

    def test_bell_with_binary_count(self):
        """Without the sanitization pass on Acquires, this will not pass."""

        config = CompilerConfig(results_format=QuantumResultsFormat().binary_count())
        results = self.execute_bell_state(config)
        assert len(results) == 1
        assert "b" in results
        assert "00" in results["b"]
        assert "11" in results["b"]
        assert results["b"]["00"] + results["b"]["11"] > 500
