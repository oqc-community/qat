# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat import QAT
from qat.pipelines.legacy.qiskit import legacy_qiskit8

from tests.qat.qasm_utils import get_qasm2


class TestQiskitPipelines:
    """Tests legacy qiskit pipelines.

    The tests are not extensive, as most of the testing for qiskit engines and builders are
    found in the `purr` package, and any adapted passes are tested by their respective unit
    tests. These tests are just some sanity checks to ensure the pipelines run smoothly.

    Qiskit builders in `purr` cannot be serialized, so we don't test any of that here.
    """

    def test_ghz_gives_expected_results(self):
        """Basic execution of a QASM file to test the pipeline works as expected."""

        qasm_str = get_qasm2("ghz.qasm")
        results, _ = QAT().run(qasm_str, pipeline=legacy_qiskit8)
        assert len(results) == 1
        assert "b" in results
        assert len(results["b"]) == 2
        assert "0000" in results["b"]
        assert "1111" in results["b"]
