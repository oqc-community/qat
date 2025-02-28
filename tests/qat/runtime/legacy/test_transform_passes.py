# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest
from compiler_config.config import CompilerConfig

from qat.purr.compiler.instructions import build_generated_name
from qat.runtime.legacy.transform_passes import (
    QiskitErrorMitigation,
    QiskitSimplifyResults,
    QiskitStripMetadata,
)


class TestQiskitErrorMitigation:
    """Doesn't test actual functionality of error mitigation - the pass just calls
    error mitigation methods in `purr`, which are tested there. We can work on more robust
    error mitigation tests when we refactor."""

    def test_skipped_with_no_config(self):
        results = {"b": {"11": 500, "00": 500}}
        results = QiskitErrorMitigation().run(
            results, compiler_config=CompilerConfig(), package=None
        )
        assert results == results

    def test_multiple_registers_throws_error(self):
        """Tests that results with many classical registers throws an error as expected. No
        actual mitigation config is needed to test this, so we just mock it up."""
        results = {"b": {"11": 500, "00": 500}, "c": {"1": 1000}}
        config = CompilerConfig(error_mitigation=True)
        with pytest.raises(ValueError):
            QiskitErrorMitigation().run(results, compiler_config=config, package=None)


class TestQiskitStripMetadata:

    def test_metadata_is_removed(self):
        """When metadata is enabled, results are returned as a tuple of execution results
        and metadata. Let's check they're stripped away!"""
        results = {"b": {"11": 500, "00": 500}}
        metadata = "test"  # the actual contents are irrelevent!
        new_results = QiskitStripMetadata().run((results, metadata))
        assert new_results == results


class TestQiskitSimplifyResults:

    def test_single_register_has_key_removed(self):
        results = {build_generated_name(): np.zeros(1000)}
        results = QiskitSimplifyResults().run(results)
        assert isinstance(results, np.ndarray)

    def test_multiple_registers(self):
        results = {build_generated_name(): [0] * 1000, build_generated_name(): [0] * 500}
        results = QiskitSimplifyResults().run(results)
        assert isinstance(results, list)
        assert len(results) == 2
