# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Canonical system-data fixtures for experimental tests."""

import pytest

from qat.experimental.system_data.canonical.schema import CanonicalSystemData, PortData


@pytest.fixture(scope="module")
def canonical_model() -> CanonicalSystemData:
    """Return the smallest canonical model accepted by the pulse constraints."""
    return CanonicalSystemData(
        calibration_id="test-calibration",
        ports=(PortData(id="port0", sample_time=1_000, block_size=8),),
    )
