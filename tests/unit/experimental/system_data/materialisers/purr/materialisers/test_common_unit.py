# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from qat.experimental.system_data.materialisers.purr.materialisers.common import (
    _seconds_to_picoseconds,
)


def test_seconds_to_picoseconds_logs_warning_on_lossy_rounding(caplog):
    with caplog.at_level("WARNING", logger="qat.purr.utils.logger"):
        result = _seconds_to_picoseconds(1.2345e-12)

    assert result == 1
    assert "Rounded duration from scaled picoseconds value" in caplog.text


def test_seconds_to_picoseconds_skips_warning_for_exact_scaling(caplog):
    with caplog.at_level("WARNING", logger="qat.purr.utils.logger"):
        result = _seconds_to_picoseconds(2e-12)

    assert result == 2
    assert "Rounded duration from scaled picoseconds value" not in caplog.text
