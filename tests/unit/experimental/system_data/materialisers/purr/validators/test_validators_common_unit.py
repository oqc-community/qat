# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from types import SimpleNamespace

from qat.experimental.system_data.materialisers.purr.validators.common import (
    _collect_qubit_index_by_id,
)


def test_collect_qubit_index_by_id_logs_skips_and_duplicates(caplog):
    dto = SimpleNamespace(
        quantum_devices={
            "raw": "bad",
            "missing_id": {"index": 1},
            "q0_a": {"id": "Q0", "index": 0},
            "q0_b": {"id": "Q0", "index": 2},
            "q1": {"id": "Q1", "index": 1},
        }
    )

    with caplog.at_level("WARNING"):
        qubit_index_by_id = _collect_qubit_index_by_id(dto)

    assert qubit_index_by_id == {"Q0": 2, "Q1": 1}
    assert "payload is not a mapping" in caplog.text
    assert "id/index are not (str, int)" in caplog.text
    assert "Duplicate qubit id 'Q0' encountered" in caplog.text
