# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from types import SimpleNamespace

import pytest

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


_SKIP_MESSAGE = "when collecting qubit indices"


def _make_resonator_payload_without_index(device_id: str) -> dict:
    """Return a resonator payload shaped like the LX5 calibration: no ``index`` key."""

    return {"id": device_id, "measure_device": None, "pulse_channels": {}}


def test_collect_qubit_index_by_id_skips_devices_without_index_silently(caplog):
    dto = SimpleNamespace(
        quantum_devices={
            "Q1": {"id": "Q1", "index": 1},
            "R1": _make_resonator_payload_without_index("R1"),
            "R2": _make_resonator_payload_without_index("R2"),
        }
    )

    with caplog.at_level("WARNING"):
        qubit_index_by_id = _collect_qubit_index_by_id(dto)

    assert qubit_index_by_id == {"Q1": 1}
    assert _SKIP_MESSAGE not in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"id": 3, "index": 0}, id="non_str_id"),
        pytest.param({"id": "Q0", "index": "0"}, id="non_int_index"),
        pytest.param({"id": "Q0", "index": None}, id="none_index"),
        pytest.param({"id": "Q0", "measure_device": {"id": "R0"}}, id="missing_index"),
    ],
)
def test_collect_qubit_index_by_id_warns_for_qubit_with_bad_id_or_index(caplog, payload):
    dto = SimpleNamespace(
        quantum_devices={
            "bad_qubit": payload,
            "R1": _make_resonator_payload_without_index("R1"),
        }
    )

    with caplog.at_level("WARNING"):
        qubit_index_by_id = _collect_qubit_index_by_id(dto)

    assert qubit_index_by_id == {}
    # A set, because the default logger writes each message to both the purr and root loggers.
    skip_messages = {
        r.getMessage() for r in caplog.records if _SKIP_MESSAGE in r.getMessage()
    }
    assert skip_messages == {
        "Skipping quantum device 'bad_qubit' when collecting qubit indices: "
        "id/index are not (str, int)."
    }


def test_collect_qubit_index_by_id_duplicate_warning_unchanged_with_resonators(caplog):
    dto = SimpleNamespace(
        quantum_devices={
            "q0_a": {"id": "Q0", "index": 0},
            "R1": _make_resonator_payload_without_index("R1"),
            "q0_b": {"id": "Q0", "index": 2},
        }
    )

    with caplog.at_level("WARNING"):
        qubit_index_by_id = _collect_qubit_index_by_id(dto)

    assert qubit_index_by_id == {"Q0": 2}
    # Only the purr records: the default logger also writes each message to the root logger.
    assert [r.getMessage() for r in caplog.records if r.name == "purr"] == [
        "Duplicate qubit id 'Q0' encountered while collecting qubit indices; "
        "overwriting index 0 with 2."
    ]
