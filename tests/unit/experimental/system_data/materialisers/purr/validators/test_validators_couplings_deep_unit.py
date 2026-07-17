# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.materialisers.errors import SourceConsistencyError
from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.system_data.materialisers.purr.validators.couplings import (
    _iter_cr_crc_pulse_views,
    _iter_normalized_cr_crc_entries,
    _normalise_cr_crc_auxiliary_target,
    _parse_cr_crc_channel_id,
    _parse_cr_crc_key,
    _validate_coupling_indices,
    _validate_cr_crc_auxiliary_targets,
    _validate_cr_crc_channel_mapping_keys,
    _validate_cr_crc_counterparts,
    _validate_cr_crc_matches_coupling_graph,
)


def _make_dto() -> PurrIngressV010:
    payload = {
        "calibration_id": "cal",
        "quantum_devices": {
            "Q0": {
                "id": "Q0",
                "index": 0,
                "pulse_channels": {
                    "Q1.cross_resonance": {
                        "pulse_channel": {
                            "id": "Q0.Q1.cross_resonance",
                            "physical_channel": {"id": "p_q0"},
                            "frequency": 5e9,
                            "auxiliary_qubit": "Q1",
                        }
                    },
                    "Q1.cross_resonance_cancellation": {
                        "pulse_channel": {
                            "id": "Q0.Q1.cross_resonance_cancellation",
                            "physical_channel": {"id": "p_q0"},
                            "frequency": 5e9,
                        }
                    },
                    "drive": {
                        "pulse_channel": {
                            "id": "Q0.drive",
                            "physical_channel": {"id": "p_q0"},
                            "frequency": 5e9,
                        }
                    },
                },
            },
            "Q1": {
                "id": "Q1",
                "index": 1,
                "pulse_channels": {},
            },
        },
        "pulse_channels": {},
        "physical_channels": {"p_q0": {"sample_time": 1e-9}},
        "basebands": {"bb0": {"frequency": 5e9}},
        "qubit_direction_couplings": [{"direction": [0, 1]}],
    }
    return PurrIngressV010.model_validate(payload)


def _mutated_dto(mutator) -> PurrIngressV010:
    payload = _make_dto().model_dump(mode="python")
    mutator(payload)
    return PurrIngressV010.model_validate(payload)


def test_parse_helpers_cover_invalid_shapes_and_auxiliary_normalisation():
    assert _parse_cr_crc_key("bad") is None
    assert _parse_cr_crc_channel_id("bad") is None

    assert _normalise_cr_crc_auxiliary_target({"auxiliary_qubit": "Q1"}) == "Q1"
    assert _normalise_cr_crc_auxiliary_target({"auxiliary_qubit": "bad"}) is None
    assert _normalise_cr_crc_auxiliary_target({"auxiliary_qubit": 1}) is None


def test_iter_cr_crc_pulse_views_and_normalized_entries_skip_non_cr_shapes():
    dto = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"].update(
            {
                "weird": "bad",
                "Q2.cross_resonance": {"pulse_channel": {"id": 7}},
                "Q3.cross_resonance": {
                    "pulse_channel": {
                        "id": "Q0.Q2.cross_resonance",
                        "physical_channel": {"id": "p_q0"},
                    }
                },
            }
        )
    )

    views = list(_iter_cr_crc_pulse_views(dto))
    assert any(pulse_key == "Q1.cross_resonance" for _, pulse_key, _, _ in views)

    normalized = list(_iter_normalized_cr_crc_entries(dto))
    # only valid, consistent CR/CRC mappings should survive normalization
    assert all(entry["source_id"] == "Q0" for entry in normalized)
    assert all(entry["target_id"] == "Q1" for entry in normalized)


def test_validate_coupling_indices_success_and_error_path():
    _validate_coupling_indices([{"direction": [0, 1]}], {0, 1})

    with pytest.raises(SourceConsistencyError, match="unknown qubit indices"):
        _validate_coupling_indices([{"direction": [0, 2]}], {0, 1})


def test_validate_cr_crc_counterparts_and_auxiliary_target_branches():
    dto_missing = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"].pop(
            "Q1.cross_resonance_cancellation"
        )
    )
    with pytest.raises(SourceConsistencyError, match="missing counterpart"):
        _validate_cr_crc_counterparts(dto_missing)

    dto_aux_bad = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["Q1.cross_resonance"][
            "pulse_channel"
        ].__setitem__("auxiliary_qubit", "Q2")
    )
    with pytest.raises(SourceConsistencyError, match="auxiliary target does not match"):
        _validate_cr_crc_auxiliary_targets(dto_aux_bad)


def test_validate_cr_crc_matches_coupling_graph_skips_unresolvable_entries_and_fails_when_missing():
    dto = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"]["Q1.cross_resonance"][
            "pulse_channel"
        ].__setitem__("id", "Q0.QX.cross_resonance")
    )
    # Invalid CR id drops mapping and should trigger missing-edge failure.
    with pytest.raises(
        SourceConsistencyError, match="missing CR/CRC pulse-channel mappings"
    ):
        _validate_cr_crc_matches_coupling_graph(dto)


def test_validate_cr_crc_channel_mapping_keys_success_case():
    dto = _make_dto()
    _validate_cr_crc_channel_mapping_keys(dto)


def test_coupling_validators_cover_remaining_skip_and_no_error_paths(caplog):
    dto = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"].update(
            {
                "Q2.cross_resonance": {
                    "pulse_channel": {
                        "id": 123,
                        "physical_channel": {"id": "p_q0"},
                    }
                },
                "Q3.cross_resonance": {
                    "pulse_channel": {
                        "id": "bad.id",
                        "physical_channel": {"id": "p_q0"},
                    }
                },
                "Q4.cross_resonance": {
                    "pulse_channel": {
                        "id": "Q9.Q4.cross_resonance",
                        "physical_channel": {"id": "p_q0"},
                    }
                },
            }
        )
    )

    # Non-dict entries and malformed CR metadata are skipped by warning/normalization helpers.
    from qat.experimental.system_data.materialisers.purr.validators.couplings import (
        _warn_missing_coupling_quality,
    )

    _warn_missing_coupling_quality(["bad", {"direction": [0]}, {"direction": [0, 1]}])
    assert any("missing quality fidelity" in rec.message for rec in caplog.records)

    # Extra malformed CR mappings do not crash normalization/auxiliary validation.
    _validate_cr_crc_auxiliary_targets(dto)


def test_coupling_validator_remaining_continue_paths():
    dto = _mutated_dto(
        lambda p: p["quantum_devices"]["Q0"]["pulse_channels"].update(
            {
                "Q1.cross_resonance.extra": {
                    "pulse_channel": {
                        "id": "Q0.Q1.cross_resonance",
                        "physical_channel": {"id": "p_q0"},
                    }
                },
                "Q5.cross_resonance": {"pulse_channel": "bad"},
                "Q5.cross_resonance_cancellation": {"pulse_channel": "bad"},
            }
        )
    )

    _validate_cr_crc_counterparts(dto)

    dto_unknown_index = _mutated_dto(lambda p: p["quantum_devices"]["Q1"].pop("index"))
    with pytest.raises(
        SourceConsistencyError, match="missing CR/CRC pulse-channel mappings"
    ):
        _validate_cr_crc_matches_coupling_graph(dto_unknown_index)
