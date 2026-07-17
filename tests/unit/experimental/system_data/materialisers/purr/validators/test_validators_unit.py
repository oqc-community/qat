# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.materialisers.errors import (
    SourceConsistencyError,
    SourceValidationError,
)
from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.system_data.materialisers.purr.validators.couplings import (
    _validate_cr_crc_channel_mapping_keys,
    _validate_cr_crc_counterparts,
)
from qat.experimental.system_data.materialisers.purr.validators.postprocess import (
    _validate_post_process_method,
    _validate_readout_mitigation,
)
from qat.experimental.system_data.materialisers.purr.validators.signal_paths import (
    _validate_basebands,
    _validate_physical_channels,
    _validate_pulse_channel_references,
    _validate_repeat_limit,
    _validate_top_level_collections,
)
from qat.experimental.system_data.materialisers.purr.validators.waveforms import (
    _validate_acquire_sync_field,
    _validate_measure_acquire_payloads,
    _validate_waveform_numeric_fields,
    _validate_waveform_payloads,
)


def _make_dto(**overrides):
    payload = {
        "calibration_id": "cal",
        "supported_acquire_modes": ["integrator"],
        "default_acquire_mode": "integrator",
        "repeat_limit": 10,
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
                            "scale": 1.0,
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
                },
                "pulse_hw_x_pi_2": {"width": 20e-9, "rise": 5e-9, "amp": 0.2},
                "pulse_hw_x_pi": {"width": 40e-9, "rise": 10e-9, "amp": 0.4},
                "pulse_measure": {"width": 100e-9, "amp": 0.8},
                "measure_acquire": {
                    "delay": 10e-9,
                    "width": 40e-9,
                    "weights": [1.0, 0.0],
                    "sync": True,
                },
                "post_process_method": {
                    "method": "max_likelihood",
                    "states": {
                        "0": {"location": 1 + 0j, "label": "g"},
                        "1": {"location": -1 + 0j, "label": "e"},
                    },
                    "noise_est": 1.0,
                    "p_min": 0.0,
                    "transform": [[1.0, 0.0], [0.0, 1.0]],
                    "offset": [0.0, 0.0],
                },
            },
            "Q1": {
                "id": "Q1",
                "index": 1,
                "pulse_channels": {},
            },
        },
        "pulse_channels": {
            "ch_global": {"physical_channel": "p_q0"},
        },
        "physical_channels": {
            "p_q0": {
                "id": "p_q0",
                "sample_time": 1e-9,
                "block_size": 4,
                "acquire_allowed": False,
            }
        },
        "basebands": {"bb0": {"frequency": 5e9}},
        "qubit_direction_couplings": [{"direction": [0, 1], "quality": 0.9}],
        "error_mitigation": {
            "readout_mitigation": {
                "linear": {
                    "0": {
                        "0|0": 0.99,
                        "1|0": 0.01,
                        "0|1": 0.02,
                        "1|1": 0.98,
                    }
                }
            }
        },
    }
    payload.update(overrides)
    return PurrIngressV010.model_validate(payload)


def test_signal_path_validators_accept_valid_payload_shapes():
    dto = _make_dto()

    _validate_top_level_collections(
        quantum_devices=dto.quantum_devices,
        physical_channels=dto.physical_channels,
        basebands=dto.basebands,
    )
    _validate_repeat_limit(dto.repeat_limit)
    _validate_physical_channels(dto.physical_channels)
    _validate_basebands(dto.basebands)
    _validate_pulse_channel_references(
        pulse_channels=dto.pulse_channels,
        physical_channel_ids=frozenset(dto.physical_channels.keys()),
    )


def test_signal_path_validators_reject_invalid_repeat_limit_and_reference():
    with pytest.raises(SourceValidationError, match="repeat_limit"):
        _validate_repeat_limit(0)

    with pytest.raises(SourceValidationError, match="must be a mapping"):
        _validate_pulse_channel_references(
            pulse_channels={"a": "bad"},
            physical_channel_ids=frozenset({"p0"}),
        )

    with pytest.raises(SourceConsistencyError, match="unknown physical channel"):
        _validate_pulse_channel_references(
            pulse_channels={"a": {"physical_channel": "missing"}},
            physical_channel_ids=frozenset({"p0"}),
        )


def test_waveform_validators_accept_valid_shapes_and_reject_invalid_sync():
    dto = _make_dto()
    _validate_waveform_payloads(dto)
    _validate_measure_acquire_payloads(dto)
    _validate_waveform_numeric_fields(dto)
    _validate_acquire_sync_field(dto)

    bad_sync = _make_dto()
    bad_sync.quantum_devices["Q0"]["measure_acquire"]["sync"] = "yes"
    with pytest.raises(SourceValidationError, match="Acquire sync"):
        _validate_acquire_sync_field(bad_sync)


def test_postprocess_validators_cover_method_and_mitigation_paths():
    dto = _make_dto()
    _validate_post_process_method(dto)
    _validate_readout_mitigation(dto.error_mitigation, {0, 1})

    bad = _make_dto()
    bad.quantum_devices["Q0"]["post_process_method"]["method"] = "unknown"
    with pytest.raises(SourceValidationError, match="supported method names"):
        _validate_post_process_method(bad)

    bad_map = _make_dto()
    bad_map.error_mitigation["readout_mitigation"]["linear"]["0"]["0|0"] = 2.0
    with pytest.raises(SourceValidationError, match=r"must lie in \[0, 1\]"):
        _validate_readout_mitigation(bad_map.error_mitigation, {0, 1})


def test_coupling_validators_cover_counterpart_and_channel_id_consistency():
    dto = _make_dto()
    _validate_cr_crc_channel_mapping_keys(dto)
    _validate_cr_crc_counterparts(dto)

    bad_counterpart = _make_dto()
    del bad_counterpart.quantum_devices["Q0"]["pulse_channels"][
        "Q1.cross_resonance_cancellation"
    ]
    with pytest.raises(SourceConsistencyError, match="missing counterpart"):
        _validate_cr_crc_counterparts(bad_counterpart)

    bad_key = _make_dto()
    bad_key.quantum_devices["Q0"]["pulse_channels"]["Q1.cross_resonance"]["pulse_channel"][
        "id"
    ] = "Q0.Q2.cross_resonance"
    with pytest.raises(SourceConsistencyError, match="does not match"):
        _validate_cr_crc_channel_mapping_keys(bad_key)
